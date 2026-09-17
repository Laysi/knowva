use super::report::{
    ArtifactIdentity, CapabilityRetryPolicies, CitationPolicy, FactMatchPolicy, GitRevision,
    LockedModelIds, LockedRunConfig, ManifestFailure, OperationalStageTotals, OperationalSummary,
    ProviderSeedSupport, REPORT_SCHEMA_VERSION, RetryPolicy, RunManifest, ThresholdSnapshot,
    TopNConfig, build_ab_report, load_threshold_snapshot, write_report_bundle,
};
use super::schema::{
    DocumentFixture, EvaluationDataset, EvaluationDatasetSnapshot, QueryFixture,
    load_and_validate_dataset_snapshot,
};
use super::scoring::{
    ContextSource, EvaluationMode, ExpectedSource, FallbackTelemetry, ManualReview,
    ManualReviewStatus, NegativeAnswerPolicy, QueryExpectation, QueryRunInput,
    RetrievalCandidate as ScoringCandidate, RetrievalStage, RetrievalTrace, RunFailure,
    ScoredQuery, ScoringConfig, StageTelemetry, StageTrace, score_query,
};
use crate::cloud_models::{
    ChatMessage, CloudModelClient, CloudModelConfig, CloudRequestTelemetry, CloudRetryPolicy,
};
use crate::ingest::{DocumentLocks, EvaluationIngestResult, ingest_evaluation_bytes};
use crate::page_artifacts::ArtifactConfig;
use crate::rag_context::{MAX_RAG_SOURCES, RAG_SYSTEM_PROMPT, SourceRef, build_rag_context};
use crate::search::{
    DEFAULT_BM25_TOP_K, DEFAULT_DENSE_TOP_K, ObservedAbRetrieval, ObservedFailureStage,
    ObservedModeRetrieval, ObservedRetrievalFailure, ObservedRetrievalStage, RRF_K,
    RetrievalCandidate, retrieve_ab_observed,
};
use crate::storage::{Storage, StorageConfig, ensure_initialization_lock};
use crate::{AppState, DEFAULT_DATABASE_URL, DEFAULT_FILES_DIR};
use anyhow::{Context, bail};
use chrono::{SecondsFormat, Utc};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::OsString;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use url::Url;
use uuid::Uuid;

const DEFAULT_MANIFEST: &str = "eval/fixtures/manifest.json";
const DEFAULT_OUTPUT: &str = "eval/results";
const DEFAULT_SEED: u64 = 42;
const DEFAULT_CONCURRENCY: usize = 1;
const PROMPT_VERSION: &str = "gemma4-rag-v1";
const MAX_ANSWER_CHARS: usize = 4_000;
const EVALUATION_DATABASE_PREFIX: &str = "knowva_eval_";
const EVALUATION_DATABASE_MARKER: &str = "knowva-evaluation-only:v1";
const GITHUB_COMMENT_API_BASE: &str = "https://api.github.com/repos/Laysi/knowva/issues/comments/";
const THRESHOLD_APPROVER_LOGIN: &str = "Ruisi-Lu";

#[cfg(test)]
static CLOUD_RUN_INVOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
static CLOUD_BOOTSTRAP_INVOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunMode {
    DryRun,
    Cloud,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOptions {
    mode: RunMode,
    manifest: PathBuf,
    output: PathBuf,
    thresholds: Option<PathBuf>,
    documents: BTreeSet<String>,
    queries: BTreeSet<String>,
    seed: u64,
    concurrency: usize,
}

struct CloudPreflight {
    thresholds: ThresholdSnapshot,
    database_url: String,
    approved_files_root: PathBuf,
}

trait ThresholdApprovalVerifier: Send + Sync {
    fn verify<'a>(&'a self, snapshot: &'a ThresholdSnapshot) -> BoxFuture<'a, anyhow::Result<()>>;
}

struct GitHubThresholdApprovalVerifier {
    http: reqwest::Client,
}

#[derive(Deserialize)]
struct GitHubIssueComment {
    html_url: String,
    body: String,
    user: GitHubCommentUser,
    author_association: String,
}

#[derive(Deserialize)]
struct GitHubCommentUser {
    login: String,
}

impl GitHubThresholdApprovalVerifier {
    fn new() -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|_| anyhow::anyhow!("threshold approval verifier could not be initialized"))?;
        Ok(Self { http })
    }
}

impl ThresholdApprovalVerifier for GitHubThresholdApprovalVerifier {
    fn verify<'a>(&'a self, snapshot: &'a ThresholdSnapshot) -> BoxFuture<'a, anyhow::Result<()>> {
        Box::pin(async move {
            let comment_id = threshold_comment_id(&snapshot.config.issue_comment_url)?;
            let endpoint = format!("{GITHUB_COMMENT_API_BASE}{comment_id}");
            let response = self
                .http
                .get(endpoint)
                .header(reqwest::header::USER_AGENT, "knowva-evaluation-runner")
                .header(reqwest::header::ACCEPT, "application/vnd.github+json")
                .send()
                .await
                .map_err(|_| anyhow::anyhow!("threshold approval could not be verified"))?;
            if !response.status().is_success() {
                bail!("threshold approval could not be verified");
            }
            let comment = response
                .json::<GitHubIssueComment>()
                .await
                .map_err(|_| anyhow::anyhow!("threshold approval response was invalid"))?;
            validate_threshold_approval(
                snapshot,
                &comment.html_url,
                &comment.body,
                &comment.user.login,
                &comment.author_association,
            )
        })
    }
}

fn threshold_comment_id(issue_comment_url: &str) -> anyhow::Result<&str> {
    issue_comment_url
        .strip_prefix("https://github.com/Laysi/knowva/issues/1#issuecomment-")
        .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .context("threshold approval URL is invalid")
}

fn validate_threshold_approval(
    snapshot: &ThresholdSnapshot,
    html_url: &str,
    body: &str,
    author_login: &str,
    author_association: &str,
) -> anyhow::Result<()> {
    threshold_comment_id(&snapshot.config.issue_comment_url)?;
    if html_url != snapshot.config.issue_comment_url {
        bail!("threshold approval comment identity did not match");
    }
    if author_login != THRESHOLD_APPROVER_LOGIN
        || !matches!(author_association, "OWNER" | "MEMBER" | "COLLABORATOR")
    {
        bail!("threshold approval author was not authorized");
    }
    let expected_marker = format!("threshold_sha256: {}", snapshot.identity.sha256);
    if !body.lines().any(|line| line.trim() == expected_marker) {
        bail!("threshold approval did not contain the locked snapshot hash");
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StableDocument {
    document_key: String,
    document_id: Uuid,
    filename: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StableUnit {
    document_key: String,
    document_id: Uuid,
    unit_id: String,
    filename: String,
    page: Option<u32>,
}

#[derive(Default)]
struct StableIndex {
    documents_by_key: HashMap<String, StableDocument>,
    units_by_database_id: HashMap<i64, StableUnit>,
    units_by_stable_key: HashMap<(String, String), StableUnit>,
}

#[derive(Debug, sqlx::FromRow)]
struct DatabaseUnitRow {
    id: i64,
    document_id: Uuid,
    kind: String,
    unit_index: i32,
    page_number: Option<i32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CloudRunSummary {
    run_id: String,
    report_json: String,
    report_csv: String,
    report_markdown: String,
    run_manifest: String,
    passed: bool,
}

struct ModeAnswer {
    answer: String,
    context_sources: BTreeMap<String, ContextSource>,
    latency_ms: u64,
    request_count: u32,
    retry_count: u32,
    failure: Option<RunFailure>,
}

#[derive(Default)]
struct OperationalAccumulator {
    ingest: OperationalStageTotals,
    query_embedding: OperationalStageTotals,
    rerank: OperationalStageTotals,
    generation: OperationalStageTotals,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            mode: RunMode::DryRun,
            manifest: PathBuf::from(DEFAULT_MANIFEST),
            output: PathBuf::from(DEFAULT_OUTPUT),
            thresholds: None,
            documents: BTreeSet::new(),
            queries: BTreeSet::new(),
            seed: DEFAULT_SEED,
            concurrency: DEFAULT_CONCURRENCY,
        }
    }
}

impl CliOptions {
    fn parse(arguments: impl IntoIterator<Item = String>) -> anyhow::Result<Self> {
        let mut options = Self::default();
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--dry-run" => options.mode = RunMode::DryRun,
                "--cloud" => options.mode = RunMode::Cloud,
                "--manifest" => {
                    options.manifest = PathBuf::from(required_value(&mut arguments, "--manifest")?);
                }
                "--output" => {
                    options.output = PathBuf::from(required_value(&mut arguments, "--output")?);
                }
                "--thresholds" => {
                    options.thresholds = Some(PathBuf::from(required_value(
                        &mut arguments,
                        "--thresholds",
                    )?));
                }
                "--document" => {
                    options
                        .documents
                        .insert(required_value(&mut arguments, "--document")?);
                }
                "--query" => {
                    options
                        .queries
                        .insert(required_value(&mut arguments, "--query")?);
                }
                "--seed" => {
                    options.seed = required_value(&mut arguments, "--seed")?
                        .parse()
                        .context("--seed must be an unsigned integer")?;
                }
                "--concurrency" => {
                    options.concurrency = required_value(&mut arguments, "--concurrency")?
                        .parse()
                        .context("--concurrency must be a positive integer")?;
                    if options.concurrency == 0 {
                        bail!("--concurrency must be a positive integer");
                    }
                }
                "--help" | "-h" => bail!(usage()),
                _ => bail!("unknown eval argument {argument:?}\n\n{}", usage()),
            }
        }
        Ok(options)
    }
}

fn required_value(
    arguments: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> anyhow::Result<String> {
    arguments
        .next()
        .filter(|value| !value.trim().is_empty() && !value.starts_with("--"))
        .with_context(|| format!("{option} requires a value"))
}

fn usage() -> &'static str {
    "usage: knowva-server eval [--dry-run|--cloud] [--manifest PATH] [--output PATH] \
     [--thresholds PATH] [--document KEY]... [--query ID]... [--seed N] \
     [--concurrency N]"
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DryRunSummary {
    status: &'static str,
    manifest: String,
    output: String,
    dataset_version: String,
    document_count: usize,
    query_count: usize,
    documents: Vec<String>,
    queries: Vec<String>,
    seed: u64,
    concurrency: usize,
}

struct Selection<'a> {
    documents: Vec<&'a DocumentFixture>,
    queries: Vec<&'a QueryFixture>,
}

fn select_dataset<'a>(
    dataset: &'a EvaluationDataset,
    options: &CliOptions,
) -> anyhow::Result<Selection<'a>> {
    let known_documents = dataset
        .documents
        .iter()
        .map(|document| document.document_key.as_str())
        .collect::<BTreeSet<_>>();
    let known_queries = dataset
        .queries
        .iter()
        .map(|query| query.query_id.as_str())
        .collect::<BTreeSet<_>>();
    for document in &options.documents {
        if !known_documents.contains(document.as_str()) {
            bail!("unknown document subset key {document:?}");
        }
    }
    for query in &options.queries {
        if !known_queries.contains(query.as_str()) {
            bail!("unknown query subset id {query:?}");
        }
    }

    let queries = dataset
        .queries
        .iter()
        .filter(|query| {
            (options.queries.is_empty() || options.queries.contains(&query.query_id))
                && (options.documents.is_empty()
                    || query
                        .document_scope
                        .iter()
                        .all(|key| options.documents.contains(key)))
        })
        .collect::<Vec<_>>();
    if queries.is_empty() {
        bail!("the requested subsets select no evaluation queries");
    }

    let required_documents = queries
        .iter()
        .flat_map(|query| query.document_scope.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    let documents = dataset
        .documents
        .iter()
        .filter(|document| required_documents.contains(document.document_key.as_str()))
        .collect::<Vec<_>>();

    Ok(Selection { documents, queries })
}

pub async fn run_cli(arguments: impl IntoIterator<Item = String>) -> anyhow::Result<()> {
    let options = CliOptions::parse(arguments)?;
    execute(options).await
}

async fn execute(options: CliOptions) -> anyhow::Result<()> {
    let snapshot =
        load_and_validate_dataset_snapshot(&options.manifest).map_err(format_dataset_errors)?;
    match options.mode {
        RunMode::DryRun => {
            let selection = select_dataset(&snapshot.dataset, &options)?;
            let summary = DryRunSummary {
                status: "valid",
                manifest: display_path(&options.manifest),
                output: display_path(&options.output),
                dataset_version: snapshot.dataset.version.clone(),
                document_count: selection.documents.len(),
                query_count: selection.queries.len(),
                documents: selection
                    .documents
                    .iter()
                    .map(|document| document.document_key.clone())
                    .collect(),
                queries: selection
                    .queries
                    .iter()
                    .map(|query| query.query_id.clone())
                    .collect(),
                seed: options.seed,
                concurrency: options.concurrency,
            };
            println!("{}", serde_json::to_string_pretty(&summary)?);
            Ok(())
        }
        RunMode::Cloud => {
            // Validate requested subsets before any cloud preflight or mutation.
            select_dataset(&snapshot.dataset, &options)?;
            run_cloud(options, snapshot).await
        }
    }
}

fn format_dataset_errors(errors: Vec<super::schema::ValidationError>) -> anyhow::Error {
    let message = errors
        .into_iter()
        .map(|error| error.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    anyhow::anyhow!("fixture validation failed:\n{message}")
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

async fn run_cloud(options: CliOptions, snapshot: EvaluationDatasetSnapshot) -> anyhow::Result<()> {
    #[cfg(test)]
    CLOUD_RUN_INVOCATIONS.fetch_add(1, Ordering::SeqCst);
    dotenvy::dotenv().ok();
    let verifier = GitHubThresholdApprovalVerifier::new()?;
    run_cloud_with_lookup_and_verifier(
        options,
        snapshot,
        |name| std::env::var(name).ok(),
        &verifier,
    )
    .await
}

async fn run_cloud_with_lookup_and_verifier(
    options: CliOptions,
    snapshot: EvaluationDatasetSnapshot,
    mut get_env: impl FnMut(&str) -> Option<String>,
    verifier: &impl ThresholdApprovalVerifier,
) -> anyhow::Result<()> {
    let current_directory = std::env::current_dir()
        .map_err(|_| anyhow::anyhow!("evaluation working directory is unavailable"))?;
    let preflight = validate_cloud_preflight(&options, &current_directory, &mut get_env)?;
    verifier.verify(&preflight.thresholds).await?;

    #[cfg(test)]
    CLOUD_BOOTSTRAP_INVOCATIONS.fetch_add(1, Ordering::SeqCst);

    bootstrap_and_ingest(&options, &snapshot, preflight, &mut get_env).await
}

fn validate_cloud_preflight(
    options: &CliOptions,
    current_directory: &Path,
    get_env: &mut impl FnMut(&str) -> Option<String>,
) -> anyhow::Result<CloudPreflight> {
    let threshold_path = options
        .thresholds
        .as_deref()
        .context("--thresholds is required for --cloud")?;
    let thresholds = load_threshold_snapshot(threshold_path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if options.concurrency != 1 {
        bail!("--cloud currently requires --concurrency 1 for exact telemetry attribution");
    }

    let database_url = required_environment(get_env, "EVAL_DATABASE_URL")?;
    let files_root = required_environment(get_env, "EVAL_FILES_DIR")?;
    let production_database_url = get_env("DATABASE_URL")
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_DATABASE_URL.to_owned());
    let evaluation_database_url = normalized_database_url(&database_url, "EVAL_DATABASE_URL")?;
    let production_database_url =
        normalized_database_url(&production_database_url, "DATABASE_URL")?;
    let default_database_url = normalized_database_url(DEFAULT_DATABASE_URL, "DATABASE_URL")?;
    if evaluation_database_url == production_database_url
        || evaluation_database_url == default_database_url
    {
        bail!("EVAL_DATABASE_URL must use a database isolated from production");
    }

    let production_files_root = get_env("FILES_DIR")
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_FILES_DIR.to_owned());
    let files_root = resolve_path_without_creating(Path::new(&files_root), current_directory)?;
    let production_files_root =
        resolve_path_without_creating(Path::new(&production_files_root), current_directory)?;
    let default_files_root =
        resolve_path_without_creating(Path::new(DEFAULT_FILES_DIR), current_directory)?;
    if paths_overlap(&files_root, &production_files_root)
        || paths_overlap(&files_root, &default_files_root)
    {
        bail!("EVAL_FILES_DIR must not equal, contain, or be contained by FILES_DIR");
    }
    if files_root == current_directory || files_root.parent().is_none() {
        bail!("EVAL_FILES_DIR must be a dedicated child directory");
    }

    Ok(CloudPreflight {
        thresholds,
        database_url,
        approved_files_root: files_root,
    })
}

fn required_environment(
    get_env: &mut impl FnMut(&str) -> Option<String>,
    name: &'static str,
) -> anyhow::Result<String> {
    get_env(name)
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .with_context(|| format!("{name} is required for --cloud"))
}

fn normalized_database_url(value: &str, name: &'static str) -> anyhow::Result<Url> {
    let url = Url::parse(value).with_context(|| format!("{name} must be a valid database URL"))?;
    if !matches!(url.scheme(), "postgres" | "postgresql") || url.fragment().is_some() {
        bail!("{name} must be a PostgreSQL database URL");
    }
    Ok(url)
}

fn resolve_path_without_creating(path: &Path, current_directory: &Path) -> anyhow::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        current_directory.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    bail!("evaluation file path could not be normalized");
                }
            }
            Component::Normal(value) => normalized.push(value),
        }
    }

    let mut existing = normalized.clone();
    let mut missing = Vec::<OsString>::new();
    while !existing.exists() {
        let name = existing
            .file_name()
            .map(ToOwned::to_owned)
            .ok_or_else(|| anyhow::anyhow!("evaluation file path has no existing ancestor"))?;
        missing.push(name);
        if !existing.pop() {
            bail!("evaluation file path has no existing ancestor");
        }
    }
    let mut resolved = existing
        .canonicalize()
        .map_err(|_| anyhow::anyhow!("evaluation file path could not be resolved"))?;
    for component in missing.into_iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left == right || left.starts_with(right) || right.starts_with(left)
}

#[cfg(target_os = "linux")]
fn reject_existing_symlink_components(path: &Path) -> anyhow::Result<()> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                bail!("EVAL_FILES_DIR must not contain symbolic-link components");
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(_) => bail!("EVAL_FILES_DIR components could not be inspected"),
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn enforce_private_directory_mode(directory: &fs::File) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mode = directory
        .metadata()
        .map_err(|_| anyhow::anyhow!("EVAL_FILES_DIR permissions could not be inspected"))?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        bail!("EVAL_FILES_DIR must not grant group or world permissions");
    }
    Ok(())
}

#[derive(Debug)]
struct RunFilesRoot {
    #[cfg(target_os = "linux")]
    _base_directory: fs::File,
    #[cfg(target_os = "linux")]
    run_directory: fs::File,
    run_id: String,
}

impl RunFilesRoot {
    #[cfg(target_os = "linux")]
    fn path(&self) -> PathBuf {
        use std::os::fd::AsRawFd;

        PathBuf::from(format!("/proc/self/fd/{}", self.run_directory.as_raw_fd()))
    }

    // Keeps non-Linux builds compiling; prepare_run_files_root refuses there.
    #[cfg(not(target_os = "linux"))]
    fn path(&self) -> PathBuf {
        unreachable!("evaluation run roots are only created on Linux")
    }

    fn run_id(&self) -> &str {
        &self.run_id
    }
}

#[cfg(target_os = "linux")]
fn open_directory(path: &Path) -> anyhow::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;

    fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| anyhow::anyhow!("EVAL_FILES_DIR could not be opened securely"))
}

#[cfg(target_os = "linux")]
fn mkdirat_private(base: &fs::File, name: &std::ffi::CStr) -> anyhow::Result<()> {
    use std::os::fd::AsRawFd;

    // SAFETY: base is a live directory fd and name is a NUL-terminated single
    // path component owned for the duration of the call.
    let result = unsafe { libc::mkdirat(base.as_raw_fd(), name.as_ptr(), 0o700) };
    if result == -1 {
        bail!("evaluation run file directory could not be created");
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn openat_directory(base: &fs::File, name: &std::ffi::CStr) -> anyhow::Result<fs::File> {
    use std::os::fd::{AsRawFd, FromRawFd};

    // SAFETY: base is a live directory fd and name is a NUL-terminated single
    // component. On success the returned fd is uniquely owned by File.
    let fd = unsafe {
        libc::openat(
            base.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd == -1 {
        bail!("evaluation run file directory could not be opened securely");
    }
    // SAFETY: openat returned a new owned descriptor and ownership is
    // transferred exactly once to File.
    Ok(unsafe { fs::File::from_raw_fd(fd) })
}

#[cfg(target_os = "linux")]
fn prepare_run_files_root_with_hook(
    approved_base: &Path,
    after_base_open: impl FnOnce(),
) -> anyhow::Result<RunFilesRoot> {
    reject_existing_symlink_components(approved_base)?;
    let created = match fs::symlink_metadata(approved_base) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                bail!("EVAL_FILES_DIR must be a directory");
            }
            false
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            use std::os::unix::fs::DirBuilderExt;

            let parent = approved_base
                .parent()
                .context("EVAL_FILES_DIR must have an existing parent directory")?;
            let parent_metadata = fs::symlink_metadata(parent)
                .map_err(|_| anyhow::anyhow!("EVAL_FILES_DIR parent directory must exist"))?;
            if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
                bail!("EVAL_FILES_DIR parent must be a real existing directory");
            }
            let mut builder = fs::DirBuilder::new();
            builder.mode(0o700);
            builder
                .create(approved_base)
                .map_err(|_| anyhow::anyhow!("EVAL_FILES_DIR could not be created"))?;
            true
        }
        Err(_) => bail!("EVAL_FILES_DIR could not be inspected"),
    };

    reject_existing_symlink_components(approved_base)?;
    let metadata = fs::symlink_metadata(approved_base)
        .map_err(|_| anyhow::anyhow!("EVAL_FILES_DIR could not be inspected"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("EVAL_FILES_DIR must be a real directory");
    }
    let base_directory = open_directory(approved_base)?;
    let base_fd_path = PathBuf::from(format!(
        "/proc/self/fd/{}",
        std::os::fd::AsRawFd::as_raw_fd(&base_directory)
    ));
    let canonical_base = base_fd_path
        .canonicalize()
        .map_err(|_| anyhow::anyhow!("EVAL_FILES_DIR could not be resolved"))?;
    if canonical_base != approved_base {
        bail!("EVAL_FILES_DIR changed after preflight approval");
    }
    if created {
        use std::os::unix::fs::PermissionsExt;

        base_directory
            .set_permissions(fs::Permissions::from_mode(0o700))
            .map_err(|_| anyhow::anyhow!("EVAL_FILES_DIR permissions could not be secured"))?;
    }
    enforce_private_directory_mode(&base_directory)?;

    after_base_open();

    let mut entries = fs::read_dir(&base_fd_path)
        .map_err(|_| anyhow::anyhow!("EVAL_FILES_DIR could not be inspected"))?;
    if entries
        .next()
        .transpose()
        .map_err(|_| anyhow::anyhow!("EVAL_FILES_DIR could not be inspected"))?
        .is_some()
    {
        bail!("EVAL_FILES_DIR must be empty before a cloud evaluation");
    }

    let run_id = format!("run-{}", Uuid::new_v4());
    let run_name = std::ffi::CString::new(run_id.as_str())
        .map_err(|_| anyhow::anyhow!("evaluation run id is invalid"))?;
    mkdirat_private(&base_directory, &run_name)?;
    let run_directory = openat_directory(&base_directory, &run_name)?;
    use std::os::unix::fs::PermissionsExt;
    run_directory
        .set_permissions(fs::Permissions::from_mode(0o700))
        .map_err(|_| {
            anyhow::anyhow!("evaluation run file directory permissions could not be secured")
        })?;

    Ok(RunFilesRoot {
        _base_directory: base_directory,
        run_directory,
        run_id,
    })
}

#[cfg(target_os = "linux")]
fn prepare_run_files_root(approved_base: &Path) -> anyhow::Result<RunFilesRoot> {
    prepare_run_files_root_with_hook(approved_base, || {})
}

#[cfg(not(target_os = "linux"))]
fn prepare_run_files_root(_approved_base: &Path) -> anyhow::Result<RunFilesRoot> {
    bail!("secure EVAL_FILES_DIR dirfd handling is supported only on Linux")
}

struct FailureMarkerGuard {
    run_root: PathBuf,
    run_id: String,
    armed: bool,
}

impl FailureMarkerGuard {
    fn new(run_root: PathBuf, run_id: String) -> Self {
        Self {
            run_root,
            run_id,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for FailureMarkerGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let marker = serde_json::json!({
            "schemaVersion": "1.0",
            "runId": self.run_id,
            "status": "failed",
            "safeMessage": "evaluation did not complete; isolated database and files were preserved"
        });
        if let Ok(bytes) = serde_json::to_vec_pretty(&marker) {
            let _ = fs::write(self.run_root.join("evaluation-failure.json"), bytes);
        }
    }
}

async fn validate_evaluation_database(pool: &sqlx::PgPool) -> anyhow::Result<()> {
    let identity: (String, String, Option<String>) = sqlx::query_as(
        "SELECT current_database(), current_user, \
                shobj_description(oid, 'pg_database') \
         FROM pg_database WHERE datname = current_database()",
    )
    .fetch_one(pool)
    .await
    .map_err(|_| anyhow::anyhow!("evaluation database identity check failed"))?;
    let (database_name, current_user, database_marker) = identity;

    let documents_relation: Option<String> =
        sqlx::query_scalar("SELECT to_regclass('public.documents')::text")
            .fetch_one(pool)
            .await
            .map_err(|_| anyhow::anyhow!("evaluation database isolation check failed"))?;
    let has_documents = if documents_relation.is_some() {
        sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM public.documents LIMIT 1)")
            .fetch_one(pool)
            .await
            .map_err(|_| anyhow::anyhow!("evaluation database isolation check failed"))?
    } else {
        false
    };
    validate_database_identity_values(
        &database_name,
        &current_user,
        database_marker.as_deref(),
        has_documents,
    )
}

fn validate_database_identity_values(
    database_name: &str,
    current_user: &str,
    database_marker: Option<&str>,
    has_documents: bool,
) -> anyhow::Result<()> {
    if !database_name.starts_with(EVALUATION_DATABASE_PREFIX)
        || database_name.len() == EVALUATION_DATABASE_PREFIX.len()
    {
        bail!("evaluation database name must use the dedicated knowva_eval_ prefix");
    }
    if current_user.trim().is_empty() {
        bail!("evaluation database user identity is invalid");
    }
    if database_marker != Some(EVALUATION_DATABASE_MARKER) {
        bail!("evaluation database is missing the required evaluation-only marker");
    }
    if has_documents {
        bail!("evaluation database must contain zero documents before a run");
    }
    Ok(())
}

async fn build_stable_index(
    pool: &sqlx::PgPool,
    documents: &[&DocumentFixture],
    ingested: &[EvaluationIngestResult],
) -> anyhow::Result<StableIndex> {
    if documents.len() != ingested.len() {
        bail!("evaluation document mapping count mismatch");
    }
    let mut index = StableIndex::default();
    for (document, result) in documents.iter().zip(ingested) {
        let filename = Path::new(&document.relative_path)
            .file_name()
            .and_then(|value| value.to_str())
            .filter(|value| !value.trim().is_empty())
            .context("evaluation fixture filename is invalid")?
            .to_owned();
        let stable_document = StableDocument {
            document_key: document.document_key.clone(),
            document_id: result.document_id,
            filename,
        };
        if index
            .documents_by_key
            .insert(document.document_key.clone(), stable_document.clone())
            .is_some()
        {
            bail!("duplicate stable document identity");
        }

        let rows = sqlx::query_as::<_, DatabaseUnitRow>(
            "SELECT id, document_id, kind, unit_index, page_number \
             FROM document_units WHERE document_id = $1 ORDER BY unit_index, id",
        )
        .bind(result.document_id)
        .fetch_all(pool)
        .await
        .map_err(|_| anyhow::anyhow!("evaluation unit mapping query failed"))?;
        if rows.len() != document.expected.units.len() {
            bail!("evaluation unit mapping count mismatch");
        }
        for row in rows {
            let stable = map_database_unit(document, &stable_document, &row)?;
            if index
                .units_by_database_id
                .insert(row.id, stable.clone())
                .is_some()
                || index
                    .units_by_stable_key
                    .insert(
                        (stable.document_key.clone(), stable.unit_id.clone()),
                        stable,
                    )
                    .is_some()
            {
                bail!("duplicate stable unit identity");
            }
        }
    }
    Ok(index)
}

fn map_database_unit(
    document: &DocumentFixture,
    stable_document: &StableDocument,
    row: &DatabaseUnitRow,
) -> anyhow::Result<StableUnit> {
    if row.document_id != stable_document.document_id || row.unit_index < 0 {
        bail!("evaluation unit database identity mismatch");
    }
    let (expected, page) = match row.kind.as_str() {
        "page" => {
            let page = row
                .page_number
                .filter(|page| *page > 0)
                .context("evaluation page unit has invalid page number")?
                as u32;
            let expected = document
                .expected
                .units
                .iter()
                .find(|unit| unit.page == Some(page))
                .context("evaluation page is absent from stable inventory")?;
            (expected, Some(page))
        }
        "text" => {
            if row.page_number.is_some() {
                bail!("evaluation text unit must not have a page");
            }
            let expected = document
                .expected
                .units
                .get(row.unit_index as usize)
                .context("evaluation text unit is absent from stable inventory")?;
            if expected.page.is_some() {
                bail!("evaluation text inventory must use null page");
            }
            (expected, None)
        }
        _ => bail!("evaluation unit has unsupported kind"),
    };
    Ok(StableUnit {
        document_key: stable_document.document_key.clone(),
        document_id: stable_document.document_id,
        unit_id: expected.unit_id.clone(),
        filename: stable_document.filename.clone(),
        page,
    })
}

fn query_expectation(
    query: &QueryFixture,
    stable: &StableIndex,
) -> anyhow::Result<QueryExpectation> {
    let expected_sources = query
        .expected_sources
        .iter()
        .map(|expected| {
            let unit = stable
                .units_by_stable_key
                .get(&(expected.document_key.clone(), expected.unit_id.clone()))
                .context("golden source is absent from ingested stable mapping")?;
            if unit.page != expected.page {
                bail!("golden source page differs from ingested stable mapping");
            }
            Ok(ExpectedSource {
                document_key: unit.document_key.clone(),
                unit_id: unit.unit_id.clone(),
                filename: unit.filename.clone(),
                page: unit.page,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(QueryExpectation {
        answerable: query.answerable,
        expected_sources,
        required_facts: query.required_facts.clone(),
    })
}

fn scoring_candidate(
    candidate: &RetrievalCandidate,
    rank: usize,
    stage: RetrievalStage,
    stable: &StableIndex,
) -> anyhow::Result<ScoringCandidate> {
    let identity = stable
        .units_by_database_id
        .get(&candidate.unit_id)
        .context("retrieval candidate is absent from stable mapping")?;
    if identity.document_id != candidate.document_id {
        bail!("retrieval candidate document identity mismatch");
    }
    let score = match stage {
        RetrievalStage::Dense => candidate.dense_score,
        RetrievalStage::Bm25 => candidate.bm25_score,
        RetrievalStage::Rrf => Some(candidate.rrf_score),
        RetrievalStage::Rerank => candidate.rerank_score.or(Some(candidate.rrf_score)),
    }
    .context("retrieval candidate is missing its stage score")?;
    Ok(ScoringCandidate {
        document_key: identity.document_key.clone(),
        unit_id: identity.unit_id.clone(),
        filename: identity.filename.clone(),
        page: identity.page,
        rank: u32::try_from(rank).context("candidate rank exceeds report range")?,
        score,
    })
}

fn scoring_stage(
    observed: &ObservedRetrievalStage,
    stage: RetrievalStage,
    stable: &StableIndex,
    request_count: u32,
    retry_count: u32,
) -> anyhow::Result<StageTrace> {
    let candidates = observed
        .candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| scoring_candidate(candidate, index + 1, stage, stable))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(StageTrace {
        candidates,
        telemetry: StageTelemetry {
            latency_ms: observed.latency_ms,
            request_count,
            retry_count,
            fallback: observed.fallback.then(|| FallbackTelemetry {
                from: "rerank".to_owned(),
                to: "rrf".to_owned(),
                reason_code: "transient_rerank_failure".to_owned(),
            }),
        },
    })
}

fn scoring_traces(
    observed: &ObservedAbRetrieval,
    stable: &StableIndex,
    cloud_delta: CloudRequestTelemetry,
) -> anyhow::Result<(RetrievalTrace, RetrievalTrace)> {
    let embedding_requests = bounded_u32(cloud_delta.embedding_attempts);
    let embedding_retries = embedding_requests.saturating_sub(1);
    let rerank_requests = bounded_u32(cloud_delta.rerank_attempts);
    let rerank_retries = rerank_requests.saturating_sub(observed.mode_b.rerank.request_count);
    let a = mode_scoring_trace(
        &observed.mode_a,
        stable,
        embedding_requests,
        embedding_retries,
        0,
        0,
    )?;
    let b = mode_scoring_trace(
        &observed.mode_b,
        stable,
        0,
        0,
        rerank_requests,
        rerank_retries,
    )?;
    Ok((a, b))
}

fn mode_scoring_trace(
    observed: &ObservedModeRetrieval,
    stable: &StableIndex,
    dense_requests: u32,
    dense_retries: u32,
    rerank_requests: u32,
    rerank_retries: u32,
) -> anyhow::Result<RetrievalTrace> {
    Ok(RetrievalTrace {
        dense: scoring_stage(
            &observed.dense,
            RetrievalStage::Dense,
            stable,
            dense_requests,
            dense_retries,
        )?,
        bm25: scoring_stage(
            &observed.bm25,
            RetrievalStage::Bm25,
            stable,
            observed.bm25.request_count,
            0,
        )?,
        rrf: scoring_stage(&observed.rrf, RetrievalStage::Rrf, stable, 0, 0)?,
        rerank: scoring_stage(
            &observed.rerank,
            RetrievalStage::Rerank,
            stable,
            rerank_requests,
            rerank_retries,
        )?,
    })
}

fn empty_trace(
    dense_latency_ms: u64,
    embedding_attempts: u64,
    rerank_attempts: u64,
) -> RetrievalTrace {
    let empty = |latency_ms, request_count, retry_count| StageTrace {
        candidates: Vec::new(),
        telemetry: StageTelemetry {
            latency_ms,
            request_count,
            retry_count,
            fallback: None,
        },
    };
    let embedding_requests = bounded_u32(embedding_attempts);
    let rerank_requests = bounded_u32(rerank_attempts);
    RetrievalTrace {
        dense: empty(
            dense_latency_ms,
            embedding_requests,
            embedding_requests.saturating_sub(1),
        ),
        bm25: empty(0, 0, 0),
        rrf: empty(0, 0, 0),
        rerank: empty(0, rerank_requests, rerank_requests.saturating_sub(1)),
    }
}

fn bounded_u32(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn report_retry_policy(policy: CloudRetryPolicy) -> RetryPolicy {
    RetryPolicy {
        max_attempts: policy.max_attempts,
        base_backoff_ms: policy.base_backoff_ms,
        max_backoff_ms: policy.max_backoff_ms,
        jitter: policy.jitter,
    }
}

fn context_sources(
    sources: &[SourceRef],
    stable: &StableIndex,
) -> anyhow::Result<BTreeMap<String, ContextSource>> {
    sources
        .iter()
        .map(|source| {
            let identity = stable
                .units_by_database_id
                .get(&source.unit_id)
                .context("RAG source is absent from stable mapping")?;
            if identity.document_id != source.document_id {
                bail!("RAG source document identity mismatch");
            }
            Ok((
                source.source_id.clone(),
                ContextSource {
                    document_key: identity.document_key.clone(),
                    unit_id: identity.unit_id.clone(),
                    filename: identity.filename.clone(),
                    page: identity.page,
                },
            ))
        })
        .collect()
}

async fn answer_mode(
    state: &AppState,
    stable: &StableIndex,
    query: &str,
    candidates: &[RetrievalCandidate],
) -> ModeAnswer {
    let context = match build_rag_context(candidates, &state.storage, &[]).await {
        Ok(context) => context,
        Err(error) => return failed_answer("rag_context_failed", error.client_message(), false),
    };
    let system_prompt = RAG_SYSTEM_PROMPT;
    let context = match state.models.fit_rag_context_to_request(
        system_prompt,
        context,
        &[ChatMessage {
            role: "user".to_owned(),
            content: query.to_owned(),
        }],
    ) {
        Ok(context) => context,
        Err(error) => {
            return failed_answer(
                "rag_request_invalid",
                error.chat_client_message(),
                error.is_retryable(),
            );
        }
    };
    let sources = match context_sources(&context.sources, stable) {
        Ok(sources) => sources,
        Err(_) => {
            return failed_answer(
                "source_mapping_failed",
                "evaluation source mapping failed".to_owned(),
                false,
            );
        }
    };
    let before = state.models.telemetry_snapshot();
    let started = std::time::Instant::now();
    let result = state
        .models
        .answer_rag_once(system_prompt, context.content, query)
        .await;
    let latency_ms = started.elapsed().as_millis() as u64;
    let delta = state.models.telemetry_snapshot().delta_since(before);
    match result {
        Ok(answer) => ModeAnswer {
            answer: answer.chars().take(MAX_ANSWER_CHARS).collect(),
            context_sources: sources,
            latency_ms,
            request_count: bounded_u32(delta.chat_attempts),
            retry_count: bounded_u32(delta.chat_attempts).saturating_sub(1),
            failure: None,
        },
        Err(error) => ModeAnswer {
            answer: String::new(),
            context_sources: sources,
            latency_ms,
            request_count: bounded_u32(delta.chat_attempts),
            retry_count: bounded_u32(delta.chat_attempts).saturating_sub(1),
            failure: Some(RunFailure {
                code: "generation_failed".to_owned(),
                stage: None,
                safe_message: error.chat_client_message(),
                retryable: error.is_retryable(),
            }),
        },
    }
}

fn failed_answer(code: &str, safe_message: String, retryable: bool) -> ModeAnswer {
    ModeAnswer {
        answer: String::new(),
        context_sources: BTreeMap::new(),
        latency_ms: 0,
        request_count: 0,
        retry_count: 0,
        failure: Some(RunFailure {
            code: code.to_owned(),
            stage: None,
            safe_message,
            retryable,
        }),
    }
}

fn failed_retrieval_answer(failure: &ObservedRetrievalFailure) -> ModeAnswer {
    let stage = match failure.stage {
        ObservedFailureStage::Bm25 => RetrievalStage::Bm25,
        ObservedFailureStage::Rerank => RetrievalStage::Rerank,
    };
    ModeAnswer {
        answer: String::new(),
        context_sources: BTreeMap::new(),
        latency_ms: 0,
        request_count: 0,
        retry_count: 0,
        failure: Some(RunFailure {
            code: failure.code.to_owned(),
            stage: Some(stage),
            safe_message: failure.safe_message.clone(),
            retryable: failure.retryable,
        }),
    }
}

fn manual_review() -> ManualReview {
    ManualReview {
        status: ManualReviewStatus::Pending,
        reviewer: None,
        notes: None,
        reviewed_at: None,
    }
}

fn add_operational(
    totals: &mut OperationalStageTotals,
    latency_ms: u64,
    request_count: u32,
    retry_count: u32,
    fallback: bool,
) {
    totals.latency_ms = totals.latency_ms.saturating_add(latency_ms);
    totals.request_count = totals.request_count.saturating_add(request_count);
    totals.retry_count = totals.retry_count.saturating_add(retry_count);
    totals.fallback_count = totals.fallback_count.saturating_add(u32::from(fallback));
}

async fn evaluate_queries(
    state: &AppState,
    queries: &[&QueryFixture],
    stable: &StableIndex,
    scoring_config: &ScoringConfig,
    operational: &mut OperationalAccumulator,
    manifest_failures: &mut Vec<ManifestFailure>,
) -> anyhow::Result<Vec<ScoredQuery>> {
    let mut scored = Vec::with_capacity(queries.len() * 2);
    for query in queries {
        let expectation = query_expectation(query, stable)?;
        let before = state.models.telemetry_snapshot();
        let retrieval_started = std::time::Instant::now();
        let retrieval = retrieve_ab_observed(state, &query.query).await;
        let retrieval_latency_ms = retrieval_started.elapsed().as_millis() as u64;
        let cloud_delta = state.models.telemetry_snapshot().delta_since(before);

        let observed = match retrieval {
            Ok(observed) => observed,
            Err(error) => {
                let retryable = error.is_retryable();
                let failure = RunFailure {
                    code: "retrieval_failed".to_owned(),
                    stage: None,
                    safe_message: error.client_message(),
                    retryable,
                };
                let embedding_requests = bounded_u32(cloud_delta.embedding_attempts);
                let rerank_requests = bounded_u32(cloud_delta.rerank_attempts);
                add_operational(
                    &mut operational.query_embedding,
                    retrieval_latency_ms,
                    embedding_requests,
                    embedding_requests.saturating_sub(1),
                    false,
                );
                add_operational(
                    &mut operational.rerank,
                    0,
                    rerank_requests,
                    rerank_requests.saturating_sub(1),
                    false,
                );
                append_failed_pair(
                    &mut scored,
                    manifest_failures,
                    query,
                    &expectation,
                    empty_trace(retrieval_latency_ms, cloud_delta.embedding_attempts, 0),
                    empty_trace(0, 0, cloud_delta.rerank_attempts),
                    failure,
                    scoring_config,
                )?;
                continue;
            }
        };

        let (trace_a, trace_b) = match scoring_traces(&observed, stable, cloud_delta) {
            Ok(traces) => traces,
            Err(_) => {
                let failure = RunFailure {
                    code: "candidate_mapping_failed".to_owned(),
                    stage: None,
                    safe_message: "evaluation candidate mapping failed".to_owned(),
                    retryable: false,
                };
                append_failed_pair(
                    &mut scored,
                    manifest_failures,
                    query,
                    &expectation,
                    empty_trace(
                        observed.mode_a.dense.latency_ms,
                        cloud_delta.embedding_attempts,
                        0,
                    ),
                    empty_trace(0, 0, cloud_delta.rerank_attempts),
                    failure,
                    scoring_config,
                )?;
                continue;
            }
        };
        let embedding_requests = bounded_u32(cloud_delta.embedding_attempts);
        let rerank_requests = bounded_u32(cloud_delta.rerank_attempts);
        add_operational(
            &mut operational.query_embedding,
            observed.mode_a.dense.latency_ms,
            embedding_requests,
            embedding_requests.saturating_sub(1),
            false,
        );
        add_operational(
            &mut operational.rerank,
            observed.mode_b.rerank.latency_ms,
            rerank_requests,
            rerank_requests.saturating_sub(observed.mode_b.rerank.request_count),
            observed.mode_b.rerank.fallback,
        );

        let answer_a = answer_mode(
            state,
            stable,
            &query.query,
            &observed.mode_a.final_candidates,
        )
        .await;
        let answer_b = if let Some(failure) = &observed.mode_b.failure {
            failed_retrieval_answer(failure)
        } else {
            answer_mode(
                state,
                stable,
                &query.query,
                &observed.mode_b.final_candidates,
            )
            .await
        };
        for answer in [&answer_a, &answer_b] {
            add_operational(
                &mut operational.generation,
                answer.latency_ms,
                answer.request_count,
                answer.retry_count,
                false,
            );
        }

        for (mode, trace, answer) in [
            (EvaluationMode::A, trace_a, answer_a),
            (EvaluationMode::B, trace_b, answer_b),
        ] {
            if let Some(failure) = &answer.failure {
                manifest_failures.push(manifest_failure(&query.query_id, mode, failure));
            }
            let input = QueryRunInput {
                query_id: query.query_id.clone(),
                mode,
                answer: answer.answer,
                expectation: expectation.clone(),
                stages: trace,
                context_sources: answer.context_sources,
                manual_review: manual_review(),
                failure: answer.failure,
            };
            scored.push(score_query(input, scoring_config).map_err(format_scoring_errors)?);
        }
    }
    Ok(scored)
}

#[allow(clippy::too_many_arguments)]
fn append_failed_pair(
    scored: &mut Vec<ScoredQuery>,
    manifest_failures: &mut Vec<ManifestFailure>,
    query: &QueryFixture,
    expectation: &QueryExpectation,
    trace_a: RetrievalTrace,
    trace_b: RetrievalTrace,
    failure: RunFailure,
    scoring_config: &ScoringConfig,
) -> anyhow::Result<()> {
    for (mode, trace) in [(EvaluationMode::A, trace_a), (EvaluationMode::B, trace_b)] {
        manifest_failures.push(manifest_failure(&query.query_id, mode, &failure));
        let input = QueryRunInput {
            query_id: query.query_id.clone(),
            mode,
            answer: String::new(),
            expectation: expectation.clone(),
            stages: trace,
            context_sources: BTreeMap::new(),
            manual_review: manual_review(),
            failure: Some(failure.clone()),
        };
        scored.push(score_query(input, scoring_config).map_err(format_scoring_errors)?);
    }
    Ok(())
}

fn manifest_failure(query_id: &str, mode: EvaluationMode, failure: &RunFailure) -> ManifestFailure {
    ManifestFailure {
        query_id: Some(query_id.to_owned()),
        mode: Some(mode),
        stage: failure.stage,
        code: failure.code.clone(),
        safe_message: failure.safe_message.clone(),
        retryable: failure.retryable,
    }
}

fn format_scoring_errors(errors: Vec<super::scoring::ScoringError>) -> anyhow::Error {
    anyhow::anyhow!(
        "evaluation scoring failed: {}",
        errors
            .into_iter()
            .map(|error| error.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    )
}

fn git_revision() -> anyhow::Result<GitRevision> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("repository root is unavailable")?;
    let sha_output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repository)
        .output()
        .map_err(|_| anyhow::anyhow!("git revision could not be read"))?;
    if !sha_output.status.success() {
        bail!("git revision could not be read");
    }
    let sha = String::from_utf8(sha_output.stdout)
        .map_err(|_| anyhow::anyhow!("git revision is invalid"))?
        .trim()
        .to_owned();
    let status_output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(repository)
        .output()
        .map_err(|_| anyhow::anyhow!("git status could not be read"))?;
    if !status_output.status.success() {
        bail!("git status could not be read");
    }
    Ok(GitRevision {
        sha,
        dirty: !status_output.stdout.is_empty(),
    })
}

fn dataset_identity(
    options: &CliOptions,
    snapshot: &EvaluationDatasetSnapshot,
) -> ArtifactIdentity {
    ArtifactIdentity {
        sha256: snapshot.manifest_sha256.clone(),
        source: display_path(&options.manifest),
        version: snapshot.dataset.version.clone(),
    }
}

fn subset_manifest(options: &CliOptions) -> Vec<String> {
    let mut subsets = Vec::new();
    subsets.extend(
        options
            .documents
            .iter()
            .map(|value| format!("document:{value}")),
    );
    subsets.extend(options.queries.iter().map(|value| format!("query:{value}")));
    if subsets.is_empty() {
        subsets.push("all".to_owned());
    }
    subsets
}

fn scoring_config_with_threshold_cutoff(threshold_k: u32) -> ScoringConfig {
    let mut config = ScoringConfig::default();
    config.cutoffs.push(threshold_k);
    config.cutoffs.sort_unstable();
    config.cutoffs.dedup();
    config
}

fn manifest_top_n(rerank_candidate_top_k: usize, rerank_top_n: usize) -> TopNConfig {
    let report_value = |value| u32::try_from(value).unwrap_or(u32::MAX);
    TopNConfig {
        dense: report_value(DEFAULT_DENSE_TOP_K.max(rerank_candidate_top_k)),
        bm25: report_value(DEFAULT_BM25_TOP_K.max(rerank_candidate_top_k)),
        rrf: report_value(rerank_candidate_top_k),
        rerank: report_value(rerank_top_n),
        context: report_value(rerank_top_n.min(MAX_RAG_SOURCES)),
    }
}

async fn bootstrap_and_ingest(
    options: &CliOptions,
    snapshot: &EvaluationDatasetSnapshot,
    preflight: CloudPreflight,
    get_env: &mut impl FnMut(&str) -> Option<String>,
) -> anyhow::Result<()> {
    let dataset = &snapshot.dataset;
    let model_config = CloudModelConfig::from_lookup(|name| get_env(name))?;
    let models = CloudModelClient::new(model_config)?.with_deterministic_evaluation_retries();
    let retry_policies = models.evaluation_retry_policies();
    let embedding_model = models.embed_model().to_owned();
    let rerank_model = models.rerank_model().to_owned();
    let chat_model = models.chat_model().to_owned();
    let request_timeout_ms =
        u64::try_from(models.request_timeout().as_millis()).unwrap_or(u64::MAX);
    let rerank_candidate_top_k = models.rerank_candidate_top_k();
    let rerank_top_n = models.rerank_top_n();
    let top_n = manifest_top_n(rerank_candidate_top_k, rerank_top_n);
    let artifact_config = ArtifactConfig::from_env()?;
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&preflight.database_url)
        .await
        .map_err(|_| anyhow::anyhow!("evaluation database connection failed"))?;
    validate_evaluation_database(&pool).await?;

    // All remaining steps may mutate the isolated database. Publish a matching
    // run-scoped files directory first and leave a safe marker on any failure.
    let run_files_root = prepare_run_files_root(&preflight.approved_files_root)?;
    let run_files_path = run_files_root.path();
    let run_id = run_files_root.run_id().to_owned();
    let mut failure_marker = FailureMarkerGuard::new(run_files_path.clone(), run_id.clone());
    sqlx::migrate!()
        .run(&pool)
        .await
        .map_err(|_| anyhow::anyhow!("evaluation database migration failed"))?;
    let document_count: i64 = sqlx::query_scalar("SELECT count(*) FROM documents")
        .fetch_one(&pool)
        .await
        .map_err(|_| anyhow::anyhow!("evaluation database isolation check failed"))?;
    if document_count != 0 {
        bail!("evaluation database must contain zero documents before a run");
    }
    // Evaluation always stores files locally under EVAL_FILES_DIR, whatever
    // STORAGE_BACKEND says, and locks its own isolated database accordingly.
    let storage_config = StorageConfig::Local {
        files_dir: run_files_path.clone(),
    };
    ensure_initialization_lock(&pool, &storage_config)
        .await
        .map_err(|_| anyhow::anyhow!("evaluation storage initialization lock failed"))?;
    let storage = Storage::local(&run_files_path)
        .map_err(|_| anyhow::anyhow!("evaluation run file storage could not be opened"))?;

    let state = Arc::new(AppState {
        pool,
        models,
        storage,
        artifact_config,
        document_locks: DocumentLocks::default(),
    });
    let selection = select_dataset(dataset, options)?;
    let ingest_before = state.models.telemetry_snapshot();
    let ingest_started = std::time::Instant::now();
    let mut documents = Vec::with_capacity(selection.documents.len());
    for document in &selection.documents {
        let filename = Path::new(&document.relative_path)
            .file_name()
            .and_then(|value| value.to_str())
            .context("validated evaluation fixture filename is unavailable")?;
        let bytes = snapshot
            .fixture_bytes(&document.document_key)
            .context("validated evaluation fixture bytes are unavailable")?;
        documents.push(
            ingest_evaluation_bytes(state.clone(), filename, bytes)
                .await
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
        );
    }
    let ingest_latency_ms = ingest_started.elapsed().as_millis() as u64;
    let ingest_delta = state.models.telemetry_snapshot().delta_since(ingest_before);
    if ingest_delta.chat_attempts != 0 || ingest_delta.rerank_attempts != 0 {
        bail!("evaluation ingest invoked an unexpected cloud capability");
    }
    let logical_ingest_requests = selection
        .documents
        .iter()
        .map(|document| document.expected.units.len() as u64)
        .sum::<u64>();
    let ingest_requests = bounded_u32(ingest_delta.embedding_attempts);
    let mut operational = OperationalAccumulator::default();
    add_operational(
        &mut operational.ingest,
        ingest_latency_ms,
        ingest_requests,
        bounded_u32(
            ingest_delta
                .embedding_attempts
                .saturating_sub(logical_ingest_requests),
        ),
        false,
    );

    let stable = build_stable_index(&state.pool, &selection.documents, &documents).await?;
    let threshold_k = preflight
        .thresholds
        .config
        .minimums
        .correct_page_recall_at_k
        .k;
    let scoring_config = scoring_config_with_threshold_cutoff(threshold_k);
    let mut manifest_failures = Vec::new();
    let scored = evaluate_queries(
        &state,
        &selection.queries,
        &stable,
        &scoring_config,
        &mut operational,
        &mut manifest_failures,
    )
    .await?;

    let manifest = RunManifest {
        schema_version: REPORT_SCHEMA_VERSION.to_owned(),
        run_id: run_id.clone(),
        created_at_utc: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        git: git_revision()?,
        dataset: dataset_identity(options, snapshot),
        thresholds: preflight.thresholds,
        locked_model_ids: LockedModelIds {
            embedding: embedding_model,
            reranker: rerank_model,
            generation_a: chat_model.clone(),
            generation_b: chat_model,
        },
        seed: options.seed,
        concurrency: options.concurrency as u32,
        retry: CapabilityRetryPolicies {
            embedding: report_retry_policy(retry_policies.embedding),
            rerank: report_retry_policy(retry_policies.rerank),
            generation: report_retry_policy(retry_policies.generation),
        },
        operational: OperationalSummary {
            ingest: operational.ingest,
            query_embedding: operational.query_embedding,
            rerank: operational.rerank,
            generation: operational.generation,
        },
        subsets: subset_manifest(options),
        top_n,
        prompt_version: PROMPT_VERSION.to_owned(),
        provider_seed_support: ProviderSeedSupport {
            mode_a: false,
            mode_b: false,
        },
        config: LockedRunConfig {
            retrieval_cutoffs: scoring_config.cutoffs.clone(),
            comparison_epsilon: 0.0,
            rrf_k: RRF_K as u32,
            max_answer_chars: MAX_ANSWER_CHARS as u32,
            request_timeout_ms,
            citation_policy: CitationPolicy::ActualBracketedSourcesOnly,
            fact_match_policy: FactMatchPolicy::DeterministicNormalizedSubstring,
            negative_answer_policy: NegativeAnswerPolicy::DeterministicAbstentionPhrases,
        },
        failures: manifest_failures,
    };
    let bundle = build_ab_report(manifest, scored)?;
    let passed = bundle.report.thresholds.overall_pass;
    let paths = write_report_bundle(&options.output, &bundle)?;
    failure_marker.disarm();
    println!(
        "{}",
        serde_json::to_string_pretty(&CloudRunSummary {
            run_id,
            report_json: display_path(&paths.report_json),
            report_csv: display_path(&paths.report_csv),
            report_markdown: display_path(&paths.report_markdown),
            run_manifest: display_path(&paths.run_manifest),
            passed,
        })?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::schema::{DocumentKind, ExpectedInventory, ExpectedUnit};
    use super::*;

    fn temporary_directory(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("knowva-eval-{label}-{}", Uuid::new_v4()));
        fs::create_dir_all(&path).expect("create temporary directory");
        path
    }

    fn write_valid_threshold(root: &Path) -> PathBuf {
        let path = root.join("thresholds.json");
        let value = serde_json::json!({
            "schema_version": "1.0",
            "version": "thresholds-locked-v1",
            "issue_comment_url": "https://github.com/Laysi/knowva/issues/1#issuecomment-12345",
            "minimums": {
                "correct_page_recall_at_k": {
                    "stage": "rerank",
                    "k": 5,
                    "minimum": 0.5
                },
                "citation_precision": 0.5,
                "citation_recall": 0.5,
                "answer_facts_recall": 0.5,
                "b_relative_improvement": 0.0
            }
        });
        fs::write(
            &path,
            serde_json::to_vec_pretty(&value).expect("serialize threshold"),
        )
        .expect("write threshold");
        path
    }

    fn cloud_options(thresholds: PathBuf) -> CliOptions {
        CliOptions {
            mode: RunMode::Cloud,
            thresholds: Some(thresholds),
            ..CliOptions::default()
        }
    }

    fn document_fixture(
        document_key: &str,
        relative_path: &str,
        kind: DocumentKind,
        pages: Vec<u32>,
        units: Vec<(&str, Option<u32>)>,
    ) -> DocumentFixture {
        DocumentFixture {
            document_key: document_key.to_owned(),
            relative_path: relative_path.to_owned(),
            sha256: "0".repeat(64),
            license: "CC0-1.0".to_owned(),
            source: "synthetic unit test".to_owned(),
            kind,
            expected: ExpectedInventory {
                pages,
                units: units
                    .into_iter()
                    .map(|(unit_id, page)| ExpectedUnit {
                        unit_id: unit_id.to_owned(),
                        page,
                    })
                    .collect(),
            },
        }
    }

    fn environment(
        eval_database: &str,
        eval_files: &Path,
        production_files: &Path,
    ) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("EVAL_DATABASE_URL".to_owned(), eval_database.to_owned()),
            (
                "EVAL_FILES_DIR".to_owned(),
                eval_files.to_string_lossy().into_owned(),
            ),
            (
                "DATABASE_URL".to_owned(),
                "postgres://knowva:knowva@localhost:5432/production".to_owned(),
            ),
            (
                "FILES_DIR".to_owned(),
                production_files.to_string_lossy().into_owned(),
            ),
        ])
    }

    #[test]
    fn defaults_to_dry_run_and_parses_repeatable_subsets() {
        let options = CliOptions::parse([
            "--document".to_owned(),
            "doc-a".to_owned(),
            "--document".to_owned(),
            "doc-b".to_owned(),
            "--query".to_owned(),
            "query-a".to_owned(),
            "--seed".to_owned(),
            "7".to_owned(),
            "--concurrency".to_owned(),
            "2".to_owned(),
        ])
        .expect("parse eval options");
        assert_eq!(options.mode, RunMode::DryRun);
        assert_eq!(options.documents.len(), 2);
        assert_eq!(options.queries.len(), 1);
        assert_eq!(options.seed, 7);
        assert_eq!(options.concurrency, 2);
    }

    #[test]
    fn cloud_is_only_enabled_by_explicit_flag() {
        assert_eq!(
            CliOptions::parse(Vec::<String>::new()).unwrap().mode,
            RunMode::DryRun
        );
        assert_eq!(
            CliOptions::parse(["--cloud".to_owned()]).unwrap().mode,
            RunMode::Cloud
        );
    }

    #[test]
    fn rejects_zero_concurrency_and_missing_values() {
        assert!(CliOptions::parse(["--concurrency".to_owned(), "0".to_owned()]).is_err());
        assert!(CliOptions::parse(["--manifest".to_owned()]).is_err());
    }

    #[test]
    fn threshold_approval_requires_the_exact_pinned_comment_and_hash_line() {
        let root = temporary_directory("approval");
        let snapshot =
            load_threshold_snapshot(write_valid_threshold(&root)).expect("load threshold snapshot");
        let pinned_url = snapshot.config.issue_comment_url.clone();
        let marker = format!("threshold_sha256: {}", snapshot.identity.sha256);
        validate_threshold_approval(
            &snapshot,
            &pinned_url,
            &marker,
            THRESHOLD_APPROVER_LOGIN,
            "OWNER",
        )
        .expect("exact approval must pass");

        let cases = [
            (
                "https://github.com/example/knowva/issues/1#issuecomment-12345",
                marker.as_str(),
            ),
            (
                "https://github.com/Laysi/knowva/issues/1#issuecomment-54321",
                marker.as_str(),
            ),
            (pinned_url.as_str(), "threshold_sha256: wrong-secret-hash"),
        ];
        for (url, body) in cases {
            let error = validate_threshold_approval(
                &snapshot,
                url,
                body,
                THRESHOLD_APPROVER_LOGIN,
                "OWNER",
            )
            .expect_err("wrong repository, comment, or hash must fail")
            .to_string();
            assert!(
                !error.contains(body),
                "approval errors must not echo comment bodies"
            );
        }

        let mut wrong_repository = snapshot.clone();
        wrong_repository.config.issue_comment_url =
            "https://github.com/example/knowva/issues/1#issuecomment-12345".to_owned();
        let body = format!("threshold_sha256: {}", wrong_repository.identity.sha256);
        let error = validate_threshold_approval(
            &wrong_repository,
            &wrong_repository.config.issue_comment_url,
            &body,
            THRESHOLD_APPROVER_LOGIN,
            "OWNER",
        )
        .expect_err("a snapshot for another repository must fail")
        .to_string();
        assert!(!error.contains(&body));

        for (login, association) in [
            ("attacker", "OWNER"),
            (THRESHOLD_APPROVER_LOGIN, "CONTRIBUTOR"),
            ("ruisi-lu", "COLLABORATOR"),
        ] {
            let error =
                validate_threshold_approval(&snapshot, &pinned_url, &marker, login, association)
                    .expect_err("wrong author identity or association must fail")
                    .to_string();
            assert!(!error.contains(login));
            assert!(!error.contains(&marker));
        }
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    fn failure_marker_is_run_scoped_safe_and_disarmable() {
        let root = temporary_directory("marker-secret-parent");
        let failed_run = root.join("run-failed");
        fs::create_dir(&failed_run).expect("create failed run root");
        {
            let _guard = FailureMarkerGuard::new(failed_run.clone(), "run-failed".to_owned());
        }

        let marker_path = failed_run.join("evaluation-failure.json");
        let marker = fs::read_to_string(&marker_path).expect("read failure marker");
        let value: serde_json::Value = serde_json::from_str(&marker).expect("parse failure marker");
        assert_eq!(value["runId"], "run-failed");
        assert_eq!(value["status"], "failed");
        assert!(!root.join("evaluation-failure.json").exists());
        assert!(!marker.contains("marker-secret-parent"));
        assert!(!marker.contains(&root.to_string_lossy().into_owned()));

        let completed_run = root.join("run-completed");
        fs::create_dir(&completed_run).expect("create completed run root");
        {
            let mut guard =
                FailureMarkerGuard::new(completed_run.clone(), "run-completed".to_owned());
            guard.disarm();
        }
        assert!(!completed_run.join("evaluation-failure.json").exists());
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    fn database_unit_mapping_keeps_same_named_documents_isolated() {
        let north_id = Uuid::new_v4();
        let south_id = Uuid::new_v4();
        let north = document_fixture(
            "north-report",
            "documents/north/quarterly-report.pdf",
            DocumentKind::NativePdf,
            vec![1],
            vec![("page-1", Some(1))],
        );
        let south = document_fixture(
            "south-report",
            "documents/south/quarterly-report.pdf",
            DocumentKind::NativePdf,
            vec![1],
            vec![("page-1", Some(1))],
        );
        let north_document = StableDocument {
            document_key: north.document_key.clone(),
            document_id: north_id,
            filename: "quarterly-report.pdf".to_owned(),
        };
        let south_document = StableDocument {
            document_key: south.document_key.clone(),
            document_id: south_id,
            filename: "quarterly-report.pdf".to_owned(),
        };
        let north_unit = map_database_unit(
            &north,
            &north_document,
            &DatabaseUnitRow {
                id: 1,
                document_id: north_id,
                kind: "page".to_owned(),
                unit_index: 0,
                page_number: Some(1),
            },
        )
        .expect("map north page");
        let south_unit = map_database_unit(
            &south,
            &south_document,
            &DatabaseUnitRow {
                id: 2,
                document_id: south_id,
                kind: "page".to_owned(),
                unit_index: 0,
                page_number: Some(1),
            },
        )
        .expect("map south page");

        assert_eq!(north_unit.filename, south_unit.filename);
        assert_ne!(north_unit.document_key, south_unit.document_key);
        assert_ne!(north_unit.document_id, south_unit.document_id);
    }

    #[test]
    fn database_page_units_are_one_based_and_must_exist_in_inventory() {
        let document_id = Uuid::new_v4();
        let document = document_fixture(
            "one-page",
            "documents/one-page.pdf",
            DocumentKind::NativePdf,
            vec![1],
            vec![("page-1", Some(1))],
        );
        let stable_document = StableDocument {
            document_key: document.document_key.clone(),
            document_id,
            filename: "one-page.pdf".to_owned(),
        };
        let row = |page_number| DatabaseUnitRow {
            id: 1,
            document_id,
            kind: "page".to_owned(),
            unit_index: 0,
            page_number,
        };

        let mapped = map_database_unit(&document, &stable_document, &row(Some(1)))
            .expect("page one must map without an off-by-one shift");
        assert_eq!(mapped.unit_id, "page-1");
        assert_eq!(mapped.page, Some(1));
        assert!(map_database_unit(&document, &stable_document, &row(Some(0))).is_err());
        assert!(map_database_unit(&document, &stable_document, &row(Some(2))).is_err());
    }

    #[test]
    fn database_text_units_use_zero_based_stable_indices_and_no_page() {
        let document_id = Uuid::new_v4();
        let document = document_fixture(
            "operations",
            "documents/operations.md",
            DocumentKind::Markdown,
            vec![],
            vec![("chunk-1", None), ("chunk-2", None)],
        );
        let stable_document = StableDocument {
            document_key: document.document_key.clone(),
            document_id,
            filename: "operations.md".to_owned(),
        };
        let mapped = map_database_unit(
            &document,
            &stable_document,
            &DatabaseUnitRow {
                id: 2,
                document_id,
                kind: "text".to_owned(),
                unit_index: 1,
                page_number: None,
            },
        )
        .expect("zero-based text index one must select the second stable unit");

        assert_eq!(mapped.unit_id, "chunk-2");
        assert_eq!(mapped.page, None);
    }

    #[test]
    fn threshold_cutoff_is_sorted_and_deduplicated() {
        assert_eq!(
            scoring_config_with_threshold_cutoff(6).cutoffs,
            vec![1, 3, 5, 6, 10]
        );
        assert_eq!(
            scoring_config_with_threshold_cutoff(5).cutoffs,
            vec![1, 3, 5, 10]
        );
    }

    #[test]
    fn manifest_top_n_reports_resolved_retrieval_limits() {
        let below_defaults = manifest_top_n(6, 6);
        assert_eq!(below_defaults.dense, DEFAULT_DENSE_TOP_K as u32);
        assert_eq!(below_defaults.bm25, DEFAULT_BM25_TOP_K as u32);
        assert_eq!(below_defaults.rrf, 6);
        assert_eq!(below_defaults.rerank, 6);
        assert_eq!(below_defaults.context, 6);

        let above_defaults = manifest_top_n(73, 8);
        assert_eq!(above_defaults.dense, 73);
        assert_eq!(above_defaults.bm25, 73);
        assert_eq!(above_defaults.rrf, 73);
        assert_eq!(above_defaults.rerank, 8);
        assert_eq!(above_defaults.context, MAX_RAG_SOURCES as u32);
    }

    #[tokio::test]
    async fn dry_run_validates_real_fixtures_without_entering_cloud_path() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures/manifest.json");
        let options = CliOptions {
            manifest,
            ..CliOptions::default()
        };
        execute(options).await.expect("offline dry-run must pass");
        assert_eq!(CLOUD_RUN_INVOCATIONS.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cloud_preflight_requires_threshold_before_reading_environment() {
        let options = CliOptions {
            mode: RunMode::Cloud,
            ..CliOptions::default()
        };
        let environment_reads = AtomicUsize::new(0);
        let error = validate_cloud_preflight(&options, Path::new("/tmp"), &mut |_| {
            environment_reads.fetch_add(1, Ordering::SeqCst);
            None
        })
        .err()
        .expect("threshold must be required");
        assert!(error.to_string().contains("--thresholds"));
        assert_eq!(environment_reads.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cloud_preflight_rejects_concurrency_before_environment_or_network() {
        let root = temporary_directory("concurrency");
        let threshold = write_valid_threshold(&root);
        let mut options = cloud_options(threshold);
        options.concurrency = 2;
        let environment_reads = AtomicUsize::new(0);
        let error = validate_cloud_preflight(&options, &root, &mut |_| {
            environment_reads.fetch_add(1, Ordering::SeqCst);
            None
        })
        .err()
        .expect("concurrency must fail closed");
        assert!(error.to_string().contains("concurrency 1"));
        assert_eq!(environment_reads.load(Ordering::SeqCst), 0);
        assert_eq!(CLOUD_BOOTSTRAP_INVOCATIONS.load(Ordering::SeqCst), 0);
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    fn cloud_preflight_rejects_missing_or_shared_database_and_overlapping_files() {
        let root = temporary_directory("isolation");
        let threshold = write_valid_threshold(&root);
        let options = cloud_options(threshold.clone());
        let missing = validate_cloud_preflight(&options, &root, &mut |_| None)
            .err()
            .expect("missing eval environment must fail");
        assert!(missing.to_string().contains("EVAL_DATABASE_URL"));

        let eval_root = root.join("eval-files");
        let production_root = root.join("production-files");
        let mut shared_database = environment(DEFAULT_DATABASE_URL, &eval_root, &production_root);
        shared_database.remove("DATABASE_URL");
        let shared = validate_cloud_preflight(&options, &root, &mut |name| {
            shared_database.get(name).cloned()
        })
        .err()
        .expect("production database reuse must fail");
        assert!(shared.to_string().contains("isolated"));
        assert!(!shared.to_string().contains(DEFAULT_DATABASE_URL));

        let default_database_environment =
            environment(DEFAULT_DATABASE_URL, &eval_root, &production_root);
        let default_database = validate_cloud_preflight(&options, &root, &mut |name| {
            default_database_environment.get(name).cloned()
        })
        .err()
        .expect("default database reuse must fail even with a custom production URL");
        assert!(default_database.to_string().contains("isolated"));

        let nested_eval_root = production_root.join("nested-eval");
        let overlap_environment = environment(
            "postgres://knowva:knowva@localhost:5432/evaluation",
            &nested_eval_root,
            &production_root,
        );
        let overlap = validate_cloud_preflight(&options, &root, &mut |name| {
            overlap_environment.get(name).cloned()
        })
        .err()
        .expect("nested file roots must fail");
        assert!(overlap.to_string().contains("must not equal"));

        let default_eval_files = root.join(DEFAULT_FILES_DIR);
        let default_files_environment = environment(
            "postgres://knowva:knowva@localhost:5432/evaluation",
            &default_eval_files,
            &production_root,
        );
        let default_files = validate_cloud_preflight(&options, &root, &mut |name| {
            default_files_environment.get(name).cloned()
        })
        .err()
        .expect("default FILES_DIR reuse must fail with a custom production path");
        assert!(default_files.to_string().contains("must not equal"));
        assert_eq!(CLOUD_BOOTSTRAP_INVOCATIONS.load(Ordering::SeqCst), 0);
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn nonempty_eval_root_is_rejected_without_creating_a_run_child() {
        use std::os::unix::fs::PermissionsExt;

        let root = temporary_directory("nonempty");
        let eval_root = root.join("eval-files");
        fs::create_dir_all(&eval_root).expect("create eval root");
        fs::set_permissions(&eval_root, fs::Permissions::from_mode(0o700))
            .expect("secure eval root");
        fs::write(eval_root.join("unexpected"), b"occupied").expect("occupy eval root");
        let error = prepare_run_files_root(&eval_root).expect_err("nonempty eval root must fail");
        assert!(error.to_string().contains("must be empty"));
        assert_eq!(fs::read_dir(&eval_root).expect("read eval root").count(), 1);
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn empty_eval_root_gets_one_run_scoped_child() {
        use std::os::unix::fs::PermissionsExt;

        let root = temporary_directory("run-root");
        let eval_root = root.join("eval-files");
        fs::create_dir(&eval_root).expect("create eval root");
        fs::set_permissions(&eval_root, fs::Permissions::from_mode(0o700))
            .expect("secure eval root");
        let run_root = prepare_run_files_root(&eval_root).expect("prepare run root");
        let run_path = run_root.path();
        let canonical_run = run_path.canonicalize().expect("canonical run root");
        let approved_base = eval_root.canonicalize().expect("canonical eval root");
        assert_eq!(canonical_run.parent(), Some(approved_base.as_path()));
        assert!(run_root.run_id().starts_with("run-"));
        assert!(run_path.is_dir());
        assert_eq!(
            fs::metadata(&run_path)
                .expect("run root metadata")
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        use std::os::fd::AsRawFd;
        // SAFETY: fcntl reads descriptor flags for live File-owned fds.
        let base_flags =
            unsafe { libc::fcntl(run_root._base_directory.as_raw_fd(), libc::F_GETFD) };
        // SAFETY: fcntl reads descriptor flags for a live File-owned fd.
        let run_flags = unsafe { libc::fcntl(run_root.run_directory.as_raw_fd(), libc::F_GETFD) };
        assert_ne!(base_flags & libc::FD_CLOEXEC, 0);
        assert_ne!(run_flags & libc::FD_CLOEXEC, 0);
        assert_eq!(fs::read_dir(&eval_root).expect("read eval root").count(), 1);
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn symlink_swap_is_rejected_without_writing_the_target() {
        use std::os::unix::fs::symlink;

        let root = temporary_directory("symlink-swap");
        let approved_base = root.join("eval-files");
        let target = root.join("redirected-target");
        fs::create_dir(&target).expect("create redirect target");
        symlink(&target, &approved_base).expect("replace approved base with symlink");

        let error = prepare_run_files_root(&approved_base)
            .expect_err("symlink replacement must fail closed");
        assert!(error.to_string().contains("symbolic-link"));
        assert_eq!(fs::read_dir(&target).expect("read target").count(), 0);
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn permissive_existing_eval_root_is_rejected() {
        use std::os::unix::fs::PermissionsExt;

        let root = temporary_directory("permissive-root");
        let eval_root = root.join("eval-files");
        fs::create_dir(&eval_root).expect("create eval root");
        fs::set_permissions(&eval_root, fs::Permissions::from_mode(0o777))
            .expect("make eval root permissive");

        let error = prepare_run_files_root(&eval_root)
            .expect_err("group/world-accessible root must fail closed");
        assert!(error.to_string().contains("group or world"));
        assert_eq!(fs::read_dir(&eval_root).expect("read eval root").count(), 0);
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn opened_base_fd_survives_path_swap_without_writing_redirect_target() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let root = temporary_directory("dirfd-race");
        let eval_root = root.join("eval-files");
        let original_root = root.join("eval-files-original");
        let redirect_target = root.join("redirect-target");
        fs::create_dir(&eval_root).expect("create eval root");
        fs::set_permissions(&eval_root, fs::Permissions::from_mode(0o700))
            .expect("secure eval root");
        fs::create_dir(&redirect_target).expect("create redirect target");

        let run_root = prepare_run_files_root_with_hook(&eval_root, || {
            fs::rename(&eval_root, &original_root).expect("rename opened eval root");
            symlink(&redirect_target, &eval_root).expect("replace eval root with symlink");
        })
        .expect("dirfd preparation must remain anchored to original root");
        let stable_run_path = run_root.path();
        let original_run_path = original_root.join(run_root.run_id());
        assert_eq!(
            stable_run_path
                .canonicalize()
                .expect("canonical stable run"),
            original_run_path
        );

        {
            let _guard = FailureMarkerGuard::new(stable_run_path, run_root.run_id().to_owned());
        }
        assert!(original_run_path.join("evaluation-failure.json").is_file());
        assert_eq!(
            fs::read_dir(&redirect_target)
                .expect("read redirect target")
                .count(),
            0
        );
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    fn database_identity_values_fail_closed_and_accept_isolated_empty_database() {
        assert!(
            validate_database_identity_values(
                "production",
                "knowva_eval_user",
                Some(EVALUATION_DATABASE_MARKER),
                false,
            )
            .is_err()
        );
        assert!(
            validate_database_identity_values(
                "knowva_eval_fixture",
                "knowva_eval_user",
                Some("wrong-marker"),
                false,
            )
            .is_err()
        );
        assert!(
            validate_database_identity_values(
                "knowva_eval_fixture",
                "knowva_eval_user",
                Some(EVALUATION_DATABASE_MARKER),
                true,
            )
            .is_err()
        );
        validate_database_identity_values(
            "knowva_eval_fixture",
            "knowva_eval_user",
            Some(EVALUATION_DATABASE_MARKER),
            false,
        )
        .expect("isolated marked empty evaluation database must pass");
    }

    #[test]
    fn production_defaults_remain_the_server_defaults() {
        assert_eq!(
            DEFAULT_DATABASE_URL,
            "postgres://knowva:knowva@localhost:5432/knowva"
        );
        assert_eq!(DEFAULT_FILES_DIR, "./data/files");
    }
}
