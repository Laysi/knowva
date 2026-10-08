use super::scoring::{EvaluationMode, NegativeAnswerPolicy, RetrievalStage, ScoredQuery};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::ErrorKind;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// 1.1 added the manifest's `profile_id` and `system_prompt_sha256`; 2.0
/// removed the `bm25` and `rrf` stages, `top_n.bm25`, `top_n.rrf` and
/// `config.rrf_k` along with BM25 retrieval.
pub const REPORT_SCHEMA_VERSION: &str = "2.0";
pub const THRESHOLD_SCHEMA_VERSION: &str = "2.0";
static NEXT_STAGING_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactIdentity {
    pub sha256: String,
    pub source: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GitRevision {
    pub sha: String,
    pub dirty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedModelIds {
    pub embedding: String,
    pub reranker: String,
    pub generation_a: String,
    pub generation_b: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub jitter: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityRetryPolicies {
    pub embedding: RetryPolicy,
    pub rerank: RetryPolicy,
    pub generation: RetryPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrectPageRecallThreshold {
    pub stage: RetrievalStage,
    pub k: u32,
    pub minimum: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdMinimums {
    pub correct_page_recall_at_k: CorrectPageRecallThreshold,
    pub citation_precision: f64,
    pub citation_recall: f64,
    pub answer_facts_recall: f64,
    pub b_relative_improvement: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdConfig {
    pub schema_version: String,
    pub version: String,
    pub minimums: ThresholdMinimums,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdSnapshot {
    pub identity: ArtifactIdentity,
    pub config: ThresholdConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationalStageTotals {
    pub latency_ms: u64,
    pub request_count: u32,
    pub retry_count: u32,
    pub fallback_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationalSummary {
    pub ingest: OperationalStageTotals,
    pub query_embedding: OperationalStageTotals,
    pub rerank: OperationalStageTotals,
    pub generation: OperationalStageTotals,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopNConfig {
    pub dense: u32,
    pub rerank: u32,
    pub context: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSeedSupport {
    pub mode_a: bool,
    pub mode_b: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CitationPolicy {
    ActualBracketedSourcesOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactMatchPolicy {
    DeterministicNormalizedSubstring,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedRunConfig {
    pub retrieval_cutoffs: Vec<u32>,
    pub comparison_epsilon: f64,
    pub max_answer_chars: u32,
    pub request_timeout_ms: u64,
    pub citation_policy: CitationPolicy,
    pub fact_match_policy: FactMatchPolicy,
    pub negative_answer_policy: NegativeAnswerPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestFailure {
    pub query_id: Option<String>,
    pub mode: Option<EvaluationMode>,
    pub stage: Option<RetrievalStage>,
    pub code: String,
    pub safe_message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifest {
    pub schema_version: String,
    pub run_id: String,
    pub created_at_utc: String,
    pub git: GitRevision,
    pub dataset: ArtifactIdentity,
    pub thresholds: ThresholdSnapshot,
    pub locked_model_ids: LockedModelIds,
    pub seed: u64,
    pub concurrency: u32,
    pub retry: CapabilityRetryPolicies,
    pub operational: OperationalSummary,
    pub subsets: Vec<String>,
    pub top_n: TopNConfig,
    pub prompt_version: String,
    /// Industry profile the run answered under (`KNOWVA_PROFILE`).
    pub profile_id: String,
    /// Lowercase hex SHA-256 of the composed system prompt sent with every answer.
    pub system_prompt_sha256: String,
    pub provider_seed_support: ProviderSeedSupport,
    pub config: LockedRunConfig,
    pub failures: Vec<ManifestFailure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonStatus {
    Improvement,
    Same,
    Regression,
    Failure,
}

impl ComparisonStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Improvement => "improvement",
            Self::Same => "same",
            Self::Regression => "regression",
            Self::Failure => "failure",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryComparison {
    pub query_id: String,
    pub status: ComparisonStatus,
    pub score_a: Option<f64>,
    pub score_b: Option<f64>,
    pub delta: Option<f64>,
    pub mode_a: Option<ScoredQuery>,
    pub mode_b: Option<ScoredQuery>,
    pub failure_reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricAggregate {
    pub paired_count: u32,
    pub mean_a: Option<f64>,
    pub mean_b: Option<f64>,
    pub mean_delta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbAggregates {
    pub total_queries: u32,
    pub improvement_count: u32,
    pub same_count: u32,
    pub regression_count: u32,
    pub failure_count: u32,
    pub scored_pair_count: u32,
    pub mean_score_a: Option<f64>,
    pub mean_score_b: Option<f64>,
    pub mean_delta: Option<f64>,
    pub metrics: BTreeMap<String, MetricAggregate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbReport {
    pub schema_version: String,
    pub run_id: String,
    pub aggregates: AbAggregates,
    pub thresholds: ThresholdResults,
    pub queries: Vec<QueryComparison>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum ThresholdMetric {
    CorrectPageRecallAtK { stage: RetrievalStage, k: u32 },
    CitationPrecision,
    CitationRecall,
    AnswerFactsRecall,
    BRelativeImprovement,
}

impl ThresholdMetric {
    fn label(&self) -> String {
        match self {
            Self::CorrectPageRecallAtK { stage, k } => {
                format!("{}.correct_page_recall@{k}", stage.as_str())
            }
            Self::CitationPrecision => "citation_precision".to_owned(),
            Self::CitationRecall => "citation_recall".to_owned(),
            Self::AnswerFactsRecall => "answer_facts_recall".to_owned(),
            Self::BRelativeImprovement => "b_relative_improvement".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdEvaluation {
    pub metric: ThresholdMetric,
    pub observed: Option<f64>,
    pub required_minimum: f64,
    pub paired_count: u32,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdResults {
    pub valid_pair_count: u32,
    pub overall_pass: bool,
    pub evaluations: Vec<ThresholdEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReportBundle {
    pub manifest: RunManifest,
    pub report: AbReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReportError {
    pub code: &'static str,
    pub path: String,
    pub message: &'static str,
}

impl ReportError {
    fn new(code: &'static str, path: impl Into<String>, message: &'static str) -> Self {
        Self {
            code,
            path: path.into(),
            message,
        }
    }
}

impl std::fmt::Display for ReportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} at {}: {}",
            self.code, self.path, self.message
        )
    }
}

impl std::error::Error for ReportError {}
pub type ReportResult<T> = Result<T, ReportError>;

pub fn load_threshold_snapshot(path: impl AsRef<Path>) -> ReportResult<ThresholdSnapshot> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| {
        if error.kind() == ErrorKind::NotFound {
            ReportError::new(
                "thresholds_not_found",
                "$",
                "threshold configuration file was not found",
            )
        } else {
            ReportError::new(
                "thresholds_read_failed",
                "$",
                "threshold configuration file could not be read",
            )
        }
    })?;
    let config = serde_json::from_slice::<ThresholdConfig>(&bytes).map_err(|_| {
        ReportError::new(
            "thresholds_invalid_json",
            "$",
            "threshold configuration is not valid JSON for the supported schema",
        )
    })?;
    validate_threshold_config(&config)?;
    let source = path.to_string_lossy().into_owned();
    validate_safe_value(&source, "$.identity.source")?;
    Ok(ThresholdSnapshot {
        identity: ArtifactIdentity {
            sha256: super::schema::sha256_hex(&bytes),
            source,
            version: config.version.clone(),
        },
        config,
    })
}

pub fn validate_threshold_config(config: &ThresholdConfig) -> ReportResult<()> {
    if config.schema_version != THRESHOLD_SCHEMA_VERSION {
        return Err(ReportError::new(
            "unsupported_threshold_version",
            "$.schema_version",
            "threshold schema version is not supported",
        ));
    }
    validate_safe_value(&config.version, "$.version")?;
    if config.minimums.correct_page_recall_at_k.k == 0 {
        return Err(ReportError::new(
            "threshold_k_invalid",
            "$.minimums.correct_page_recall_at_k.k",
            "threshold cutoff must be greater than zero",
        ));
    }
    for (path, value) in [
        (
            "$.minimums.correct_page_recall_at_k.minimum",
            config.minimums.correct_page_recall_at_k.minimum,
        ),
        (
            "$.minimums.citation_precision",
            config.minimums.citation_precision,
        ),
        (
            "$.minimums.citation_recall",
            config.minimums.citation_recall,
        ),
        (
            "$.minimums.answer_facts_recall",
            config.minimums.answer_facts_recall,
        ),
        (
            "$.minimums.b_relative_improvement",
            config.minimums.b_relative_improvement,
        ),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(ReportError::new(
                "threshold_value_invalid",
                path,
                "threshold values must be finite and between zero and one",
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReportPaths {
    pub directory: PathBuf,
    pub report_json: PathBuf,
    pub report_csv: PathBuf,
    pub report_markdown: PathBuf,
    pub run_manifest: PathBuf,
}

#[derive(Default)]
struct Pair {
    mode_a: Option<ScoredQuery>,
    mode_b: Option<ScoredQuery>,
    failures: Vec<String>,
}

#[derive(Default)]
struct MetricAccumulator {
    count: u32,
    sum_a: f64,
    sum_b: f64,
    sum_delta: f64,
}

pub fn build_ab_report(
    manifest: RunManifest,
    results: Vec<ScoredQuery>,
) -> ReportResult<ReportBundle> {
    validate_manifest(&manifest)?;
    let mut pairs = BTreeMap::<String, Pair>::new();
    for scored in results {
        let query_id = scored.input.query_id.clone();
        let pair = pairs.entry(query_id).or_default();
        let slot = match scored.input.mode {
            EvaluationMode::A => &mut pair.mode_a,
            EvaluationMode::B => &mut pair.mode_b,
        };
        if slot.is_some() {
            pair.failures.push(format!(
                "duplicate_mode_{}",
                match scored.input.mode {
                    EvaluationMode::A => "a",
                    EvaluationMode::B => "b",
                }
            ));
        } else {
            *slot = Some(scored);
        }
    }

    for failure in &manifest.failures {
        if let Some(query_id) = &failure.query_id {
            pairs
                .entry(query_id.clone())
                .or_default()
                .failures
                .push(format!("manifest:{}", failure.code));
        }
    }

    let epsilon = manifest.config.comparison_epsilon;
    let mut queries = Vec::with_capacity(pairs.len());
    for (query_id, mut pair) in pairs {
        if pair.mode_a.is_none() {
            pair.failures.push("missing_mode_a".to_owned());
        }
        if pair.mode_b.is_none() {
            pair.failures.push("missing_mode_b".to_owned());
        }
        if let Some(mode_a) = &pair.mode_a
            && let Some(failure) = &mode_a.input.failure
        {
            pair.failures.push(format!("mode_a:{}", failure.code));
        }
        if let Some(mode_b) = &pair.mode_b
            && let Some(failure) = &mode_b.input.failure
        {
            pair.failures.push(format!("mode_b:{}", failure.code));
        }
        if let (Some(mode_a), Some(mode_b)) = (&pair.mode_a, &pair.mode_b)
            && mode_a.input.expectation != mode_b.input.expectation
        {
            pair.failures.push("expectation_mismatch".to_owned());
        }

        let score_a = pair
            .mode_a
            .as_ref()
            .and_then(|query| query.metrics.as_ref())
            .map(|metrics| metrics.composite_score);
        let score_b = pair
            .mode_b
            .as_ref()
            .and_then(|query| query.metrics.as_ref())
            .map(|metrics| metrics.composite_score);
        if score_a.is_none() && pair.mode_a.is_some() {
            pair.failures.push("mode_a_unscored".to_owned());
        }
        if score_b.is_none() && pair.mode_b.is_some() {
            pair.failures.push("mode_b_unscored".to_owned());
        }
        pair.failures.sort();
        pair.failures.dedup();

        let delta = score_a.zip(score_b).map(|(a, b)| b - a);
        let status = if !pair.failures.is_empty() {
            ComparisonStatus::Failure
        } else if delta.is_some_and(|value| value > epsilon) {
            ComparisonStatus::Improvement
        } else if delta.is_some_and(|value| value < -epsilon) {
            ComparisonStatus::Regression
        } else {
            ComparisonStatus::Same
        };
        queries.push(QueryComparison {
            query_id,
            status,
            score_a,
            score_b,
            delta,
            mode_a: pair.mode_a,
            mode_b: pair.mode_b,
            failure_reasons: pair.failures,
        });
    }

    let aggregates = aggregate_queries(&queries);
    let thresholds = evaluate_thresholds(&aggregates, &manifest.thresholds.config.minimums);
    Ok(ReportBundle {
        report: AbReport {
            schema_version: REPORT_SCHEMA_VERSION.to_owned(),
            run_id: manifest.run_id.clone(),
            aggregates,
            thresholds,
            queries,
        },
        manifest,
    })
}

fn evaluate_thresholds(
    aggregates: &AbAggregates,
    minimums: &ThresholdMinimums,
) -> ThresholdResults {
    let page = &minimums.correct_page_recall_at_k;
    let page_name = format!("{}.page_recall@{}", page.stage.as_str(), page.k);
    let specifications = [
        (
            ThresholdMetric::CorrectPageRecallAtK {
                stage: page.stage,
                k: page.k,
            },
            aggregates.metrics.get(&page_name),
            page.minimum,
            false,
        ),
        (
            ThresholdMetric::CitationPrecision,
            aggregates.metrics.get("citation.exact.precision"),
            minimums.citation_precision,
            false,
        ),
        (
            ThresholdMetric::CitationRecall,
            aggregates.metrics.get("citation.exact.recall"),
            minimums.citation_recall,
            false,
        ),
        (
            ThresholdMetric::AnswerFactsRecall,
            aggregates.metrics.get("facts.recall"),
            minimums.answer_facts_recall,
            false,
        ),
        (
            ThresholdMetric::BRelativeImprovement,
            None,
            minimums.b_relative_improvement,
            true,
        ),
    ];
    let evaluations = specifications
        .into_iter()
        .map(
            |(metric, aggregate, required_minimum, is_composite_delta)| {
                let (observed, paired_count) = if is_composite_delta {
                    (
                        relative_improvement(aggregates.mean_score_a, aggregates.mean_score_b),
                        aggregates.scored_pair_count,
                    )
                } else {
                    (
                        aggregate.and_then(|value| value.mean_b),
                        aggregate.map_or(0, |value| value.paired_count),
                    )
                };
                let passed = aggregates.scored_pair_count > 0
                    && paired_count > 0
                    && observed.is_some_and(|value| value.is_finite() && value >= required_minimum);
                ThresholdEvaluation {
                    metric,
                    observed,
                    required_minimum,
                    paired_count,
                    passed,
                }
            },
        )
        .collect::<Vec<_>>();
    ThresholdResults {
        valid_pair_count: aggregates.scored_pair_count,
        overall_pass: aggregates.scored_pair_count > 0
            && evaluations.iter().all(|evaluation| evaluation.passed),
        evaluations,
    }
}

fn relative_improvement(mean_a: Option<f64>, mean_b: Option<f64>) -> Option<f64> {
    mean_a.zip(mean_b).map(|(a, b)| {
        if a == 0.0 {
            if b > 0.0 { 1.0 } else { 0.0 }
        } else {
            (b - a) / a
        }
    })
}

fn aggregate_queries(queries: &[QueryComparison]) -> AbAggregates {
    let mut score_a = Vec::new();
    let mut score_b = Vec::new();
    let mut deltas = Vec::new();
    let mut metric_values = BTreeMap::<String, MetricAccumulator>::new();

    for query in queries {
        if query.status == ComparisonStatus::Failure {
            continue;
        }
        if let (Some(a), Some(b), Some(delta)) = (query.score_a, query.score_b, query.delta) {
            score_a.push(a);
            score_b.push(b);
            deltas.push(delta);
        }
        if let (Some(a), Some(b)) = (&query.mode_a, &query.mode_b) {
            let metrics_a = flatten_metrics(a);
            let metrics_b = flatten_metrics(b);
            for (name, value_a) in metrics_a {
                if let Some(value_b) = metrics_b.get(&name) {
                    let accumulator = metric_values.entry(name).or_default();
                    accumulator.count += 1;
                    accumulator.sum_a += value_a;
                    accumulator.sum_b += value_b;
                    accumulator.sum_delta += value_b - value_a;
                }
            }
        }
    }

    let metrics = metric_values
        .into_iter()
        .map(|(name, value)| {
            let divisor = f64::from(value.count);
            (
                name,
                MetricAggregate {
                    paired_count: value.count,
                    mean_a: Some(value.sum_a / divisor),
                    mean_b: Some(value.sum_b / divisor),
                    mean_delta: Some(value.sum_delta / divisor),
                },
            )
        })
        .collect();

    AbAggregates {
        total_queries: queries.len() as u32,
        improvement_count: status_count(queries, ComparisonStatus::Improvement),
        same_count: status_count(queries, ComparisonStatus::Same),
        regression_count: status_count(queries, ComparisonStatus::Regression),
        failure_count: status_count(queries, ComparisonStatus::Failure),
        scored_pair_count: deltas.len() as u32,
        mean_score_a: mean(&score_a),
        mean_score_b: mean(&score_b),
        mean_delta: mean(&deltas),
        metrics,
    }
}

fn status_count(queries: &[QueryComparison], status: ComparisonStatus) -> u32 {
    queries
        .iter()
        .filter(|query| query.status == status)
        .count() as u32
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn flatten_metrics(query: &ScoredQuery) -> BTreeMap<String, f64> {
    let Some(metrics) = &query.metrics else {
        return BTreeMap::new();
    };
    let mut values = BTreeMap::new();
    values.insert("composite_score".to_owned(), metrics.composite_score);
    insert_optional(
        &mut values,
        "citation.exact.precision",
        metrics.citations.exact.precision,
    );
    insert_optional(
        &mut values,
        "citation.exact.recall",
        metrics.citations.exact.recall,
    );
    insert_optional(&mut values, "citation.exact.f1", metrics.citations.exact.f1);
    insert_optional(
        &mut values,
        "citation.filename.precision",
        metrics.citations.filename.precision,
    );
    insert_optional(
        &mut values,
        "citation.filename.recall",
        metrics.citations.filename.recall,
    );
    insert_optional(
        &mut values,
        "citation.filename.f1",
        metrics.citations.filename.f1,
    );
    insert_optional(
        &mut values,
        "citation.page.precision",
        metrics.citations.page.precision,
    );
    insert_optional(
        &mut values,
        "citation.page.recall",
        metrics.citations.page.recall,
    );
    insert_optional(&mut values, "citation.page.f1", metrics.citations.page.f1);
    values.insert(
        "citation.negative_error".to_owned(),
        metrics.citations.negative_citation_error,
    );
    insert_optional(&mut values, "facts.recall", metrics.facts.recall);
    for stage in &metrics.retrieval {
        let prefix = stage.stage.as_str();
        insert_optional(&mut values, &format!("{prefix}.mrr"), stage.metrics.mrr);
        for score in &stage.metrics.source_recall_at_k {
            insert_optional(
                &mut values,
                &format!("{prefix}.source_recall@{}", score.k),
                score.value,
            );
        }
        for score in &stage.metrics.page_recall_at_k {
            insert_optional(
                &mut values,
                &format!("{prefix}.page_recall@{}", score.k),
                score.value,
            );
        }
        for score in &stage.metrics.ndcg_at_k {
            insert_optional(
                &mut values,
                &format!("{prefix}.ndcg@{}", score.k),
                score.value,
            );
        }
    }
    values
}

fn insert_optional(values: &mut BTreeMap<String, f64>, name: &str, value: Option<f64>) {
    if let Some(value) = value {
        values.insert(name.to_owned(), value);
    }
}

pub fn validate_manifest(manifest: &RunManifest) -> ReportResult<()> {
    if manifest.schema_version != REPORT_SCHEMA_VERSION {
        return Err(ReportError::new(
            "unsupported_report_version",
            "$.schema_version",
            "report schema version is not supported",
        ));
    }
    validate_run_id(&manifest.run_id)?;
    validate_safe_value(&manifest.created_at_utc, "$.created_at_utc")?;
    if !(7..=64).contains(&manifest.git.sha.len())
        || !manifest
            .git
            .sha
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ReportError::new(
            "git_sha_invalid",
            "$.git.sha",
            "git sha must contain 7 to 64 hexadecimal characters",
        ));
    }
    validate_artifact(&manifest.dataset, "$.dataset")?;
    validate_threshold_snapshot(&manifest.thresholds)?;
    for (path, value) in [
        (
            "$.locked_model_ids.embedding",
            &manifest.locked_model_ids.embedding,
        ),
        (
            "$.locked_model_ids.reranker",
            &manifest.locked_model_ids.reranker,
        ),
        (
            "$.locked_model_ids.generation_a",
            &manifest.locked_model_ids.generation_a,
        ),
        (
            "$.locked_model_ids.generation_b",
            &manifest.locked_model_ids.generation_b,
        ),
        ("$.prompt_version", &manifest.prompt_version),
        ("$.profile_id", &manifest.profile_id),
    ] {
        validate_safe_value(value, path)?;
    }
    validate_sha256(&manifest.system_prompt_sha256, "$.system_prompt_sha256")?;
    if manifest.concurrency == 0 {
        return Err(ReportError::new(
            "concurrency_invalid",
            "$.concurrency",
            "concurrency must be greater than zero",
        ));
    }
    for (capability, policy) in [
        ("embedding", &manifest.retry.embedding),
        ("rerank", &manifest.retry.rerank),
        ("generation", &manifest.retry.generation),
    ] {
        validate_retry_policy(policy, &format!("$.retry.{capability}"))?;
    }
    for (name, totals) in [
        ("ingest", &manifest.operational.ingest),
        ("query_embedding", &manifest.operational.query_embedding),
        ("rerank", &manifest.operational.rerank),
        ("generation", &manifest.operational.generation),
    ] {
        validate_operational_totals(totals, &format!("$.operational.{name}"))?;
    }
    if [
        manifest.top_n.dense,
        manifest.top_n.rerank,
        manifest.top_n.context,
    ]
    .contains(&0)
    {
        return Err(ReportError::new(
            "top_n_invalid",
            "$.top_n",
            "top_n values must be greater than zero",
        ));
    }
    if !manifest.config.comparison_epsilon.is_finite()
        || manifest.config.comparison_epsilon < 0.0
        || manifest.config.max_answer_chars == 0
        || manifest.config.request_timeout_ms == 0
    {
        return Err(ReportError::new(
            "locked_config_invalid",
            "$.config",
            "locked run configuration is invalid",
        ));
    }
    let mut cutoffs = BTreeSet::new();
    if manifest.config.retrieval_cutoffs.is_empty()
        || manifest
            .config
            .retrieval_cutoffs
            .iter()
            .any(|cutoff| *cutoff == 0 || !cutoffs.insert(*cutoff))
    {
        return Err(ReportError::new(
            "cutoffs_invalid",
            "$.config.retrieval_cutoffs",
            "retrieval cutoffs must be unique and greater than zero",
        ));
    }
    let mut subsets = BTreeSet::new();
    for (index, subset) in manifest.subsets.iter().enumerate() {
        validate_safe_value(subset, &format!("$.subsets[{index}]"))?;
        if !subsets.insert(subset) {
            return Err(ReportError::new(
                "duplicate_subset",
                format!("$.subsets[{index}]"),
                "subsets must be unique",
            ));
        }
    }
    for (index, failure) in manifest.failures.iter().enumerate() {
        if let Some(query_id) = &failure.query_id {
            validate_safe_value(query_id, &format!("$.failures[{index}].query_id"))?;
        }
        validate_safe_value(&failure.code, &format!("$.failures[{index}].code"))?;
        validate_safe_value(
            &failure.safe_message,
            &format!("$.failures[{index}].safe_message"),
        )?;
    }
    Ok(())
}

fn validate_retry_policy(policy: &RetryPolicy, path: &str) -> ReportResult<()> {
    if policy.max_attempts == 0 || policy.base_backoff_ms > policy.max_backoff_ms || policy.jitter {
        Err(ReportError::new(
            "retry_policy_invalid",
            path,
            "retry policy values are invalid or non-deterministic",
        ))
    } else {
        Ok(())
    }
}

fn validate_threshold_snapshot(snapshot: &ThresholdSnapshot) -> ReportResult<()> {
    validate_artifact(&snapshot.identity, "$.thresholds.identity")?;
    validate_threshold_config(&snapshot.config)?;
    if snapshot.identity.version != snapshot.config.version {
        return Err(ReportError::new(
            "threshold_version_mismatch",
            "$.thresholds",
            "threshold identity and configuration versions must match",
        ));
    }
    Ok(())
}

fn validate_operational_totals(totals: &OperationalStageTotals, path: &str) -> ReportResult<()> {
    if totals.retry_count > totals.request_count || totals.fallback_count > totals.request_count {
        Err(ReportError::new(
            "operational_totals_invalid",
            path,
            "retry and fallback totals cannot exceed request totals",
        ))
    } else {
        Ok(())
    }
}

fn validate_sha256(value: &str, path: &str) -> ReportResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ReportError::new(
            "sha256_invalid",
            path,
            "sha256 must contain 64 lowercase hexadecimal characters",
        ));
    }
    Ok(())
}

fn validate_artifact(artifact: &ArtifactIdentity, path: &str) -> ReportResult<()> {
    validate_sha256(&artifact.sha256, &format!("{path}.sha256"))?;
    validate_safe_value(&artifact.source, &format!("{path}.source"))?;
    validate_safe_value(&artifact.version, &format!("{path}.version"))
}

fn validate_run_id(run_id: &str) -> ReportResult<()> {
    let valid = !run_id.is_empty()
        && run_id.len() <= 128
        && run_id != "."
        && run_id != ".."
        && run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'));
    if valid {
        Ok(())
    } else {
        Err(ReportError::new(
            "run_id_invalid",
            "$.run_id",
            "run_id must be one safe path component",
        ))
    }
}

fn validate_safe_value(value: &str, path: &str) -> ReportResult<()> {
    let lowercase = value.to_ascii_lowercase();
    let sensitive = [
        "authorization",
        "bearer ",
        "api_key",
        "api-key",
        "apikey",
        "token=",
        "token:",
        "secret=",
        "secret:",
        "base64",
        "provider_response",
        "provider response",
        "://",
    ];
    if value.is_empty()
        || value.len() > 512
        || value.chars().any(char::is_control)
        || sensitive.iter().any(|marker| lowercase.contains(marker))
    {
        Err(ReportError::new(
            "unsafe_manifest_value",
            path,
            "manifest value is empty, unsafe, or contains prohibited data",
        ))
    } else {
        Ok(())
    }
}

pub fn render_report_json(report: &AbReport) -> ReportResult<String> {
    serde_json::to_string_pretty(report).map_err(|_| {
        ReportError::new(
            "report_json_failed",
            "$",
            "report could not be serialized as JSON",
        )
    })
}

pub fn render_manifest_json(manifest: &RunManifest) -> ReportResult<String> {
    validate_manifest(manifest)?;
    serde_json::to_string_pretty(manifest).map_err(|_| {
        ReportError::new(
            "manifest_json_failed",
            "$",
            "run manifest could not be serialized as JSON",
        )
    })
}

pub fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\r', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

pub fn render_report_csv(report: &AbReport) -> ReportResult<String> {
    let mut output = "record_type,identifier,status,observed,required_minimum,passed,paired_count,score_a,score_b,delta,mode_a_scored_query,mode_b_scored_query,failure_reasons\n".to_owned();
    append_csv_row(
        &mut output,
        &[
            "overall".to_owned(),
            "thresholds".to_owned(),
            if report.thresholds.overall_pass {
                "pass".to_owned()
            } else {
                "fail".to_owned()
            },
            String::new(),
            String::new(),
            report.thresholds.overall_pass.to_string(),
            report.thresholds.valid_pair_count.to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ],
    );
    for evaluation in &report.thresholds.evaluations {
        append_csv_row(
            &mut output,
            &[
                "threshold".to_owned(),
                evaluation.metric.label(),
                if evaluation.passed {
                    "pass".to_owned()
                } else {
                    "fail".to_owned()
                },
                format_number(evaluation.observed),
                format_number(Some(evaluation.required_minimum)),
                evaluation.passed.to_string(),
                evaluation.paired_count.to_string(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            ],
        );
    }
    for query in &report.queries {
        let mode_a = serialize_csv_json(&query.mode_a)?;
        let mode_b = serialize_csv_json(&query.mode_b)?;
        let fields = [
            "query".to_owned(),
            query.query_id.clone(),
            query.status.as_str().to_owned(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            format_number(query.score_a),
            format_number(query.score_b),
            format_number(query.delta),
            mode_a,
            mode_b,
            query.failure_reasons.join("|"),
        ];
        append_csv_row(&mut output, &fields);
    }
    Ok(output)
}

fn append_csv_row(output: &mut String, fields: &[String]) {
    output.push_str(
        &fields
            .iter()
            .map(|field| csv_escape(field))
            .collect::<Vec<_>>()
            .join(","),
    );
    output.push('\n');
}

fn serialize_csv_json<T: Serialize>(value: &T) -> ReportResult<String> {
    serde_json::to_string(value).map_err(|_| {
        ReportError::new(
            "report_csv_failed",
            "$",
            "report could not be serialized as CSV",
        )
    })
}

fn format_number(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.8}"))
}

pub fn render_report_markdown(report: &AbReport) -> String {
    let mut output = String::new();
    let aggregate = &report.aggregates;
    writeln!(&mut output, "# A/B evaluation report\n").expect("write String");
    writeln!(&mut output, "- Run: {}", markdown_cell(&report.run_id)).expect("write String");
    writeln!(
        &mut output,
        "- Overall threshold result: **{}**",
        if report.thresholds.overall_pass {
            "PASS"
        } else {
            "FAIL"
        }
    )
    .expect("write String");
    writeln!(
        &mut output,
        "- Valid A/B pairs: {}",
        report.thresholds.valid_pair_count
    )
    .expect("write String");
    writeln!(&mut output, "- Queries: {}", aggregate.total_queries).expect("write String");
    writeln!(
        &mut output,
        "- Improvement / same / regression / failure: {} / {} / {} / {}\n",
        aggregate.improvement_count,
        aggregate.same_count,
        aggregate.regression_count,
        aggregate.failure_count
    )
    .expect("write String");
    output.push_str("## Thresholds\n\n");
    output.push_str("| Metric | Observed | Required minimum | Pairs | Result |\n");
    output.push_str("|---|---:|---:|---:|---:|\n");
    for evaluation in &report.thresholds.evaluations {
        writeln!(
            &mut output,
            "| {} | {} | {:.8} | {} | {} |",
            markdown_cell(&evaluation.metric.label()),
            format_number(evaluation.observed),
            evaluation.required_minimum,
            evaluation.paired_count,
            if evaluation.passed { "PASS" } else { "FAIL" }
        )
        .expect("write String");
    }
    output.push('\n');
    output.push_str("| Query | Status | A | B | Delta |\n|---|---:|---:|---:|---:|\n");
    for query in &report.queries {
        writeln!(
            &mut output,
            "| {} | {} | {} | {} | {} |",
            markdown_cell(&query.query_id),
            query.status.as_str(),
            format_number(query.score_a),
            format_number(query.score_b),
            format_number(query.delta)
        )
        .expect("write String");
    }
    for query in &report.queries {
        writeln!(&mut output, "\n## {}\n", markdown_cell(&query.query_id)).expect("write String");
        writeln!(&mut output, "Status: **{}**", query.status.as_str()).expect("write String");
        if !query.failure_reasons.is_empty() {
            writeln!(
                &mut output,
                "\nFailures: {}",
                markdown_cell(&query.failure_reasons.join(", "))
            )
            .expect("write String");
        }
        write_mode_markdown(&mut output, "A", query.mode_a.as_ref());
        write_mode_markdown(&mut output, "B", query.mode_b.as_ref());
    }
    output
}

fn write_mode_markdown(output: &mut String, label: &str, query: Option<&ScoredQuery>) {
    writeln!(output, "\n### Mode {label}\n").expect("write String");
    let Some(query) = query else {
        output.push_str("Missing result.\n");
        return;
    };
    writeln!(
        output,
        "Composite: {}\n\nAnswer: {}\n",
        query
            .metrics
            .as_ref()
            .map(|metrics| format!("{:.8}", metrics.composite_score))
            .unwrap_or_else(|| "failure".to_owned()),
        markdown_cell(&query.input.answer)
    )
    .expect("write String");
    for stage in RetrievalStage::ALL {
        let trace = query.input.stages.get(stage);
        writeln!(
            output,
            "#### {} candidates\n\nLatency {} ms; requests {}; retries {}; fallback {}.\n",
            stage.as_str(),
            trace.telemetry.latency_ms,
            trace.telemetry.request_count,
            trace.telemetry.retry_count,
            trace
                .telemetry
                .fallback
                .as_ref()
                .map(|fallback| fallback.reason_code.as_str())
                .unwrap_or("none")
        )
        .expect("write String");
        output.push_str("| Rank | Document | Unit | Filename | Page | Score |\n");
        output.push_str("|---:|---|---|---|---:|---:|\n");
        for candidate in &trace.candidates {
            writeln!(
                output,
                "| {} | {} | {} | {} | {} | {:.8} |",
                candidate.rank,
                markdown_cell(&candidate.document_key),
                markdown_cell(&candidate.unit_id),
                markdown_cell(&candidate.filename),
                candidate
                    .page
                    .map_or_else(|| "null".to_owned(), |page| page.to_string()),
                candidate.score
            )
            .expect("write String");
        }
    }
}

fn markdown_cell(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "\\|")
        .replace("\r\n", "<br>")
        .replace(['\r', '\n'], "<br>")
}

pub fn write_report_bundle(
    base_directory: impl AsRef<Path>,
    bundle: &ReportBundle,
) -> ReportResult<ReportPaths> {
    validate_manifest(&bundle.manifest)?;
    if bundle.report.run_id != bundle.manifest.run_id {
        return Err(ReportError::new(
            "run_id_mismatch",
            "$.report.run_id",
            "report and manifest run_id values must match",
        ));
    }
    let report_json = render_report_json(&bundle.report)?;
    let report_csv = render_report_csv(&bundle.report)?;
    let report_markdown = render_report_markdown(&bundle.report);
    let run_manifest = render_manifest_json(&bundle.manifest)?;

    fs::create_dir_all(base_directory.as_ref()).map_err(|_| {
        ReportError::new(
            "output_base_create_failed",
            "$.output",
            "output base directory could not be created",
        )
    })?;
    let base = base_directory.as_ref().canonicalize().map_err(|_| {
        ReportError::new(
            "output_base_invalid",
            "$.output",
            "output base directory could not be resolved",
        )
    })?;
    let final_directory = base.join(&bundle.manifest.run_id);
    if final_directory.exists() {
        return Err(ReportError::new(
            "output_exists",
            "$.output",
            "run output directory already exists",
        ));
    }

    let suffix = NEXT_STAGING_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let staging = base.join(format!(
        ".{}.tmp-{}-{suffix}",
        bundle.manifest.run_id,
        std::process::id()
    ));
    fs::create_dir(&staging).map_err(|_| {
        ReportError::new(
            "output_staging_failed",
            "$.output",
            "staging output directory could not be created",
        )
    })?;
    let writes = [
        ("report.json", report_json.as_bytes()),
        ("report.csv", report_csv.as_bytes()),
        ("report.md", report_markdown.as_bytes()),
        ("run-manifest.json", run_manifest.as_bytes()),
    ];
    for (filename, bytes) in writes {
        if write_new_file(&staging.join(filename), bytes).is_err() {
            let _ = fs::remove_dir_all(&staging);
            return Err(ReportError::new(
                "output_write_failed",
                "$.output",
                "report output file could not be written",
            ));
        }
    }
    if fs::rename(&staging, &final_directory).is_err() {
        let _ = fs::remove_dir_all(&staging);
        return Err(ReportError::new(
            "output_publish_failed",
            "$.output",
            "report output directory could not be published",
        ));
    }
    if let Ok(directory) = fs::File::open(&base) {
        let _ = directory.sync_all();
    }
    Ok(ReportPaths {
        report_json: final_directory.join("report.json"),
        report_csv: final_directory.join("report.csv"),
        report_markdown: final_directory.join("report.md"),
        run_manifest: final_directory.join("run-manifest.json"),
        directory: final_directory,
    })
}

fn write_new_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::super::scoring::{
        ContextSource, ExpectedSource, FallbackTelemetry, ManualReview, ManualReviewStatus,
        QueryExpectation, QueryRunInput, RetrievalCandidate, RetrievalTrace, RunFailure,
        ScoringConfig, StageTelemetry, StageTrace, score_query,
    };
    use super::*;

    fn manifest() -> RunManifest {
        RunManifest {
            schema_version: REPORT_SCHEMA_VERSION.to_owned(),
            run_id: "run-20260713".to_owned(),
            created_at_utc: "2026-07-13T04:00:00Z".to_owned(),
            git: GitRevision {
                sha: "0123456789abcdef0123456789abcdef01234567".to_owned(),
                dirty: true,
            },
            dataset: ArtifactIdentity {
                sha256: "a".repeat(64),
                source: "backend/eval/fixtures/manifest.json".to_owned(),
                version: "1.0".to_owned(),
            },
            thresholds: ThresholdSnapshot {
                identity: ArtifactIdentity {
                    sha256: "b".repeat(64),
                    source: "evaluation-thresholds.json".to_owned(),
                    version: "thresholds-2026-07-13".to_owned(),
                },
                config: threshold_config(),
            },
            locked_model_ids: LockedModelIds {
                embedding: "embed-v1".to_owned(),
                reranker: "rerank-v1".to_owned(),
                generation_a: "answer-a-v1".to_owned(),
                generation_b: "answer-b-v1".to_owned(),
            },
            seed: 42,
            concurrency: 2,
            retry: CapabilityRetryPolicies {
                embedding: RetryPolicy {
                    max_attempts: 3,
                    base_backoff_ms: 100,
                    max_backoff_ms: 1_000,
                    jitter: false,
                },
                rerank: RetryPolicy {
                    max_attempts: 3,
                    base_backoff_ms: 100,
                    max_backoff_ms: 1_000,
                    jitter: false,
                },
                generation: RetryPolicy {
                    max_attempts: 1,
                    base_backoff_ms: 0,
                    max_backoff_ms: 0,
                    jitter: false,
                },
            },
            operational: OperationalSummary {
                ingest: operational_totals(),
                query_embedding: operational_totals(),
                rerank: operational_totals(),
                generation: operational_totals(),
            },
            subsets: vec!["all".to_owned()],
            top_n: TopNConfig {
                dense: 20,
                rerank: 10,
                context: 5,
            },
            prompt_version: "prompt-v1".to_owned(),
            profile_id: "default".to_owned(),
            system_prompt_sha256: "c".repeat(64),
            provider_seed_support: ProviderSeedSupport {
                mode_a: true,
                mode_b: false,
            },
            config: LockedRunConfig {
                retrieval_cutoffs: vec![1, 3, 5],
                comparison_epsilon: 0.001,
                max_answer_chars: 4_000,
                request_timeout_ms: 30_000,
                citation_policy: CitationPolicy::ActualBracketedSourcesOnly,
                fact_match_policy: FactMatchPolicy::DeterministicNormalizedSubstring,
                negative_answer_policy: NegativeAnswerPolicy::DeterministicAbstentionPhrases,
            },
            failures: vec![],
        }
    }

    fn operational_totals() -> OperationalStageTotals {
        OperationalStageTotals {
            latency_ms: 100,
            request_count: 2,
            retry_count: 1,
            fallback_count: 0,
        }
    }

    fn threshold_config() -> ThresholdConfig {
        ThresholdConfig {
            schema_version: THRESHOLD_SCHEMA_VERSION.to_owned(),
            version: "thresholds-2026-07-13".to_owned(),
            minimums: ThresholdMinimums {
                correct_page_recall_at_k: CorrectPageRecallThreshold {
                    stage: RetrievalStage::Rerank,
                    k: 1,
                    minimum: 0.5,
                },
                citation_precision: 0.5,
                citation_recall: 0.5,
                answer_facts_recall: 0.5,
                b_relative_improvement: 0.0,
            },
        }
    }

    fn stage(good: bool) -> StageTrace {
        StageTrace {
            candidates: vec![RetrievalCandidate {
                document_key: if good { "doc" } else { "wrong" }.to_owned(),
                unit_id: "page-1".to_owned(),
                filename: "report.pdf".to_owned(),
                page: Some(1),
                rank: 1,
                score: 1.0,
            }],
            telemetry: StageTelemetry {
                latency_ms: 10,
                request_count: 1,
                retry_count: 0,
                fallback: None::<FallbackTelemetry>,
            },
        }
    }

    fn scored(query_id: &str, mode: EvaluationMode, good: bool, failed: bool) -> ScoredQuery {
        let trace = stage(good);
        let mut sources = BTreeMap::new();
        sources.insert(
            "S1".to_owned(),
            ContextSource {
                document_key: "doc".to_owned(),
                unit_id: "page-1".to_owned(),
                filename: "report.pdf".to_owned(),
                page: Some(1),
            },
        );
        score_query(
            QueryRunInput {
                query_id: query_id.to_owned(),
                mode,
                answer: if good {
                    "required fact [S1]"
                } else {
                    "wrong, \"answer\"\nnext"
                }
                .to_owned(),
                expectation: QueryExpectation {
                    answerable: true,
                    expected_sources: vec![ExpectedSource {
                        document_key: "doc".to_owned(),
                        unit_id: "page-1".to_owned(),
                        filename: "report.pdf".to_owned(),
                        page: Some(1),
                    }],
                    required_facts: vec!["required fact".to_owned()],
                },
                stages: RetrievalTrace {
                    dense: trace.clone(),
                    rerank: trace,
                },
                context_sources: sources,
                manual_review: ManualReview {
                    status: ManualReviewStatus::Pending,
                    reviewer: None,
                    notes: None,
                    reviewed_at: None,
                },
                failure: failed.then(|| RunFailure {
                    code: "generation_failed".to_owned(),
                    stage: None,
                    safe_message: "generation did not complete".to_owned(),
                    retryable: true,
                }),
            },
            &ScoringConfig { cutoffs: vec![1] },
        )
        .expect("score test query")
    }

    fn report() -> ReportBundle {
        build_ab_report(
            manifest(),
            vec![
                scored("improved", EvaluationMode::A, false, false),
                scored("improved", EvaluationMode::B, true, false),
                scored("regressed", EvaluationMode::A, true, false),
                scored("regressed", EvaluationMode::B, false, false),
                scored("failed", EvaluationMode::A, true, false),
                scored("failed", EvaluationMode::B, true, true),
            ],
        )
        .expect("build report")
    }

    fn passing_report() -> ReportBundle {
        build_ab_report(
            manifest(),
            vec![
                scored("passing", EvaluationMode::A, true, false),
                scored("passing", EvaluationMode::B, true, false),
            ],
        )
        .expect("build passing report")
    }

    #[test]
    fn json_round_trip_preserves_scored_queries_and_candidates() {
        let bundle = report();
        let json = render_report_json(&bundle.report).expect("render JSON");
        let decoded: AbReport = serde_json::from_str(&json).expect("decode JSON");
        assert_eq!(decoded, bundle.report);
        assert_eq!(
            decoded.queries[0]
                .mode_a
                .as_ref()
                .unwrap()
                .input
                .stages
                .dense
                .candidates
                .len(),
            1
        );
    }

    #[test]
    fn csv_escapes_commas_quotes_and_newlines() {
        assert_eq!(csv_escape("a,\"b\"\nnext"), "\"a,\"\"b\"\"\nnext\"");
        let csv = render_report_csv(&report().report).expect("render CSV");
        assert!(csv.contains("wrong,"));
        assert!(csv.contains("\"\"answer"));
        assert!(csv.contains("\\nnext"));
    }

    #[test]
    fn comparisons_include_improvement_regression_and_failure() {
        let report = report().report;
        assert_eq!(report.aggregates.improvement_count, 1);
        assert_eq!(report.aggregates.regression_count, 1);
        assert_eq!(report.aggregates.failure_count, 1);
        assert_eq!(report.queries[0].status, ComparisonStatus::Failure);
        assert_eq!(report.queries[1].status, ComparisonStatus::Improvement);
        assert_eq!(report.queries[2].status, ComparisonStatus::Regression);
    }

    #[test]
    fn markdown_contains_stage_candidates() {
        let markdown = render_report_markdown(&report().report);
        assert!(markdown.contains("dense candidates"));
        assert!(markdown.contains("report.pdf"));
        assert!(markdown.contains("improvement"));
    }

    #[test]
    fn threshold_snapshot_loads_validates_and_hashes_exact_input() {
        let bytes = serde_json::to_vec_pretty(&threshold_config()).expect("serialize thresholds");
        let suffix = NEXT_STAGING_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "knowva-threshold-snapshot-{}-{suffix}.json",
            std::process::id()
        ));
        fs::write(&path, &bytes).expect("write thresholds");
        let snapshot = load_threshold_snapshot(&path).expect("load thresholds");
        fs::remove_file(path).expect("remove thresholds");

        assert_eq!(snapshot.config, threshold_config());
        assert_eq!(
            snapshot.identity.sha256,
            super::super::schema::sha256_hex(&bytes)
        );
        assert_eq!(snapshot.identity.version, snapshot.config.version);
    }

    #[test]
    fn threshold_validation_rejects_invalid_values() {
        let mut config = threshold_config();
        config.minimums.citation_precision = f64::NAN;
        assert_eq!(
            validate_threshold_config(&config).unwrap_err().code,
            "threshold_value_invalid"
        );
        config.minimums.citation_precision = 1.01;
        assert_eq!(
            validate_threshold_config(&config).unwrap_err().code,
            "threshold_value_invalid"
        );
    }

    #[test]
    fn thresholds_report_observed_values_and_overall_pass_in_all_formats() {
        let report = passing_report().report;
        assert!(report.thresholds.overall_pass);
        assert_eq!(report.thresholds.valid_pair_count, 1);
        assert_eq!(report.thresholds.evaluations.len(), 5);
        assert!(
            report
                .thresholds
                .evaluations
                .iter()
                .all(|evaluation| evaluation.passed && evaluation.observed.is_some())
        );

        let json = render_report_json(&report).expect("render JSON");
        assert!(json.contains("\"overall_pass\": true"));
        assert!(json.contains("\"observed\": 1.0"));
        let csv = render_report_csv(&report).expect("render CSV");
        assert!(csv.contains("threshold,citation_precision,pass,1.00000000,0.50000000,true,1"));
        assert!(csv.contains("overall,thresholds,pass,,,true,1"));
        let markdown = render_report_markdown(&report);
        assert!(markdown.contains("Overall threshold result: **PASS**"));
        assert!(markdown.contains("| citation_precision | 1.00000000 | 0.50000000 | 1 | PASS |"));
    }

    #[test]
    fn thresholds_fail_without_valid_pairs_or_when_any_minimum_is_missed() {
        let empty = build_ab_report(manifest(), vec![]).expect("build empty report");
        assert!(!empty.report.thresholds.overall_pass);
        assert_eq!(empty.report.thresholds.valid_pair_count, 0);
        assert!(
            empty
                .report
                .thresholds
                .evaluations
                .iter()
                .all(|evaluation| !evaluation.passed)
        );

        let mut strict = manifest();
        strict.thresholds.config.minimums.b_relative_improvement = 0.01;
        let report = build_ab_report(
            strict,
            vec![
                scored("same", EvaluationMode::A, true, false),
                scored("same", EvaluationMode::B, true, false),
            ],
        )
        .expect("build strict report");
        assert!(!report.report.thresholds.overall_pass);
        let relative = report
            .report
            .thresholds
            .evaluations
            .iter()
            .find(|evaluation| evaluation.metric == ThresholdMetric::BRelativeImprovement)
            .expect("relative threshold");
        assert_eq!(relative.observed, Some(0.0));
        assert!(!relative.passed);
    }

    #[test]
    fn relative_improvement_uses_aggregate_means_and_preserves_pair_count() {
        let aggregates = AbAggregates {
            total_queries: 3,
            improvement_count: 3,
            same_count: 0,
            regression_count: 0,
            failure_count: 0,
            scored_pair_count: 3,
            mean_score_a: Some(0.4),
            mean_score_b: Some(0.5),
            mean_delta: Some(0.1),
            metrics: BTreeMap::new(),
        };
        let results = evaluate_thresholds(&aggregates, &threshold_config().minimums);
        let relative = results
            .evaluations
            .iter()
            .find(|evaluation| evaluation.metric == ThresholdMetric::BRelativeImprovement)
            .expect("relative threshold");
        assert!(
            relative
                .observed
                .is_some_and(|value| (value - 0.25).abs() < f64::EPSILON)
        );
        assert_eq!(relative.paired_count, 3);
    }

    #[test]
    fn relative_improvement_has_explicit_zero_baseline_behavior() {
        assert_eq!(relative_improvement(Some(0.0), Some(0.25)), Some(1.0));
        assert_eq!(relative_improvement(Some(0.0), Some(0.0)), Some(0.0));
        assert_eq!(relative_improvement(None, Some(0.25)), None);
    }

    #[test]
    fn manifest_requires_deterministic_retry_and_valid_operational_totals() {
        let mut value = manifest();
        value.retry.generation.jitter = true;
        assert_eq!(
            validate_manifest(&value).unwrap_err().code,
            "retry_policy_invalid"
        );

        let mut value = manifest();
        value.retry.embedding.max_attempts = 0;
        let error = validate_manifest(&value).unwrap_err();
        assert_eq!(error.code, "retry_policy_invalid");
        assert_eq!(error.path, "$.retry.embedding");

        let mut value = manifest();
        value.operational.rerank.retry_count = value.operational.rerank.request_count + 1;
        assert_eq!(
            validate_manifest(&value).unwrap_err().code,
            "operational_totals_invalid"
        );
    }

    #[test]
    fn manifest_api_has_no_sensitive_fields_and_rejects_endpoint_values() {
        let mut value = manifest();
        let json = render_manifest_json(&value).expect("render manifest");
        let decoded: RunManifest = serde_json::from_str(&json).expect("decode manifest");
        assert_eq!(decoded, value);
        assert!(json.contains("\"negative_answer_policy\": \"deterministic_abstention_phrases\""));
        assert!(json.contains("\"generation\": {"));
        for forbidden in [
            "\"token\"",
            "\"endpoint\"",
            "\"base64\"",
            "\"document_content\"",
            "\"provider_response\"",
        ] {
            assert!(!json.contains(forbidden));
        }
        value.dataset.source = "https://example.invalid/dataset".to_owned();
        assert_eq!(
            validate_manifest(&value).unwrap_err().code,
            "unsafe_manifest_value"
        );
    }

    #[test]
    fn manifest_records_the_profile_and_a_well_formed_prompt_hash() {
        let json = render_manifest_json(&manifest()).expect("render manifest");
        assert!(json.contains("\"profile_id\": \"default\""), "{json}");
        assert!(
            json.contains(&format!("\"system_prompt_sha256\": \"{}\"", "c".repeat(64))),
            "{json}"
        );

        let mut value = manifest();
        value.system_prompt_sha256 = "C".repeat(64);
        let error = validate_manifest(&value).unwrap_err();
        assert_eq!(error.code, "sha256_invalid");
        assert_eq!(error.path, "$.system_prompt_sha256");

        let mut value = manifest();
        value.profile_id = "https://example.invalid/profile".to_owned();
        let error = validate_manifest(&value).unwrap_err();
        assert_eq!(error.code, "unsafe_manifest_value");
        assert_eq!(error.path, "$.profile_id");
    }
}
