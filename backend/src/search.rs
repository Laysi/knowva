use crate::AppState;
use crate::cloud_models::{
    CloudModelClient, CloudModelError, CloudModelResult, MAX_EMBED_IMAGE_BYTES, RerankPassage,
    RerankResponse,
};
use crate::page_vision::page_uses_image;
use crate::storage::{IO_CONCURRENCY, Storage, StorageError};
use futures::future::BoxFuture;
use futures::{StreamExt, TryStreamExt, stream};
use pgvector::HalfVector;
use serde::Serialize;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateMetadata {
    pub unit_id: i64,
    pub unit_index: i32,
    pub document_id: Uuid,
    pub filename: String,
    pub kind: String,
    pub page_number: Option<i32>,
    pub image_path: Option<String>,
    pub content: Option<String>,
    pub needs_vision: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankedCandidate {
    pub metadata: CandidateMetadata,
    pub rank: usize,
    pub score: f64,
}

/// Complete page/unit-level contract passed from retrieval to reranking/answering.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalCandidate {
    pub unit_id: i64,
    pub unit_index: i32,
    pub document_id: Uuid,
    pub filename: String,
    pub kind: String,
    pub page_number: Option<i32>,
    pub image_path: Option<String>,
    pub content: Option<String>,
    /// Set when model stages must see the page image; text-only pages are
    /// reranked and answered from `content` alone.
    pub needs_vision: bool,
    pub dense_rank: Option<usize>,
    pub dense_score: Option<f64>,
    pub rerank_score: Option<f64>,
    pub rerank_rank: Option<usize>,
}

impl RetrievalCandidate {
    /// Whether model stages send this candidate's page image.
    pub fn uses_page_image(&self) -> bool {
        self.kind == "page" && page_uses_image(self.needs_vision, self.content.as_deref())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RerankConfig {
    pub candidate_top_k: usize,
    pub top_n: usize,
    pub batch_size: usize,
}

impl RerankConfig {
    pub fn new(candidate_top_k: usize, top_n: usize, batch_size: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            candidate_top_k > 0,
            "candidate_top_k must be greater than zero"
        );
        anyhow::ensure!(
            (crate::cloud_models::MIN_RERANK_TOP_N..=crate::cloud_models::MAX_RERANK_TOP_N)
                .contains(&top_n),
            "top_n must be between {} and {}",
            crate::cloud_models::MIN_RERANK_TOP_N,
            crate::cloud_models::MAX_RERANK_TOP_N
        );
        anyhow::ensure!(
            candidate_top_k >= top_n,
            "candidate_top_k ({candidate_top_k}) must be at least top_n ({top_n})"
        );
        anyhow::ensure!(batch_size > 0, "batch_size must be greater than zero");
        anyhow::ensure!(
            batch_size <= crate::cloud_models::MAX_RERANK_BATCH_SIZE,
            "batch_size must not exceed {}",
            crate::cloud_models::MAX_RERANK_BATCH_SIZE
        );
        Ok(Self {
            candidate_top_k,
            top_n,
            batch_size,
        })
    }

    fn from_client(client: &CloudModelClient) -> Self {
        Self {
            candidate_top_k: client.rerank_candidate_top_k(),
            top_n: client.rerank_top_n(),
            batch_size: client.rerank_batch_size(),
        }
    }
}

#[derive(Debug)]
pub enum RerankStageError {
    Artifact {
        unit_id: i64,
        stage: &'static str,
        message: String,
    },
    InvalidCandidate {
        unit_id: i64,
        message: String,
    },
    Model(CloudModelError),
    Mapping(String),
}

impl RerankStageError {
    fn artifact(unit_id: i64, stage: &'static str, error: impl fmt::Display) -> Self {
        Self::Artifact {
            unit_id,
            stage,
            message: error.to_string(),
        }
    }

    fn is_transient(&self) -> bool {
        matches!(self, Self::Model(error) if error.is_retryable())
    }

    fn safe_kind(&self) -> &'static str {
        match self {
            Self::Artifact { .. } => "artifact",
            Self::InvalidCandidate { .. } => "invalid_candidate",
            Self::Mapping(_) => "contract_mapping",
            Self::Model(CloudModelError::Http { .. }) => "model_http",
            Self::Model(CloudModelError::Timeout { .. }) => "model_timeout",
            Self::Model(CloudModelError::Transport { .. }) => "model_transport",
            Self::Model(CloudModelError::InvalidRequest { .. }) => "model_invalid_request",
            Self::Model(CloudModelError::InvalidResponse { .. }) => "model_invalid_response",
        }
    }

    fn http_status(&self) -> Option<u16> {
        match self {
            Self::Model(error) => error.status().map(|status| status.as_u16()),
            _ => None,
        }
    }

    fn unit_id(&self) -> Option<i64> {
        match self {
            Self::Artifact { unit_id, .. } | Self::InvalidCandidate { unit_id, .. } => {
                Some(*unit_id)
            }
            Self::Model(_) | Self::Mapping(_) => None,
        }
    }

    fn client_message(&self) -> String {
        match self {
            Self::Artifact { unit_id, stage, .. } => format!(
                "精排頁面檔案無法讀取（錯誤代碼：rerank_artifact；unit：{unit_id}；stage：{stage}）"
            ),
            Self::InvalidCandidate { unit_id, .. } => {
                format!("精排候選資料無效（錯誤代碼：rerank_invalid_candidate；unit：{unit_id}）")
            }
            Self::Mapping(_) => {
                "精排結果無法對應來源（錯誤代碼：rerank_contract_mapping）".to_string()
            }
            Self::Model(CloudModelError::Http { status, .. }) => format!(
                "精排服務拒絕請求（錯誤代碼：rerank_model_http；HTTP {}）",
                status.as_u16()
            ),
            Self::Model(CloudModelError::Timeout { .. }) => {
                "精排服務逾時（錯誤代碼：rerank_model_timeout）".to_string()
            }
            Self::Model(CloudModelError::Transport { .. }) => {
                "精排服務連線失敗（錯誤代碼：rerank_model_transport）".to_string()
            }
            Self::Model(CloudModelError::InvalidRequest { .. }) => {
                "精排服務請求設定無效（錯誤代碼：rerank_model_invalid_request）".to_string()
            }
            Self::Model(CloudModelError::InvalidResponse { .. }) => {
                "精排服務回應格式無效（錯誤代碼：rerank_model_invalid_response）".to_string()
            }
        }
    }
}

impl fmt::Display for RerankStageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact {
                unit_id,
                stage,
                message,
            } => write!(
                formatter,
                "rerank artifact {stage} failed for unit {unit_id}: {message}"
            ),
            Self::InvalidCandidate { unit_id, message } => {
                write!(
                    formatter,
                    "invalid rerank candidate unit {unit_id}: {message}"
                )
            }
            Self::Model(error) => write!(formatter, "rerank model failed: {error}"),
            Self::Mapping(message) => {
                write!(formatter, "rerank response mapping failed: {message}")
            }
        }
    }
}

impl std::error::Error for RerankStageError {}

#[derive(Debug)]
pub enum CandidateRetrievalError {
    DenseModel(CloudModelError),
    Dense(anyhow::Error),
    Catalog(anyhow::Error),
    Configuration(anyhow::Error),
}

impl CandidateRetrievalError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::DenseModel(error) if error.is_retryable())
    }
}

impl fmt::Display for CandidateRetrievalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DenseModel(error) => write!(formatter, "dense model retrieval failed: {error}"),
            Self::Dense(error) => write!(formatter, "dense database retrieval failed: {error}"),
            Self::Catalog(error) => write!(formatter, "catalog database retrieval failed: {error}"),
            Self::Configuration(error) => {
                write!(formatter, "retrieval configuration failed: {error}")
            }
        }
    }
}

impl std::error::Error for CandidateRetrievalError {}

#[derive(Debug)]
pub enum RetrieveError {
    Candidates(CandidateRetrievalError),
    Rerank(RerankStageError),
}

impl RetrieveError {
    pub fn client_message(&self) -> String {
        match self {
            Self::Candidates(_) => {
                "知識庫檢索暫時無法使用（錯誤代碼：candidate_retrieval）".to_string()
            }
            Self::Rerank(error) => error.client_message(),
        }
    }

    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Candidates(error) => error.is_retryable(),
            Self::Rerank(error) => error.is_transient(),
        }
    }
}

impl fmt::Display for RetrieveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Candidates(error) => write!(formatter, "candidate retrieval failed: {error}"),
            Self::Rerank(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RetrieveError {}

pub trait RerankBackend: Send + Sync {
    fn model_id(&self) -> &str;

    fn rerank<'a>(
        &'a self,
        query: &'a str,
        passages: &'a [RerankPassage],
    ) -> BoxFuture<'a, CloudModelResult<RerankResponse>>;
}

impl RerankBackend for CloudModelClient {
    fn model_id(&self) -> &str {
        self.rerank_model()
    }

    fn rerank<'a>(
        &'a self,
        query: &'a str,
        passages: &'a [RerankPassage],
    ) -> BoxFuture<'a, CloudModelResult<RerankResponse>> {
        Box::pin(async move { CloudModelClient::rerank(self, query, passages).await })
    }
}

#[derive(Debug, sqlx::FromRow)]
struct CatalogDocumentRow {
    document_id: Uuid,
    filename: String,
    summary: Option<String>,
    tags: Option<Vec<String>>,
    unit_id: Option<i64>,
}

/// The text a catalog answer sees for one document. Opening that citation
/// (`GET /api/documents/{id}/units/{unitId}/catalog`) shows the same text.
pub fn catalog_card(filename: &str, summary: Option<&str>, tags: Option<&[String]>) -> String {
    let mut content = format!("文件名稱：{filename}");
    if let Some(summary) = summary.filter(|value| !value.trim().is_empty()) {
        content.push_str(&format!("\n摘要：{}", summary.trim()));
    }
    if let Some(tags) = tags.filter(|values| !values.is_empty()) {
        content.push_str(&format!("\n標籤：{}", tags.join("、")));
    }
    content
}

/// Returns one compact, text-only source per ready document for catalog questions.
pub async fn catalog_candidates(pool: &sqlx::PgPool) -> anyhow::Result<Vec<RetrievalCandidate>> {
    let rows = sqlx::query_as::<_, CatalogDocumentRow>(
        "SELECT d.id AS document_id, d.filename, d.summary, d.tags, MIN(u.id) AS unit_id \
         FROM documents AS d \
         LEFT JOIN document_units AS u ON u.document_id = d.id AND u.retrieval_ready \
         WHERE d.status = 'ready' \
         GROUP BY d.id, d.filename, d.summary, d.tags, d.created_at \
         ORDER BY d.created_at, d.id",
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| {
            let content = catalog_card(&row.filename, row.summary.as_deref(), row.tags.as_deref());
            let unit_id = row.unit_id.unwrap_or(0);
            RetrievalCandidate {
                unit_id,
                unit_index: 0,
                document_id: row.document_id,
                filename: row.filename,
                kind: "text".to_string(),
                page_number: None,
                image_path: None,
                content: Some(content),
                needs_vision: false,
                dense_rank: None,
                dense_score: None,
                rerank_score: None,
                rerank_rank: Some(index + 1),
            }
        })
        .collect::<Vec<_>>())
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct ObservedRetrievalStage {
    pub candidates: Vec<RetrievalCandidate>,
    pub latency_ms: u64,
    /// Logical backend calls made for this stage. A reused shared stage has zero.
    pub request_count: u32,
    pub reused: bool,
    pub fallback: bool,
}

impl ObservedRetrievalStage {
    fn empty() -> Self {
        Self {
            candidates: Vec::new(),
            latency_ms: 0,
            request_count: 0,
            reused: false,
            fallback: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct ObservedModeRetrieval {
    pub dense: ObservedRetrievalStage,
    pub rerank: ObservedRetrievalStage,
    pub final_candidates: Vec<RetrievalCandidate>,
    pub failure: Option<ObservedRetrievalFailure>,
}

/// A mode failure after the shared dense stage, which only rerank can cause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedRetrievalFailure {
    pub code: &'static str,
    pub safe_message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct ObservedAbRetrieval {
    pub mode_a: ObservedModeRetrieval,
    pub mode_b: ObservedModeRetrieval,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservedRerankOutcome {
    pub candidates: Vec<RetrievalCandidate>,
    /// Logical batch calls into the rerank backend. HTTP attempts, including
    /// retries, are exposed separately by `CloudModelClient` telemetry.
    pub request_count: u32,
    pub fallback: bool,
}

/// Injectable boundary around the dense candidate stage.
pub trait CandidateRetrievalBackend: Send + Sync {
    fn dense_candidates<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> BoxFuture<'a, Result<Vec<RankedCandidate>, CandidateRetrievalError>>;
}

struct PostgresRetrievalBackend<'a> {
    state: &'a AppState,
}

struct CountingRerankBackend<'a, B: RerankBackend + ?Sized> {
    inner: &'a B,
    calls: AtomicU32,
}

impl<'a, B: RerankBackend + ?Sized> CountingRerankBackend<'a, B> {
    fn new(inner: &'a B) -> Self {
        Self {
            inner,
            calls: AtomicU32::new(0),
        }
    }

    fn request_count(&self) -> u32 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl<B: RerankBackend + ?Sized> RerankBackend for CountingRerankBackend<'_, B> {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn rerank<'a>(
        &'a self,
        query: &'a str,
        passages: &'a [RerankPassage],
    ) -> BoxFuture<'a, CloudModelResult<RerankResponse>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.rerank(query, passages)
    }
}

impl CandidateRetrievalBackend for PostgresRetrievalBackend<'_> {
    fn dense_candidates<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> BoxFuture<'a, Result<Vec<RankedCandidate>, CandidateRetrievalError>> {
        Box::pin(async move {
            let embedding = self
                .state
                .models
                .embed_query(query)
                .await
                .map_err(CandidateRetrievalError::DenseModel)?;
            let embedding =
                crate::embedding::to_halfvec(embedding).map_err(CandidateRetrievalError::Dense)?;
            dense_candidates_from_pool(&self.state.pool, &embedding, top_k)
                .await
                .map_err(CandidateRetrievalError::Dense)
        })
    }
}

#[derive(Debug, sqlx::FromRow)]
struct CandidateRow {
    unit_id: i64,
    unit_index: i32,
    document_id: Uuid,
    filename: String,
    kind: String,
    page_number: Option<i32>,
    image_path: Option<String>,
    content: Option<String>,
    needs_vision: bool,
    score: f64,
}

impl CandidateRow {
    fn into_ranked(self, rank: usize) -> RankedCandidate {
        RankedCandidate {
            metadata: CandidateMetadata {
                unit_id: self.unit_id,
                unit_index: self.unit_index,
                document_id: self.document_id,
                filename: self.filename,
                kind: self.kind,
                page_number: self.page_number,
                image_path: self.image_path,
                content: self.content,
                needs_vision: self.needs_vision,
            },
            rank,
            score: self.score,
        }
    }
}

pub(crate) async fn dense_candidates_from_pool(
    pool: &sqlx::PgPool,
    query_embedding: &HalfVector,
    top_k: usize,
) -> anyhow::Result<Vec<RankedCandidate>> {
    let limit = i64::try_from(top_k)?;
    let rows = sqlx::query_as::<_, CandidateRow>(
        "WITH nearest AS MATERIALIZED ( \
           SELECT u.id AS unit_id, u.unit_index, u.document_id, u.kind, u.page_number, \
                  u.image_path, u.content, u.needs_vision, \
                  (u.embedding <=> $1)::double precision AS distance \
           FROM document_units AS u \
           WHERE u.retrieval_ready \
           ORDER BY u.embedding <=> $1 \
           LIMIT $2 \
         ) \
         SELECT nearest.unit_id, nearest.unit_index, nearest.document_id, d.filename, nearest.kind, \
                nearest.page_number, nearest.image_path, nearest.content, nearest.needs_vision, \
                (1.0 - nearest.distance)::double precision AS score \
         FROM nearest \
         JOIN documents AS d ON d.id = nearest.document_id \
         WHERE d.status = 'ready' \
         ORDER BY nearest.distance, nearest.unit_id \
         LIMIT $2",
    )
    .bind(query_embedding)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| row.into_ranked(index + 1))
        .collect())
}

/// Evaluates the control and treatment retrieval paths over one shared dense call.
///
/// Mode A is strictly dense-only: its context is the dense prefix and its
/// rerank observation is empty with zero requests. Mode B reuses the exact
/// dense candidate values and reranks them with the locked rerank backend,
/// which is the production path. The reused B dense observation has
/// `request_count = 0`, so summing the two mode observations still represents
/// the one physical dense backend call.
#[cfg_attr(not(test), allow(dead_code))]
pub async fn retrieve_ab_with_backends<
    C: CandidateRetrievalBackend + ?Sized,
    R: RerankBackend + ?Sized,
>(
    candidate_backend: &C,
    rerank_backend: &R,
    query: &str,
    storage: &Storage,
    rerank_config: RerankConfig,
) -> Result<ObservedAbRetrieval, RetrieveError> {
    let rerank_config = RerankConfig::new(
        rerank_config.candidate_top_k,
        rerank_config.top_n,
        rerank_config.batch_size,
    )
    .map_err(|error| RetrieveError::Candidates(CandidateRetrievalError::Configuration(error)))?;

    let dense_started = Instant::now();
    let dense = candidate_backend
        .dense_candidates(query, rerank_config.candidate_top_k)
        .await
        .map_err(RetrieveError::Candidates)?;
    let dense_latency_ms = dense_started.elapsed().as_millis() as u64;
    let dense_candidates = dense_route_candidates(&dense);
    let mode_a_final = dense_candidates
        .iter()
        .take(rerank_config.top_n)
        .cloned()
        .collect::<Vec<_>>();
    let mode_a_dense = ObservedRetrievalStage {
        candidates: dense_candidates.clone(),
        latency_ms: dense_latency_ms,
        request_count: 1,
        reused: false,
        fallback: false,
    };
    let mode_b_dense = ObservedRetrievalStage {
        candidates: dense_candidates,
        latency_ms: dense_latency_ms,
        request_count: 0,
        reused: true,
        fallback: false,
    };
    let mode_a = ObservedModeRetrieval {
        dense: mode_a_dense,
        rerank: ObservedRetrievalStage::empty(),
        final_candidates: mode_a_final,
        failure: None,
    };

    let rerank_started = Instant::now();
    let counted_reranker = CountingRerankBackend::new(rerank_backend);
    let reranked = match rerank_candidates_observed(
        &counted_reranker,
        query,
        &mode_b_dense.candidates,
        storage,
        rerank_config,
    )
    .await
    {
        Ok(reranked) => reranked,
        Err(error) => {
            return Ok(ObservedAbRetrieval {
                mode_a,
                mode_b: ObservedModeRetrieval {
                    dense: mode_b_dense,
                    rerank: ObservedRetrievalStage {
                        candidates: Vec::new(),
                        latency_ms: rerank_started.elapsed().as_millis() as u64,
                        request_count: counted_reranker.request_count(),
                        reused: false,
                        fallback: false,
                    },
                    final_candidates: Vec::new(),
                    failure: Some(ObservedRetrievalFailure {
                        code: "rerank_failed",
                        safe_message: error.client_message(),
                        retryable: error.is_transient(),
                    }),
                },
            });
        }
    };
    let mode_b_rerank = ObservedRetrievalStage {
        candidates: reranked.candidates.clone(),
        latency_ms: rerank_started.elapsed().as_millis() as u64,
        request_count: reranked.request_count,
        reused: false,
        fallback: reranked.fallback,
    };

    Ok(ObservedAbRetrieval {
        mode_a,
        mode_b: ObservedModeRetrieval {
            dense: mode_b_dense,
            rerank: mode_b_rerank,
            final_candidates: reranked.candidates,
            failure: None,
        },
    })
}

/// Production-backed entry point for the evaluation runner. The PostgreSQL
/// backend embeds the query inside its one dense call; the B path only reuses
/// those candidates and therefore never embeds the query again.
#[allow(dead_code)]
pub async fn retrieve_ab_observed(
    state: &AppState,
    query: &str,
) -> Result<ObservedAbRetrieval, RetrieveError> {
    let backend = PostgresRetrievalBackend { state };
    retrieve_ab_with_backends(
        &backend,
        &state.models,
        query,
        &state.storage,
        RerankConfig::from_client(&state.models),
    )
    .await
}

/// Retrieves the dense candidate pool and returns its top 3-8 candidates,
/// reranked with the locked VL model unless rerank is disabled.
pub async fn retrieve(
    state: &AppState,
    query: &str,
) -> Result<Vec<RetrievalCandidate>, RetrieveError> {
    let backend = PostgresRetrievalBackend { state };
    retrieve_with_backends(
        &backend,
        &state.models,
        query,
        &state.storage,
        state.models.rerank_enabled(),
        RerankConfig::from_client(&state.models),
    )
    .await
}

/// With rerank the dense pool is the first `candidate_top_k` units; without it
/// only the first `top_n` are fetched and returned in dense order. Exhausted
/// transient rerank failures fall back to the unchanged dense prefix;
/// contract/artifact failures remain diagnostic errors.
async fn retrieve_with_backends<
    C: CandidateRetrievalBackend + ?Sized,
    R: RerankBackend + ?Sized,
>(
    candidate_backend: &C,
    rerank_backend: &R,
    query: &str,
    storage: &Storage,
    rerank: bool,
    rerank_config: RerankConfig,
) -> Result<Vec<RetrievalCandidate>, RetrieveError> {
    let pool_size = if rerank {
        rerank_config.candidate_top_k
    } else {
        rerank_config.top_n
    };
    let started = Instant::now();
    let dense = candidate_backend
        .dense_candidates(query, pool_size)
        .await
        .map_err(RetrieveError::Candidates)?;
    tracing::info!(
        stage = "dense",
        candidate_count = dense.len(),
        latency_ms = started.elapsed().as_millis() as u64,
        "retrieval stage completed"
    );
    let pool = dense_route_candidates(&dense);
    if !rerank {
        return Ok(pool.into_iter().take(rerank_config.top_n).collect());
    }
    rerank_candidates(rerank_backend, query, &pool, storage, rerank_config)
        .await
        .map_err(RetrieveError::Rerank)
}

pub async fn rerank_candidates<B: RerankBackend + ?Sized>(
    backend: &B,
    query: &str,
    pool: &[RetrievalCandidate],
    storage: &Storage,
    config: RerankConfig,
) -> Result<Vec<RetrievalCandidate>, RerankStageError> {
    rerank_candidates_observed(backend, query, pool, storage, config)
        .await
        .map(|outcome| outcome.candidates)
}

pub async fn rerank_candidates_observed<B: RerankBackend + ?Sized>(
    backend: &B,
    query: &str,
    pool: &[RetrievalCandidate],
    storage: &Storage,
    config: RerankConfig,
) -> Result<ObservedRerankOutcome, RerankStageError> {
    let started = Instant::now();
    let candidate_count = pool.len().min(config.candidate_top_k);
    let result = rerank_candidates_inner(backend, query, pool, storage, config).await;
    if let Err(error) = &result {
        tracing::error!(
            stage = "rerank",
            candidate_count,
            latency_ms = started.elapsed().as_millis() as u64,
            fallback = false,
            model_id = backend.model_id(),
            error_kind = error.safe_kind(),
            http_status = ?error.http_status(),
            unit_id = ?error.unit_id(),
            "rerank failed permanently"
        );
    }
    result
}

async fn rerank_candidates_inner<B: RerankBackend + ?Sized>(
    backend: &B,
    query: &str,
    pool: &[RetrievalCandidate],
    storage: &Storage,
    config: RerankConfig,
) -> Result<ObservedRerankOutcome, RerankStageError> {
    let config = RerankConfig::new(config.candidate_top_k, config.top_n, config.batch_size)
        .map_err(|error| RerankStageError::Mapping(error.to_string()))?;
    let candidates = &pool[..pool.len().min(config.candidate_top_k)];
    if candidates.is_empty() {
        tracing::info!(
            stage = "rerank",
            candidate_count = 0,
            output_count = 0,
            batch_count = 0,
            latency_ms = 0,
            fallback = false,
            model_id = backend.model_id(),
            "retrieval stage completed"
        );
        return Ok(ObservedRerankOutcome {
            candidates: Vec::new(),
            request_count: 0,
            fallback: false,
        });
    }

    let started = Instant::now();
    let batches = modality_batches(candidates, config.batch_size);
    let batch_count = batches.len();
    let image_passage_count = candidates
        .iter()
        .filter(|candidate| candidate.uses_page_image())
        .count();
    let text_passage_count = candidates.len() - image_passage_count;
    let mut scores = vec![None; candidates.len()];
    let mut request_count = 0_u32;

    for (batch_index, batch) in batches.iter().enumerate() {
        let batch_started = Instant::now();
        let batch_candidates = batch
            .iter()
            .map(|index| &candidates[*index])
            .collect::<Vec<_>>();
        let passages = prepare_rerank_passages(&batch_candidates, storage).await?;
        request_count = request_count.saturating_add(1);
        let response = match backend.rerank(query, &passages).await {
            Ok(response) => response,
            Err(error) => {
                let stage_error = RerankStageError::Model(error);
                if stage_error.is_transient() {
                    tracing::warn!(
                        stage = "rerank",
                        candidate_count = candidates.len(),
                        batch_index,
                        batch_count,
                        fallback = true,
                        model_id = backend.model_id(),
                        error_kind = stage_error.safe_kind(),
                        http_status = ?stage_error.http_status(),
                        "rerank transient failure exhausted retries; using the dense order"
                    );
                    let fallback = dense_fallback(candidates, config.top_n);
                    tracing::info!(
                        stage = "rerank",
                        candidate_count = candidates.len(),
                        image_passage_count,
                        text_passage_count,
                        output_count = fallback.len(),
                        batch_count,
                        latency_ms = started.elapsed().as_millis() as u64,
                        fallback = true,
                        model_id = backend.model_id(),
                        "retrieval stage completed"
                    );
                    return Ok(ObservedRerankOutcome {
                        candidates: fallback,
                        request_count,
                        fallback: true,
                    });
                }
                return Err(stage_error);
            }
        };
        validate_batch_rankings(&response, batch.len(), batch_index)?;

        for ranking in response.rankings {
            // Validation above guarantees the index is inside this batch.
            let global_index = batch[ranking.index];
            let slot = scores.get_mut(global_index).ok_or_else(|| {
                RerankStageError::Mapping(format!(
                    "batch {batch_index} mapped outside the candidate list"
                ))
            })?;
            if slot.replace(ranking.logit).is_some() {
                return Err(RerankStageError::Mapping(format!(
                    "candidate index {global_index} received more than one score"
                )));
            }
        }
        tracing::info!(
            stage = "rerank_batch",
            batch_index,
            batch_candidate_count = batch.len(),
            latency_ms = batch_started.elapsed().as_millis() as u64,
            model_id = backend.model_id(),
            "rerank batch completed"
        );
    }

    if let Some(missing_index) = scores.iter().position(Option::is_none) {
        return Err(RerankStageError::Mapping(format!(
            "candidate index {missing_index} did not receive a score"
        )));
    }

    let mut reranked = candidates
        .iter()
        .cloned()
        .enumerate()
        .map(|(pool_index, mut candidate)| {
            candidate.rerank_score = scores[pool_index];
            (pool_index, candidate)
        })
        .collect::<Vec<_>>();
    reranked.sort_by(|(left_pool_index, left), (right_pool_index, right)| {
        right
            .rerank_score
            .expect("all rerank scores were validated")
            .total_cmp(&left.rerank_score.expect("all rerank scores were validated"))
            .then_with(|| left_pool_index.cmp(right_pool_index))
    });
    let mut reranked = reranked
        .into_iter()
        .enumerate()
        .map(|(index, (_, mut candidate))| {
            candidate.rerank_rank = Some(index + 1);
            candidate
        })
        .collect::<Vec<_>>();
    reranked.truncate(config.top_n);

    tracing::info!(
        stage = "rerank",
        candidate_count = candidates.len(),
        image_passage_count,
        text_passage_count,
        output_count = reranked.len(),
        batch_count,
        latency_ms = started.elapsed().as_millis() as u64,
        fallback = false,
        model_id = backend.model_id(),
        "retrieval stage completed"
    );
    Ok(ObservedRerankOutcome {
        candidates: reranked,
        request_count,
        fallback: false,
    })
}

/// Splits candidates into batches of one modality, each listing candidate
/// indices in pool order. A rerank request is processed as a single modality,
/// so mixing would make text passages pay the image cost and let their scores
/// depend on batch-mates. Groups follow the order of their first candidate, so
/// a single-modality list batches exactly as plain chunking would.
fn modality_batches(candidates: &[RetrievalCandidate], batch_size: usize) -> Vec<Vec<usize>> {
    let mut groups: Vec<(bool, Vec<usize>)> = Vec::with_capacity(2);
    for (index, candidate) in candidates.iter().enumerate() {
        let uses_image = candidate.uses_page_image();
        match groups.iter_mut().find(|(group, _)| *group == uses_image) {
            Some((_, indices)) => indices.push(index),
            None => groups.push((uses_image, vec![index])),
        }
    }
    groups
        .into_iter()
        .flat_map(|(_, indices)| {
            indices
                .chunks(batch_size)
                .map(<[usize]>::to_vec)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn validate_batch_rankings(
    response: &RerankResponse,
    expected_count: usize,
    batch_index: usize,
) -> Result<(), RerankStageError> {
    if response.rankings.len() != expected_count {
        return Err(RerankStageError::Mapping(format!(
            "batch {batch_index} expected {expected_count} rankings, received {}",
            response.rankings.len()
        )));
    }
    let mut seen = vec![false; expected_count];
    for ranking in &response.rankings {
        if ranking.index >= expected_count {
            return Err(RerankStageError::Mapping(format!(
                "batch {batch_index} returned out-of-range index {}",
                ranking.index
            )));
        }
        if seen[ranking.index] {
            return Err(RerankStageError::Mapping(format!(
                "batch {batch_index} returned duplicate index {}",
                ranking.index
            )));
        }
        if !ranking.logit.is_finite() {
            return Err(RerankStageError::Mapping(format!(
                "batch {batch_index} returned a non-finite score for index {}",
                ranking.index
            )));
        }
        seen[ranking.index] = true;
    }
    Ok(())
}

/// Prepares one batch with bounded concurrent page reads. `buffered` keeps the
/// original candidate order and reports the first failure in that order.
async fn prepare_rerank_passages(
    candidates: &[&RetrievalCandidate],
    storage: &Storage,
) -> Result<Vec<RerankPassage>, RerankStageError> {
    // Futures are built eagerly (they stay lazy until polled) so the stream type
    // carries no borrowing closure; otherwise spawned callers fail the Send check.
    let passages = candidates
        .iter()
        .map(|candidate| prepare_rerank_passage(candidate, storage))
        .collect::<Vec<_>>();
    stream::iter(passages)
        .buffered(IO_CONCURRENCY)
        .try_collect()
        .await
}

async fn prepare_rerank_passage(
    candidate: &RetrievalCandidate,
    storage: &Storage,
) -> Result<RerankPassage, RerankStageError> {
    let invalid = |message: String| RerankStageError::InvalidCandidate {
        unit_id: candidate.unit_id,
        message,
    };
    match candidate.kind.as_str() {
        "page" if !candidate.uses_page_image() => candidate
            .content
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(RerankPassage::text)
            .ok_or_else(|| invalid("text-only page is missing non-empty content".to_string())),
        "page" => {
            let key = candidate
                .image_path
                .as_deref()
                .ok_or_else(|| invalid("page candidate is missing image_path".to_string()))?;
            let image = storage
                .get(candidate.document_id, key, MAX_EMBED_IMAGE_BYTES)
                .await
                .map_err(|error| match error {
                    StorageError::InvalidKey(_) | StorageError::TooLarge { .. } => {
                        invalid(error.to_string())
                    }
                    error => RerankStageError::artifact(candidate.unit_id, "read", error),
                })?;
            match candidate
                .content
                .as_deref()
                .map(str::trim)
                .filter(|text| !text.is_empty())
            {
                Some(text) => RerankPassage::image_text(text, &image),
                None => RerankPassage::image(&image),
            }
            .map_err(RerankStageError::Model)
        }
        "text" => candidate
            .content
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(RerankPassage::text)
            .ok_or_else(|| invalid("text candidate is missing non-empty content".to_string())),
        kind => Err(invalid(format!("unsupported candidate kind {kind:?}"))),
    }
}

fn dense_route_candidates(candidates: &[RankedCandidate]) -> Vec<RetrievalCandidate> {
    candidates
        .iter()
        .map(|ranked| RetrievalCandidate {
            unit_id: ranked.metadata.unit_id,
            unit_index: ranked.metadata.unit_index,
            document_id: ranked.metadata.document_id,
            filename: ranked.metadata.filename.clone(),
            kind: ranked.metadata.kind.clone(),
            page_number: ranked.metadata.page_number,
            image_path: ranked.metadata.image_path.clone(),
            content: ranked.metadata.content.clone(),
            needs_vision: ranked.metadata.needs_vision,
            dense_rank: Some(ranked.rank),
            dense_score: Some(ranked.score),
            rerank_score: None,
            rerank_rank: None,
        })
        .collect()
}

fn dense_fallback(candidates: &[RetrievalCandidate], top_n: usize) -> Vec<RetrievalCandidate> {
    candidates
        .iter()
        .take(top_n)
        .cloned()
        .map(|mut candidate| {
            candidate.rerank_score = None;
            candidate.rerank_rank = None;
            candidate
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud_models::{ModelOperation, RerankResult};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    fn metadata(unit_id: i64) -> CandidateMetadata {
        CandidateMetadata {
            unit_id,
            unit_index: unit_id as i32 - 1,
            document_id: Uuid::from_u128(unit_id as u128 + 100),
            filename: format!("document-{unit_id}.pdf"),
            kind: "page".to_string(),
            page_number: Some(unit_id as i32),
            image_path: Some(format!("/pages/{unit_id}.png")),
            content: Some(format!("content {unit_id}")),
            needs_vision: true,
        }
    }

    fn ranked(unit_id: i64, rank: usize, score: f64) -> RankedCandidate {
        RankedCandidate {
            metadata: metadata(unit_id),
            rank,
            score,
        }
    }

    fn text_ranked(unit_id: i64, rank: usize, score: f64) -> RankedCandidate {
        let mut candidate = ranked(unit_id, rank, score);
        candidate.metadata.kind = "text".to_owned();
        candidate.metadata.page_number = None;
        candidate.metadata.image_path = None;
        candidate
    }

    fn retrieval_candidate(
        unit_id: i64,
        kind: &str,
        image_path: Option<String>,
        content: Option<&str>,
    ) -> RetrievalCandidate {
        RetrievalCandidate {
            unit_id,
            unit_index: unit_id as i32 - 1,
            document_id: Uuid::from_u128(unit_id as u128 + 100),
            filename: format!("document-{unit_id}.pdf"),
            kind: kind.to_string(),
            page_number: (kind == "page").then_some(unit_id as i32),
            image_path,
            content: content.map(str::to_string),
            needs_vision: kind == "page",
            dense_rank: Some(unit_id as usize),
            dense_score: Some(1.0 / unit_id as f64),
            rerank_score: None,
            rerank_rank: None,
        }
    }

    struct FakeReranker {
        responses: Mutex<VecDeque<CloudModelResult<RerankResponse>>>,
        requests: Mutex<Vec<serde_json::Value>>,
    }

    impl FakeReranker {
        fn new(responses: Vec<CloudModelResult<RerankResponse>>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    impl RerankBackend for FakeReranker {
        fn model_id(&self) -> &str {
            crate::cloud_models::DEFAULT_RERANK_MODEL_ID
        }

        fn rerank<'a>(
            &'a self,
            _query: &'a str,
            passages: &'a [RerankPassage],
        ) -> BoxFuture<'a, CloudModelResult<RerankResponse>> {
            Box::pin(async move {
                self.requests
                    .lock()
                    .expect("requests lock")
                    .push(serde_json::to_value(passages).expect("serialize passages"));
                if let Some(response) = self.responses.lock().expect("responses lock").pop_front() {
                    return response;
                }
                Ok(RerankResponse {
                    rankings: (0..passages.len())
                        .rev()
                        .map(|index| RerankResult {
                            index,
                            logit: index as f64,
                        })
                        .collect(),
                })
            })
        }
    }

    #[test]
    fn dense_route_keeps_rank_score_and_metadata() {
        let candidates = dense_route_candidates(&[ranked(10, 1, 0.9), ranked(20, 2, 0.8)]);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| (
                    candidate.unit_id,
                    candidate.dense_rank,
                    candidate.dense_score
                ))
                .collect::<Vec<_>>(),
            vec![(10, Some(1), Some(0.9)), (20, Some(2), Some(0.8))]
        );
        assert_eq!(candidates[0].filename, "document-10.pdf");
        assert_eq!(candidates[0].page_number, Some(10));
        assert!(candidates[0].needs_vision);
    }

    #[test]
    fn serialized_candidate_keeps_complete_nullable_contract() {
        let candidates = dense_route_candidates(&[ranked(1, 1, 0.9)]);
        let value = serde_json::to_value(&candidates[0]).expect("serialize candidate");
        let fields = [
            "unitId",
            "unitIndex",
            "documentId",
            "filename",
            "kind",
            "pageNumber",
            "imagePath",
            "content",
            "needsVision",
            "denseRank",
            "denseScore",
            "rerankScore",
            "rerankRank",
        ];
        assert_eq!(
            value.as_object().expect("candidate object").len(),
            fields.len()
        );
        for field in fields {
            assert!(
                value.get(field).is_some(),
                "missing candidate field {field}"
            );
        }
        assert!(value["rerankScore"].is_null());
        assert!(value["rerankRank"].is_null());
    }

    #[test]
    fn rerank_config_rejects_a_pool_below_top_n_and_oversized_batches() {
        assert!(RerankConfig::new(5, 6, 8).is_err());
        assert!(
            RerankConfig::new(
                crate::cloud_models::MAX_RERANK_BATCH_SIZE + 1,
                3,
                crate::cloud_models::MAX_RERANK_BATCH_SIZE + 1,
            )
            .is_err()
        );
        assert!(RerankConfig::new(6, 6, 8).is_ok());
    }

    fn text_only_page(unit_id: i64, image_path: String, content: &str) -> RetrievalCandidate {
        RetrievalCandidate {
            needs_vision: false,
            ..retrieval_candidate(unit_id, "page", Some(image_path), Some(content))
        }
    }

    #[tokio::test]
    async fn rerank_payload_covers_every_modality_in_single_modality_batches() {
        let storage = Storage::memory();
        let mixed_image = format!("{}/artifacts-run/pages/0001.png", Uuid::from_u128(101));
        let image_only = format!("{}/artifacts-run/pages/0002.png", Uuid::from_u128(102));
        // Never stored: a text-only page must not read its image.
        let text_page_image = format!("{}/artifacts-run/pages/0004.png", Uuid::from_u128(104));
        let png = b"\x89PNG\r\n\x1a\nmock-rerank-page";
        storage
            .put(&mixed_image, png.to_vec())
            .await
            .expect("store mixed image");
        storage
            .put(&image_only, png.to_vec())
            .await
            .expect("store image-only image");
        let candidates = vec![
            retrieval_candidate(1, "page", Some(mixed_image.clone()), Some("same-page text")),
            retrieval_candidate(2, "page", Some(image_only.clone()), None),
            retrieval_candidate(3, "text", None, Some("markdown chunk")),
            text_only_page(4, text_page_image.clone(), "  text-only page  "),
        ];
        let reranker = FakeReranker::new(Vec::new());
        let result = rerank_candidates(
            &reranker,
            "query",
            &candidates,
            &storage,
            RerankConfig::new(4, 4, 4).expect("config"),
        )
        .await
        .expect("rerank all modalities");
        assert_eq!(result.len(), 4);
        for candidate in &result {
            let original = candidates
                .iter()
                .find(|original| original.unit_id == candidate.unit_id)
                .expect("reranked unit must come from the original candidate list");
            assert_eq!(candidate.document_id, original.document_id);
            assert_eq!(candidate.filename, original.filename);
            assert_eq!(candidate.page_number, original.page_number);
            assert_eq!(candidate.image_path, original.image_path);
            assert_eq!(candidate.content, original.content);
            assert_eq!(candidate.needs_vision, original.needs_vision);
        }

        {
            let requests = reranker.requests.lock().expect("requests lock");
            // One request per modality, even though all four fit one batch.
            assert_eq!(requests.len(), 2);
            let image_passages = requests[0].as_array().expect("image passages array");
            assert_eq!(image_passages.len(), 2);
            assert_eq!(image_passages[0]["text"], "same-page text");
            assert!(
                image_passages[0]["image"]
                    .as_str()
                    .is_some_and(|image| image.starts_with("data:image/png;base64,"))
            );
            assert!(image_passages[1].get("text").is_none());
            assert!(
                image_passages[1]["image"]
                    .as_str()
                    .is_some_and(|image| image.starts_with("data:image/png;base64,"))
            );
            let text_passages = requests[1].as_array().expect("text passages array");
            assert_eq!(
                text_passages,
                &vec![
                    serde_json::json!({ "text": "markdown chunk" }),
                    serde_json::json!({ "text": "text-only page" }),
                ]
            );
            assert!(image_passages.iter().all(|passage| {
                !passage
                    .get("image")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|image| {
                        image.contains(&mixed_image) || image.contains(&image_only)
                    })
            }));
        }
    }

    #[tokio::test]
    async fn interleaved_modalities_map_scores_back_to_their_candidates() {
        let storage = Storage::memory();
        let png = b"\x89PNG\r\n\x1a\nmock-rerank-page";
        let mut candidates = Vec::new();
        for unit_id in 1..=4 {
            let image_path = format!(
                "{}/artifacts-run/pages/{unit_id:04}.png",
                Uuid::from_u128(unit_id as u128 + 100)
            );
            let content = format!("page {unit_id}");
            if unit_id % 2 == 1 {
                storage
                    .put(&image_path, png.to_vec())
                    .await
                    .expect("store vision page image");
                candidates.push(retrieval_candidate(
                    unit_id,
                    "page",
                    Some(image_path),
                    Some(&content),
                ));
            } else {
                candidates.push(text_only_page(unit_id, image_path, &content));
            }
        }
        // Pool order is [1 vision, 2 text, 3 vision, 4 text]; batches are [1, 3]
        // then [2, 4]. Batch-local index 0 of the second batch is unit 2.
        let responses = vec![
            Ok(RerankResponse {
                rankings: vec![
                    RerankResult {
                        index: 0,
                        logit: 0.1,
                    },
                    RerankResult {
                        index: 1,
                        logit: 0.7,
                    },
                ],
            }),
            Ok(RerankResponse {
                rankings: vec![
                    RerankResult {
                        index: 1,
                        logit: 0.4,
                    },
                    RerankResult {
                        index: 0,
                        logit: 0.9,
                    },
                ],
            }),
        ];
        let reranker = FakeReranker::new(responses);
        let result = rerank_candidates(
            &reranker,
            "query",
            &candidates,
            &storage,
            RerankConfig::new(4, 4, 2).expect("config"),
        )
        .await
        .expect("rerank interleaved modalities");

        assert_eq!(
            result
                .iter()
                .map(|candidate| (candidate.unit_id, candidate.rerank_score))
                .collect::<Vec<_>>(),
            vec![
                (2, Some(0.9)),
                (3, Some(0.7)),
                (4, Some(0.4)),
                (1, Some(0.1))
            ]
        );
        let requests = reranker.requests.lock().expect("requests lock");
        assert_eq!(requests.len(), 2);
        assert!(
            requests[0]
                .as_array()
                .expect("vision batch")
                .iter()
                .all(|passage| passage.get("image").is_some())
        );
        assert_eq!(
            requests[1],
            serde_json::json!([{ "text": "page 2" }, { "text": "page 4" }])
        );
    }

    #[test]
    fn single_modality_lists_batch_like_plain_chunking() {
        let pages = (1..=5)
            .map(|unit_id| retrieval_candidate(unit_id, "page", None, Some("text")))
            .collect::<Vec<_>>();
        assert_eq!(
            modality_batches(&pages, 2),
            vec![vec![0, 1], vec![2, 3], vec![4]]
        );
        let mixed = vec![
            retrieval_candidate(1, "text", None, Some("chunk")),
            retrieval_candidate(2, "page", None, Some("text")),
            retrieval_candidate(3, "text", None, Some("chunk")),
        ];
        // Groups follow the pool position of their first member.
        assert_eq!(modality_batches(&mixed, 8), vec![vec![0, 2], vec![1]]);
    }

    #[test]
    fn blank_text_only_pages_still_use_their_image() {
        let blank = RetrievalCandidate {
            needs_vision: false,
            ..retrieval_candidate(1, "page", None, Some("   "))
        };
        assert!(blank.uses_page_image());
        assert!(!text_only_page(2, "page.png".to_string(), "text").uses_page_image());
        assert!(!retrieval_candidate(3, "text", None, Some("chunk")).uses_page_image());
    }

    #[tokio::test]
    async fn cross_batch_reordering_keeps_scores_on_original_metadata() {
        let candidates = (1..=5)
            .map(|unit_id| {
                retrieval_candidate(unit_id, "text", None, Some(&format!("chunk {unit_id}")))
            })
            .collect::<Vec<_>>();
        let responses = vec![
            Ok(RerankResponse {
                rankings: vec![
                    RerankResult {
                        index: 1,
                        logit: 0.2,
                    },
                    RerankResult {
                        index: 0,
                        logit: 0.1,
                    },
                ],
            }),
            Ok(RerankResponse {
                rankings: vec![
                    RerankResult {
                        index: 1,
                        logit: 0.3,
                    },
                    RerankResult {
                        index: 0,
                        logit: 0.9,
                    },
                ],
            }),
            Ok(RerankResponse {
                rankings: vec![RerankResult {
                    index: 0,
                    logit: 0.4,
                }],
            }),
        ];
        let reranker = FakeReranker::new(responses);
        let result = rerank_candidates(
            &reranker,
            "query",
            &candidates,
            &Storage::memory(),
            RerankConfig::new(5, 3, 2).expect("config"),
        )
        .await
        .expect("cross-batch rerank");

        assert_eq!(
            result
                .iter()
                .map(|candidate| candidate.unit_id)
                .collect::<Vec<_>>(),
            vec![3, 5, 4]
        );
        assert_eq!(
            result
                .iter()
                .map(|candidate| candidate.rerank_score)
                .collect::<Vec<_>>(),
            vec![Some(0.9), Some(0.4), Some(0.3)]
        );
        assert_eq!(
            result
                .iter()
                .map(|candidate| candidate.rerank_rank)
                .collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(3)]
        );
        for candidate in &result {
            assert_eq!(
                candidate.document_id,
                Uuid::from_u128(candidate.unit_id as u128 + 100)
            );
            assert_eq!(
                candidate.content.as_deref(),
                Some(format!("chunk {}", candidate.unit_id).as_str())
            );
        }
        assert_eq!(reranker.requests.lock().expect("requests lock").len(), 3);
    }

    #[tokio::test]
    async fn equal_rerank_scores_keep_the_original_dense_order() {
        let candidates = (1..=4)
            .map(|unit_id| retrieval_candidate(unit_id, "text", None, Some("chunk")))
            .collect::<Vec<_>>();
        let responses = vec![
            Ok(RerankResponse {
                rankings: vec![
                    RerankResult {
                        index: 1,
                        logit: 0.5,
                    },
                    RerankResult {
                        index: 0,
                        logit: 0.5,
                    },
                ],
            }),
            Ok(RerankResponse {
                rankings: vec![
                    RerankResult {
                        index: 1,
                        logit: 0.5,
                    },
                    RerankResult {
                        index: 0,
                        logit: 0.5,
                    },
                ],
            }),
        ];
        let reranker = FakeReranker::new(responses);
        let result = rerank_candidates(
            &reranker,
            "query",
            &candidates,
            &Storage::memory(),
            RerankConfig::new(4, 3, 2).expect("config"),
        )
        .await
        .expect("equal scores must remain deterministic");

        assert_eq!(
            result
                .iter()
                .map(|candidate| candidate.unit_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            result
                .iter()
                .map(|candidate| candidate.rerank_rank)
                .collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(3)]
        );
    }

    #[tokio::test]
    async fn rerank_top_n_supports_three_six_eight_and_short_candidate_lists() {
        let candidates = (1..=10)
            .map(|unit_id| retrieval_candidate(unit_id, "text", None, Some("chunk")))
            .collect::<Vec<_>>();
        for top_n in [3, 6, 8] {
            let reranker = FakeReranker::new(Vec::new());
            let result = rerank_candidates(
                &reranker,
                "query",
                &candidates,
                &Storage::memory(),
                RerankConfig::new(10, top_n, 4).expect("config"),
            )
            .await
            .expect("supported top_n");
            assert_eq!(result.len(), top_n);
            assert!(result.iter().all(|candidate| {
                candidate.rerank_score.is_some() && candidate.rerank_rank.is_some()
            }));
        }

        let reranker = FakeReranker::new(Vec::new());
        let short = rerank_candidates(
            &reranker,
            "query",
            &candidates[..2],
            &Storage::memory(),
            RerankConfig::new(8, 8, 4).expect("config"),
        )
        .await
        .expect("candidate list shorter than top_n");
        assert_eq!(short.len(), 2);
        assert_eq!(short[0].rerank_rank, Some(1));
        assert_eq!(short[1].rerank_rank, Some(2));
    }

    #[tokio::test]
    async fn exhausted_transient_error_uses_exact_dense_prefix_without_fake_scores() {
        let candidates = (1..=6)
            .map(|unit_id| retrieval_candidate(unit_id, "text", None, Some("chunk")))
            .collect::<Vec<_>>();
        let reranker = FakeReranker::new(vec![
            Ok(RerankResponse {
                rankings: vec![
                    RerankResult {
                        index: 1,
                        logit: 0.8,
                    },
                    RerankResult {
                        index: 0,
                        logit: 0.7,
                    },
                ],
            }),
            Err(CloudModelError::Timeout {
                operation: ModelOperation::Rerank,
                timeout: std::time::Duration::from_millis(10),
            }),
        ]);
        let result = rerank_candidates(
            &reranker,
            "query",
            &candidates,
            &Storage::memory(),
            RerankConfig::new(6, 3, 2).expect("config"),
        )
        .await
        .expect("timeout should fall back");
        assert_eq!(result, candidates[..3]);
        assert!(result.iter().all(|candidate| {
            candidate.rerank_score.is_none() && candidate.rerank_rank.is_none()
        }));
    }

    #[tokio::test]
    async fn contract_errors_never_fall_back() {
        let candidates = vec![
            retrieval_candidate(1, "text", None, Some("one")),
            retrieval_candidate(2, "text", None, Some("two")),
        ];
        let invalid_responses = vec![
            RerankResponse {
                rankings: vec![RerankResult {
                    index: 0,
                    logit: 0.1,
                }],
            },
            RerankResponse {
                rankings: vec![
                    RerankResult {
                        index: 0,
                        logit: 0.1,
                    },
                    RerankResult {
                        index: 0,
                        logit: 0.2,
                    },
                ],
            },
            RerankResponse {
                rankings: vec![
                    RerankResult {
                        index: 0,
                        logit: 0.1,
                    },
                    RerankResult {
                        index: 2,
                        logit: 0.2,
                    },
                ],
            },
            RerankResponse {
                rankings: vec![
                    RerankResult {
                        index: 0,
                        logit: f64::NAN,
                    },
                    RerankResult {
                        index: 1,
                        logit: 0.2,
                    },
                ],
            },
        ];
        for response in invalid_responses {
            let reranker = FakeReranker::new(vec![Ok(response)]);
            let error = rerank_candidates(
                &reranker,
                "query",
                &candidates,
                &Storage::memory(),
                RerankConfig::new(3, 3, 2).expect("config"),
            )
            .await
            .expect_err("invalid response must be permanent");
            assert!(matches!(error, RerankStageError::Mapping(_)));
        }

        let reranker = FakeReranker::new(vec![Err(CloudModelError::InvalidResponse {
            operation: ModelOperation::Rerank,
            message: "malformed hosted response".to_string(),
        })]);
        let error = rerank_candidates(
            &reranker,
            "query",
            &candidates,
            &Storage::memory(),
            RerankConfig::new(3, 3, 2).expect("config"),
        )
        .await
        .expect_err("malformed model response must be permanent");
        assert!(matches!(
            error,
            RerankStageError::Model(CloudModelError::InvalidResponse { .. })
        ));
    }

    #[test]
    fn permanent_rerank_client_messages_do_not_expose_provider_or_artifact_details() {
        let provider_detail = "echoed private document content";
        let error = RetrieveError::Rerank(RerankStageError::Model(CloudModelError::Http {
            operation: ModelOperation::Rerank,
            status: reqwest::StatusCode::BAD_REQUEST,
            response_snippet: provider_detail.to_string(),
            retryable: false,
        }));
        let message = error.client_message();
        assert!(message.contains("rerank_model_http"));
        assert!(message.contains("400"));
        assert!(!message.contains(provider_detail));

        let local_detail = "/srv/customer-a/private/page-0001.png";
        let error = RetrieveError::Rerank(RerankStageError::Artifact {
            unit_id: 42,
            stage: "read",
            message: local_detail.to_string(),
        });
        let message = error.client_message();
        assert!(message.contains("rerank_artifact"));
        assert!(message.contains("42"));
        assert!(!message.contains(local_detail));
    }

    #[tokio::test]
    async fn page_artifact_failure_is_diagnostic_and_never_degrades_to_text_only() {
        let candidate = retrieval_candidate(
            1,
            "page",
            Some(format!(
                "{}/artifacts-run/pages/missing.png",
                Uuid::from_u128(101)
            )),
            Some("text must not mask missing image"),
        );
        let reranker = FakeReranker::new(Vec::new());
        let error = rerank_candidates(
            &reranker,
            "query",
            &[candidate],
            &Storage::memory(),
            RerankConfig::new(3, 3, 2).expect("config"),
        )
        .await
        .expect_err("missing page artifact must fail");
        assert!(matches!(error, RerankStageError::Artifact { .. }));
        assert!(reranker.requests.lock().expect("requests lock").is_empty());
    }

    #[tokio::test]
    async fn page_artifact_must_belong_to_the_candidate_document_directory() {
        let storage = Storage::memory();
        let other_document_image = format!("{}/artifacts-run/pages/0001.png", Uuid::from_u128(102));
        storage
            .put(
                &other_document_image,
                b"\x89PNG\r\n\x1a\nother-document".to_vec(),
            )
            .await
            .expect("store other document image");

        // Unit 1 maps to document UUID 101 in the fixture helper, so UUID 102
        // must never be accepted even though it is a valid key in the same store.
        for key in [
            other_document_image.clone(),
            format!("{}/../{other_document_image}", Uuid::from_u128(101)),
            format!("/{other_document_image}"),
        ] {
            let candidate = retrieval_candidate(1, "page", Some(key.clone()), None);
            let reranker = FakeReranker::new(Vec::new());
            let error = rerank_candidates(
                &reranker,
                "query",
                &[candidate],
                &storage,
                RerankConfig::new(3, 3, 2).expect("config"),
            )
            .await
            .expect_err("cross-document artifact mapping must fail");
            assert!(matches!(error, RerankStageError::InvalidCandidate { .. }));
            assert!(!error.to_string().contains(&key));
            assert!(!error.client_message().contains(&other_document_image));
            assert!(reranker.requests.lock().expect("requests lock").is_empty());
        }
    }

    #[derive(Debug)]
    struct FakeBackend {
        dense: Vec<RankedCandidate>,
        calls: Mutex<Vec<usize>>,
    }

    impl FakeBackend {
        fn new(dense: Vec<RankedCandidate>) -> Self {
            Self {
                dense,
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<usize> {
            self.calls.lock().expect("calls lock").clone()
        }
    }

    impl CandidateRetrievalBackend for FakeBackend {
        fn dense_candidates<'a>(
            &'a self,
            _query: &'a str,
            top_k: usize,
        ) -> BoxFuture<'a, Result<Vec<RankedCandidate>, CandidateRetrievalError>> {
            Box::pin(async move {
                self.calls.lock().expect("calls lock").push(top_k);
                Ok(self.dense.iter().take(top_k).cloned().collect())
            })
        }
    }

    struct TransientDenseFailureBackend;

    impl CandidateRetrievalBackend for TransientDenseFailureBackend {
        fn dense_candidates<'a>(
            &'a self,
            _query: &'a str,
            _top_k: usize,
        ) -> BoxFuture<'a, Result<Vec<RankedCandidate>, CandidateRetrievalError>> {
            Box::pin(async {
                Err(CandidateRetrievalError::DenseModel(
                    CloudModelError::Timeout {
                        operation: ModelOperation::Embed,
                        timeout: std::time::Duration::from_millis(10),
                    },
                ))
            })
        }
    }

    #[tokio::test]
    async fn transient_dense_model_failure_is_a_retryable_retrieve_error() {
        let error = retrieve_ab_with_backends(
            &TransientDenseFailureBackend,
            &FakeReranker::new(Vec::new()),
            "fixture query",
            &Storage::memory(),
            RerankConfig::new(3, 3, 8).expect("rerank config"),
        )
        .await
        .expect_err("dense timeout must fail candidate retrieval");

        assert!(matches!(
            &error,
            RetrieveError::Candidates(CandidateRetrievalError::DenseModel(
                CloudModelError::Timeout { .. }
            ))
        ));
        assert!(error.is_retryable());
    }

    #[test]
    fn database_and_permanent_candidate_errors_are_not_retryable() {
        let errors = [
            RetrieveError::Candidates(CandidateRetrievalError::Dense(anyhow::anyhow!(
                "database unavailable"
            ))),
            RetrieveError::Candidates(CandidateRetrievalError::Catalog(anyhow::anyhow!(
                "database unavailable"
            ))),
            RetrieveError::Candidates(CandidateRetrievalError::DenseModel(
                CloudModelError::InvalidResponse {
                    operation: ModelOperation::Embed,
                    message: "malformed embedding response".to_owned(),
                },
            )),
        ];

        assert!(errors.iter().all(|error| !error.is_retryable()));
    }

    fn dense_text_pool(count: i64) -> FakeBackend {
        FakeBackend::new(
            (1..=count)
                .map(|id| text_ranked(id, id as usize, 1.0 - id as f64 / 100.0))
                .collect(),
        )
    }

    #[tokio::test]
    async fn unset_retrieval_settings_rerank_the_top_twenty_dense_candidates() {
        let config = crate::cloud_models::CloudModelConfig::from_lookup(|name| match name {
            "GEMMA_4_MODEL_ID" => Some("hosted/gemma-4-test".to_owned()),
            "MODEL_BASE_URL" => Some("http://127.0.0.1:9".to_owned()),
            _ => None,
        })
        .expect("default model config");
        let client = CloudModelClient::new(config).expect("cloud client");
        let backend = dense_text_pool(30);
        let reranker = FakeReranker::new(Vec::new());

        let output = retrieve_with_backends(
            &backend,
            &reranker,
            "fixture query",
            &Storage::memory(),
            client.rerank_enabled(),
            RerankConfig::from_client(&client),
        )
        .await
        .expect("dense reranked retrieval");

        assert_eq!(backend.calls(), vec![20]);
        // Twenty text candidates are reranked, eight per request.
        let requests = reranker.requests.lock().expect("requests lock");
        assert_eq!(
            requests
                .iter()
                .map(|request| request.as_array().expect("passages").len())
                .collect::<Vec<_>>(),
            vec![8, 8, 4]
        );
        assert_eq!(output.len(), 6);
        assert!(
            output
                .iter()
                .all(|candidate| candidate.rerank_rank.is_some() && candidate.unit_id <= 20)
        );
    }

    #[tokio::test]
    async fn disabled_rerank_returns_the_dense_prefix_without_calling_the_reranker() {
        let backend = dense_text_pool(30);
        let reranker = FakeReranker::new(Vec::new());
        let output = retrieve_with_backends(
            &backend,
            &reranker,
            "fixture query",
            &Storage::memory(),
            false,
            RerankConfig::new(20, 6, 8).expect("rerank config"),
        )
        .await
        .expect("dense retrieval");

        assert_eq!(backend.calls(), vec![6]);
        assert_eq!(
            output
                .iter()
                .map(|candidate| candidate.unit_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert!(reranker.requests.lock().expect("requests lock").is_empty());
        assert!(
            output
                .iter()
                .all(|candidate| candidate.rerank_rank.is_none())
        );
    }

    #[tokio::test]
    async fn ab_observation_shares_dense_and_keeps_mode_a_strictly_dense_only() {
        let backend = FakeBackend::new(vec![text_ranked(1, 1, 0.9), text_ranked(2, 2, 0.8)]);
        let reranker = FakeReranker::new(vec![Ok(RerankResponse {
            rankings: vec![
                RerankResult {
                    index: 0,
                    logit: 0.2,
                },
                RerankResult {
                    index: 1,
                    logit: 0.9,
                },
            ],
        })]);
        let observed = retrieve_ab_with_backends(
            &backend,
            &reranker,
            "fixture query",
            &Storage::memory(),
            RerankConfig::new(3, 3, 8).expect("rerank config"),
        )
        .await
        .expect("observed A/B retrieval");

        assert_eq!(backend.calls(), vec![3]);
        assert_eq!(reranker.requests.lock().expect("requests lock").len(), 1);

        assert_eq!(observed.mode_a.dense.request_count, 1);
        assert!(!observed.mode_a.dense.reused);
        assert!(observed.mode_a.rerank.candidates.is_empty());
        assert_eq!(observed.mode_a.rerank.request_count, 0);
        assert!(!observed.mode_a.rerank.fallback);
        assert_eq!(observed.mode_a.final_candidates.len(), 2);
        assert_eq!(
            observed.mode_a.dense.candidates,
            observed.mode_b.dense.candidates
        );
        assert!(observed.mode_b.dense.reused);
        assert_eq!(observed.mode_b.dense.request_count, 0);
        assert_eq!(observed.mode_b.rerank.request_count, 1);
        assert!(!observed.mode_b.rerank.fallback);
        // A keeps the dense order; B's rerank scores put unit 2 first.
        assert_eq!(observed.mode_a.final_candidates[0].unit_id, 1);
        assert_eq!(observed.mode_b.final_candidates[0].unit_id, 2);
    }

    #[tokio::test]
    async fn observed_ab_marks_transient_rerank_fallback_explicitly() {
        let backend = FakeBackend::new(vec![text_ranked(1, 1, 0.9)]);
        let reranker = FakeReranker::new(vec![Err(CloudModelError::Timeout {
            operation: ModelOperation::Rerank,
            timeout: std::time::Duration::from_millis(1),
        })]);
        let observed = retrieve_ab_with_backends(
            &backend,
            &reranker,
            "fixture query",
            &Storage::memory(),
            RerankConfig::new(3, 3, 8).expect("rerank config"),
        )
        .await
        .expect("fallback is a successful observed outcome");

        assert!(observed.mode_b.rerank.fallback);
        assert_eq!(observed.mode_b.rerank.request_count, 1);
        assert_eq!(
            observed.mode_b.final_candidates,
            observed.mode_b.dense.candidates
        );
        assert!(
            observed
                .mode_b
                .final_candidates
                .iter()
                .all(|candidate| candidate.rerank_rank.is_none())
        );
    }

    #[tokio::test]
    async fn permanent_b_rerank_failure_preserves_the_dense_control() {
        let backend = FakeBackend::new(vec![text_ranked(1, 1, 0.9)]);
        let reranker = FakeReranker::new(vec![Err(CloudModelError::InvalidResponse {
            operation: ModelOperation::Rerank,
            message: "malformed provider payload".to_owned(),
        })]);
        let observed = retrieve_ab_with_backends(
            &backend,
            &reranker,
            "fixture query",
            &Storage::memory(),
            RerankConfig::new(3, 3, 8).expect("rerank config"),
        )
        .await
        .expect("B failure must remain an observed pair");

        assert!(observed.mode_a.failure.is_none());
        assert_eq!(observed.mode_a.final_candidates.len(), 1);
        let failure = observed.mode_b.failure.expect("B failure");
        assert_eq!(failure.code, "rerank_failed");
        assert!(!failure.retryable);
        assert_eq!(observed.mode_b.rerank.request_count, 1);
        assert!(observed.mode_b.final_candidates.is_empty());
    }

    #[tokio::test]
    async fn legacy_rerank_api_returns_the_observed_candidates_unchanged() {
        let response = RerankResponse {
            rankings: vec![RerankResult {
                index: 0,
                logit: 0.7,
            }],
        };
        let candidates = dense_route_candidates(&[text_ranked(1, 1, 0.9)]);
        let legacy = rerank_candidates(
            &FakeReranker::new(vec![Ok(RerankResponse {
                rankings: response.rankings.clone(),
            })]),
            "fixture query",
            &candidates,
            &Storage::memory(),
            RerankConfig::new(3, 3, 8).expect("rerank config"),
        )
        .await
        .expect("legacy rerank");
        let observed = rerank_candidates_observed(
            &FakeReranker::new(vec![Ok(response)]),
            "fixture query",
            &candidates,
            &Storage::memory(),
            RerankConfig::new(3, 3, 8).expect("rerank config"),
        )
        .await
        .expect("observed rerank");
        assert_eq!(legacy, observed.candidates);
        assert!(!observed.fallback);
        assert_eq!(observed.request_count, 1);
    }
}
