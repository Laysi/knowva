use crate::AppState;
use crate::cloud_models::{
    CloudModelClient, CloudModelError, CloudModelResult, MAX_EMBED_IMAGE_BYTES, RerankPassage,
    RerankResponse,
};
use futures::future::BoxFuture;
use pgvector::HalfVector;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Instant;
use uuid::Uuid;

pub const DEFAULT_DENSE_TOP_K: usize = 50;
pub const DEFAULT_BM25_TOP_K: usize = 50;
pub const DEFAULT_FINAL_TOP_N: usize = 50;
pub const RRF_K: f64 = 60.0;

// Okapi BM25 defaults used by the database-side scoring expression.
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

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
    pub dense_rank: Option<usize>,
    pub dense_score: Option<f64>,
    pub bm25_rank: Option<usize>,
    pub bm25_score: Option<f64>,
    pub rrf_score: f64,
    pub rerank_score: Option<f64>,
    pub rerank_rank: Option<usize>,
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
pub enum RetrieveError {
    Candidates(anyhow::Error),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetrievalConfig {
    pub dense_top_k: usize,
    pub bm25_top_k: usize,
    pub final_top_n: usize,
}

impl RetrievalConfig {
    pub fn new(dense_top_k: usize, bm25_top_k: usize, final_top_n: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(final_top_n > 0, "final_top_n must be greater than zero");
        anyhow::ensure!(
            dense_top_k >= final_top_n,
            "dense_top_k ({dense_top_k}) must be at least final_top_n ({final_top_n})"
        );
        anyhow::ensure!(
            bm25_top_k >= final_top_n,
            "bm25_top_k ({bm25_top_k}) must be at least final_top_n ({final_top_n})"
        );
        Ok(Self {
            dense_top_k,
            bm25_top_k,
            final_top_n,
        })
    }
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            dense_top_k: DEFAULT_DENSE_TOP_K,
            bm25_top_k: DEFAULT_BM25_TOP_K,
            final_top_n: DEFAULT_FINAL_TOP_N,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalDebug {
    pub dense: Vec<RankedCandidate>,
    pub bm25: Vec<RankedCandidate>,
    pub fused: Vec<RetrievalCandidate>,
}

/// Injectable boundary around the two independently testable candidate stages.
pub trait CandidateRetrievalBackend: Send + Sync {
    fn dense_candidates<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> BoxFuture<'a, anyhow::Result<Vec<RankedCandidate>>>;

    fn bm25_candidates<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> BoxFuture<'a, anyhow::Result<Vec<RankedCandidate>>>;
}

struct PostgresRetrievalBackend<'a> {
    state: &'a AppState,
}

impl CandidateRetrievalBackend for PostgresRetrievalBackend<'_> {
    fn dense_candidates<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> BoxFuture<'a, anyhow::Result<Vec<RankedCandidate>>> {
        Box::pin(async move {
            let embedding = self.state.models.embed_query(query).await?;
            let embedding = crate::embedding::to_halfvec(embedding)?;
            dense_candidates_from_pool(&self.state.pool, &embedding, top_k).await
        })
    }

    fn bm25_candidates<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> BoxFuture<'a, anyhow::Result<Vec<RankedCandidate>>> {
        Box::pin(async move { bm25_candidates_from_pool(&self.state.pool, query, top_k).await })
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
                  u.image_path, u.content, \
                  (u.embedding <=> $1)::double precision AS distance \
           FROM document_units AS u \
           WHERE u.retrieval_ready \
           ORDER BY u.embedding <=> $1 \
           LIMIT $2 \
         ) \
         SELECT nearest.unit_id, nearest.unit_index, nearest.document_id, d.filename, nearest.kind, \
                nearest.page_number, nearest.image_path, nearest.content, \
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

pub(crate) async fn bm25_candidates_from_pool(
    pool: &sqlx::PgPool,
    query: &str,
    top_k: usize,
) -> anyhow::Result<Vec<RankedCandidate>> {
    let limit = i64::try_from(top_k)?;
    let rows = sqlx::query_as::<_, CandidateRow>(
        "WITH query_terms AS MATERIALIZED ( \
           SELECT DISTINCT token AS term \
           FROM unnest(km_bm25_tokenize($1)) AS token \
           WHERE token <> '' \
         ), query_token_array AS ( \
           SELECT COALESCE(array_agg(term ORDER BY term), ARRAY[]::text[]) AS terms \
           FROM query_terms \
         ), candidate_units AS MATERIALIZED ( \
           SELECT u.id AS unit_id, u.unit_index, u.document_id, u.kind, \
                  u.page_number, u.image_path, u.content, u.bm25_tokens, u.bm25_length \
           FROM document_units AS u \
           CROSS JOIN query_token_array AS query \
           WHERE u.retrieval_ready \
             AND u.content IS NOT NULL \
             AND btrim(u.content) <> '' \
             AND u.bm25_tokens && query.terms \
         ), scored AS ( \
           SELECT candidate.unit_id, candidate.unit_index, candidate.document_id, document.filename, \
                  candidate.kind, candidate.page_number, candidate.image_path, \
                  candidate.content, \
                  sum( \
                    ln(1.0 + ( \
                      (greatest(corpus.document_count - term_stats.document_frequency, 0)::double precision + 0.5) \
                      / (term_stats.document_frequency::double precision + 0.5) \
                    )) \
                    * ((frequency.tf * ($3::double precision + 1.0)) \
                      / (frequency.tf + $3::double precision * ( \
                        1.0 - $4::double precision \
                        + $4::double precision * candidate.bm25_length::double precision \
                          / greatest( \
                              corpus.total_length::double precision \
                                / corpus.document_count::double precision, \
                              1.0 \
                            ) \
                      ))) \
                  )::double precision AS score \
           FROM candidate_units AS candidate \
           JOIN documents AS document ON document.id = candidate.document_id \
           CROSS JOIN km_bm25_corpus_stats AS corpus \
           JOIN query_terms AS query ON query.term = ANY(candidate.bm25_tokens) \
           JOIN km_bm25_term_stats AS term_stats ON term_stats.term = query.term \
           CROSS JOIN LATERAL ( \
             SELECT cardinality(array_positions(candidate.bm25_tokens, query.term))::double precision AS tf \
           ) AS frequency \
           WHERE corpus.singleton \
             AND document.status = 'ready' \
             AND corpus.document_count > 0 \
             AND frequency.tf > 0 \
           GROUP BY candidate.unit_id, candidate.unit_index, candidate.document_id, document.filename, \
                    candidate.kind, candidate.page_number, candidate.image_path, \
                    candidate.content \
         ) \
         SELECT unit_id, unit_index, document_id, filename, kind, page_number, image_path, content, score \
         FROM scored \
         ORDER BY score DESC, unit_id \
         LIMIT $2",
    )
    .bind(query)
    .bind(limit)
    .bind(BM25_K1)
    .bind(BM25_B)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| row.into_ranked(index + 1))
        .collect())
}

/// Runs dense and lexical retrieval once each, then performs one deterministic fusion.
pub async fn retrieve_with_backend<B: CandidateRetrievalBackend + ?Sized>(
    backend: &B,
    query: &str,
    config: RetrievalConfig,
) -> anyhow::Result<RetrievalDebug> {
    // Validate externally constructed configs too, not only values made via new().
    let config = RetrievalConfig::new(config.dense_top_k, config.bm25_top_k, config.final_top_n)?;
    let dense = async {
        let started = Instant::now();
        let candidates = backend.dense_candidates(query, config.dense_top_k).await?;
        tracing::info!(
            stage = "dense",
            candidate_count = candidates.len(),
            latency_ms = started.elapsed().as_millis() as u64,
            "retrieval stage completed"
        );
        Ok::<_, anyhow::Error>(candidates)
    };
    let bm25 = async {
        let started = Instant::now();
        let candidates = backend.bm25_candidates(query, config.bm25_top_k).await?;
        tracing::info!(
            stage = "bm25",
            candidate_count = candidates.len(),
            latency_ms = started.elapsed().as_millis() as u64,
            "retrieval stage completed"
        );
        Ok::<_, anyhow::Error>(candidates)
    };
    let (dense, bm25) = tokio::try_join!(dense, bm25)?;
    let fusion_started = Instant::now();
    let fused = reciprocal_rank_fusion(&dense, &bm25, config.final_top_n);
    tracing::info!(
        stage = "rrf",
        dense_candidate_count = dense.len(),
        bm25_candidate_count = bm25.len(),
        candidate_count = fused.len(),
        latency_ms = fusion_started.elapsed().as_millis() as u64,
        "retrieval stage completed"
    );
    Ok(RetrievalDebug { dense, bm25, fused })
}

/// Retrieves a stable RRF pool, reranks it with the locked VL model, and
/// returns the configured top 3-8 candidates. Exhausted transient model
/// failures fall back to the unchanged RRF prefix; contract/artifact failures
/// remain diagnostic errors.
pub async fn retrieve(
    state: &AppState,
    query: &str,
) -> Result<Vec<RetrievalCandidate>, RetrieveError> {
    let rerank_config = RerankConfig::from_client(&state.models);
    let route_top_k = DEFAULT_DENSE_TOP_K.max(rerank_config.candidate_top_k);
    let retrieval_config = RetrievalConfig::new(
        route_top_k,
        DEFAULT_BM25_TOP_K.max(rerank_config.candidate_top_k),
        rerank_config.candidate_top_k,
    )
    .map_err(RetrieveError::Candidates)?;
    let debug = retrieve_debug(state, query, retrieval_config)
        .await
        .map_err(RetrieveError::Candidates)?;
    rerank_candidates(
        &state.models,
        query,
        &debug.fused,
        Path::new(&state.files_dir),
        rerank_config,
    )
    .await
    .map_err(RetrieveError::Rerank)
}

/// Runs the production PostgreSQL stages while retaining both route rankings
/// for diagnostics and evaluation fixtures.
pub async fn retrieve_debug(
    state: &AppState,
    query: &str,
    config: RetrievalConfig,
) -> anyhow::Result<RetrievalDebug> {
    let backend = PostgresRetrievalBackend { state };
    retrieve_with_backend(&backend, query, config).await
}

pub async fn rerank_candidates<B: RerankBackend + ?Sized>(
    backend: &B,
    query: &str,
    rrf_candidates: &[RetrievalCandidate],
    artifact_root: &Path,
    config: RerankConfig,
) -> Result<Vec<RetrievalCandidate>, RerankStageError> {
    let started = Instant::now();
    let candidate_count = rrf_candidates.len().min(config.candidate_top_k);
    let result =
        rerank_candidates_inner(backend, query, rrf_candidates, artifact_root, config).await;
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
    rrf_candidates: &[RetrievalCandidate],
    artifact_root: &Path,
    config: RerankConfig,
) -> Result<Vec<RetrievalCandidate>, RerankStageError> {
    let config = RerankConfig::new(config.candidate_top_k, config.top_n, config.batch_size)
        .map_err(|error| RerankStageError::Mapping(error.to_string()))?;
    let candidates = &rrf_candidates[..rrf_candidates.len().min(config.candidate_top_k)];
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
        return Ok(Vec::new());
    }

    let started = Instant::now();
    let batch_count = candidates.len().div_ceil(config.batch_size);
    let canonical_root = if let Some(page) =
        candidates.iter().find(|candidate| candidate.kind == "page")
    {
        Some(
            tokio::fs::canonicalize(artifact_root)
                .await
                .map_err(|error| RerankStageError::artifact(page.unit_id, "root-resolve", error))?,
        )
    } else {
        None
    };
    let mut scores = vec![None; candidates.len()];

    for (batch_index, batch) in candidates.chunks(config.batch_size).enumerate() {
        let batch_started = Instant::now();
        let passages = prepare_rerank_passages(batch, canonical_root.as_deref()).await?;
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
                        "rerank transient failure exhausted retries; using stable RRF fallback"
                    );
                    let fallback = rrf_fallback(candidates, config.top_n);
                    tracing::info!(
                        stage = "rerank",
                        candidate_count = candidates.len(),
                        output_count = fallback.len(),
                        batch_count,
                        latency_ms = started.elapsed().as_millis() as u64,
                        fallback = true,
                        model_id = backend.model_id(),
                        "retrieval stage completed"
                    );
                    return Ok(fallback);
                }
                return Err(stage_error);
            }
        };
        validate_batch_rankings(&response, batch.len(), batch_index)?;

        let batch_start = batch_index * config.batch_size;
        for ranking in response.rankings {
            let global_index = batch_start.checked_add(ranking.index).ok_or_else(|| {
                RerankStageError::Mapping(format!(
                    "batch {batch_index} ranking index overflowed the candidate list"
                ))
            })?;
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
        .map(|(rrf_index, mut candidate)| {
            candidate.rerank_score = scores[rrf_index];
            (rrf_index, candidate)
        })
        .collect::<Vec<_>>();
    reranked.sort_by(|(left_rrf_index, left), (right_rrf_index, right)| {
        right
            .rerank_score
            .expect("all rerank scores were validated")
            .total_cmp(&left.rerank_score.expect("all rerank scores were validated"))
            .then_with(|| left_rrf_index.cmp(right_rrf_index))
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
        output_count = reranked.len(),
        batch_count,
        latency_ms = started.elapsed().as_millis() as u64,
        fallback = false,
        model_id = backend.model_id(),
        "retrieval stage completed"
    );
    Ok(reranked)
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

async fn prepare_rerank_passages(
    candidates: &[RetrievalCandidate],
    canonical_root: Option<&Path>,
) -> Result<Vec<RerankPassage>, RerankStageError> {
    let mut passages = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        match candidate.kind.as_str() {
            "page" => {
                let root = canonical_root.ok_or_else(|| RerankStageError::InvalidCandidate {
                    unit_id: candidate.unit_id,
                    message: "page candidate requires an artifact root".to_string(),
                })?;
                let image_path = candidate.image_path.as_deref().ok_or_else(|| {
                    RerankStageError::InvalidCandidate {
                        unit_id: candidate.unit_id,
                        message: "page candidate is missing image_path".to_string(),
                    }
                })?;
                let canonical_path = tokio::fs::canonicalize(PathBuf::from(image_path))
                    .await
                    .map_err(|error| {
                        RerankStageError::artifact(candidate.unit_id, "path-resolve", error)
                    })?;
                if !canonical_path.starts_with(root) {
                    return Err(RerankStageError::InvalidCandidate {
                        unit_id: candidate.unit_id,
                        message: "page image resolves outside FILES_DIR".to_string(),
                    });
                }
                let expected_document_root = root.join(candidate.document_id.to_string());
                if !canonical_path.starts_with(&expected_document_root) {
                    return Err(RerankStageError::InvalidCandidate {
                        unit_id: candidate.unit_id,
                        message: "page image resolves outside the candidate document directory"
                            .to_string(),
                    });
                }
                let metadata = tokio::fs::metadata(&canonical_path)
                    .await
                    .map_err(|error| {
                        RerankStageError::artifact(candidate.unit_id, "metadata", error)
                    })?;
                if !metadata.is_file() {
                    return Err(RerankStageError::InvalidCandidate {
                        unit_id: candidate.unit_id,
                        message: "page image is not a regular file".to_string(),
                    });
                }
                if metadata.len() > MAX_EMBED_IMAGE_BYTES as u64 {
                    return Err(RerankStageError::InvalidCandidate {
                        unit_id: candidate.unit_id,
                        message: format!(
                            "page image is {} bytes; maximum is {MAX_EMBED_IMAGE_BYTES} bytes",
                            metadata.len()
                        ),
                    });
                }
                let image = tokio::fs::read(&canonical_path).await.map_err(|error| {
                    RerankStageError::artifact(candidate.unit_id, "read", error)
                })?;
                if image.len() > MAX_EMBED_IMAGE_BYTES {
                    return Err(RerankStageError::InvalidCandidate {
                        unit_id: candidate.unit_id,
                        message: format!(
                            "page image changed to {} bytes while reading; maximum is {MAX_EMBED_IMAGE_BYTES} bytes",
                            image.len()
                        ),
                    });
                }
                let passage = match candidate
                    .content
                    .as_deref()
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                {
                    Some(text) => RerankPassage::image_text(text, &image),
                    None => RerankPassage::image(&image),
                }
                .map_err(RerankStageError::Model)?;
                passages.push(passage);
            }
            "text" => {
                let text = candidate
                    .content
                    .as_deref()
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .ok_or_else(|| RerankStageError::InvalidCandidate {
                        unit_id: candidate.unit_id,
                        message: "text candidate is missing non-empty content".to_string(),
                    })?;
                passages.push(RerankPassage::text(text));
            }
            kind => {
                return Err(RerankStageError::InvalidCandidate {
                    unit_id: candidate.unit_id,
                    message: format!("unsupported candidate kind {kind:?}"),
                });
            }
        }
    }
    Ok(passages)
}

fn rrf_fallback(candidates: &[RetrievalCandidate], top_n: usize) -> Vec<RetrievalCandidate> {
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

/// Reciprocal Rank Fusion: score(d) = sum(1 / (60 + rank_i(d))).
///
/// A unit contributes at most once per ranking. Equal fused scores are ordered
/// by best contributing rank and finally unit_id, making fixtures/UI stable.
pub fn reciprocal_rank_fusion(
    dense: &[RankedCandidate],
    bm25: &[RankedCandidate],
    final_top_n: usize,
) -> Vec<RetrievalCandidate> {
    #[derive(Debug)]
    struct Accumulator {
        candidate: RetrievalCandidate,
    }

    fn add_ranking(
        accumulators: &mut HashMap<i64, Accumulator>,
        ranking: &[RankedCandidate],
        dense: bool,
    ) {
        let mut best_by_unit: HashMap<i64, &RankedCandidate> = HashMap::new();
        for ranked in ranking {
            if ranked.rank == 0 {
                continue;
            }
            best_by_unit
                .entry(ranked.metadata.unit_id)
                .and_modify(|current| {
                    if ranked.rank < current.rank
                        || (ranked.rank == current.rank
                            && ranked.score.total_cmp(&current.score).is_gt())
                    {
                        *current = ranked;
                    }
                })
                .or_insert(ranked);
        }

        for ranked in best_by_unit.into_values() {
            let accumulator = accumulators
                .entry(ranked.metadata.unit_id)
                .or_insert_with(|| Accumulator {
                    candidate: RetrievalCandidate {
                        unit_id: ranked.metadata.unit_id,
                        unit_index: ranked.metadata.unit_index,
                        document_id: ranked.metadata.document_id,
                        filename: ranked.metadata.filename.clone(),
                        kind: ranked.metadata.kind.clone(),
                        page_number: ranked.metadata.page_number,
                        image_path: ranked.metadata.image_path.clone(),
                        content: ranked.metadata.content.clone(),
                        dense_rank: None,
                        dense_score: None,
                        bm25_rank: None,
                        bm25_score: None,
                        rrf_score: 0.0,
                        rerank_score: None,
                        rerank_rank: None,
                    },
                });

            accumulator.candidate.rrf_score += 1.0 / (RRF_K + ranked.rank as f64);
            if dense {
                accumulator.candidate.dense_rank = Some(ranked.rank);
                accumulator.candidate.dense_score = Some(ranked.score);
            } else {
                accumulator.candidate.bm25_rank = Some(ranked.rank);
                accumulator.candidate.bm25_score = Some(ranked.score);
            }
        }
    }

    let mut accumulators = HashMap::new();
    add_ranking(&mut accumulators, dense, true);
    add_ranking(&mut accumulators, bm25, false);

    let mut fused: Vec<_> = accumulators
        .into_values()
        .map(|accumulator| accumulator.candidate)
        .collect();
    fused.sort_by(|left, right| {
        right
            .rrf_score
            .total_cmp(&left.rrf_score)
            .then_with(|| best_rank(left).cmp(&best_rank(right)))
            .then_with(|| left.unit_id.cmp(&right.unit_id))
    });
    fused.truncate(final_top_n);
    fused
}

fn best_rank(candidate: &RetrievalCandidate) -> usize {
    candidate
        .dense_rank
        .into_iter()
        .chain(candidate.bm25_rank)
        .min()
        .unwrap_or(usize::MAX)
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
        }
    }

    fn ranked(unit_id: i64, rank: usize, score: f64) -> RankedCandidate {
        RankedCandidate {
            metadata: metadata(unit_id),
            rank,
            score,
        }
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
            dense_rank: Some(unit_id as usize),
            dense_score: Some(1.0 / unit_id as f64),
            bm25_rank: None,
            bm25_score: None,
            rrf_score: 1.0 / (RRF_K + unit_id as f64),
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
    fn dense_only_and_bm25_only_keep_rank_and_scores() {
        let dense = reciprocal_rank_fusion(&[ranked(10, 1, 0.9), ranked(20, 2, 0.8)], &[], 10);
        assert_eq!(
            dense
                .iter()
                .map(|candidate| candidate.unit_id)
                .collect::<Vec<_>>(),
            vec![10, 20]
        );
        assert_eq!(dense[0].dense_rank, Some(1));
        assert_eq!(dense[0].dense_score, Some(0.9));
        assert_eq!(dense[0].bm25_rank, None);

        let bm25 = reciprocal_rank_fusion(&[], &[ranked(30, 1, 4.2), ranked(40, 2, 3.1)], 10);
        assert_eq!(
            bm25.iter()
                .map(|candidate| candidate.unit_id)
                .collect::<Vec<_>>(),
            vec![30, 40]
        );
        assert_eq!(bm25[0].bm25_rank, Some(1));
        assert_eq!(bm25[0].bm25_score, Some(4.2));
        assert_eq!(bm25[0].dense_rank, None);
    }

    #[test]
    fn overlap_is_deduplicated_and_outranks_single_route_candidates() {
        let fused = reciprocal_rank_fusion(
            &[ranked(1, 1, 0.99), ranked(2, 2, 0.8)],
            &[ranked(3, 1, 6.0), ranked(1, 2, 5.0)],
            10,
        );
        assert_eq!(
            fused
                .iter()
                .map(|candidate| candidate.unit_id)
                .collect::<Vec<_>>(),
            vec![1, 3, 2]
        );
        assert_eq!(fused[0].dense_rank, Some(1));
        assert_eq!(fused[0].bm25_rank, Some(2));
        assert!((fused[0].rrf_score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn disjoint_equal_scores_use_stable_unit_id_tie_breaker() {
        let dense = vec![ranked(20, 1, 0.9), ranked(40, 2, 0.8)];
        let bm25 = vec![ranked(10, 1, 5.0), ranked(30, 2, 4.0)];
        let expected = vec![10, 20, 30, 40];
        for _ in 0..100 {
            let fused = reciprocal_rank_fusion(&dense, &bm25, 10);
            assert_eq!(
                fused
                    .iter()
                    .map(|candidate| candidate.unit_id)
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn duplicate_unit_in_one_route_contributes_only_at_best_occurrence() {
        let fused = reciprocal_rank_fusion(&[ranked(1, 2, 0.8), ranked(1, 1, 0.9)], &[], 10);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].dense_rank, Some(1));
        assert_eq!(fused[0].rrf_score, 1.0 / 61.0);
    }

    #[test]
    fn serialized_candidate_keeps_complete_nullable_contract() {
        let fused = reciprocal_rank_fusion(&[ranked(1, 1, 0.9)], &[], 10);
        let value = serde_json::to_value(&fused[0]).expect("serialize candidate");
        for field in [
            "unitId",
            "unitIndex",
            "documentId",
            "filename",
            "kind",
            "pageNumber",
            "imagePath",
            "content",
            "denseRank",
            "denseScore",
            "bm25Rank",
            "bm25Score",
            "rrfScore",
            "rerankScore",
            "rerankRank",
        ] {
            assert!(
                value.get(field).is_some(),
                "missing candidate field {field}"
            );
        }
        assert!(value["bm25Rank"].is_null());
        assert!(value["bm25Score"].is_null());
        assert!(value["rerankScore"].is_null());
        assert!(value["rerankRank"].is_null());
    }

    #[test]
    fn config_rejects_route_limits_below_final_top_n() {
        assert!(RetrievalConfig::new(7, 8, 8).is_err());
        assert!(RetrievalConfig::new(8, 7, 8).is_err());
        assert!(
            RerankConfig::new(
                crate::cloud_models::MAX_RERANK_BATCH_SIZE + 1,
                3,
                crate::cloud_models::MAX_RERANK_BATCH_SIZE + 1,
            )
            .is_err()
        );
        assert_eq!(
            RetrievalConfig::default(),
            RetrievalConfig::new(50, 50, 50).expect("valid defaults")
        );
    }

    #[tokio::test]
    async fn rerank_payload_covers_image_text_image_only_and_text_only() {
        let root = std::env::temp_dir().join(format!("km-rerank-modalities-{}", Uuid::new_v4()));
        let mixed_pages = root.join(Uuid::from_u128(101).to_string()).join("pages");
        let image_only_pages = root.join(Uuid::from_u128(102).to_string()).join("pages");
        tokio::fs::create_dir_all(&mixed_pages)
            .await
            .expect("create mixed artifact fixture");
        tokio::fs::create_dir_all(&image_only_pages)
            .await
            .expect("create image-only artifact fixture");
        let mixed_image = mixed_pages.join("0001.png");
        let image_only = image_only_pages.join("0002.png");
        let png = b"\x89PNG\r\n\x1a\nmock-rerank-page";
        tokio::fs::write(&mixed_image, png)
            .await
            .expect("write mixed image");
        tokio::fs::write(&image_only, png)
            .await
            .expect("write image-only image");
        let candidates = vec![
            retrieval_candidate(
                1,
                "page",
                Some(mixed_image.to_string_lossy().into_owned()),
                Some("same-page text"),
            ),
            retrieval_candidate(
                2,
                "page",
                Some(image_only.to_string_lossy().into_owned()),
                None,
            ),
            retrieval_candidate(3, "text", None, Some("markdown chunk")),
        ];
        let reranker = FakeReranker::new(Vec::new());
        let result = rerank_candidates(
            &reranker,
            "query",
            &candidates,
            &root,
            RerankConfig::new(3, 3, 3).expect("config"),
        )
        .await
        .expect("rerank all modalities");
        assert_eq!(result.len(), 3);
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
        }

        {
            let requests = reranker.requests.lock().expect("requests lock");
            assert_eq!(requests.len(), 1);
            let passages = requests[0].as_array().expect("passages array");
            assert_eq!(passages[0]["text"], "same-page text");
            assert!(
                passages[0]["image"]
                    .as_str()
                    .is_some_and(|image| image.starts_with("data:image/png;base64,"))
            );
            assert!(passages[1].get("text").is_none());
            assert!(
                passages[1]["image"]
                    .as_str()
                    .is_some_and(|image| image.starts_with("data:image/png;base64,"))
            );
            assert_eq!(passages[2], serde_json::json!({ "text": "markdown chunk" }));
            assert!(passages.iter().all(|passage| {
                !passage
                    .get("image")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|image| image.contains(root.to_string_lossy().as_ref()))
            }));
        }
        tokio::fs::remove_dir_all(root)
            .await
            .expect("remove artifact fixture");
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
            Path::new("unused-for-text"),
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
    async fn equal_rerank_scores_keep_the_original_rrf_order() {
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
            Path::new("unused-for-text"),
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
                Path::new("unused-for-text"),
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
            Path::new("unused-for-text"),
            RerankConfig::new(8, 8, 4).expect("config"),
        )
        .await
        .expect("candidate list shorter than top_n");
        assert_eq!(short.len(), 2);
        assert_eq!(short[0].rerank_rank, Some(1));
        assert_eq!(short[1].rerank_rank, Some(2));
    }

    #[tokio::test]
    async fn exhausted_transient_error_uses_exact_rrf_prefix_without_fake_scores() {
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
            Path::new("unused-for-text"),
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
                Path::new("unused-for-text"),
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
            Path::new("unused-for-text"),
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
        let root = std::env::temp_dir().join(format!("km-rerank-missing-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root)
            .await
            .expect("create fixture root");
        let candidate = retrieval_candidate(
            1,
            "page",
            Some(root.join("missing.png").to_string_lossy().into_owned()),
            Some("text must not mask missing image"),
        );
        let reranker = FakeReranker::new(Vec::new());
        let error = rerank_candidates(
            &reranker,
            "query",
            &[candidate],
            &root,
            RerankConfig::new(3, 3, 2).expect("config"),
        )
        .await
        .expect_err("missing page artifact must fail");
        assert!(matches!(error, RerankStageError::Artifact { .. }));
        assert!(reranker.requests.lock().expect("requests lock").is_empty());
        tokio::fs::remove_dir_all(root)
            .await
            .expect("remove fixture root");
    }

    #[tokio::test]
    async fn page_artifact_must_belong_to_the_candidate_document_directory() {
        let root =
            std::env::temp_dir().join(format!("km-rerank-cross-document-{}", Uuid::new_v4()));
        let other_document_pages = root.join(Uuid::from_u128(102).to_string()).join("pages");
        tokio::fs::create_dir_all(&other_document_pages)
            .await
            .expect("create other document fixture");
        let other_document_image = other_document_pages.join("0001.png");
        tokio::fs::write(&other_document_image, b"\x89PNG\r\n\x1a\nother-document")
            .await
            .expect("write other document image");

        // Unit 1 maps to document UUID 101 in the fixture helper, so UUID 102
        // must never be accepted even though it remains inside FILES_DIR.
        let candidate = retrieval_candidate(
            1,
            "page",
            Some(other_document_image.to_string_lossy().into_owned()),
            None,
        );
        let reranker = FakeReranker::new(Vec::new());
        let error = rerank_candidates(
            &reranker,
            "query",
            &[candidate],
            &root,
            RerankConfig::new(3, 3, 2).expect("config"),
        )
        .await
        .expect_err("cross-document artifact mapping must fail");
        assert!(matches!(error, RerankStageError::InvalidCandidate { .. }));
        assert!(reranker.requests.lock().expect("requests lock").is_empty());

        tokio::fs::remove_dir_all(root)
            .await
            .expect("remove fixture root");
    }

    #[derive(Debug)]
    struct FakeBackend {
        dense: Vec<RankedCandidate>,
        bm25: Vec<RankedCandidate>,
        calls: Mutex<Vec<(&'static str, usize)>>,
    }

    impl CandidateRetrievalBackend for FakeBackend {
        fn dense_candidates<'a>(
            &'a self,
            _query: &'a str,
            top_k: usize,
        ) -> BoxFuture<'a, anyhow::Result<Vec<RankedCandidate>>> {
            Box::pin(async move {
                self.calls
                    .lock()
                    .expect("calls lock")
                    .push(("dense", top_k));
                Ok(self.dense.clone())
            })
        }

        fn bm25_candidates<'a>(
            &'a self,
            _query: &'a str,
            top_k: usize,
        ) -> BoxFuture<'a, anyhow::Result<Vec<RankedCandidate>>> {
            Box::pin(async move {
                self.calls.lock().expect("calls lock").push(("bm25", top_k));
                Ok(self.bm25.clone())
            })
        }
    }

    #[tokio::test]
    async fn injected_routes_are_each_called_once_and_debug_keeps_rankings() {
        let backend = FakeBackend {
            dense: vec![ranked(1, 1, 0.9)],
            bm25: vec![ranked(2, 1, 3.0)],
            calls: Mutex::new(Vec::new()),
        };
        let debug = retrieve_with_backend(&backend, "fixture query", RetrievalConfig::default())
            .await
            .expect("hybrid retrieval");
        assert_eq!(debug.dense, backend.dense);
        assert_eq!(debug.bm25, backend.bm25);
        assert_eq!(debug.fused.len(), 2);
        let mut calls = backend.calls.lock().expect("calls lock").clone();
        calls.sort_unstable();
        assert_eq!(calls, vec![("bm25", 50), ("dense", 50)]);
    }
}
