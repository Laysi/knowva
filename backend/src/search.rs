use crate::AppState;
use futures::future::BoxFuture;
use pgvector::HalfVector;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

pub const DEFAULT_DENSE_TOP_K: usize = 50;
pub const DEFAULT_BM25_TOP_K: usize = 50;
pub const DEFAULT_FINAL_TOP_N: usize = 50;
pub const CHAT_CONTEXT_TOP_N: usize = 8;
pub const RRF_K: f64 = 60.0;

// Okapi BM25 defaults used by the database-side scoring expression.
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateMetadata {
    pub unit_id: i64,
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
}

// Compatibility name retained for the existing chat/prompt call sites.
pub type RetrievedChunk = RetrievalCandidate;

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
           SELECT u.id AS unit_id, u.document_id, u.kind, u.page_number, \
                  u.image_path, u.content, \
                  (u.embedding <=> $1)::double precision AS distance \
           FROM document_units AS u \
           WHERE u.retrieval_ready \
           ORDER BY u.embedding <=> $1 \
           LIMIT $2 \
         ) \
         SELECT nearest.unit_id, nearest.document_id, d.filename, nearest.kind, \
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
           SELECT u.id AS unit_id, u.document_id, u.kind, \
                  u.page_number, u.image_path, u.content, u.bm25_tokens, u.bm25_length \
           FROM document_units AS u \
           CROSS JOIN query_token_array AS query \
           WHERE u.retrieval_ready \
             AND u.content IS NOT NULL \
             AND btrim(u.content) <> '' \
             AND u.bm25_tokens && query.terms \
         ), scored AS ( \
           SELECT candidate.unit_id, candidate.document_id, document.filename, \
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
           GROUP BY candidate.unit_id, candidate.document_id, document.filename, \
                    candidate.kind, candidate.page_number, candidate.image_path, \
                    candidate.content \
         ) \
         SELECT unit_id, document_id, filename, kind, page_number, image_path, content, score \
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
    let (dense, bm25) = tokio::try_join!(
        backend.dense_candidates(query, config.dense_top_k),
        backend.bm25_candidates(query, config.bm25_top_k),
    )?;
    let fused = reciprocal_rank_fusion(&dense, &bm25, config.final_top_n);
    Ok(RetrievalDebug { dense, bm25, fused })
}

/// Embeds the query as 2048-dimensional halfvec input, retrieves 50 candidates
/// from dense and BM25 independently, and returns one stable RRF top-50 list.
pub async fn retrieve(state: &AppState, query: &str) -> anyhow::Result<Vec<RetrievedChunk>> {
    Ok(retrieve_debug(state, query, RetrievalConfig::default())
        .await?
        .fused)
}

/// Runs the production PostgreSQL stages while retaining both route rankings
/// for diagnostics, evaluation fixtures, and the next reranking integration.
pub async fn retrieve_debug(
    state: &AppState,
    query: &str,
    config: RetrievalConfig,
) -> anyhow::Result<RetrievalDebug> {
    let backend = PostgresRetrievalBackend { state };
    retrieve_with_backend(&backend, query, config).await
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

#[derive(Debug, Clone, Serialize)]
pub struct SourceRef {
    #[serde(rename = "documentId")]
    pub document_id: Uuid,
    pub filename: String,
    #[serde(rename = "pageNumber", skip_serializing_if = "Option::is_none")]
    pub page_number: Option<i32>,
}

/// Builds the RAG-augmented system prompt from retrieved candidates.
pub fn build_rag_system_prompt(chunks: &[RetrievedChunk]) -> String {
    let mut prompt = String::new();
    prompt.push_str(crate::cloud_models::SYSTEM_PROMPT);
    prompt.push_str(
        "\n\n以下是知識庫檢索到的段落,回答時優先依據這些內容,並在回答中標註出處檔名;若段落與問題無關或不足,請直接說明知識庫沒有相關資料。\n\n",
    );
    for (i, chunk) in chunks.iter().enumerate() {
        let page = chunk
            .page_number
            .map(|number| format!(" 第 {number} 頁"))
            .unwrap_or_default();
        let content = chunk
            .content
            .as_deref()
            .unwrap_or("[此頁只有頁面影像，尚無文字層]");
        prompt.push_str(&format!(
            "[{}] {}{}\n{}\n\n",
            i + 1,
            chunk.filename,
            page,
            content
        ));
    }
    prompt
}

/// Dedupe retrieved units by document/page, preserving retrieval rank order.
pub fn dedupe_sources(chunks: &[RetrievedChunk]) -> Vec<SourceRef> {
    let mut seen = HashSet::new();
    let mut sources = Vec::new();
    for chunk in chunks {
        if seen.insert((chunk.document_id, chunk.page_number)) {
            sources.push(SourceRef {
                document_id: chunk.document_id,
                filename: chunk.filename.clone(),
                page_number: chunk.page_number,
            });
        }
    }
    sources
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn metadata(unit_id: i64) -> CandidateMetadata {
        CandidateMetadata {
            unit_id,
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
        ] {
            assert!(
                value.get(field).is_some(),
                "missing candidate field {field}"
            );
        }
        assert!(value["bm25Rank"].is_null());
        assert!(value["bm25Score"].is_null());
    }

    #[test]
    fn config_rejects_route_limits_below_final_top_n() {
        assert!(RetrievalConfig::new(7, 8, 8).is_err());
        assert!(RetrievalConfig::new(8, 7, 8).is_err());
        assert_eq!(
            RetrievalConfig::default(),
            RetrievalConfig::new(50, 50, 50).expect("valid defaults")
        );
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
