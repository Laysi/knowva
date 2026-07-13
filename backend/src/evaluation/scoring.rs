use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationMode {
    A,
    B,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalStage {
    Dense,
    Bm25,
    Rrf,
    Rerank,
}

impl RetrievalStage {
    pub const ALL: [Self; 4] = [Self::Dense, Self::Bm25, Self::Rrf, Self::Rerank];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Bm25 => "bm25",
            Self::Rrf => "rrf",
            Self::Rerank => "rerank",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalCandidate {
    pub document_key: String,
    pub unit_id: String,
    pub filename: String,
    pub page: Option<u32>,
    pub rank: u32,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FallbackTelemetry {
    pub from: String,
    pub to: String,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageTelemetry {
    pub latency_ms: u64,
    pub request_count: u32,
    pub retry_count: u32,
    pub fallback: Option<FallbackTelemetry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageTrace {
    pub candidates: Vec<RetrievalCandidate>,
    pub telemetry: StageTelemetry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalTrace {
    pub dense: StageTrace,
    pub bm25: StageTrace,
    pub rrf: StageTrace,
    pub rerank: StageTrace,
}

impl RetrievalTrace {
    pub fn get(&self, stage: RetrievalStage) -> &StageTrace {
        match stage {
            RetrievalStage::Dense => &self.dense,
            RetrievalStage::Bm25 => &self.bm25,
            RetrievalStage::Rrf => &self.rrf,
            RetrievalStage::Rerank => &self.rerank,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextSource {
    pub document_key: String,
    pub unit_id: String,
    pub filename: String,
    pub page: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedSource {
    pub document_key: String,
    pub unit_id: String,
    pub filename: String,
    pub page: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryExpectation {
    pub answerable: bool,
    pub expected_sources: Vec<ExpectedSource>,
    pub required_facts: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManualReviewStatus {
    Pending,
    Pass,
    Fail,
    NeedsDiscussion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NegativeAnswerPolicy {
    DeterministicAbstentionPhrases,
}

const DETERMINISTIC_ABSTENTION_PHRASES: [&str; 6] = [
    "資料不足",
    "資訊不足",
    "無法回答",
    "找不到相關資料",
    "insufficient information",
    "cannot answer",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManualReview {
    pub status: ManualReviewStatus,
    pub reviewer: Option<String>,
    pub notes: Option<String>,
    pub reviewed_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunFailure {
    pub code: String,
    pub stage: Option<RetrievalStage>,
    pub safe_message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryRunInput {
    pub query_id: String,
    pub mode: EvaluationMode,
    pub answer: String,
    pub expectation: QueryExpectation,
    pub stages: RetrievalTrace,
    pub context_sources: BTreeMap<String, ContextSource>,
    pub manual_review: ManualReview,
    pub failure: Option<RunFailure>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoringConfig {
    pub cutoffs: Vec<u32>,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            cutoffs: vec![1, 3, 5, 10],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtKScore {
    pub k: u32,
    pub value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalMetrics {
    pub source_recall_at_k: Vec<AtKScore>,
    pub page_recall_at_k: Vec<AtKScore>,
    pub mrr: Option<f64>,
    pub ndcg_at_k: Vec<AtKScore>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageMetrics {
    pub stage: RetrievalStage,
    pub metrics: RetrievalMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CitationOutcome {
    Correct,
    UnknownReference,
    UnexpectedForNegative,
    WrongDocument,
    WrongUnit,
    WrongFilename,
    WrongPage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CitationAssessment {
    pub label: String,
    pub outcome: CitationOutcome,
    pub resolved_source: Option<ContextSource>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionRecall {
    pub correct: u32,
    pub predicted: u32,
    pub relevant: u32,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CitationMetrics {
    pub cited_labels: Vec<String>,
    pub unknown_labels: Vec<String>,
    pub assessments: Vec<CitationAssessment>,
    pub exact: PrecisionRecall,
    pub filename: PrecisionRecall,
    pub page: PrecisionRecall,
    pub negative_citation_error: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredFactMatch {
    pub required_fact: String,
    pub normalized_fact: String,
    pub matched: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactMetrics {
    pub normalized_answer: String,
    pub matches: Vec<RequiredFactMatch>,
    pub recall: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryMetrics {
    pub retrieval: Vec<StageMetrics>,
    pub citations: CitationMetrics,
    pub facts: FactMetrics,
    pub composite_score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoredQuery {
    pub input: QueryRunInput,
    pub metrics: Option<QueryMetrics>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScoringError {
    pub code: &'static str,
    pub path: String,
    pub message: &'static str,
}

impl ScoringError {
    fn new(code: &'static str, path: impl Into<String>, message: &'static str) -> Self {
        Self {
            code,
            path: path.into(),
            message,
        }
    }
}

impl std::fmt::Display for ScoringError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} at {}: {}",
            self.code, self.path, self.message
        )
    }
}

impl std::error::Error for ScoringError {}

pub type ScoringResult<T> = Result<T, Vec<ScoringError>>;

pub fn score_query(input: QueryRunInput, config: &ScoringConfig) -> ScoringResult<ScoredQuery> {
    validate_input(&input, config)?;
    if input.failure.is_some() {
        return Ok(ScoredQuery {
            input,
            metrics: None,
        });
    }

    let retrieval: Vec<StageMetrics> = RetrievalStage::ALL
        .into_iter()
        .map(|stage| StageMetrics {
            stage,
            metrics: score_retrieval_stage(
                &input.stages.get(stage).candidates,
                &input.expectation,
                &config.cutoffs,
            ),
        })
        .collect();
    let citations = score_citations(&input.answer, &input.context_sources, &input.expectation);
    let facts = score_required_facts(&input.answer, &input.expectation.required_facts);
    let composite_score = composite_score(
        &retrieval,
        &citations,
        &facts,
        &input.answer,
        input.expectation.answerable,
        input.mode,
    );

    Ok(ScoredQuery {
        input,
        metrics: Some(QueryMetrics {
            retrieval,
            citations,
            facts,
            composite_score,
        }),
    })
}

fn validate_input(input: &QueryRunInput, config: &ScoringConfig) -> ScoringResult<()> {
    let mut errors = Vec::new();
    if input.query_id.trim().is_empty() {
        errors.push(ScoringError::new(
            "query_id_empty",
            "$.query_id",
            "query_id must not be empty",
        ));
    }

    let mut cutoffs = BTreeSet::new();
    if config.cutoffs.is_empty() {
        errors.push(ScoringError::new(
            "cutoffs_empty",
            "$.config.cutoffs",
            "at least one retrieval cutoff is required",
        ));
    }
    for (index, cutoff) in config.cutoffs.iter().copied().enumerate() {
        if cutoff == 0 {
            errors.push(ScoringError::new(
                "cutoff_not_one_based",
                format!("$.config.cutoffs[{index}]"),
                "retrieval cutoffs must be greater than zero",
            ));
        } else if !cutoffs.insert(cutoff) {
            errors.push(ScoringError::new(
                "duplicate_cutoff",
                format!("$.config.cutoffs[{index}]"),
                "retrieval cutoffs must be unique",
            ));
        }
    }

    if input.expectation.answerable && input.expectation.expected_sources.is_empty() {
        errors.push(ScoringError::new(
            "answerable_sources_empty",
            "$.expectation.expected_sources",
            "answerable query must have expected sources",
        ));
    }
    if !input.expectation.answerable && !input.expectation.expected_sources.is_empty() {
        errors.push(ScoringError::new(
            "negative_query_has_sources",
            "$.expectation.expected_sources",
            "negative query must not have expected sources",
        ));
    }

    let mut expected_identities = BTreeSet::new();
    for (index, source) in input.expectation.expected_sources.iter().enumerate() {
        let base = format!("$.expectation.expected_sources[{index}]");
        validate_source_fields(
            &source.document_key,
            &source.unit_id,
            &source.filename,
            source.page,
            &base,
            &mut errors,
        );
        if !expected_identities.insert((source.document_key.as_str(), source.unit_id.as_str())) {
            errors.push(ScoringError::new(
                "duplicate_expected_source",
                base,
                "expected source identities must be unique",
            ));
        }
    }

    for stage in RetrievalStage::ALL {
        validate_stage(stage, input.stages.get(stage), &mut errors);
    }

    for (label, source) in &input.context_sources {
        let base = format!("$.context_sources.{label}");
        if !is_citation_label(label) {
            errors.push(ScoringError::new(
                "citation_label_invalid",
                &base,
                "context source key must use canonical S followed by a positive integer",
            ));
        }
        validate_source_fields(
            &source.document_key,
            &source.unit_id,
            &source.filename,
            source.page,
            &base,
            &mut errors,
        );
    }

    if let Some(failure) = &input.failure {
        if failure.code.trim().is_empty() {
            errors.push(ScoringError::new(
                "failure_code_empty",
                "$.failure.code",
                "failure code must not be empty",
            ));
        }
        if failure.safe_message.trim().is_empty() {
            errors.push(ScoringError::new(
                "failure_message_empty",
                "$.failure.safe_message",
                "safe failure message must not be empty",
            ));
        }
    }

    errors.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then_with(|| left.code.cmp(right.code))
    });
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_stage(stage: RetrievalStage, trace: &StageTrace, errors: &mut Vec<ScoringError>) {
    let base = format!("$.stages.{}", stage.as_str());
    if trace.telemetry.retry_count > trace.telemetry.request_count {
        errors.push(ScoringError::new(
            "retry_count_invalid",
            format!("{base}.telemetry.retry_count"),
            "retry_count cannot exceed request_count",
        ));
    }
    if let Some(fallback) = &trace.telemetry.fallback
        && (fallback.from.trim().is_empty()
            || fallback.to.trim().is_empty()
            || fallback.reason_code.trim().is_empty())
    {
        errors.push(ScoringError::new(
            "fallback_invalid",
            format!("{base}.telemetry.fallback"),
            "fallback fields must not be empty",
        ));
    }

    let mut ranks = BTreeSet::new();
    let mut identities = BTreeSet::new();
    for (index, candidate) in trace.candidates.iter().enumerate() {
        let candidate_base = format!("{base}.candidates[{index}]");
        validate_source_fields(
            &candidate.document_key,
            &candidate.unit_id,
            &candidate.filename,
            candidate.page,
            &candidate_base,
            errors,
        );
        if candidate.rank == 0 {
            errors.push(ScoringError::new(
                "rank_not_one_based",
                format!("{candidate_base}.rank"),
                "candidate rank must be greater than zero",
            ));
        } else if !ranks.insert(candidate.rank) {
            errors.push(ScoringError::new(
                "duplicate_rank",
                format!("{candidate_base}.rank"),
                "candidate ranks must be unique within a stage",
            ));
        }
        if !candidate.score.is_finite() {
            errors.push(ScoringError::new(
                "score_not_finite",
                format!("{candidate_base}.score"),
                "candidate score must be finite",
            ));
        }
        if !identities.insert((
            candidate.document_key.as_str(),
            candidate.unit_id.as_str(),
            candidate.page,
        )) {
            errors.push(ScoringError::new(
                "duplicate_candidate",
                candidate_base,
                "candidate identities must be unique within a stage",
            ));
        }
    }
    if ranks
        .iter()
        .copied()
        .ne((1..=trace.candidates.len()).map(|rank| rank as u32))
    {
        errors.push(ScoringError::new(
            "ranks_not_contiguous",
            format!("{base}.candidates"),
            "complete candidate ranks must be contiguous and start at 1",
        ));
    }
}

fn validate_source_fields(
    document_key: &str,
    unit_id: &str,
    filename: &str,
    page: Option<u32>,
    base: &str,
    errors: &mut Vec<ScoringError>,
) {
    if document_key.trim().is_empty() {
        errors.push(ScoringError::new(
            "document_key_empty",
            format!("{base}.document_key"),
            "document_key must not be empty",
        ));
    }
    if unit_id.trim().is_empty() {
        errors.push(ScoringError::new(
            "unit_id_empty",
            format!("{base}.unit_id"),
            "unit_id must not be empty",
        ));
    }
    if filename.trim().is_empty() {
        errors.push(ScoringError::new(
            "filename_empty",
            format!("{base}.filename"),
            "filename must not be empty",
        ));
    }
    if page == Some(0) {
        errors.push(ScoringError::new(
            "page_not_one_based",
            format!("{base}.page"),
            "page numbers are 1-based and must be greater than zero",
        ));
    }
}

fn is_citation_label(label: &str) -> bool {
    let Some(digits) = label.strip_prefix('S') else {
        return false;
    };
    !digits.is_empty()
        && !digits.starts_with('0')
        && digits.bytes().all(|byte| byte.is_ascii_digit())
}

pub fn parse_answer_citations(answer: &str) -> Vec<String> {
    let bytes = answer.as_bytes();
    let mut labels = BTreeSet::new();
    let mut index = 0;
    while index + 3 < bytes.len() {
        if bytes[index] != b'[' || bytes[index + 1] != b'S' {
            index += 1;
            continue;
        }
        let digit_start = index + 2;
        let mut cursor = digit_start;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            cursor += 1;
        }
        if cursor > digit_start && cursor < bytes.len() && bytes[cursor] == b']' {
            let digits = &answer[digit_start..cursor];
            labels.insert(format!("S{digits}"));
            index = cursor + 1;
        } else {
            index += 1;
        }
    }
    labels.into_iter().collect()
}

fn score_retrieval_stage(
    candidates: &[RetrievalCandidate],
    expectation: &QueryExpectation,
    cutoffs: &[u32],
) -> RetrievalMetrics {
    let mut ranked = candidates.iter().collect::<Vec<_>>();
    ranked.sort_by_key(|candidate| candidate.rank);
    let relevant_count = expectation.expected_sources.len();
    let paged_relevant = expectation
        .expected_sources
        .iter()
        .filter(|source| source.page.is_some())
        .count();

    let source_recall_at_k = cutoffs
        .iter()
        .copied()
        .map(|k| {
            let mut matched = BTreeSet::new();
            for candidate in ranked
                .iter()
                .copied()
                .filter(|candidate| candidate.rank <= k)
            {
                if let Some(index) = expectation
                    .expected_sources
                    .iter()
                    .position(|source| source_identity_matches(candidate, source))
                {
                    matched.insert(index);
                }
            }
            AtKScore {
                k,
                value: ratio(matched.len(), relevant_count),
            }
        })
        .collect();

    let page_recall_at_k = cutoffs
        .iter()
        .copied()
        .map(|k| {
            let mut matched = BTreeSet::new();
            for candidate in ranked
                .iter()
                .copied()
                .filter(|candidate| candidate.rank <= k)
            {
                if let Some(index) = expectation.expected_sources.iter().position(|source| {
                    source.page.is_some()
                        && source_identity_matches(candidate, source)
                        && candidate.page == source.page
                }) {
                    matched.insert(index);
                }
            }
            AtKScore {
                k,
                value: ratio(matched.len(), paged_relevant),
            }
        })
        .collect();

    let mrr = if relevant_count == 0 {
        None
    } else {
        let first_rank = ranked
            .iter()
            .copied()
            .find(|candidate| {
                expectation
                    .expected_sources
                    .iter()
                    .any(|source| source_identity_matches(candidate, source))
            })
            .map(|candidate| candidate.rank);
        Some(first_rank.map_or(0.0, |rank| 1.0 / f64::from(rank)))
    };

    let ndcg_at_k = cutoffs
        .iter()
        .copied()
        .map(|k| AtKScore {
            k,
            value: ndcg(&ranked, &expectation.expected_sources, k),
        })
        .collect();

    RetrievalMetrics {
        source_recall_at_k,
        page_recall_at_k,
        mrr,
        ndcg_at_k,
    }
}

fn source_identity_matches(candidate: &RetrievalCandidate, expected: &ExpectedSource) -> bool {
    candidate.document_key == expected.document_key && candidate.unit_id == expected.unit_id
}

fn ndcg(ranked: &[&RetrievalCandidate], expected: &[ExpectedSource], cutoff: u32) -> Option<f64> {
    if expected.is_empty() {
        return None;
    }
    let mut matched = BTreeSet::new();
    let mut dcg = 0.0;
    for candidate in ranked
        .iter()
        .copied()
        .filter(|candidate| candidate.rank <= cutoff)
    {
        if let Some(index) = expected
            .iter()
            .position(|source| source_identity_matches(candidate, source))
            && matched.insert(index)
        {
            dcg += discount(candidate.rank);
        }
    }
    let ideal_count = expected.len().min(cutoff as usize);
    let ideal = (1..=ideal_count)
        .map(|rank| discount(rank as u32))
        .sum::<f64>();
    Some(if ideal == 0.0 { 0.0 } else { dcg / ideal })
}

fn discount(rank: u32) -> f64 {
    1.0 / f64::from(rank + 1).log2()
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then_some(numerator as f64 / denominator as f64)
}

fn score_citations(
    answer: &str,
    context_sources: &BTreeMap<String, ContextSource>,
    expectation: &QueryExpectation,
) -> CitationMetrics {
    let cited_labels = parse_answer_citations(answer);
    let mut unknown_labels = Vec::new();
    let mut assessments = Vec::new();
    let mut exact_correct = 0;
    let mut filename_correct = 0;
    let mut page_correct = 0;
    let mut exact_matched = BTreeSet::new();
    let mut filename_matched = BTreeSet::new();
    let mut page_matched = BTreeSet::new();

    for label in &cited_labels {
        let Some(actual) = context_sources.get(label) else {
            unknown_labels.push(label.clone());
            assessments.push(CitationAssessment {
                label: label.clone(),
                outcome: CitationOutcome::UnknownReference,
                resolved_source: None,
            });
            continue;
        };

        let identity_index = expectation.expected_sources.iter().position(|expected| {
            actual.document_key == expected.document_key && actual.unit_id == expected.unit_id
        });
        if let Some(index) = identity_index {
            let expected = &expectation.expected_sources[index];
            if actual.filename == expected.filename {
                filename_correct += 1;
                filename_matched.insert(index);
            }
            if actual.page == expected.page {
                page_correct += 1;
                page_matched.insert(index);
            }
            if actual.filename == expected.filename && actual.page == expected.page {
                exact_correct += 1;
                exact_matched.insert(index);
            }
        }

        let outcome = if !expectation.answerable {
            CitationOutcome::UnexpectedForNegative
        } else {
            classify_citation(actual, &expectation.expected_sources)
        };
        assessments.push(CitationAssessment {
            label: label.clone(),
            outcome,
            resolved_source: Some(actual.clone()),
        });
    }

    let predicted = cited_labels.len();
    let relevant = expectation.expected_sources.len();
    CitationMetrics {
        cited_labels,
        unknown_labels,
        assessments,
        exact: precision_recall(exact_correct, exact_matched.len(), predicted, relevant),
        filename: precision_recall(
            filename_correct,
            filename_matched.len(),
            predicted,
            relevant,
        ),
        page: precision_recall(page_correct, page_matched.len(), predicted, relevant),
        negative_citation_error: if !expectation.answerable && predicted > 0 {
            1.0
        } else {
            0.0
        },
    }
}

fn classify_citation(actual: &ContextSource, expected: &[ExpectedSource]) -> CitationOutcome {
    let same_document = expected
        .iter()
        .filter(|source| source.document_key == actual.document_key)
        .collect::<Vec<_>>();
    if same_document.is_empty() {
        return CitationOutcome::WrongDocument;
    }
    let Some(exact_unit) = same_document
        .into_iter()
        .find(|source| source.unit_id == actual.unit_id)
    else {
        return CitationOutcome::WrongUnit;
    };
    if exact_unit.filename != actual.filename {
        CitationOutcome::WrongFilename
    } else if exact_unit.page != actual.page {
        CitationOutcome::WrongPage
    } else {
        CitationOutcome::Correct
    }
}

fn precision_recall(
    correct_predictions: usize,
    matched_relevant: usize,
    predicted: usize,
    relevant: usize,
) -> PrecisionRecall {
    let precision = ratio(correct_predictions, predicted);
    let recall = ratio(matched_relevant, relevant);
    let f1 = match (precision, recall) {
        (Some(precision), Some(recall)) if precision + recall > 0.0 => {
            Some(2.0 * precision * recall / (precision + recall))
        }
        (Some(_), Some(_)) => Some(0.0),
        (None, Some(_)) => Some(0.0),
        _ => None,
    };
    PrecisionRecall {
        correct: correct_predictions as u32,
        predicted: predicted as u32,
        relevant: relevant as u32,
        precision,
        recall,
        f1,
    }
}

pub fn normalize_fact(value: &str) -> String {
    let mut output = String::new();
    let mut needs_space = false;
    for character in value.chars() {
        if character.is_alphanumeric() {
            if needs_space && !output.is_empty() {
                output.push(' ');
            }
            output.extend(character.to_lowercase());
            needs_space = false;
        } else {
            needs_space = !output.is_empty();
        }
    }
    output
}

fn score_required_facts(answer: &str, required_facts: &[String]) -> FactMetrics {
    let normalized_answer = normalize_fact(answer);
    let compact_answer = normalized_answer.replace(' ', "");
    let matches = required_facts
        .iter()
        .map(|required_fact| {
            let normalized_fact = normalize_fact(required_fact);
            let compact_fact = normalized_fact.replace(' ', "");
            let matched = !normalized_fact.is_empty()
                && (normalized_answer.contains(&normalized_fact)
                    || compact_answer.contains(&compact_fact));
            RequiredFactMatch {
                required_fact: required_fact.clone(),
                normalized_fact,
                matched,
            }
        })
        .collect::<Vec<_>>();
    let matched_count = matches.iter().filter(|fact| fact.matched).count();
    FactMetrics {
        normalized_answer,
        recall: ratio(matched_count, matches.len()),
        matches,
    }
}

fn composite_score(
    retrieval: &[StageMetrics],
    citations: &CitationMetrics,
    facts: &FactMetrics,
    answer: &str,
    answerable: bool,
    mode: EvaluationMode,
) -> f64 {
    if !answerable {
        return if citations.negative_citation_error == 0.0
            && matches_negative_answer_policy(
                answer,
                NegativeAnswerPolicy::DeterministicAbstentionPhrases,
            ) {
            1.0
        } else {
            0.0
        };
    }
    let mut values = Vec::new();
    let final_stage = match mode {
        EvaluationMode::A => RetrievalStage::Dense,
        EvaluationMode::B => RetrievalStage::Rerank,
    };
    if let Some(rerank) = retrieval.iter().find(|stage| stage.stage == final_stage)
        && let Some(value) = rerank
            .metrics
            .ndcg_at_k
            .last()
            .and_then(|score| score.value)
    {
        values.push(value);
    }
    if let Some(value) = citations.exact.f1 {
        values.push(value);
    }
    if let Some(value) = facts.recall {
        values.push(value);
    }
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

pub fn matches_negative_answer_policy(answer: &str, policy: NegativeAnswerPolicy) -> bool {
    match policy {
        NegativeAnswerPolicy::DeterministicAbstentionPhrases => {
            let normalized_answer = normalize_fact(answer);
            let compact_answer = normalized_answer.replace(' ', "");
            DETERMINISTIC_ABSTENTION_PHRASES.iter().any(|phrase| {
                let normalized_phrase = normalize_fact(phrase);
                normalized_answer.contains(&normalized_phrase)
                    || compact_answer.contains(&normalized_phrase.replace(' ', ""))
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        document_key: &str,
        unit_id: &str,
        filename: &str,
        page: u32,
        rank: u32,
    ) -> RetrievalCandidate {
        RetrievalCandidate {
            document_key: document_key.to_owned(),
            unit_id: unit_id.to_owned(),
            filename: filename.to_owned(),
            page: Some(page),
            rank,
            score: 1.0 / f64::from(rank),
        }
    }

    fn trace(candidates: Vec<RetrievalCandidate>) -> RetrievalTrace {
        let stage = StageTrace {
            candidates,
            telemetry: StageTelemetry {
                latency_ms: 10,
                request_count: 1,
                retry_count: 0,
                fallback: None,
            },
        };
        RetrievalTrace {
            dense: stage.clone(),
            bm25: stage.clone(),
            rrf: stage.clone(),
            rerank: stage,
        }
    }

    fn expected(document_key: &str, unit_id: &str, filename: &str, page: u32) -> ExpectedSource {
        ExpectedSource {
            document_key: document_key.to_owned(),
            unit_id: unit_id.to_owned(),
            filename: filename.to_owned(),
            page: Some(page),
        }
    }

    fn context(document_key: &str, unit_id: &str, filename: &str, page: u32) -> ContextSource {
        ContextSource {
            document_key: document_key.to_owned(),
            unit_id: unit_id.to_owned(),
            filename: filename.to_owned(),
            page: Some(page),
        }
    }

    fn input(
        answer: &str,
        expected_sources: Vec<ExpectedSource>,
        candidates: Vec<RetrievalCandidate>,
    ) -> QueryRunInput {
        QueryRunInput {
            query_id: "query-1".to_owned(),
            mode: EvaluationMode::A,
            answer: answer.to_owned(),
            expectation: QueryExpectation {
                answerable: true,
                expected_sources,
                required_facts: vec![],
            },
            stages: trace(candidates),
            context_sources: BTreeMap::new(),
            manual_review: ManualReview {
                status: ManualReviewStatus::Pending,
                reviewer: None,
                notes: None,
                reviewed_at: None,
            },
            failure: None,
        }
    }

    #[test]
    fn off_by_one_page_is_wrong() {
        let mut value = input(
            "答案 [S1]",
            vec![expected("north", "page-2", "report.pdf", 2)],
            vec![candidate("north", "page-2", "report.pdf", 2, 1)],
        );
        value
            .context_sources
            .insert("S1".to_owned(), context("north", "page-2", "report.pdf", 3));
        let scored = score_query(value, &ScoringConfig::default()).unwrap();
        let citations = &scored.metrics.unwrap().citations;
        assert_eq!(citations.assessments[0].outcome, CitationOutcome::WrongPage);
        assert_eq!(citations.page.precision, Some(0.0));
        assert_eq!(citations.exact.precision, Some(0.0));
    }

    #[test]
    fn same_filename_different_document_is_wrong() {
        let mut value = input(
            "答案 [S1]",
            vec![expected("north", "page-2", "quarterly-report.pdf", 2)],
            vec![candidate("north", "page-2", "quarterly-report.pdf", 2, 1)],
        );
        value.context_sources.insert(
            "S1".to_owned(),
            context("south", "page-2", "quarterly-report.pdf", 2),
        );
        let scored = score_query(value, &ScoringConfig::default()).unwrap();
        let citations = &scored.metrics.unwrap().citations;
        assert_eq!(
            citations.assessments[0].outcome,
            CitationOutcome::WrongDocument
        );
        assert_eq!(citations.filename.precision, Some(0.0));
    }

    #[test]
    fn multi_page_sources_receive_full_recall() {
        let expected_sources = vec![
            expected("north", "page-2", "report.pdf", 2),
            expected("north", "page-3", "report.pdf", 3),
        ];
        let candidates = vec![
            candidate("north", "page-2", "report.pdf", 2, 1),
            candidate("north", "page-3", "report.pdf", 3, 2),
        ];
        let mut value = input(
            "專案代號是極光17，交付期限是2026年9月30日。[S1][S2]",
            expected_sources,
            candidates,
        );
        value.expectation.required_facts = vec![
            "專案代號是極光-17".to_owned(),
            "交付期限是 2026 年 9 月 30 日".to_owned(),
        ];
        value
            .context_sources
            .insert("S1".to_owned(), context("north", "page-2", "report.pdf", 2));
        value
            .context_sources
            .insert("S2".to_owned(), context("north", "page-3", "report.pdf", 3));
        let scored = score_query(
            value,
            &ScoringConfig {
                cutoffs: vec![1, 2],
            },
        )
        .unwrap();
        let metrics = scored.metrics.unwrap();
        assert_eq!(metrics.citations.exact.recall, Some(1.0));
        assert_eq!(metrics.facts.recall, Some(1.0));
        assert_eq!(
            metrics.retrieval[0].metrics.page_recall_at_k[1].value,
            Some(1.0)
        );
    }

    #[test]
    fn unknown_s99_is_counted_only_when_present_in_answer() {
        let mut value = input(
            "答案 [S99]",
            vec![expected("north", "page-2", "report.pdf", 2)],
            vec![candidate("north", "page-2", "report.pdf", 2, 1)],
        );
        value
            .context_sources
            .insert("S1".to_owned(), context("north", "page-2", "report.pdf", 2));
        let scored = score_query(value, &ScoringConfig::default()).unwrap();
        let citations = &scored.metrics.unwrap().citations;
        assert_eq!(citations.cited_labels, vec!["S99"]);
        assert_eq!(citations.unknown_labels, vec!["S99"]);
        assert_eq!(
            citations.assessments[0].outcome,
            CitationOutcome::UnknownReference
        );
    }

    #[test]
    fn any_negative_citation_is_an_error() {
        let mut value = input("找不到資料 [S1]", vec![], vec![]);
        value.expectation.answerable = false;
        value
            .context_sources
            .insert("S1".to_owned(), context("north", "page-1", "report.pdf", 1));
        let scored = score_query(value, &ScoringConfig::default()).unwrap();
        let metrics = scored.metrics.unwrap();
        assert_eq!(metrics.citations.negative_citation_error, 1.0);
        assert_eq!(metrics.composite_score, 0.0);
    }

    #[test]
    fn negative_query_requires_a_deterministic_abstention_phrase() {
        for answer in [
            "目前資料不足，無法確認。",
            "根據提供內容無法回答這個問題。",
            "Insufficient information to answer.",
        ] {
            let mut value = input(answer, vec![], vec![]);
            value.expectation.answerable = false;
            let scored = score_query(value, &ScoringConfig::default()).unwrap();
            assert_eq!(scored.metrics.unwrap().composite_score, 1.0, "{answer}");
        }
    }

    #[test]
    fn uncited_hallucination_does_not_pass_a_negative_query() {
        let mut value = input("專案負責人是王小明。", vec![], vec![]);
        value.expectation.answerable = false;
        let scored = score_query(value, &ScoringConfig::default()).unwrap();
        let metrics = scored.metrics.unwrap();
        assert_eq!(metrics.citations.negative_citation_error, 0.0);
        assert_eq!(metrics.composite_score, 0.0);
    }

    #[test]
    fn required_fact_matching_is_normalized_and_deterministic() {
        assert_eq!(normalize_fact("OPS—AMBER, 15 分鐘！"), "ops amber 15 分鐘");
        let facts = score_required_facts(
            "通知代碼：OPS AMBER；連續 15 分鐘未恢復。",
            &[
                "通知代碼 OPS-AMBER".to_owned(),
                "連續15分鐘未恢復".to_owned(),
                "負責人林怡君".to_owned(),
            ],
        );
        assert_eq!(
            facts
                .matches
                .iter()
                .map(|fact| fact.matched)
                .collect::<Vec<_>>(),
            vec![true, true, false]
        );
        assert_eq!(facts.recall, Some(2.0 / 3.0));
    }
}
