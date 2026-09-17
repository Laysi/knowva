use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::ErrorKind;
use std::path::{Component, Path};

pub const DATASET_VERSION: &str = "1.0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationDataset {
    pub version: String,
    pub documents: Vec<DocumentFixture>,
    pub queries: Vec<QueryFixture>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentFixture {
    pub document_key: String,
    pub relative_path: String,
    pub sha256: String,
    pub license: String,
    pub source: String,
    pub kind: DocumentKind,
    pub expected: ExpectedInventory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentKind {
    NativePdf,
    ScannedPdf,
    Presentation,
    TableChart,
    Image,
    Markdown,
    Text,
}

impl DocumentKind {
    fn expected_extension(self) -> &'static str {
        match self {
            Self::NativePdf | Self::ScannedPdf | Self::TableChart => "pdf",
            Self::Presentation => "pptx",
            Self::Image => "png",
            Self::Markdown => "md",
            Self::Text => "txt",
        }
    }

    fn is_unpaged_text(self) -> bool {
        matches!(self, Self::Markdown | Self::Text)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedInventory {
    pub pages: Vec<u32>,
    pub units: Vec<ExpectedUnit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedUnit {
    pub unit_id: String,
    pub page: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryFixture {
    pub query_id: String,
    pub query: String,
    pub document_scope: Vec<String>,
    pub expected_sources: Vec<ExpectedSource>,
    pub required_facts: Vec<String>,
    pub answerable: bool,
    #[serde(rename = "type")]
    pub query_type: QueryType,
    pub difficulty: Difficulty,
    pub requires_visual: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedSource {
    pub document_key: String,
    pub unit_id: String,
    pub page: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryType {
    FactLookup,
    MultiSource,
    VisualReasoning,
    NotFound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValidationError {
    pub code: &'static str,
    pub path: String,
    pub message: &'static str,
}

impl ValidationError {
    fn new(code: &'static str, path: impl Into<String>, message: &'static str) -> Self {
        Self {
            code,
            path: path.into(),
            message,
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} at {}: {}",
            self.code, self.path, self.message
        )
    }
}

impl std::error::Error for ValidationError {}

pub type DatasetResult<T> = Result<T, Vec<ValidationError>>;

/// Immutable input captured during the initial validation pass.
///
/// Cloud evaluation must consume this snapshot rather than reopening user-controlled
/// paths after preflight validation.
#[derive(Debug, Clone)]
pub struct EvaluationDatasetSnapshot {
    pub dataset: EvaluationDataset,
    pub manifest_sha256: String,
    fixture_bytes: BTreeMap<String, Vec<u8>>,
}

impl EvaluationDatasetSnapshot {
    pub fn fixture_bytes(&self, document_key: &str) -> Option<&[u8]> {
        self.fixture_bytes.get(document_key).map(Vec::as_slice)
    }
}

pub fn load_dataset(manifest_path: impl AsRef<Path>) -> DatasetResult<EvaluationDataset> {
    let bytes = read_manifest_bytes(manifest_path.as_ref())?;
    parse_manifest(&bytes)
}

fn read_manifest_bytes(manifest_path: &Path) -> DatasetResult<Vec<u8>> {
    Ok(match fs::read(manifest_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return Err(vec![ValidationError::new(
                "manifest_not_found",
                "$",
                "dataset manifest file was not found",
            )]);
        }
        Err(_) => {
            return Err(vec![ValidationError::new(
                "manifest_read_failed",
                "$",
                "dataset manifest file could not be read",
            )]);
        }
    })
}

fn parse_manifest(bytes: &[u8]) -> DatasetResult<EvaluationDataset> {
    serde_json::from_slice(bytes).map_err(|_| {
        vec![ValidationError::new(
            "manifest_invalid_json",
            "$",
            "dataset manifest is not valid JSON for the supported schema",
        )]
    })
}

pub fn load_and_validate_dataset(
    manifest_path: impl AsRef<Path>,
) -> DatasetResult<EvaluationDataset> {
    load_and_validate_dataset_snapshot(manifest_path).map(|snapshot| snapshot.dataset)
}

pub fn load_and_validate_dataset_snapshot(
    manifest_path: impl AsRef<Path>,
) -> DatasetResult<EvaluationDatasetSnapshot> {
    let manifest_path = manifest_path.as_ref();
    let manifest_bytes = read_manifest_bytes(manifest_path)?;
    let dataset = parse_manifest(&manifest_bytes)?;
    let fixture_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let fixture_bytes = validate_dataset_and_capture_fixtures(&dataset, fixture_root)?;
    Ok(EvaluationDatasetSnapshot {
        dataset,
        manifest_sha256: sha256_hex(&manifest_bytes),
        fixture_bytes,
    })
}

pub fn validate_dataset_dry_run(
    dataset: &EvaluationDataset,
    fixture_root: impl AsRef<Path>,
) -> DatasetResult<()> {
    validate_dataset_and_capture_fixtures(dataset, fixture_root).map(|_| ())
}

fn validate_dataset_and_capture_fixtures(
    dataset: &EvaluationDataset,
    fixture_root: impl AsRef<Path>,
) -> DatasetResult<BTreeMap<String, Vec<u8>>> {
    let fixture_root = fixture_root.as_ref();
    let canonical_root = fixture_root.canonicalize().ok();
    let mut errors = Vec::new();

    if dataset.version != DATASET_VERSION {
        errors.push(ValidationError::new(
            "unsupported_version",
            "$.version",
            "dataset version is not supported",
        ));
    }
    if dataset.documents.is_empty() {
        errors.push(ValidationError::new(
            "documents_empty",
            "$.documents",
            "dataset must declare at least one document",
        ));
    }
    if dataset.queries.is_empty() {
        errors.push(ValidationError::new(
            "queries_empty",
            "$.queries",
            "dataset must declare at least one query",
        ));
    }

    let mut document_by_key = BTreeMap::<String, usize>::new();
    let mut document_by_path = BTreeMap::<String, usize>::new();
    let mut unit_by_key = BTreeMap::<(String, String), Option<u32>>::new();
    let mut fixture_bytes = BTreeMap::new();

    for (document_index, document) in dataset.documents.iter().enumerate() {
        let base = format!("$.documents[{document_index}]");
        let key_path = format!("{base}.document_key");
        validate_identifier(&document.document_key, &key_path, &mut errors);
        if document_by_key.contains_key(&document.document_key) {
            errors.push(ValidationError::new(
                "duplicate_document_key",
                key_path,
                "document_key must be unique",
            ));
        } else {
            document_by_key.insert(document.document_key.clone(), document_index);
        }

        let relative_path_path = format!("{base}.relative_path");
        let safe_relative_path =
            validate_relative_path(&document.relative_path, &relative_path_path, &mut errors);
        if document_by_path.contains_key(&document.relative_path) {
            errors.push(ValidationError::new(
                "duplicate_relative_path",
                &relative_path_path,
                "relative_path must be unique",
            ));
        } else {
            document_by_path.insert(document.relative_path.clone(), document_index);
        }

        if extension(&document.relative_path) != Some(document.kind.expected_extension()) {
            errors.push(ValidationError::new(
                "document_extension_mismatch",
                &relative_path_path,
                "document extension does not match kind",
            ));
        }
        if document.license.trim().is_empty() {
            errors.push(ValidationError::new(
                "license_empty",
                format!("{base}.license"),
                "license must not be empty",
            ));
        }
        if document.source.trim().is_empty() {
            errors.push(ValidationError::new(
                "source_empty",
                format!("{base}.source"),
                "source must not be empty",
            ));
        }

        validate_inventory(document, document_index, &mut unit_by_key, &mut errors);

        let checksum_valid = is_lower_hex_sha256(&document.sha256);
        if !checksum_valid {
            errors.push(ValidationError::new(
                "sha256_invalid",
                format!("{base}.sha256"),
                "sha256 must contain 64 lowercase hexadecimal characters",
            ));
        }

        if safe_relative_path
            && let Some(bytes) = validate_fixture_file(
                document,
                document_index,
                fixture_root,
                canonical_root.as_deref(),
                checksum_valid,
                &mut errors,
            )
        {
            fixture_bytes.insert(document.document_key.clone(), bytes);
        }
    }

    let mut query_ids = BTreeSet::new();
    for (query_index, query) in dataset.queries.iter().enumerate() {
        let base = format!("$.queries[{query_index}]");
        let query_id_path = format!("{base}.query_id");
        validate_identifier(&query.query_id, &query_id_path, &mut errors);
        if !query_ids.insert(query.query_id.clone()) {
            errors.push(ValidationError::new(
                "duplicate_query_id",
                query_id_path,
                "query_id must be unique",
            ));
        }
        if query.query.trim().is_empty() {
            errors.push(ValidationError::new(
                "query_empty",
                format!("{base}.query"),
                "query must not be empty",
            ));
        }

        validate_query_scope(query, query_index, &document_by_key, &mut errors);
        validate_required_facts(query, query_index, &mut errors);

        if query.answerable {
            if query.expected_sources.is_empty() {
                errors.push(ValidationError::new(
                    "answerable_sources_empty",
                    format!("{base}.expected_sources"),
                    "answerable query must declare at least one expected source",
                ));
            }
            if query.required_facts.is_empty() {
                errors.push(ValidationError::new(
                    "answerable_facts_empty",
                    format!("{base}.required_facts"),
                    "answerable query must declare at least one required fact",
                ));
            }
            if query.query_type == QueryType::NotFound {
                errors.push(ValidationError::new(
                    "query_type_mismatch",
                    format!("{base}.type"),
                    "answerable query cannot use not_found type",
                ));
            }
        } else {
            if !query.expected_sources.is_empty() {
                errors.push(ValidationError::new(
                    "negative_query_has_sources",
                    format!("{base}.expected_sources"),
                    "not-found query must not declare expected sources",
                ));
            }
            if !query.required_facts.is_empty() {
                errors.push(ValidationError::new(
                    "negative_query_has_facts",
                    format!("{base}.required_facts"),
                    "not-found query must not declare required facts",
                ));
            }
            if query.query_type != QueryType::NotFound {
                errors.push(ValidationError::new(
                    "query_type_mismatch",
                    format!("{base}.type"),
                    "unanswerable query must use not_found type",
                ));
            }
        }

        validate_expected_sources(
            query,
            query_index,
            &document_by_key,
            &unit_by_key,
            &mut errors,
        );
    }

    errors.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then_with(|| left.code.cmp(right.code))
    });
    if errors.is_empty() {
        Ok(fixture_bytes)
    } else {
        Err(errors)
    }
}

fn validate_identifier(value: &str, path: &str, errors: &mut Vec<ValidationError>) {
    let valid = !value.is_empty()
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
        });
    if !valid {
        errors.push(ValidationError::new(
            "identifier_invalid",
            path,
            "identifier must use lowercase ASCII letters, digits, dot, dash, or underscore",
        ));
    }
}

fn validate_relative_path(value: &str, path: &str, errors: &mut Vec<ValidationError>) -> bool {
    let candidate = Path::new(value);
    let valid = !value.is_empty()
        && !value.contains('\\')
        && !candidate.is_absolute()
        && candidate
            .components()
            .all(|component| matches!(component, Component::Normal(_)));
    if !valid {
        errors.push(ValidationError::new(
            "relative_path_invalid",
            path,
            "relative_path must be a normalized relative path",
        ));
    }
    valid
}

fn extension(path: &str) -> Option<&str> {
    Path::new(path).extension().and_then(|value| value.to_str())
}

fn validate_inventory(
    document: &DocumentFixture,
    document_index: usize,
    unit_by_key: &mut BTreeMap<(String, String), Option<u32>>,
    errors: &mut Vec<ValidationError>,
) {
    let base = format!("$.documents[{document_index}].expected");
    let is_text = document.kind.is_unpaged_text();
    let mut page_indices = BTreeMap::<u32, usize>::new();

    if is_text && !document.expected.pages.is_empty() {
        errors.push(ValidationError::new(
            "text_document_has_pages",
            format!("{base}.pages"),
            "Markdown and text documents must have an empty page inventory",
        ));
    } else if !is_text && document.expected.pages.is_empty() {
        errors.push(ValidationError::new(
            "page_inventory_empty",
            format!("{base}.pages"),
            "paged document must declare at least one page",
        ));
    }

    for (page_index, page) in document.expected.pages.iter().copied().enumerate() {
        let page_path = format!("{base}.pages[{page_index}]");
        if page == 0 {
            errors.push(ValidationError::new(
                "page_not_one_based",
                page_path,
                "page numbers are 1-based and must be greater than zero",
            ));
        } else if page_indices.insert(page, page_index).is_some() {
            errors.push(ValidationError::new(
                "duplicate_page",
                page_path,
                "page inventory must not contain duplicates",
            ));
        }
    }

    if !is_text
        && !page_indices.is_empty()
        && page_indices
            .keys()
            .copied()
            .ne((1..=page_indices.len()).map(|value| value as u32))
    {
        errors.push(ValidationError::new(
            "page_inventory_not_contiguous",
            format!("{base}.pages"),
            "page inventory must be contiguous and start at page 1",
        ));
    }

    if document.expected.units.is_empty() {
        errors.push(ValidationError::new(
            "unit_inventory_empty",
            format!("{base}.units"),
            "document must declare at least one unit",
        ));
    }

    let mut unit_ids = BTreeSet::new();
    let mut pages_with_units = BTreeSet::new();
    for (unit_index, unit) in document.expected.units.iter().enumerate() {
        let unit_base = format!("{base}.units[{unit_index}]");
        validate_identifier(&unit.unit_id, &format!("{unit_base}.unit_id"), errors);
        if !unit_ids.insert(unit.unit_id.clone()) {
            errors.push(ValidationError::new(
                "duplicate_unit_id",
                format!("{unit_base}.unit_id"),
                "unit_id must be unique within a document",
            ));
        } else {
            unit_by_key.insert(
                (document.document_key.clone(), unit.unit_id.clone()),
                unit.page,
            );
        }

        match (is_text, unit.page) {
            (true, Some(_)) => errors.push(ValidationError::new(
                "text_unit_has_page",
                format!("{unit_base}.page"),
                "Markdown and text units must use null page",
            )),
            (false, None) => errors.push(ValidationError::new(
                "paged_unit_missing_page",
                format!("{unit_base}.page"),
                "paged document units must declare a page",
            )),
            (false, Some(0)) => errors.push(ValidationError::new(
                "page_not_one_based",
                format!("{unit_base}.page"),
                "page numbers are 1-based and must be greater than zero",
            )),
            (false, Some(page)) => {
                if !page_indices.contains_key(&page) {
                    errors.push(ValidationError::new(
                        "unit_page_not_in_inventory",
                        format!("{unit_base}.page"),
                        "unit page must exist in the document page inventory",
                    ));
                } else {
                    pages_with_units.insert(page);
                }
            }
            (true, None) => {}
        }
    }

    if !is_text {
        for (page, page_index) in page_indices {
            if !pages_with_units.contains(&page) {
                errors.push(ValidationError::new(
                    "page_has_no_unit",
                    format!("{base}.pages[{page_index}]"),
                    "each declared page must have at least one unit",
                ));
            }
        }
    }
}

fn validate_fixture_file(
    document: &DocumentFixture,
    document_index: usize,
    fixture_root: &Path,
    canonical_root: Option<&Path>,
    checksum_valid: bool,
    errors: &mut Vec<ValidationError>,
) -> Option<Vec<u8>> {
    let relative_path_path = format!("$.documents[{document_index}].relative_path");
    let fixture_path = fixture_root.join(&document.relative_path);
    let metadata = match fs::metadata(&fixture_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            errors.push(ValidationError::new(
                "fixture_file_missing",
                relative_path_path,
                "declared fixture file was not found",
            ));
            return None;
        }
        Err(_) => {
            errors.push(ValidationError::new(
                "fixture_file_unreadable",
                relative_path_path,
                "declared fixture file could not be read",
            ));
            return None;
        }
    };
    if !metadata.is_file() {
        errors.push(ValidationError::new(
            "fixture_not_regular_file",
            relative_path_path,
            "declared fixture must be a regular file",
        ));
        return None;
    }

    if let (Some(root), Ok(canonical_fixture)) = (canonical_root, fixture_path.canonicalize())
        && !canonical_fixture.starts_with(root)
    {
        errors.push(ValidationError::new(
            "fixture_outside_root",
            relative_path_path,
            "declared fixture must remain inside the fixture root",
        ));
        return None;
    }

    let bytes = match fs::read(&fixture_path) {
        Ok(bytes) => bytes,
        Err(_) => {
            errors.push(ValidationError::new(
                "fixture_file_unreadable",
                relative_path_path,
                "declared fixture file could not be read",
            ));
            return None;
        }
    };
    if checksum_valid && sha256_hex(&bytes) != document.sha256 {
        errors.push(ValidationError::new(
            "sha256_mismatch",
            format!("$.documents[{document_index}].sha256"),
            "fixture checksum does not match sha256",
        ));
    }
    Some(bytes)
}

fn validate_query_scope(
    query: &QueryFixture,
    query_index: usize,
    document_by_key: &BTreeMap<String, usize>,
    errors: &mut Vec<ValidationError>,
) {
    let path = format!("$.queries[{query_index}].document_scope");
    if query.document_scope.is_empty() {
        errors.push(ValidationError::new(
            "document_scope_empty",
            path,
            "query must declare at least one document in scope",
        ));
        return;
    }
    let mut seen = BTreeSet::new();
    for (scope_index, document_key) in query.document_scope.iter().enumerate() {
        let item_path = format!("{path}[{scope_index}]");
        if !seen.insert(document_key) {
            errors.push(ValidationError::new(
                "duplicate_scope_document",
                item_path.clone(),
                "document_scope must not contain duplicates",
            ));
        }
        if !document_by_key.contains_key(document_key) {
            errors.push(ValidationError::new(
                "scope_document_not_found",
                item_path,
                "document_scope references an unknown document_key",
            ));
        }
    }
}

fn validate_required_facts(
    query: &QueryFixture,
    query_index: usize,
    errors: &mut Vec<ValidationError>,
) {
    let mut facts = BTreeSet::new();
    for (fact_index, fact) in query.required_facts.iter().enumerate() {
        let path = format!("$.queries[{query_index}].required_facts[{fact_index}]");
        if fact.trim().is_empty() {
            errors.push(ValidationError::new(
                "required_fact_empty",
                path,
                "required fact must not be empty",
            ));
        } else if !facts.insert(fact) {
            errors.push(ValidationError::new(
                "duplicate_required_fact",
                path,
                "required facts must not contain duplicates",
            ));
        }
    }
}

fn validate_expected_sources(
    query: &QueryFixture,
    query_index: usize,
    document_by_key: &BTreeMap<String, usize>,
    unit_by_key: &BTreeMap<(String, String), Option<u32>>,
    errors: &mut Vec<ValidationError>,
) {
    let scope: BTreeSet<&str> = query.document_scope.iter().map(String::as_str).collect();
    let mut seen = BTreeSet::new();
    for (source_index, source) in query.expected_sources.iter().enumerate() {
        let base = format!("$.queries[{query_index}].expected_sources[{source_index}]");
        let identity = (
            source.document_key.clone(),
            source.unit_id.clone(),
            source.page,
        );
        if !seen.insert(identity) {
            errors.push(ValidationError::new(
                "duplicate_expected_source",
                &base,
                "expected_sources must not contain duplicates",
            ));
        }
        if !scope.contains(source.document_key.as_str()) {
            errors.push(ValidationError::new(
                "source_outside_scope",
                format!("{base}.document_key"),
                "expected source document must be present in document_scope",
            ));
        }
        if !document_by_key.contains_key(&source.document_key) {
            errors.push(ValidationError::new(
                "source_document_not_found",
                format!("{base}.document_key"),
                "expected source references an unknown document_key",
            ));
            continue;
        }
        if source.page == Some(0) {
            errors.push(ValidationError::new(
                "page_not_one_based",
                format!("{base}.page"),
                "page numbers are 1-based and must be greater than zero",
            ));
        }
        match unit_by_key.get(&(source.document_key.clone(), source.unit_id.clone())) {
            None => errors.push(ValidationError::new(
                "source_unit_not_found",
                format!("{base}.unit_id"),
                "expected source references an unknown unit_id",
            )),
            Some(expected_page) if *expected_page != source.page => {
                errors.push(ValidationError::new(
                    "source_page_mismatch",
                    format!("{base}.page"),
                    "expected source page does not match the unit inventory",
                ));
            }
            Some(_) => {}
        }
    }
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn sha256_hex(input: &[u8]) -> String {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = INITIAL;
    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (index, bytes) in chunk.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temporary1 = h
                .wrapping_add(sum1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary2 = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }

    let mut output = String::with_capacity(64);
    for word in state {
        use std::fmt::Write as _;
        write!(&mut output, "{word:08x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn temporary_directory() -> PathBuf {
        let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "knowva-evaluation-schema-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(path.join("documents")).expect("create test fixture directory");
        path
    }

    fn sample_dataset(bytes: &[u8]) -> EvaluationDataset {
        EvaluationDataset {
            version: DATASET_VERSION.to_owned(),
            documents: vec![DocumentFixture {
                document_key: "notes".to_owned(),
                relative_path: "documents/notes.md".to_owned(),
                sha256: sha256_hex(bytes),
                license: "CC0-1.0".to_owned(),
                source: "synthetic test fixture".to_owned(),
                kind: DocumentKind::Markdown,
                expected: ExpectedInventory {
                    pages: vec![],
                    units: vec![ExpectedUnit {
                        unit_id: "chunk-1".to_owned(),
                        page: None,
                    }],
                },
            }],
            queries: vec![QueryFixture {
                query_id: "notes-owner".to_owned(),
                query: "Who owns the notes?".to_owned(),
                document_scope: vec!["notes".to_owned()],
                expected_sources: vec![ExpectedSource {
                    document_key: "notes".to_owned(),
                    unit_id: "chunk-1".to_owned(),
                    page: None,
                }],
                required_facts: vec!["The owner is Lin.".to_owned()],
                answerable: true,
                query_type: QueryType::FactLookup,
                difficulty: Difficulty::Easy,
                requires_visual: false,
            }],
        }
    }

    fn validate_sample(dataset: &EvaluationDataset, bytes: Option<&[u8]>) -> Vec<ValidationError> {
        let root = temporary_directory();
        if let Some(bytes) = bytes {
            fs::write(root.join("documents/notes.md"), bytes).expect("write test fixture");
        }
        let result = validate_dataset_dry_run(dataset, &root);
        fs::remove_dir_all(root).expect("remove test fixture directory");
        result.err().unwrap_or_default()
    }

    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn accepts_valid_unpaged_dataset() {
        let bytes = b"owner: Lin\n";
        let errors = validate_sample(&sample_dataset(bytes), Some(bytes));
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn validated_snapshot_is_unchanged_when_input_paths_are_replaced() {
        let root = temporary_directory();
        let original_fixture = b"owner: Lin\n";
        let manifest_path = root.join("dataset.json");
        let original_manifest =
            serde_json::to_vec(&sample_dataset(original_fixture)).expect("serialize manifest");
        fs::write(root.join("documents/notes.md"), original_fixture).expect("write fixture");
        fs::write(&manifest_path, &original_manifest).expect("write manifest");

        let snapshot =
            load_and_validate_dataset_snapshot(&manifest_path).expect("validate snapshot");
        fs::write(root.join("documents/notes.md"), b"owner: attacker\n").expect("replace fixture");
        fs::write(&manifest_path, b"{}\n").expect("replace manifest");

        assert_eq!(
            snapshot.fixture_bytes("notes"),
            Some(original_fixture.as_slice())
        );
        assert_eq!(snapshot.manifest_sha256, sha256_hex(&original_manifest));
        fs::remove_dir_all(root).expect("remove test fixture directory");
    }

    #[test]
    fn missing_manifest_and_fixture_have_stable_safe_errors() {
        let root = temporary_directory();
        let manifest_error = load_dataset(root.join("absent.json")).unwrap_err();
        assert_eq!(manifest_error[0].code, "manifest_not_found");
        assert_eq!(manifest_error[0].path, "$");
        assert!(
            !manifest_error[0]
                .message
                .contains(root.to_string_lossy().as_ref())
        );
        fs::remove_dir_all(root).expect("remove test fixture directory");

        let errors = validate_sample(&sample_dataset(b"owner: Lin\n"), None);
        let error = errors
            .iter()
            .find(|error| error.code == "fixture_file_missing")
            .expect("missing fixture error");
        assert_eq!(error.path, "$.documents[0].relative_path");
        assert_eq!(error.message, "declared fixture file was not found");
    }

    #[test]
    fn rejects_checksum_mismatch() {
        let errors = validate_sample(&sample_dataset(b"expected"), Some(b"actual"));
        assert!(errors.iter().any(|error| {
            error.code == "sha256_mismatch" && error.path == "$.documents[0].sha256"
        }));
    }

    #[test]
    fn rejects_duplicate_document_query_and_unit_keys() {
        let bytes = b"owner: Lin\n";
        let mut dataset = sample_dataset(bytes);
        let duplicate_unit = dataset.documents[0].expected.units[0].clone();
        dataset.documents[0].expected.units.push(duplicate_unit);
        dataset.documents.push(dataset.documents[0].clone());
        dataset.queries.push(dataset.queries[0].clone());
        let errors = validate_sample(&dataset, Some(bytes));
        for code in [
            "duplicate_document_key",
            "duplicate_relative_path",
            "duplicate_unit_id",
            "duplicate_query_id",
        ] {
            assert!(
                errors.iter().any(|error| error.code == code),
                "missing {code}: {errors:?}"
            );
        }
    }

    #[test]
    fn enforces_one_based_inventory_and_null_text_pages() {
        let bytes = b"owner: Lin\n";
        let mut dataset = sample_dataset(bytes);
        dataset.documents[0].expected.pages = vec![0];
        dataset.documents[0].expected.units[0].page = Some(0);
        let errors = validate_sample(&dataset, Some(bytes));
        assert!(
            errors
                .iter()
                .any(|error| error.code == "text_document_has_pages")
        );
        assert!(
            errors
                .iter()
                .any(|error| error.code == "text_unit_has_page")
        );

        dataset.documents[0].kind = DocumentKind::NativePdf;
        dataset.documents[0].relative_path = "documents/notes.pdf".to_owned();
        dataset.documents[0].expected.pages = vec![0, 2];
        dataset.documents[0].expected.units[0].page = Some(0);
        let errors = validate_sample(&dataset, None);
        assert!(
            errors
                .iter()
                .any(|error| error.code == "page_not_one_based")
        );
        assert!(
            errors
                .iter()
                .any(|error| error.code == "page_inventory_not_contiguous")
        );
    }

    #[test]
    fn negative_query_cannot_claim_sources_or_facts() {
        let bytes = b"owner: Lin\n";
        let mut dataset = sample_dataset(bytes);
        dataset.queries[0].answerable = false;
        dataset.queries[0].query_type = QueryType::NotFound;
        let errors = validate_sample(&dataset, Some(bytes));
        assert!(
            errors
                .iter()
                .any(|error| error.code == "negative_query_has_sources")
        );
        assert!(
            errors
                .iter()
                .any(|error| error.code == "negative_query_has_facts")
        );
    }
}
