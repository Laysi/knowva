//! Industry profiles: the deployment-time defaults one image ships for every industry.
//!
//! A profile is the directory `PROFILES_DIR/<KNOWVA_PROFILE>`. Its `profile.json` names the
//! frontend theme to serve and carries the interface copy, the suggested questions, and the
//! catalog keywords; its `prompt.md` is the industry section appended after the core answer
//! rules. The profile is read and validated once at startup, before the database is
//! touched, so a broken profile stops the server instead of reaching a user.
//!
//! Prompts never come from the browser: `/api/chat` has no authentication, and the industry
//! section reaches Gemma only from this file, composed after the core rules it cannot replace.

use anyhow::{anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::rag_context::{
    RAG_CATALOG_SYSTEM_PROMPT, RAG_NO_SOURCES_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT,
    compose_system_prompt,
};

pub const DEFAULT_PROFILE_ID: &str = "default";
/// Resolved against the backend's working directory, like the frontend build.
pub const DEFAULT_PROFILES_DIR: &str = "../profiles";
/// Upper bound on the industry section, counted in Unicode scalar values.
pub const MAX_INDUSTRY_SECTION_CHARS: usize = 2_000;
const MAX_NAME_LEN: usize = 64;
const PROFILE_FILE: &str = "profile.json";
const PROMPT_FILE: &str = "prompt.md";

/// The core rules an answer is held to, chosen once retrieval has finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptKind {
    /// The context holds at least one retrieved source.
    Answer,
    /// Retrieval left the context empty.
    NoSources,
    /// A whole-knowledge-base overview question.
    Catalog,
}

#[derive(Debug)]
pub struct Profile {
    id: String,
    ui: ProfileUi,
    catalog_keywords: CatalogKeywords,
    industry_section_chars: usize,
    answer_prompt: String,
    no_sources_prompt: String,
    catalog_prompt: String,
}

/// `profile.json` exactly as written. Every field is required and an unknown field is an
/// error, so a typo cannot silently fall back to a default.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ProfileFile {
    theme: String,
    brand: Brand,
    pages: Pages,
    suggestions: Vec<Suggestion>,
    composer_placeholder: String,
    catalog_keywords: CatalogKeywords,
}

/// The interface fields `GET /api/profile` returns. Prompt text and catalog keywords are not
/// part of this type, so they cannot be serialized to a browser by accident.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProfileUi {
    id: String,
    theme: String,
    brand: Brand,
    pages: Pages,
    suggestions: Vec<Suggestion>,
    composer_placeholder: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Brand {
    name: String,
    tagline: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Pages {
    chat: PageCopy,
    documents: PageCopy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct PageCopy {
    title: String,
}

/// A suggested question. `catalog` declares that the question is a whole-knowledge-base
/// overview under this profile's catalog keywords; tests hold every profile to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Suggestion {
    pub(crate) text: String,
    pub(crate) catalog: bool,
}

/// Terms that make a question a catalog query: it must contain one `scope` term and one
/// `intent` term. Stored trimmed and lowercased, matching the lowercased query.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct CatalogKeywords {
    scope: Vec<String>,
    intent: Vec<String>,
}

impl CatalogKeywords {
    pub fn scope(&self) -> &[String] {
        &self.scope
    }

    pub fn intent(&self) -> &[String] {
        &self.intent
    }
}

impl Profile {
    /// Loads `KNOWVA_PROFILE` (default `default`) from `PROFILES_DIR` (default `../profiles`).
    pub fn from_env() -> anyhow::Result<Self> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// Like [`Profile::from_env`]; empty values count as unset.
    pub fn from_lookup(mut get: impl FnMut(&str) -> Option<String>) -> anyhow::Result<Self> {
        let mut setting = |name: &str| {
            get(name)
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        };
        let id = setting("KNOWVA_PROFILE").unwrap_or_else(|| DEFAULT_PROFILE_ID.to_owned());
        let profiles_dir =
            setting("PROFILES_DIR").unwrap_or_else(|| DEFAULT_PROFILES_DIR.to_owned());
        Self::load(Path::new(&profiles_dir), &id)
    }

    pub fn load(profiles_dir: &Path, id: &str) -> anyhow::Result<Self> {
        // The id and the theme both become path segments, so neither may be `..`, contain a
        // separator, or be anything other than a plain lowercase name.
        validate_name("KNOWVA_PROFILE", id)?;
        let directory = profiles_dir.join(id);
        if !directory.is_dir() {
            bail!(
                "profile {id:?} not found: {} is not a directory; check KNOWVA_PROFILE and PROFILES_DIR",
                directory.display()
            );
        }

        let profile_path = directory.join(PROFILE_FILE);
        let bytes = std::fs::read(&profile_path).map_err(|error| {
            anyhow!(
                "profile {id:?}: {} could not be read: {error}",
                profile_path.display()
            )
        })?;
        let file: ProfileFile = serde_json::from_slice(&bytes).map_err(|error| {
            anyhow!(
                "profile {id:?}: {} is invalid: {error}",
                profile_path.display()
            )
        })?;
        validate_name("theme", &file.theme).map_err(|error| anyhow!("profile {id:?}: {error}"))?;
        validate_copy(&file).map_err(|error| anyhow!("profile {id:?}: {error}"))?;
        let catalog_keywords = normalize_catalog_keywords(file.catalog_keywords)
            .map_err(|error| anyhow!("profile {id:?}: {error}"))?;

        let prompt_path = directory.join(PROMPT_FILE);
        let raw_prompt = std::fs::read_to_string(&prompt_path).map_err(|error| {
            anyhow!(
                "profile {id:?}: {} could not be read: {error}",
                prompt_path.display()
            )
        })?;
        let industry_section = normalize_industry_section(&raw_prompt);
        let industry_section_chars = industry_section.chars().count();
        if industry_section_chars > MAX_INDUSTRY_SECTION_CHARS {
            bail!(
                "profile {id:?}: the industry section in {} has {industry_section_chars} characters, \
                 over the {MAX_INDUSTRY_SECTION_CHARS}-character limit",
                prompt_path.display()
            );
        }

        Ok(Self {
            id: id.to_owned(),
            ui: ProfileUi {
                id: id.to_owned(),
                theme: file.theme,
                brand: file.brand,
                pages: file.pages,
                suggestions: file.suggestions,
                composer_placeholder: file.composer_placeholder,
            },
            catalog_keywords,
            industry_section_chars,
            answer_prompt: compose_system_prompt(RAG_SYSTEM_PROMPT, &industry_section),
            no_sources_prompt: compose_system_prompt(
                RAG_NO_SOURCES_SYSTEM_PROMPT,
                &industry_section,
            ),
            catalog_prompt: compose_system_prompt(RAG_CATALOG_SYSTEM_PROMPT, &industry_section),
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn theme(&self) -> &str {
        &self.ui.theme
    }

    pub fn ui(&self) -> &ProfileUi {
        &self.ui
    }

    pub fn catalog_keywords(&self) -> &CatalogKeywords {
        &self.catalog_keywords
    }

    pub fn industry_section_chars(&self) -> usize {
        self.industry_section_chars
    }

    /// The composed system prompt for `kind`; the only source of answer prompts for both
    /// `/api/chat` and the evaluation runner.
    pub fn system_prompt(&self, kind: PromptKind) -> &str {
        match kind {
            PromptKind::Answer => &self.answer_prompt,
            PromptKind::NoSources => &self.no_sources_prompt,
            PromptKind::Catalog => &self.catalog_prompt,
        }
    }

    /// Lowercase hex SHA-256 of the composed system prompt for `kind`, recorded so a run or a
    /// log line identifies exactly which prompt was sent.
    pub fn system_prompt_sha256(&self, kind: PromptKind) -> String {
        crate::evaluation::schema::sha256_hex(self.system_prompt(kind).as_bytes())
    }

    /// Resolves this profile's theme build under `dist_root` (`../frontend/dist`).
    ///
    /// Without any `dist_root` the frontend is served separately (Vite in development), so
    /// `None` means API only. A `dist_root` that lacks this theme is an image built without
    /// it, a deployment error that must stop startup rather than serve the wrong interface.
    pub fn theme_dist_dir(&self, dist_root: &Path) -> anyhow::Result<Option<PathBuf>> {
        if !dist_root.exists() {
            return Ok(None);
        }
        let theme_dir = dist_root.join(self.theme());
        let index = theme_dir.join("index.html");
        if !index.is_file() {
            bail!(
                "profile {:?} selects theme {:?}, but the frontend build has no such theme: {} not found",
                self.id,
                self.theme(),
                index.display()
            );
        }
        Ok(Some(theme_dir))
    }

    #[cfg(test)]
    pub(crate) fn suggestions(&self) -> &[Suggestion] {
        &self.ui.suggestions
    }

    /// The repository's default profile, for tests that need an application state.
    #[cfg(test)]
    pub(crate) fn for_test() -> std::sync::Arc<Self> {
        std::sync::Arc::new(
            Self::load(&repository_profiles_dir(), DEFAULT_PROFILE_ID)
                .expect("the repository default profile loads"),
        )
    }
}

/// `profiles/` at the repository root, independent of the test's working directory.
#[cfg(test)]
pub(crate) fn repository_profiles_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../profiles")
}

fn validate_name(label: &str, value: &str) -> anyhow::Result<()> {
    let valid = !value.is_empty()
        && value.len() <= MAX_NAME_LEN
        && !value.starts_with('-')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-');
    if !valid {
        bail!(
            "{label} {value:?} is invalid: use 1 to {MAX_NAME_LEN} lowercase ASCII letters, \
             digits, or '-', not starting with '-'"
        );
    }
    Ok(())
}

fn validate_copy(file: &ProfileFile) -> anyhow::Result<()> {
    for (field, value) in [
        ("brand.name", &file.brand.name),
        ("brand.tagline", &file.brand.tagline),
        ("pages.chat.title", &file.pages.chat.title),
        ("pages.documents.title", &file.pages.documents.title),
        ("composerPlaceholder", &file.composer_placeholder),
    ] {
        if value.trim().is_empty() {
            bail!("{PROFILE_FILE} field {field} must not be empty");
        }
    }
    let mut seen = BTreeSet::new();
    for (index, suggestion) in file.suggestions.iter().enumerate() {
        if suggestion.text.trim().is_empty() {
            bail!("{PROFILE_FILE} field suggestions[{index}].text must not be empty");
        }
        if !seen.insert(suggestion.text.as_str()) {
            bail!(
                "{PROFILE_FILE} field suggestions[{index}].text repeats {:?}",
                suggestion.text
            );
        }
    }
    Ok(())
}

fn normalize_catalog_keywords(keywords: CatalogKeywords) -> anyhow::Result<CatalogKeywords> {
    let normalize = |field: &str, terms: Vec<String>| -> anyhow::Result<Vec<String>> {
        if terms.is_empty() {
            bail!("{PROFILE_FILE} field catalogKeywords.{field} must list at least one keyword");
        }
        terms
            .into_iter()
            .enumerate()
            .map(|(index, term)| {
                // An empty term would be contained in every query.
                let term = term.trim().to_lowercase();
                if term.is_empty() {
                    bail!(
                        "{PROFILE_FILE} field catalogKeywords.{field}[{index}] must not be empty"
                    );
                }
                Ok(term)
            })
            .collect()
    };
    Ok(CatalogKeywords {
        scope: normalize("scope", keywords.scope)?,
        intent: normalize("intent", keywords.intent)?,
    })
}

/// Editor artifacts (a byte-order mark, CRLF line ends, surrounding blank lines) must not
/// change the prompt, and a file holding only whitespace is an empty industry section.
fn normalize_industry_section(raw: &str) -> String {
    raw.trim_start_matches('\u{feff}')
        .replace("\r\n", "\n")
        .trim()
        .to_owned()
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{PROFILE_FILE, PROMPT_FILE};
    use std::fs;
    use std::path::{Path, PathBuf};

    pub(crate) struct TempDir(pub(crate) PathBuf);

    impl TempDir {
        pub(crate) fn new(label: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("knowva-profile-{label}-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&path).expect("create temporary directory");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    pub(crate) fn valid_profile_json(theme: &str) -> serde_json::Value {
        serde_json::json!({
            "theme": theme,
            "brand": { "name": "測試品牌", "tagline": "測試標語" },
            "pages": {
                "chat": { "title": "問答頁" },
                "documents": { "title": "文件頁" }
            },
            "suggestions": [
                { "text": "知識庫有哪些主題？", "catalog": true },
                { "text": "怎麼申請請假？", "catalog": false }
            ],
            "composerPlaceholder": "輸入問題",
            "catalogKeywords": { "scope": ["知識庫"], "intent": ["主題"] }
        })
    }

    pub(crate) fn write_profile(root: &Path, id: &str, json: &serde_json::Value, prompt: &str) {
        let directory = root.join(id);
        fs::create_dir_all(&directory).expect("create profile directory");
        fs::write(
            directory.join(PROFILE_FILE),
            serde_json::to_vec_pretty(json).expect("serialize profile"),
        )
        .expect("write profile.json");
        fs::write(directory.join(PROMPT_FILE), prompt).expect("write prompt.md");
    }

    /// A valid profile that does not depend on the repository's `profiles/`, for tests that
    /// run where only `backend/` is present.
    pub(crate) fn standalone_profile() -> std::sync::Arc<super::Profile> {
        let root = TempDir::new("standalone");
        write_profile(&root.0, "standalone", &valid_profile_json("default"), "");
        std::sync::Arc::new(
            super::Profile::load(&root.0, "standalone").expect("load the standalone profile"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{TempDir, valid_profile_json, write_profile};
    use super::*;
    use crate::rag_context::INDUSTRY_SECTION_HEADING;
    use std::fs;

    const KINDS: [(PromptKind, &str); 3] = [
        (PromptKind::Answer, RAG_SYSTEM_PROMPT),
        (PromptKind::NoSources, RAG_NO_SOURCES_SYSTEM_PROMPT),
        (PromptKind::Catalog, RAG_CATALOG_SYSTEM_PROMPT),
    ];

    fn load_error(root: &Path, id: &str) -> String {
        format!(
            "{:#}",
            Profile::load(root, id).expect_err("the profile must be rejected")
        )
    }

    #[test]
    fn default_profile_composes_the_core_prompts_byte_for_byte() {
        let profile = Profile::for_test();
        assert_eq!(profile.id(), "default");
        assert_eq!(profile.theme(), "default");
        assert_eq!(profile.industry_section_chars(), 0);
        for (kind, core) in KINDS {
            assert_eq!(profile.system_prompt(kind).as_bytes(), core.as_bytes());
        }
    }

    #[test]
    fn industry_section_is_appended_after_the_complete_core_rules() {
        let root = TempDir::new("industry");
        let industry = "你是醫療院所的文件助理。\n回答時先列出結論，再列出依據。";
        // Surrounding blank lines and CRLF line ends are editor artifacts, not prompt text.
        write_profile(
            &root.0,
            "clinic",
            &valid_profile_json("default"),
            &format!("\u{feff}\r\n{}\r\n\r\n", industry.replace('\n', "\r\n")),
        );
        let profile = Profile::load(&root.0, "clinic").expect("load profile");
        assert_eq!(profile.industry_section_chars(), industry.chars().count());
        for (kind, core) in KINDS {
            let prompt = profile.system_prompt(kind);
            assert!(prompt.starts_with(core), "core rules come first, unchanged");
            assert_eq!(
                &prompt[core.len()..],
                format!("\n\n{INDUSTRY_SECTION_HEADING}\n{industry}")
            );
            assert_ne!(
                profile.system_prompt_sha256(kind),
                Profile::for_test().system_prompt_sha256(kind)
            );
        }
    }

    #[test]
    fn whitespace_only_prompt_is_an_empty_industry_section() {
        let root = TempDir::new("blank-prompt");
        write_profile(
            &root.0,
            "blank",
            &valid_profile_json("default"),
            " \n\r\n\t",
        );
        let profile = Profile::load(&root.0, "blank").expect("load profile");
        for (kind, core) in KINDS {
            assert_eq!(profile.system_prompt(kind), core);
        }
    }

    #[test]
    fn missing_profile_directory_stops_startup() {
        let root = TempDir::new("missing");
        let error = load_error(&root.0, "manufacturing");
        assert!(
            error.contains("profile \"manufacturing\" not found"),
            "{error}"
        );
        assert!(error.contains("is not a directory"), "{error}");

        let error = format!(
            "{:#}",
            Profile::from_lookup(|name| match name {
                "PROFILES_DIR" => Some(root.0.display().to_string()),
                _ => None,
            })
            .expect_err("an unset KNOWVA_PROFILE still needs profiles/default")
        );
        assert!(error.contains("profile \"default\" not found"), "{error}");
    }

    #[test]
    fn profile_id_and_theme_cannot_escape_their_directories() {
        let root = TempDir::new("names");
        write_profile(&root.0, "safe", &valid_profile_json("default"), "");
        for id in [
            "..", "../safe", "safe/..", "a/b", "a\\b", "Safe", "-x", "safe.d", "",
        ] {
            let error = load_error(&root.0, id);
            assert!(error.contains("KNOWVA_PROFILE"), "{id:?}: {error}");
            assert!(error.contains("is invalid"), "{id:?}: {error}");
        }
        for theme in ["../default", "..", "a/b", "Default", ""] {
            write_profile(&root.0, "themed", &valid_profile_json(theme), "");
            let error = load_error(&root.0, "themed");
            assert!(error.contains("theme"), "{theme:?}: {error}");
            assert!(error.contains("is invalid"), "{theme:?}: {error}");
        }
    }

    #[test]
    fn unknown_or_missing_profile_fields_stop_startup() {
        let root = TempDir::new("fields");
        let mut unknown = valid_profile_json("default");
        unknown["welcomeTitle"] = serde_json::json!("歡迎");
        write_profile(&root.0, "unknown", &unknown, "");
        let error = load_error(&root.0, "unknown");
        assert!(error.contains("profile.json is invalid"), "{error}");
        assert!(error.contains("unknown field `welcomeTitle`"), "{error}");

        let mut nested = valid_profile_json("default");
        nested["brand"]["logo"] = serde_json::json!("logo.svg");
        write_profile(&root.0, "nested", &nested, "");
        assert!(load_error(&root.0, "nested").contains("unknown field `logo`"));

        let mut missing = valid_profile_json("default");
        missing
            .as_object_mut()
            .expect("profile object")
            .remove("composerPlaceholder");
        write_profile(&root.0, "missing", &missing, "");
        let error = load_error(&root.0, "missing");
        assert!(
            error.contains("missing field `composerPlaceholder`"),
            "{error}"
        );

        let mut unmarked = valid_profile_json("default");
        unmarked["suggestions"][0]
            .as_object_mut()
            .expect("suggestion object")
            .remove("catalog");
        write_profile(&root.0, "unmarked", &unmarked, "");
        assert!(load_error(&root.0, "unmarked").contains("missing field `catalog`"));

        let mut blank = valid_profile_json("default");
        blank["brand"]["name"] = serde_json::json!("  ");
        write_profile(&root.0, "blank", &blank, "");
        assert!(load_error(&root.0, "blank").contains("brand.name must not be empty"));

        let directory = root.0.join("no-prompt");
        fs::create_dir_all(&directory).expect("create profile directory");
        fs::write(
            directory.join(PROFILE_FILE),
            valid_profile_json("default").to_string(),
        )
        .expect("write profile.json");
        let error = load_error(&root.0, "no-prompt");
        assert!(error.contains("prompt.md could not be read"), "{error}");
    }

    #[test]
    fn industry_section_over_the_limit_stops_startup() {
        let root = TempDir::new("limit");
        // CJK characters are three bytes each: the limit counts characters, not bytes.
        let at_limit = "產".repeat(MAX_INDUSTRY_SECTION_CHARS);
        write_profile(
            &root.0,
            "at-limit",
            &valid_profile_json("default"),
            &at_limit,
        );
        let profile = Profile::load(&root.0, "at-limit").expect("2,000 characters are allowed");
        assert_eq!(profile.industry_section_chars(), MAX_INDUSTRY_SECTION_CHARS);

        let over = format!("{at_limit}業");
        write_profile(&root.0, "over", &valid_profile_json("default"), &over);
        let error = load_error(&root.0, "over");
        assert!(error.contains("has 2001 characters"), "{error}");
        assert!(error.contains("2000-character limit"), "{error}");
    }

    #[test]
    fn empty_catalog_keywords_stop_startup() {
        let root = TempDir::new("keywords");
        for (field, keywords, expected) in [
            (
                "scope",
                serde_json::json!({ "scope": [], "intent": ["主題"] }),
                "catalogKeywords.scope must list at least one keyword",
            ),
            (
                "intent",
                serde_json::json!({ "scope": ["知識庫"], "intent": [] }),
                "catalogKeywords.intent must list at least one keyword",
            ),
            (
                "blank",
                serde_json::json!({ "scope": ["知識庫", " "], "intent": ["主題"] }),
                "catalogKeywords.scope[1] must not be empty",
            ),
        ] {
            let mut json = valid_profile_json("default");
            json["catalogKeywords"] = keywords;
            write_profile(&root.0, field, &json, "");
            let error = load_error(&root.0, field);
            assert!(error.contains(expected), "{field}: {error}");
        }

        let mut mixed_case = valid_profile_json("default");
        mixed_case["catalogKeywords"] =
            serde_json::json!({ "scope": [" Wiki "], "intent": ["Topics"] });
        write_profile(&root.0, "mixed-case", &mixed_case, "");
        let profile = Profile::load(&root.0, "mixed-case").expect("load profile");
        assert_eq!(profile.catalog_keywords().scope(), ["wiki"]);
        assert_eq!(profile.catalog_keywords().intent(), ["topics"]);
    }

    #[test]
    fn frontend_dist_without_the_theme_stops_startup() {
        let root = TempDir::new("dist");
        write_profile(&root.0, "alpha", &valid_profile_json("alpha"), "");
        let profile = Profile::load(&root.0, "alpha").expect("load profile");

        let missing_dist = root.0.join("no-dist");
        assert_eq!(
            profile
                .theme_dist_dir(&missing_dist)
                .expect("no dist means API only"),
            None
        );

        let dist = root.0.join("dist");
        fs::create_dir_all(dist.join("beta")).expect("create other theme");
        fs::write(dist.join("beta/index.html"), "beta").expect("write other theme");
        let error = format!(
            "{:#}",
            profile
                .theme_dist_dir(&dist)
                .expect_err("a dist without the theme is a deployment error")
        );
        assert!(
            error.contains("selects theme \"alpha\", but the frontend build has no such theme"),
            "{error}"
        );

        fs::create_dir_all(dist.join("alpha")).expect("create theme without index");
        assert!(profile.theme_dist_dir(&dist).is_err());

        fs::write(dist.join("alpha/index.html"), "alpha").expect("write theme index");
        assert_eq!(
            profile.theme_dist_dir(&dist).expect("theme is built"),
            Some(dist.join("alpha"))
        );
    }

    #[test]
    fn empty_settings_fall_back_to_the_defaults() {
        let root = TempDir::new("defaults");
        write_profile(&root.0, "default", &valid_profile_json("default"), "");
        let profile = Profile::from_lookup(|name| match name {
            "KNOWVA_PROFILE" => Some("  ".to_owned()),
            "PROFILES_DIR" => Some(root.0.display().to_string()),
            _ => None,
        })
        .expect("load default profile");
        assert_eq!(profile.id(), DEFAULT_PROFILE_ID);
    }
}
