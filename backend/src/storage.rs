//! File storage selected once at initialization: a local directory or an
//! S3-compatible bucket, both driven through `object_store`.
//!
//! Every object is addressed by a relative key below `{document_id}/`. The
//! backend identity is locked in the database on first start because stored
//! keys are only meaningful inside that backend; `FILES_DIR` and the S3
//! endpoint are not part of the lock, so mounts and endpoints may move.

use anyhow::{Context, anyhow, bail};
use futures::{StreamExt, TryStreamExt};
use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::client::{HttpError, HttpErrorKind};
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use object_store::prefix::PrefixStore;
use object_store::{ObjectStore, ObjectStoreExt};
use sqlx::PgPool;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

/// Concurrent object transfers issued by one ingest run or one chat request.
pub(crate) const IO_CONCURRENCY: usize = 4;

const LOCKED_MESSAGE: &str = "儲存後端已於初始化時鎖定，變更需重新初始化 DB 與儲存空間";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageConfig {
    Local {
        files_dir: PathBuf,
    },
    S3 {
        bucket: String,
        prefix: Option<String>,
    },
}

impl StorageConfig {
    /// Reads `STORAGE_BACKEND`, `FILES_DIR`, `S3_BUCKET` and `S3_PREFIX`.
    /// Credentials, region and endpoint stay in the standard `AWS_*` variables.
    pub fn from_env() -> anyhow::Result<Self> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(get: impl Fn(&str) -> Option<String>) -> anyhow::Result<Self> {
        let value = |name: &str| {
            get(name)
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        };
        match value("STORAGE_BACKEND").as_deref().unwrap_or("local") {
            "local" => Ok(Self::Local {
                files_dir: value("FILES_DIR")
                    .unwrap_or_else(|| crate::DEFAULT_FILES_DIR.to_owned())
                    .into(),
            }),
            "s3" => {
                let bucket =
                    value("S3_BUCKET").context("S3_BUCKET is required when STORAGE_BACKEND=s3")?;
                let prefix = value("S3_PREFIX")
                    .map(|prefix| prefix.trim_matches('/').to_owned())
                    .filter(|prefix| !prefix.is_empty());
                if let Some(prefix) = &prefix {
                    ObjectPath::parse(prefix).map_err(|_| {
                        anyhow!("S3_PREFIX must be a relative key prefix without empty, `.` or `..` segments")
                    })?;
                }
                Ok(Self::S3 { bucket, prefix })
            }
            _ => bail!("STORAGE_BACKEND must be `local` or `s3`"),
        }
    }

    /// Opens the store before the database is touched. S3 access is verified
    /// once so a wrong bucket, prefix, endpoint or credential stops startup.
    pub async fn open(&self) -> anyhow::Result<Storage> {
        match self {
            Self::Local { files_dir } => {
                tracing::info!(
                    storage_backend = "local",
                    files_dir = %files_dir.display(),
                    "file storage configured"
                );
                Storage::local(files_dir)
            }
            Self::S3 { bucket, prefix } => {
                open_s3(
                    AmazonS3Builder::from_env().with_bucket_name(bucket),
                    bucket,
                    prefix.as_deref(),
                )
                .await
            }
        }
    }

    fn lock_identity(&self) -> (&'static str, Option<&str>, Option<&str>) {
        match self {
            Self::Local { .. } => ("local", None, None),
            Self::S3 { bucket, prefix } => ("s3", Some(bucket), prefix.as_deref()),
        }
    }
}

async fn open_s3(
    builder: AmazonS3Builder,
    bucket: &str,
    prefix: Option<&str>,
) -> anyhow::Result<Storage> {
    let endpoint = builder
        .get_config_value(&AmazonS3ConfigKey::S3Endpoint)
        .or_else(|| builder.get_config_value(&AmazonS3ConfigKey::Endpoint))
        .map(|endpoint| endpoint_for_log(&endpoint))
        .unwrap_or_else(|| "aws-default".to_owned());
    tracing::info!(
        storage_backend = "s3",
        bucket,
        prefix = prefix.unwrap_or(""),
        endpoint = %endpoint,
        "object storage configured"
    );
    // Builder errors only name missing or malformed settings, never their values.
    let s3 = builder
        .build()
        .map_err(|error| anyhow!("S3 storage configuration is invalid: {error}"))?;
    let store: Arc<dyn ObjectStore> = match prefix {
        Some(prefix) => Arc::new(PrefixStore::new(
            s3,
            ObjectPath::parse(prefix).map_err(|_| anyhow!("S3_PREFIX is invalid"))?,
        )),
        None => Arc::new(s3),
    };
    store.list_with_delimiter(None).await.map_err(|error| {
        anyhow!(
            "S3 bucket is not accessible (bucket={bucket}, prefix={}, endpoint={endpoint}): {}",
            prefix.unwrap_or(""),
            StorageError::from(error)
        )
    })?;
    Ok(Storage::new(store))
}

fn endpoint_for_log(endpoint: &str) -> String {
    match url::Url::parse(endpoint) {
        Ok(mut url) => {
            let _ = url.set_username("");
            let _ = url.set_password(None);
            url.set_query(None);
            url.set_fragment(None);
            url.to_string()
        }
        Err(_) => "invalid-url".to_owned(),
    }
}

/// Records the storage backend on first start and refuses any later change.
pub async fn ensure_initialization_lock(
    pool: &PgPool,
    config: &StorageConfig,
) -> anyhow::Result<()> {
    let configured = config.lock_identity();
    let mut transaction = pool.begin().await?;
    // Concurrent first starts serialize here, so exactly one backend is recorded.
    sqlx::query("LOCK TABLE storage_settings IN EXCLUSIVE MODE")
        .execute(&mut *transaction)
        .await?;
    let recorded: Option<(String, Option<String>, Option<String>)> =
        sqlx::query_as("SELECT backend, s3_bucket, s3_prefix FROM storage_settings")
            .fetch_optional(&mut *transaction)
            .await?;
    match recorded {
        Some((backend, bucket, prefix)) => {
            let recorded = (backend.as_str(), bucket.as_deref(), prefix.as_deref());
            if recorded != configured {
                bail!(
                    "{LOCKED_MESSAGE}（已鎖定：{}；目前設定：{}）",
                    describe(recorded),
                    describe(configured)
                );
            }
        }
        None => {
            let has_documents: bool = sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM documents)")
                .fetch_one(&mut *transaction)
                .await?;
            if has_documents {
                bail!(
                    "資料庫已有文件但沒有儲存後端初始化紀錄；此版本不轉換舊資料，需清空 DB 與檔案後重新初始化"
                );
            }
            sqlx::query(
                "INSERT INTO storage_settings (backend, s3_bucket, s3_prefix) VALUES ($1, $2, $3)",
            )
            .bind(configured.0)
            .bind(configured.1)
            .bind(configured.2)
            .execute(&mut *transaction)
            .await?;
        }
    }
    transaction.commit().await?;
    Ok(())
}

fn describe((backend, bucket, prefix): (&str, Option<&str>, Option<&str>)) -> String {
    match bucket {
        Some(bucket) => format!(
            "backend={backend}, bucket={bucket}, prefix={}",
            prefix.unwrap_or("")
        ),
        None => format!("backend={backend}"),
    }
}

/// Cloneable handle to the configured object store. Intentionally not `Debug`:
/// the S3 client holds credentials.
#[derive(Clone)]
pub struct Storage {
    store: Arc<dyn ObjectStore>,
    /// Canonical `FILES_DIR` of the local backend. `LocalFileSystem` follows
    /// symlinks, so local reads re-check containment and deletes avoid it.
    local_root: Option<PathBuf>,
    #[cfg(test)]
    fail_puts_containing: Option<&'static str>,
}

impl Storage {
    fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self {
            store,
            local_root: None,
            #[cfg(test)]
            fail_puts_containing: None,
        }
    }

    pub fn local(root: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(root)
            .with_context(|| format!("FILES_DIR {} could not be created", root.display()))?;
        let root = std::fs::canonicalize(root)
            .with_context(|| format!("FILES_DIR {} could not be resolved", root.display()))?;
        let store = LocalFileSystem::new_with_prefix(&root)
            .with_context(|| format!("FILES_DIR {} could not be opened", root.display()))?;
        Ok(Self {
            local_root: Some(root),
            ..Self::new(Arc::new(store))
        })
    }

    #[cfg(test)]
    pub(crate) fn memory() -> Self {
        Self::new(Arc::new(object_store::memory::InMemory::new()))
    }

    /// Test hook: every `put` whose key contains `needle` fails.
    #[cfg(test)]
    pub(crate) fn failing_puts_containing(mut self, needle: &'static str) -> Self {
        self.fail_puts_containing = Some(needle);
        self
    }

    pub async fn put(&self, key: &str, bytes: Vec<u8>) -> Result<(), StorageError> {
        #[cfg(test)]
        if self
            .fail_puts_containing
            .is_some_and(|needle| key.contains(needle))
        {
            return Err(StorageError::Backend("injected_failure"));
        }
        self.store.put(&object_path(key)?, bytes.into()).await?;
        Ok(())
    }

    /// Reads one document object. The key must stay inside `{document_id}/`,
    /// and an object over `max_bytes` is rejected before its body is read.
    pub async fn get(
        &self,
        document_id: Uuid,
        key: &str,
        max_bytes: usize,
    ) -> Result<Vec<u8>, StorageError> {
        let path = validate_document_key(document_id, key)?;
        if let Some(root) = &self.local_root {
            // A symlink under FILES_DIR (including a symlinked document directory)
            // must not pull a file from outside this document's directory.
            let resolved = tokio::fs::canonicalize(root.join(key))
                .await
                .map_err(|error| match error.kind() {
                    std::io::ErrorKind::NotFound => StorageError::NotFound,
                    _ => StorageError::Backend("local_resolve_failed"),
                })?;
            if !resolved.starts_with(root.join(document_id.to_string())) {
                return Err(StorageError::InvalidKey(
                    "key resolves outside its document directory",
                ));
            }
        }
        let limit = max_bytes as u64;
        let result = self.store.get(&path).await?;
        if result.meta.size > limit {
            return Err(StorageError::TooLarge {
                size: result.meta.size,
                limit,
            });
        }
        let bytes = result.bytes().await?;
        if bytes.len() as u64 > limit {
            return Err(StorageError::TooLarge {
                size: bytes.len() as u64,
                limit,
            });
        }
        Ok(bytes.into())
    }

    /// Lists every object key below `prefix` (whole segments, recursive).
    pub async fn list(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        let prefix = object_path(prefix)?;
        Ok(self
            .store
            .list(Some(&prefix))
            .map_ok(|meta| meta.location.to_string())
            .try_collect()
            .await?)
    }

    /// Deletes every object below `prefix`; objects already gone count as deleted.
    pub async fn delete_prefix(&self, prefix: &str) -> Result<(), StorageError> {
        let prefix = object_path(prefix)?;
        if let Some(root) = &self.local_root {
            // Listing follows symlinks and would delete their targets outside
            // FILES_DIR; remove_dir_all removes a symlink itself, never its target.
            return match tokio::fs::remove_dir_all(root.join(prefix.as_ref())).await {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(_) => Err(StorageError::Backend("local_delete_failed")),
            };
        }
        let locations = self
            .store
            .list(Some(&prefix))
            .map_ok(|meta| Ok(meta.location))
            .try_collect::<Vec<_>>()
            .await?;
        let mut deleted = self
            .store
            .delete_stream(futures::stream::iter(locations).boxed());
        while let Some(result) = deleted.next().await {
            match result {
                Ok(_) | Err(object_store::Error::NotFound { .. }) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }
}

/// Accepts only a relative key strictly inside `{document_id}/`. Replaces the
/// former canonicalize/symlink checks: persisted keys are data, not paths.
pub fn validate_document_key(document_id: Uuid, key: &str) -> Result<ObjectPath, StorageError> {
    if key.contains("..") {
        return Err(StorageError::InvalidKey("key must not contain `..`"));
    }
    let path = object_path(key)?;
    let inside = key
        .strip_prefix(&format!("{document_id}/"))
        .is_some_and(|rest| !rest.is_empty());
    if !inside {
        return Err(StorageError::InvalidKey(
            "key is outside its document prefix",
        ));
    }
    Ok(path)
}

fn object_path(key: &str) -> Result<ObjectPath, StorageError> {
    if key.is_empty() || key.starts_with('/') {
        return Err(StorageError::InvalidKey(
            "key must be a non-empty relative path",
        ));
    }
    ObjectPath::parse(key).map_err(|_| StorageError::InvalidKey("key is not a valid object path"))
}

/// Storage failure whose text never contains keys, paths, response bodies or
/// credentials, so it is safe for logs, persisted errors and API messages.
#[derive(Debug)]
pub enum StorageError {
    InvalidKey(&'static str),
    TooLarge { size: u64, limit: u64 },
    NotFound,
    HttpStatus(u16),
    Backend(&'static str),
}

impl fmt::Display for StorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKey(reason) => write!(formatter, "invalid storage key: {reason}"),
            Self::TooLarge { size, limit } => {
                write!(
                    formatter,
                    "object is {size} bytes; maximum is {limit} bytes"
                )
            }
            Self::NotFound => formatter.write_str("not found"),
            Self::HttpStatus(status) => write!(formatter, "storage backend error: HTTP {status}"),
            Self::Backend(kind) => write!(formatter, "storage backend error: {kind}"),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<object_store::Error> for StorageError {
    // object_store messages embed filesystem paths and object keys, and S3
    // error bodies may echo the request signature; keep only a stable kind.
    fn from(error: object_store::Error) -> Self {
        match &error {
            object_store::Error::NotFound { .. } => return Self::NotFound,
            object_store::Error::InvalidPath { .. } => {
                return Self::InvalidKey("key is not a valid object path");
            }
            object_store::Error::PermissionDenied { .. } => {
                return Self::Backend("permission_denied");
            }
            object_store::Error::Unauthenticated { .. } => return Self::Backend("unauthenticated"),
            _ => {}
        }
        let mut source = std::error::Error::source(&error);
        while let Some(inner) = source {
            if let Some(http) = inner.downcast_ref::<HttpError>() {
                return Self::Backend(match http.kind() {
                    HttpErrorKind::Connect => "connect_failed",
                    HttpErrorKind::Timeout => "timeout",
                    _ => "transport_error",
                });
            }
            source = inner.source();
        }
        // S3 list failures stay `Generic`. Keep only the status digits so a
        // 403 (credentials) and a 404 (bucket) remain distinguishable.
        error
            .to_string()
            .split_once("status code: ")
            .and_then(|(_, rest)| rest.get(..3)?.parse().ok())
            .map_or(Self::Backend("backend_error"), Self::HttpStatus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn config_from(values: &[(&str, &str)]) -> anyhow::Result<StorageConfig> {
        let values = values
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect::<HashMap<_, _>>();
        StorageConfig::from_lookup(|name| values.get(name).cloned())
    }

    #[test]
    fn config_defaults_to_local_and_validates_s3_settings() {
        assert_eq!(
            config_from(&[]).unwrap(),
            StorageConfig::Local {
                files_dir: crate::DEFAULT_FILES_DIR.into()
            }
        );
        assert_eq!(
            config_from(&[("STORAGE_BACKEND", "local"), ("FILES_DIR", "/srv/km")]).unwrap(),
            StorageConfig::Local {
                files_dir: "/srv/km".into()
            }
        );
        assert_eq!(
            config_from(&[
                ("STORAGE_BACKEND", "s3"),
                ("S3_BUCKET", "km-files"),
                ("S3_PREFIX", "/tenant-a/prod/"),
            ])
            .unwrap(),
            StorageConfig::S3 {
                bucket: "km-files".to_owned(),
                prefix: Some("tenant-a/prod".to_owned())
            }
        );

        // Endpoint and credentials stay in AWS_* and never enter the locked identity.
        let with_endpoint = |endpoint| {
            config_from(&[
                ("STORAGE_BACKEND", "s3"),
                ("S3_BUCKET", "km-files"),
                ("AWS_ENDPOINT", endpoint),
            ])
            .unwrap()
        };
        assert_eq!(
            with_endpoint("http://minio-a:9000"),
            with_endpoint("https://account.r2.cloudflarestorage.com")
        );

        let missing_bucket = config_from(&[("STORAGE_BACKEND", "s3")]).unwrap_err();
        assert!(missing_bucket.to_string().contains("S3_BUCKET"));
        let bad_prefix = config_from(&[
            ("STORAGE_BACKEND", "s3"),
            ("S3_BUCKET", "km-files"),
            ("S3_PREFIX", "a/../b"),
        ])
        .unwrap_err();
        assert!(bad_prefix.to_string().contains("S3_PREFIX"));
        for backend in ["gcs", "S3", "file"] {
            let error = config_from(&[("STORAGE_BACKEND", backend)]).unwrap_err();
            assert!(error.to_string().contains("STORAGE_BACKEND"));
        }
    }

    #[tokio::test]
    async fn reads_reject_keys_outside_the_document_without_echoing_them() {
        let storage = Storage::memory();
        let document_id = Uuid::new_v4();
        let other_id = Uuid::new_v4();
        let other_key = format!("{other_id}/secret-original.pdf");
        storage
            .put(&other_key, b"other document".to_vec())
            .await
            .unwrap();

        for key in [
            String::new(),
            "/etc/passwd".to_owned(),
            format!("/{document_id}/pages/0001.png"),
            format!("{document_id}/../{other_id}/secret-original.pdf"),
            format!("{document_id}/pages/..hidden"),
            other_key.clone(),
            document_id.to_string(),
            format!("{document_id}/"),
            format!("{document_id}//pages/0001.png"),
        ] {
            let error = storage
                .get(document_id, &key, usize::MAX)
                .await
                .expect_err("key must be rejected");
            assert!(matches!(error, StorageError::InvalidKey(_)), "{key:?}");
            let message = error.to_string();
            assert!(!message.contains(&other_id.to_string()), "{message}");
            assert!(!message.contains(&document_id.to_string()), "{message}");
            assert!(!message.contains("secret") && !message.contains("passwd"));
        }

        let missing = storage
            .get(
                document_id,
                &format!("{document_id}/pages/0001.png"),
                usize::MAX,
            )
            .await
            .expect_err("missing object");
        assert!(matches!(missing, StorageError::NotFound));
    }

    #[tokio::test]
    async fn reads_enforce_the_size_limit_and_prefix_delete_is_idempotent() {
        let storage = Storage::memory();
        let document_id = Uuid::new_v4();
        let key = format!("{document_id}/artifacts-a/pages/0001.png");
        storage.put(&key, vec![7; 10]).await.unwrap();
        storage
            .put(&format!("{document_id}/original.png"), vec![1])
            .await
            .unwrap();
        let other_document = Uuid::new_v4().to_string();
        let unrelated = format!("{other_document}/original.png");
        storage.put(&unrelated, vec![2]).await.unwrap();

        assert!(matches!(
            storage.get(document_id, &key, 9).await,
            Err(StorageError::TooLarge { size: 10, limit: 9 })
        ));
        assert_eq!(
            storage.get(document_id, &key, 10).await.unwrap(),
            vec![7; 10]
        );

        let prefix = document_id.to_string();
        assert_eq!(storage.list(&prefix).await.unwrap().len(), 2);
        storage.delete_prefix(&prefix).await.unwrap();
        storage.delete_prefix(&prefix).await.unwrap();
        assert!(storage.list(&prefix).await.unwrap().is_empty());
        assert_eq!(
            storage.list(&other_document).await.unwrap(),
            vec![unrelated]
        );
    }

    #[tokio::test]
    async fn local_prefix_delete_leaves_no_empty_document_directory() {
        let root = std::env::temp_dir().join(format!("km-storage-local-{}", Uuid::new_v4()));
        let storage = Storage::local(&root).unwrap();
        let document_id = Uuid::new_v4();
        for key in [
            format!("{document_id}/original.pdf"),
            format!("{document_id}/artifacts-a/manifest.json"),
            format!("{document_id}/artifacts-a/pages/0001.png"),
        ] {
            storage.put(&key, b"fixture".to_vec()).await.unwrap();
        }
        assert!(root.join(document_id.to_string()).is_dir());

        storage
            .delete_prefix(&document_id.to_string())
            .await
            .unwrap();
        assert!(!root.join(document_id.to_string()).exists());
        assert!(root.is_dir(), "the FILES_DIR root itself must remain");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn local_symlinks_never_escape_the_document_directory() {
        let base = std::env::temp_dir().join(format!("km-storage-symlink-{}", Uuid::new_v4()));
        let root = base.join("files");
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.png"), b"outside").unwrap();
        let storage = Storage::local(&root).unwrap();

        let document_id = Uuid::new_v4();
        storage
            .put(&format!("{document_id}/original.pdf"), b"fixture".to_vec())
            .await
            .unwrap();
        std::os::unix::fs::symlink(&outside, root.join(format!("{document_id}/pages"))).unwrap();
        let escaped = storage
            .get(
                document_id,
                &format!("{document_id}/pages/secret.png"),
                usize::MAX,
            )
            .await
            .expect_err("a symlink out of FILES_DIR must be rejected");
        assert!(matches!(escaped, StorageError::InvalidKey(_)));

        // A symlinked document directory must not read another document's objects.
        let aliased_id = Uuid::new_v4();
        std::os::unix::fs::symlink(
            root.join(document_id.to_string()),
            root.join(aliased_id.to_string()),
        )
        .unwrap();
        let aliased = storage
            .get(
                aliased_id,
                &format!("{aliased_id}/original.pdf"),
                usize::MAX,
            )
            .await
            .expect_err("a symlinked document directory must be rejected");
        assert!(matches!(aliased, StorageError::InvalidKey(_)));

        storage
            .delete_prefix(&document_id.to_string())
            .await
            .unwrap();
        assert!(!root.join(document_id.to_string()).exists());
        assert_eq!(
            std::fs::read(outside.join("secret.png")).unwrap(),
            b"outside"
        );
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn backend_errors_drop_paths_bodies_and_signatures() {
        let body = "<Error><Code>SignatureDoesNotMatch</Code>\
                    <SignatureProvided>0123abcd-secret</SignatureProvided></Error>";
        let list_failure = format!(
            "Error performing GET http://minio.internal/private-bucket?list-type=2 in 3ms \
             - Server returned non-2xx status code: 403 Forbidden: {body}"
        );
        for (error, expected) in [
            (
                object_store::Error::PermissionDenied {
                    path: "/srv/private/files/document.pdf".to_owned(),
                    source: body.into(),
                },
                "storage backend error: permission_denied",
            ),
            (
                object_store::Error::Generic {
                    store: "S3",
                    source: body.into(),
                },
                "storage backend error: backend_error",
            ),
            (
                object_store::Error::Generic {
                    store: "S3",
                    source: list_failure.into(),
                },
                "storage backend error: HTTP 403",
            ),
            (
                object_store::Error::NotFound {
                    path: "/srv/private/files/document.pdf".to_owned(),
                    source: body.into(),
                },
                "not found",
            ),
        ] {
            assert_eq!(StorageError::from(error).to_string(), expected);
        }
    }

    /// Opt-in: needs the MinIO/S3 environment from the README smoke command.
    #[tokio::test]
    #[ignore = "requires MinIO/S3 via STORAGE_BACKEND=s3, S3_BUCKET and AWS_*"]
    async fn minio_smoke_startup_rejects_bad_access_without_leaking_secrets() {
        let StorageConfig::S3 { bucket, prefix } = StorageConfig::from_env().unwrap() else {
            panic!("set STORAGE_BACKEND=s3 for the MinIO smoke test");
        };
        let real_secret = std::env::var("AWS_SECRET_ACCESS_KEY").unwrap_or_default();
        let wrong_secret = "km-smoke-wrong-secret-value";
        let missing_bucket = format!("km-missing-{}", Uuid::new_v4());
        let cases = [
            (
                AmazonS3Builder::from_env()
                    .with_bucket_name(&bucket)
                    .with_secret_access_key(wrong_secret),
                bucket.as_str(),
                "HTTP 403",
            ),
            (
                AmazonS3Builder::from_env().with_bucket_name(&missing_bucket),
                missing_bucket.as_str(),
                "HTTP 404",
            ),
        ];
        for (builder, bucket, status) in cases {
            let error = open_s3(builder, bucket, prefix.as_deref())
                .await
                .err()
                .expect("inaccessible bucket must stop startup")
                .to_string();
            println!("startup error: {error}");
            assert!(error.contains("S3 bucket is not accessible"), "{error}");
            assert!(error.ends_with(status), "{error}");
            assert!(!error.contains(wrong_secret), "{error}");
            assert!(
                real_secret.is_empty() || !error.contains(&real_secret),
                "{error}"
            );
            assert!(!error.contains("Signature"), "{error}");
        }
    }

    #[test]
    fn logged_endpoints_never_carry_credentials() {
        let endpoint = endpoint_for_log("https://access:secret@minio.internal:9000");
        assert_eq!(endpoint, "https://minio.internal:9000/");
        let endpoint =
            endpoint_for_log("https://minio.internal:9000/?X-Amz-Signature=secret#token=secret");
        assert_eq!(endpoint, "https://minio.internal:9000/");
    }
}
