use image::{DynamicImage, ImageDecoder, ImageFormat, Limits, metadata::Orientation};
use pdfium_render::prelude::{PdfRenderConfig, Pdfium};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::process::Command;
use uuid::Uuid;

const DEFAULT_RENDER_DPI: u16 = 180;
const DEFAULT_CONVERSION_TIMEOUT_SECS: u64 = 120;
const MAX_ERROR_CHARS: usize = 1_000;
const MAX_IMAGE_DIMENSION: u32 = 20_000;
const MAX_IMAGE_ALLOCATION_BYTES: u64 = 256 * 1024 * 1024;
const MAX_PDF_RENDER_DIMENSION: i32 = 8_192;

static PDFIUM: OnceLock<Result<Pdfium, String>> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct ArtifactConfig {
    pub render_dpi: u16,
    pub conversion_timeout: Duration,
    pub pdfium_library_path: Option<PathBuf>,
    pub libreoffice_binary: PathBuf,
}

impl ArtifactConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let render_dpi = parse_env("PDF_RENDER_DPI", DEFAULT_RENDER_DPI)?;
        anyhow::ensure!(
            (72..=600).contains(&render_dpi),
            "PDF_RENDER_DPI must be between 72 and 600"
        );

        let timeout_secs = parse_env(
            "DOCUMENT_CONVERSION_TIMEOUT_SECS",
            DEFAULT_CONVERSION_TIMEOUT_SECS,
        )?;
        anyhow::ensure!(
            timeout_secs > 0,
            "DOCUMENT_CONVERSION_TIMEOUT_SECS must be a positive integer"
        );

        let pdfium_library_path = std::env::var_os("PDFIUM_LIBRARY_PATH")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        let libreoffice_binary = std::env::var_os("LIBREOFFICE_BIN")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("libreoffice"));

        Ok(Self {
            render_dpi,
            conversion_timeout: Duration::from_secs(timeout_secs),
            pdfium_library_path,
            libreoffice_binary,
        })
    }

    #[cfg(test)]
    pub fn for_test() -> Self {
        Self {
            render_dpi: DEFAULT_RENDER_DPI,
            conversion_timeout: Duration::from_secs(5),
            pdfium_library_path: None,
            libreoffice_binary: PathBuf::from("libreoffice"),
        }
    }
}

fn parse_env<T>(name: &str, default: T) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .map_err(|error| anyhow::anyhow!("{name} is invalid: {error}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(anyhow::anyhow!("{name} is invalid: {error}")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    Pdf,
    Image,
    Office,
    Text,
}

impl SourceKind {
    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension {
            "pdf" => Some(Self::Pdf),
            "png" | "jpg" | "jpeg" | "webp" => Some(Self::Image),
            "docx" | "pptx" | "xlsx" => Some(Self::Office),
            "md" | "markdown" | "txt" => Some(Self::Text),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Image => "image",
            Self::Office => "office",
            Self::Text => "text",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageArtifact {
    pub document_id: Uuid,
    pub page_number: i32,
    pub image_path: String,
    pub width: i32,
    pub height: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extracted_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextArtifact {
    pub document_id: Uuid,
    pub unit_index: i32,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactManifest {
    pub document_id: Uuid,
    pub source_kind: SourceKind,
    pub original_path: String,
    pub converted_pdf_path: Option<String>,
    pub pages: Vec<PageArtifact>,
    pub text_units: Vec<TextArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPage {
    pub page_number: i32,
    pub relative_image_path: PathBuf,
    pub width: i32,
    pub height: i32,
    pub extracted_text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactError {
    stage: &'static str,
    message: String,
}

impl ArtifactError {
    pub fn new(stage: &'static str, message: impl Into<String>) -> Self {
        Self {
            stage,
            message: message.into(),
        }
    }

    #[cfg(test)]
    pub fn stage(&self) -> &'static str {
        self.stage
    }
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.stage, self.message)
    }
}

impl std::error::Error for ArtifactError {}

pub async fn render_pdf(
    config: &ArtifactConfig,
    pdf_path: &Path,
    run_dir: &Path,
) -> Result<Vec<RenderedPage>, ArtifactError> {
    let pdf_path = pdf_path.to_owned();
    let run_dir = run_dir.to_owned();
    let cleanup_dir = run_dir.clone();
    let dpi = config.render_dpi;
    let library_path = config.pdfium_library_path.clone();
    let cancelled = std::sync::Arc::new(AtomicBool::new(false));
    let blocking_cancelled = cancelled.clone();

    let mut task = tokio::task::spawn_blocking(move || {
        let result = render_pdf_blocking(
            &pdf_path,
            &run_dir,
            dpi,
            library_path.as_deref(),
            &blocking_cancelled,
        );
        if blocking_cancelled.load(Ordering::Relaxed) {
            let _ = std::fs::remove_dir_all(&run_dir);
        }
        result
    });

    match tokio::time::timeout(config.conversion_timeout, &mut task).await {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => Err(ArtifactError::new(
            "pdf-render",
            format!("PDF 轉圖工作異常結束: {error}"),
        )),
        Err(_) => {
            cancelled.store(true, Ordering::Relaxed);
            let timeout = config.conversion_timeout.as_secs();
            tokio::spawn(async move {
                let _ = task.await;
                let _ = tokio::fs::remove_dir_all(cleanup_dir).await;
            });
            Err(ArtifactError::new(
                "pdf-render",
                format!("PDF 轉圖超過 {timeout} 秒，已取消本次處理"),
            ))
        }
    }
}

fn render_pdf_blocking(
    pdf_path: &Path,
    run_dir: &Path,
    dpi: u16,
    library_path: Option<&Path>,
    cancelled: &AtomicBool,
) -> Result<Vec<RenderedPage>, ArtifactError> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(ArtifactError::new("pdf-render", "本次處理已取消"));
    }

    let pages_dir = run_dir.join("pages");
    std::fs::create_dir_all(&pages_dir)
        .map_err(|error| ArtifactError::new("pdf-render", format!("無法建立頁圖目錄: {error}")))?;

    let pdfium = load_pdfium(library_path)?;
    let document = pdfium
        .load_pdf_from_file(pdf_path, None)
        .map_err(|error| ArtifactError::new("pdf-open", format!("無法開啟 PDF: {error}")))?;
    if document.pages().is_empty() {
        return Err(ArtifactError::new("pdf-open", "PDF 不含任何頁面"));
    }

    let render_config = PdfRenderConfig::new()
        .scale_page_by_factor(f32::from(dpi) / 72.0)
        // Pdfium allocates the native bitmap before Rust can inspect it. Keep
        // extreme MediaBox values from creating an unbounded allocation while
        // preserving the configured DPI for ordinary document pages.
        .set_maximum_width(MAX_PDF_RENDER_DIMENSION)
        .set_maximum_height(MAX_PDF_RENDER_DIMENSION)
        .render_form_data(true)
        .render_annotations(true);
    let mut rendered_pages = Vec::with_capacity(document.pages().len() as usize);

    for (index, page) in document.pages().iter().enumerate() {
        if cancelled.load(Ordering::Relaxed) {
            return Err(ArtifactError::new("pdf-render", "本次處理已取消"));
        }

        let page_number = i32::try_from(index + 1)
            .map_err(|_| ArtifactError::new("pdf-render", "PDF 頁數超過系統可追蹤範圍"))?;
        let relative_image_path = PathBuf::from(format!("pages/{page_number:04}.png"));
        let image_path = run_dir.join(&relative_image_path);
        let image = page
            .render_with_config(&render_config)
            .and_then(|bitmap| bitmap.as_image())
            .map_err(|error| {
                ArtifactError::new(
                    "pdf-render",
                    format!("第 {page_number} 頁轉圖失敗: {error}"),
                )
            })?;
        let width = i32::try_from(image.width()).map_err(|_| {
            ArtifactError::new("pdf-render", format!("第 {page_number} 頁寬度過大"))
        })?;
        let height = i32::try_from(image.height()).map_err(|_| {
            ArtifactError::new("pdf-render", format!("第 {page_number} 頁高度過大"))
        })?;
        image
            .save_with_format(&image_path, ImageFormat::Png)
            .map_err(|error| {
                ArtifactError::new(
                    "pdf-render",
                    format!("第 {page_number} 頁圖寫入失敗: {error}"),
                )
            })?;

        // Text extraction is explicitly best-effort. Scanned pages and PDFs with a
        // broken text layer still retain a valid page image.
        let extracted_text = page
            .text()
            .ok()
            .map(|text| text.all())
            .filter(|text| !text.trim().is_empty());

        rendered_pages.push(RenderedPage {
            page_number,
            relative_image_path,
            width,
            height,
            extracted_text,
        });
    }

    Ok(rendered_pages)
}

fn load_pdfium(library_path: Option<&Path>) -> Result<&'static Pdfium, ArtifactError> {
    let configured_path = library_path.map(Path::to_owned);
    let result = PDFIUM.get_or_init(move || {
        let bindings = if let Some(path) = configured_path {
            let path = if path.is_dir() {
                Pdfium::pdfium_platform_library_name_at_path(&path)
            } else {
                path
            };
            Pdfium::bind_to_library(&path)
                .map_err(|error| format!("無法載入 {}: {error}", path.display()))
        } else {
            let adjacent = std::env::current_exe()
                .ok()
                .and_then(|path| path.parent().map(Path::to_owned))
                .map(|path| Pdfium::pdfium_platform_library_name_at_path(&path));
            match adjacent {
                Some(path) => Pdfium::bind_to_library(&path)
                    .or_else(|_| Pdfium::bind_to_system_library())
                    .map_err(|error| error.to_string()),
                None => Pdfium::bind_to_system_library().map_err(|error| error.to_string()),
            }
        }?;
        Ok(Pdfium::new(bindings))
    });

    result.as_ref().map_err(|error| {
        ArtifactError::new(
            "pdfium-load",
            format!(
                "找不到可用的 Pdfium 動態函式庫；請安裝 libpdfium 或設定 PDFIUM_LIBRARY_PATH（{error}）"
            ),
        )
    })
}

pub async fn normalize_image(
    config: &ArtifactConfig,
    source_path: &Path,
    run_dir: &Path,
) -> Result<Vec<RenderedPage>, ArtifactError> {
    let source_path = source_path.to_owned();
    let run_dir = run_dir.to_owned();
    let cleanup_dir = run_dir.clone();
    let cancelled = std::sync::Arc::new(AtomicBool::new(false));
    let blocking_cancelled = cancelled.clone();

    let mut task = tokio::task::spawn_blocking(move || {
        let result = normalize_image_blocking(&source_path, &run_dir, &blocking_cancelled);
        if blocking_cancelled.load(Ordering::Relaxed) {
            let _ = std::fs::remove_dir_all(&run_dir);
        }
        result
    });

    match tokio::time::timeout(config.conversion_timeout, &mut task).await {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => Err(ArtifactError::new(
            "image-normalize",
            format!("圖片處理工作異常結束: {error}"),
        )),
        Err(_) => {
            cancelled.store(true, Ordering::Relaxed);
            let timeout = config.conversion_timeout.as_secs();
            tokio::spawn(async move {
                let _ = task.await;
                let _ = tokio::fs::remove_dir_all(cleanup_dir).await;
            });
            Err(ArtifactError::new(
                "image-normalize",
                format!("圖片處理超過 {timeout} 秒，已取消本次處理"),
            ))
        }
    }
}

fn normalize_image_blocking(
    source_path: &Path,
    run_dir: &Path,
    cancelled: &AtomicBool,
) -> Result<Vec<RenderedPage>, ArtifactError> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(ArtifactError::new("image-normalize", "本次處理已取消"));
    }

    let mut reader = image::ImageReader::open(source_path)
        .and_then(|reader| reader.with_guessed_format())
        .map_err(|error| ArtifactError::new("image-decode", format!("無法讀取圖片: {error}")))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
    limits.max_alloc = Some(MAX_IMAGE_ALLOCATION_BYTES);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().map_err(|error| {
        ArtifactError::new("image-decode", format!("圖片格式或內容無效: {error}"))
    })?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut image = DynamicImage::from_decoder(decoder).map_err(|error| {
        ArtifactError::new("image-decode", format!("圖片格式或內容無效: {error}"))
    })?;
    image.apply_orientation(orientation);
    let width = i32::try_from(image.width())
        .map_err(|_| ArtifactError::new("image-decode", "圖片寬度過大"))?;
    let height = i32::try_from(image.height())
        .map_err(|_| ArtifactError::new("image-decode", "圖片高度過大"))?;
    if width == 0 || height == 0 {
        return Err(ArtifactError::new("image-decode", "圖片尺寸不可為零"));
    }

    let relative_image_path = PathBuf::from("pages/0001.png");
    let output_path = run_dir.join(&relative_image_path);
    std::fs::create_dir_all(
        output_path
            .parent()
            .expect("page image always has a parent"),
    )
    .map_err(|error| ArtifactError::new("image-normalize", format!("無法建立頁圖目錄: {error}")))?;
    image
        .save_with_format(&output_path, ImageFormat::Png)
        .map_err(|error| {
            ArtifactError::new("image-normalize", format!("無法寫入 PNG 頁圖: {error}"))
        })?;

    Ok(vec![RenderedPage {
        page_number: 1,
        relative_image_path,
        width,
        height,
        extracted_text: None,
    }])
}

pub async fn convert_office_to_pdf(
    config: &ArtifactConfig,
    source_path: &Path,
    run_dir: &Path,
) -> Result<PathBuf, ArtifactError> {
    let profile_dir = run_dir.join("libreoffice-profile");
    tokio::fs::create_dir_all(&profile_dir)
        .await
        .map_err(|error| {
            ArtifactError::new(
                "office-conversion",
                format!("無法建立 LibreOffice 暫存設定目錄: {error}"),
            )
        })?;
    let absolute_profile_dir = tokio::fs::canonicalize(&profile_dir)
        .await
        .map_err(|error| {
            ArtifactError::new(
                "office-conversion",
                format!("無法解析 LibreOffice 暫存設定目錄: {error}"),
            )
        })?;
    let profile_url = url::Url::from_directory_path(&absolute_profile_dir).map_err(|_| {
        ArtifactError::new("office-conversion", "無法建立 LibreOffice 隔離設定目錄 URL")
    })?;

    let mut command = Command::new(&config.libreoffice_binary);
    command
        .kill_on_drop(true)
        .arg("--headless")
        .arg("--nologo")
        .arg("--nodefault")
        .arg("--nolockcheck")
        .arg("--nofirststartwizard")
        .arg(format!("-env:UserInstallation={profile_url}"))
        .arg("--convert-to")
        .arg("pdf")
        .arg("--outdir")
        .arg(run_dir)
        .arg(source_path);

    let output = match tokio::time::timeout(config.conversion_timeout, command.output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(ArtifactError::new(
                "office-conversion",
                format!(
                    "找不到 LibreOffice 執行檔「{}」；請安裝 LibreOffice 或設定 LIBREOFFICE_BIN",
                    config.libreoffice_binary.display()
                ),
            ));
        }
        Ok(Err(error)) => {
            return Err(ArtifactError::new(
                "office-conversion",
                format!("無法啟動 LibreOffice: {error}"),
            ));
        }
        Err(_) => {
            return Err(ArtifactError::new(
                "office-conversion",
                format!(
                    "LibreOffice 轉檔超過 {} 秒，已終止處理",
                    config.conversion_timeout.as_secs()
                ),
            ));
        }
    };

    let stderr = summarize_output(&output.stderr);
    if !output.status.success() {
        return Err(ArtifactError::new(
            "office-conversion",
            format!(
                "LibreOffice 轉檔失敗（{}）；stderr: {stderr}",
                output.status
            ),
        ));
    }

    let produced_name = source_path
        .file_stem()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("original"))
        .with_extension("pdf");
    let produced_path = run_dir.join(produced_name);
    let converted_path = run_dir.join("converted.pdf");
    if produced_path != converted_path {
        tokio::fs::rename(&produced_path, &converted_path)
            .await
            .map_err(|error| {
                ArtifactError::new(
                    "office-conversion",
                    format!(
                        "LibreOffice 未產生預期 PDF（{}）；stderr: {stderr}; {error}",
                        produced_path.display()
                    ),
                )
            })?;
    }
    let metadata = tokio::fs::metadata(&converted_path)
        .await
        .map_err(|error| {
            ArtifactError::new(
                "office-conversion",
                format!("找不到轉出的 PDF；stderr: {stderr}; {error}"),
            )
        })?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(ArtifactError::new(
            "office-conversion",
            format!("LibreOffice 產生的 PDF 為空；stderr: {stderr}"),
        ));
    }

    // The isolated user profile is an implementation detail, not a durable artifact.
    let _ = tokio::fs::remove_dir_all(profile_dir).await;
    Ok(converted_path)
}

fn summarize_output(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut summary = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if summary.is_empty() {
        return "（無輸出）".to_string();
    }
    if summary.chars().count() > MAX_ERROR_CHARS {
        summary = summary.chars().take(MAX_ERROR_CHARS).collect();
        summary.push('…');
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("km-page-artifacts-{label}-{}", Uuid::new_v4()))
    }

    #[test]
    fn supported_extensions_are_explicit() {
        assert_eq!(SourceKind::from_extension("pdf"), Some(SourceKind::Pdf));
        assert_eq!(SourceKind::from_extension("webp"), Some(SourceKind::Image));
        assert_eq!(SourceKind::from_extension("pptx"), Some(SourceKind::Office));
        assert_eq!(SourceKind::from_extension("md"), Some(SourceKind::Text));
        assert_eq!(SourceKind::from_extension("csv"), None);
        assert_eq!(SourceKind::from_extension("html"), None);
        assert_eq!(SourceKind::from_extension("ppt"), None);
    }

    #[tokio::test]
    async fn image_is_normalized_as_exactly_one_png_page() -> anyhow::Result<()> {
        let root = test_dir("image");
        let source = root.join("original.webp");
        let run_dir = root.join("run");
        tokio::fs::create_dir_all(&root).await?;
        image::RgbaImage::from_pixel(13, 7, image::Rgba([10, 20, 30, 255]))
            .save_with_format(&source, ImageFormat::WebP)?;

        let pages = normalize_image(&ArtifactConfig::for_test(), &source, &run_dir).await?;
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].page_number, 1);
        assert_eq!((pages[0].width, pages[0].height), (13, 7));
        assert!(pages[0].extracted_text.is_none());
        assert_eq!(pages[0].relative_image_path, Path::new("pages/0001.png"));
        assert!(run_dir.join("pages/0001.png").is_file());

        tokio::fs::remove_dir_all(root).await?;
        Ok(())
    }

    #[tokio::test]
    async fn missing_libreoffice_has_a_staged_user_facing_error() -> anyhow::Result<()> {
        let root = test_dir("missing-office");
        let source = root.join("original.docx");
        let run_dir = root.join("run");
        tokio::fs::create_dir_all(&run_dir).await?;
        tokio::fs::write(&source, b"fixture").await?;
        let mut config = ArtifactConfig::for_test();
        config.libreoffice_binary = root.join("definitely-not-libreoffice");

        let error = convert_office_to_pdf(&config, &source, &run_dir)
            .await
            .expect_err("missing binary must fail");
        assert_eq!(error.stage(), "office-conversion");
        assert!(error.to_string().contains("找不到 LibreOffice"));

        tokio::fs::remove_dir_all(root).await?;
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires Pdfium and LibreOffice"]
    async fn generated_pdf_and_office_fixtures_preserve_page_order() -> anyhow::Result<()> {
        let root = test_dir("document-fixtures");
        let fixtures = crate::test_fixtures::write_all(&root)?;
        // Use the documented environment variables so callers can select an
        // explicit Pdfium library and LibreOffice executable for this smoke.
        let config = ArtifactConfig::from_env()?;

        let native_run = root.join("native-run");
        tokio::fs::create_dir(&native_run).await?;
        let native_pages = render_pdf(&config, &fixtures.native_pdf, &native_run).await?;
        assert_eq!(native_pages.len(), 2);
        assert_eq!(
            native_pages
                .iter()
                .map(|page| page.page_number)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(
            native_pages[0]
                .extracted_text
                .as_deref()
                .is_some_and(|text| text.contains("Native page one"))
        );
        assert!(
            native_pages[1]
                .extracted_text
                .as_deref()
                .is_some_and(|text| text.contains("Native page two"))
        );

        let scanned_run = root.join("scanned-run");
        tokio::fs::create_dir(&scanned_run).await?;
        let scanned_pages = render_pdf(&config, &fixtures.scanned_pdf, &scanned_run).await?;
        assert_eq!(scanned_pages.len(), 1);
        assert_eq!(scanned_pages[0].page_number, 1);
        assert!(scanned_pages[0].extracted_text.is_none());
        assert!(scanned_run.join("pages/0001.png").is_file());

        let office_cases = [
            (
                "docx",
                fixtures.docx.as_path(),
                vec![vec!["DOCX page one"], vec!["DOCX page two"]],
            ),
            (
                "pptx",
                fixtures.pptx.as_path(),
                vec![vec!["PPTX slide one"], vec!["PPTX slide two"]],
            ),
            (
                "xlsx",
                fixtures.xlsx.as_path(),
                vec![vec!["XLSX first row", "XLSX second row"]],
            ),
        ];
        for (label, source, expected_page_markers) in office_cases {
            let run_dir = root.join(format!("{label}-run"));
            tokio::fs::create_dir(&run_dir).await?;
            let converted = convert_office_to_pdf(&config, source, &run_dir).await?;
            let pages = render_pdf(&config, &converted, &run_dir).await?;
            assert_eq!(
                pages.len(),
                expected_page_markers.len(),
                "{label} page count"
            );
            assert_eq!(
                pages
                    .iter()
                    .map(|page| page.page_number)
                    .collect::<Vec<_>>(),
                (1..=expected_page_markers.len() as i32).collect::<Vec<_>>(),
                "{label} page order"
            );
            for (page, markers) in pages.iter().zip(expected_page_markers) {
                let text = page
                    .extracted_text
                    .as_deref()
                    .unwrap_or_else(|| panic!("{label} page {} has no text", page.page_number));
                // Office PDF exports may collapse visual word spacing in the
                // text layer (notably Impress). Compare content markers after
                // whitespace normalization while keeping page boundaries and
                // ordering strict.
                let normalized_text = text
                    .chars()
                    .filter(|character| !character.is_whitespace())
                    .collect::<String>();
                for marker in markers {
                    let normalized_marker = marker
                        .chars()
                        .filter(|character| !character.is_whitespace())
                        .collect::<String>();
                    assert!(
                        normalized_text.contains(&normalized_marker),
                        "{label} page {} is missing marker {marker:?}: {text:?}",
                        page.page_number
                    );
                }
            }
        }

        tokio::fs::remove_dir_all(root).await?;
        Ok(())
    }

    #[test]
    fn stderr_summary_is_bounded_and_single_line() {
        let input = format!("first\n{}\tlast", "x".repeat(MAX_ERROR_CHARS + 100));
        let summary = summarize_output(input.as_bytes());
        assert!(!summary.contains('\n'));
        assert!(summary.chars().count() <= MAX_ERROR_CHARS + 1);
        assert!(summary.ends_with('…'));
    }
}
