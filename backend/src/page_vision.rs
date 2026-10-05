//! Decides once per rendered page whether its image carries information that the
//! text layer does not, so text-only pages skip image inference at every stage.
//!
//! The decision is deterministic and model-free. It reads Pdfium page objects:
//! a page needs vision when its text layer is missing or unreliable, or when
//! raster images, real vector graphics, or visible annotations cover enough of
//! it. Anything the walker cannot interpret counts as vision, because a wrong
//! "vision" only costs one inference while a wrong "text" loses information.
//!
//! Graphics are judged per document as well: an image or a shape that is
//! drawn at the same place on most pages and looks the same there is page
//! furniture (slide master art, letterhead, footer bars) and does not count
//! toward any page. Objects are grouped by their content and bounds, then
//! confirmed from a 72 dpi rendering of each page, with every text box of the
//! document masked out so that text written over the furniture does not
//! matter. Comparing pixels covers everything that changes what is painted
//! (`/Decode`, palettes, masks, clipping, patterns, transforms, transparency).

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

use pdfium_render::prelude::{
    PdfMatrix, PdfPage, PdfPageAnnotationCommon, PdfPageAnnotationType, PdfPageImageObject,
    PdfPageObject, PdfPageObjectCommon, PdfPageObjectsCommon, PdfPagePathObject, PdfPathFillMode,
    PdfPathSegmentType, PdfPathSegments, PdfPoints, PdfRect, PdfRenderConfig,
};

/// Fewer valid non-whitespace characters than this means the text layer cannot
/// stand in for the page image (scans, outlined text, picture-only pages).
pub const MIN_TEXT_CHARS: usize = 20;
/// Fonts without a ToUnicode map extract as U+FFFD, controls, or private-use
/// characters; past this share the text layer is treated as garbage.
pub const MAX_UNRELIABLE_TEXT_RATIO: f64 = 0.05;
/// Symbol fonts without a ToUnicode map (Wingdings, Symbol) extract at
/// U+F000-U+F0FF. One such character standing alone between whitespace is a
/// bullet or an icon rather than garbled text, so it counts as neither.
pub const SYMBOL_FONT_RANGE: std::ops::RangeInclusive<char> = '\u{F000}'..='\u{F0FF}';
/// Images smaller than this share of the page (logos, bullets, icons) are ignored.
pub const MIN_RASTER_OBJECT_RATIO: f64 = 0.005;
/// A graphic drawn with the same content and bounds, and looking the same, on
/// at least this many pages and at least `MIN_FURNITURE_PAGE_SHARE` of the
/// document's pages is page furniture.
pub const MIN_FURNITURE_PAGES: usize = 3;
pub const MIN_FURNITURE_PAGE_SHARE: f64 = 0.5;
/// Images covering this share of the page or more are never furniture: a
/// repeated full-page background can be the content itself, such as a scanned
/// form under its filled-in values.
pub const MAX_FURNITURE_IMAGE_RATIO: f64 = 0.5;
/// Furniture candidates tracked per page; further graphics always count, which
/// bounds memory on pathological pages.
pub const MAX_FURNITURE_CANDIDATES: usize = 256;
/// Pixels one document may render, mask, and compare at 72 dpi to confirm
/// furniture; pages past the budget keep all their graphics, which bounds
/// conversion time.
pub const MAX_FURNITURE_RENDER_PIXELS: u64 = 300_000_000;
/// Text boxes one document may keep for the mask. Past this every box is
/// dropped and nothing in the document is furniture, which bounds memory.
pub const MAX_FURNITURE_TEXT_BOXES: usize = 200_000;
/// A graphic is compared only when at least this share of its pixels lies
/// outside the text mask; a graphic hidden under the mask would always match,
/// so it keeps counting. Slide masters often sit under body text: the
/// background shapes of the OSHA "Machine Guarding" deck are up to 90% masked.
pub const MIN_FURNITURE_UNMASKED_SHARE: f64 = 0.05;
/// Side of the square tiles whose hashes make up an object's appearance.
const FURNITURE_TILE_PIXELS: usize = 8;
/// Text boxes are widened by this many pixels to cover anti-aliasing.
const TEXT_MASK_MARGIN_PIXELS: i64 = 2;
/// Longest side of a 72 dpi furniture rendering.
const FURNITURE_RENDER_MAX_PIXELS: i32 = 4_096;
/// Combined share of the page covered by counted images that requires vision.
pub const MIN_RASTER_COVERAGE: f64 = 0.03;
/// Combined share of the page covered by counted vector graphics that requires vision.
pub const MIN_VECTOR_COVERAGE: f64 = 0.05;
/// Number of counted vector graphics that requires vision regardless of their area.
pub const MIN_VECTOR_OBJECTS: usize = 20;
/// Paths whose shorter side is below this are borders, rules, and underlines.
pub const MIN_VECTOR_SIDE_POINTS: f64 = 3.0;
/// Paths and shadings covering this share of the page or more are backgrounds.
pub const BACKGROUND_COVERAGE: f64 = 0.9;
/// A fill whose every channel is at least this bright is treated as invisible.
pub const NEAR_WHITE_CHANNEL: u8 = 245;
/// A line segment whose minor delta exceeds this share of its major delta is
/// diagonal rather than horizontal or vertical.
pub const AXIS_ALIGNED_TOLERANCE: f64 = 0.02;
/// Stroked paths with more segments than this are treated as drawings without
/// inspecting the rest.
pub const MAX_PATH_SEGMENTS: usize = 4_096;
/// Pages with more objects than this are not walked further and need vision.
pub const MAX_PAGE_OBJECTS: usize = 20_000;
/// Form XObjects nested deeper than this are not walked and need vision.
pub const MAX_FORM_DEPTH: usize = 8;

const MATRIX_EPSILON: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionReason {
    /// The source is an uploaded image; there is no text layer at all.
    ImageUpload,
    /// The page could not be fully interpreted.
    Unsupported,
    SparseText,
    UnreliableText,
    Annotation,
    Raster,
    Vector,
}

impl VisionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ImageUpload => "image-upload",
            Self::Unsupported => "unsupported",
            Self::SparseText => "sparse-text",
            Self::UnreliableText => "unreliable-text",
            Self::Annotation => "annotation",
            Self::Raster => "raster",
            Self::Vector => "vector",
        }
    }
}

/// Outcome of the per-page decision; `reason` is `None` exactly when the page
/// is text-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageVision {
    pub needs_vision: bool,
    pub reason: Option<VisionReason>,
}

impl PageVision {
    pub const TEXT: Self = Self {
        needs_vision: false,
        reason: None,
    };

    pub const fn vision(reason: VisionReason) -> Self {
        Self {
            needs_vision: true,
            reason: Some(reason),
        }
    }
}

/// The single rule every model stage uses to decide whether to send a page
/// image. A blank text layer always falls back to the image, so a text-only
/// page can never reach a model as an empty passage.
pub fn page_uses_image(needs_vision: bool, text: Option<&str>) -> bool {
    needs_vision || text.is_none_or(|text| text.trim().is_empty())
}

/// Axis-aligned rectangle in PDF user space (points).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub left: f64,
    pub bottom: f64,
    pub right: f64,
    pub top: f64,
}

impl Rect {
    pub fn new(left: f64, bottom: f64, right: f64, top: f64) -> Self {
        Self {
            left: left.min(right),
            bottom: bottom.min(top),
            right: left.max(right),
            top: bottom.max(top),
        }
    }

    fn width(&self) -> f64 {
        self.right - self.left
    }

    fn height(&self) -> f64 {
        self.top - self.bottom
    }

    fn area(&self) -> f64 {
        self.width() * self.height()
    }

    fn is_finite(&self) -> bool {
        [self.left, self.bottom, self.right, self.top]
            .iter()
            .all(|value| value.is_finite())
    }

    fn intersect(&self, other: &Rect) -> Option<Rect> {
        let clipped = Rect {
            left: self.left.max(other.left),
            bottom: self.bottom.max(other.bottom),
            right: self.right.min(other.right),
            top: self.top.min(other.top),
        };
        (clipped.width() > 0.0 && clipped.height() > 0.0).then_some(clipped)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}

impl Rgba {
    fn is_invisible(&self) -> bool {
        self.alpha == 0
            || [self.red, self.green, self.blue]
                .iter()
                .all(|channel| *channel >= NEAR_WHITE_CHANNEL)
    }
}

/// A drawn shape reduced to what the rule needs, already in page space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorShape {
    pub rect: Rect,
    pub filled: bool,
    /// `None` when the fill color is unknown, which counts as visible.
    pub fill: Option<Rgba>,
    pub stroked: bool,
    /// The outline contains a curve or a diagonal line, or it is drawn under a
    /// rotating or skewing transform; only axis-aligned straight strokes are
    /// table grids and frames.
    pub stroke_is_drawing: bool,
    /// Hash of the outline, transforms, and paint; `None` when it cannot be
    /// read, so the shape can never be page furniture.
    pub content: Option<u64>,
}

/// A placed image, already in page space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterImage {
    pub rect: Rect,
    /// Hash of the encoded image stream; `None` when it cannot be read, so the
    /// image can never be page furniture.
    pub content: Option<u64>,
}

/// One page object as the rule sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Primitive {
    Raster(RasterImage),
    Vector(VectorShape),
    Unsupported,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TextSignals {
    pub valid: usize,
    pub unreliable: usize,
    /// Standalone symbol-font characters, which count as neither.
    pub symbols: usize,
}

pub fn text_signals(text: Option<&str>) -> TextSignals {
    let mut signals = TextSignals::default();
    let mut previous: Option<char> = None;
    let mut characters = text.unwrap_or_default().chars().peekable();
    while let Some(character) = characters.next() {
        let standalone = previous.is_none_or(char::is_whitespace)
            && characters.peek().is_none_or(|next| next.is_whitespace());
        if standalone && SYMBOL_FONT_RANGE.contains(&character) {
            signals.symbols += 1;
        } else if is_unreliable_char(character) {
            signals.unreliable += 1;
        } else if !character.is_whitespace() {
            signals.valid += 1;
        }
        previous = Some(character);
    }
    signals
}

fn is_unreliable_char(character: char) -> bool {
    character == '\u{FFFD}'
        || (character.is_control() && !character.is_whitespace())
        || matches!(
            character,
            '\u{E000}'..='\u{F8FF}' | '\u{F0000}'..='\u{FFFFD}' | '\u{100000}'..='\u{10FFFD}'
        )
}

/// Page coverage accumulated from the counted primitives.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Coverage {
    /// Share of the page covered by counted images, page furniture included.
    pub raster_ratio: f64,
    pub vector_ratio: f64,
    pub vector_count: usize,
    pub unsupported: bool,
    /// Counted graphics that may turn out to be page furniture.
    pub furniture_candidates: Vec<FurnitureCandidate>,
    /// The part of the counts above that the document marked as furniture.
    pub furniture: Furniture,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Furniture {
    pub raster_ratio: f64,
    pub vector_ratio: f64,
    pub vector_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectKind {
    Raster,
    Vector,
}

/// The same kind of graphic with the same content at the same rounded bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectKey {
    pub kind: ObjectKind,
    pub content: u64,
    pub bounds: [i64; 4],
}

impl ObjectKey {
    fn new(kind: ObjectKind, content: u64, rect: &Rect) -> Self {
        Self {
            kind,
            content,
            bounds: [rect.left, rect.bottom, rect.right, rect.top]
                .map(|value| value.round() as i64),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FurnitureCandidate {
    pub key: ObjectKey,
    /// Counted share of the page.
    pub ratio: f64,
    /// Bounds clipped to the page, where the appearance is compared.
    pub visible: Rect,
}

pub fn accumulate(primitives: impl IntoIterator<Item = Primitive>, crop: Rect) -> Coverage {
    let mut coverage = Coverage::default();
    let page_area = crop.area();
    if !crop.is_finite() || page_area <= 0.0 {
        coverage.unsupported = true;
        return coverage;
    }
    for primitive in primitives {
        match primitive {
            Primitive::Unsupported => coverage.unsupported = true,
            Primitive::Raster(image) => {
                if !image.rect.is_finite() {
                    coverage.unsupported = true;
                    continue;
                }
                let Some(visible) = image.rect.intersect(&crop) else {
                    continue;
                };
                let ratio = visible.area() / page_area;
                if ratio < MIN_RASTER_OBJECT_RATIO {
                    continue;
                }
                coverage.raster_ratio += ratio;
                if ratio < MAX_FURNITURE_IMAGE_RATIO {
                    coverage.add_candidate(
                        ObjectKind::Raster,
                        image.content,
                        &image.rect,
                        ratio,
                        visible,
                    );
                }
            }
            Primitive::Vector(shape) => {
                if !shape.rect.is_finite() {
                    coverage.unsupported = true;
                    continue;
                }
                let Some(visible) = shape.rect.intersect(&crop) else {
                    continue;
                };
                if vector_counts(&shape, &visible, page_area) {
                    let ratio = visible.area() / page_area;
                    coverage.vector_ratio += ratio;
                    coverage.vector_count += 1;
                    coverage.add_candidate(
                        ObjectKind::Vector,
                        shape.content,
                        &shape.rect,
                        ratio,
                        visible,
                    );
                }
            }
        }
    }
    coverage
}

impl Coverage {
    fn add_candidate(
        &mut self,
        kind: ObjectKind,
        content: Option<u64>,
        rect: &Rect,
        ratio: f64,
        visible: Rect,
    ) {
        if let Some(content) = content
            && self.furniture_candidates.len() < MAX_FURNITURE_CANDIDATES
        {
            self.furniture_candidates.push(FurnitureCandidate {
                key: ObjectKey::new(kind, content, rect),
                ratio,
                visible,
            });
        }
    }

    /// Image coverage that carries page content.
    pub fn content_raster_ratio(&self) -> f64 {
        self.raster_ratio - self.furniture.raster_ratio
    }

    /// Vector coverage and count that carry page content.
    pub fn content_vectors(&self) -> (f64, usize) {
        (
            self.vector_ratio - self.furniture.vector_ratio,
            self.vector_count - self.furniture.vector_count,
        )
    }
}

fn vector_counts(shape: &VectorShape, visible: &Rect, page_area: f64) -> bool {
    if visible.width().min(visible.height()) < MIN_VECTOR_SIDE_POINTS {
        return false;
    }
    if visible.area() / page_area >= BACKGROUND_COVERAGE {
        return false;
    }
    let visible_fill = shape.filled && !shape.fill.is_some_and(|fill| fill.is_invisible());
    visible_fill || (shape.stroked && shape.stroke_is_drawing)
}

pub fn classify(text: TextSignals, coverage: &Coverage, visual_annotation: bool) -> PageVision {
    let total = text.valid + text.unreliable;
    if coverage.unsupported {
        PageVision::vision(VisionReason::Unsupported)
    } else if total > 0 && text.unreliable as f64 / total as f64 > MAX_UNRELIABLE_TEXT_RATIO {
        PageVision::vision(VisionReason::UnreliableText)
    } else if text.valid < MIN_TEXT_CHARS {
        PageVision::vision(VisionReason::SparseText)
    } else if visual_annotation {
        PageVision::vision(VisionReason::Annotation)
    } else if coverage.content_raster_ratio() >= MIN_RASTER_COVERAGE {
        PageVision::vision(VisionReason::Raster)
    } else if let (ratio, count) = coverage.content_vectors()
        && (ratio >= MIN_VECTOR_COVERAGE || count >= MIN_VECTOR_OBJECTS)
    {
        PageVision::vision(VisionReason::Vector)
    } else {
        PageVision::TEXT
    }
}

/// What one page contributes to the document's decisions. It holds no text
/// content, so it is safe to log.
#[derive(Debug, Clone, PartialEq)]
pub struct PageObservation {
    pub text: TextSignals,
    pub coverage: Coverage,
    pub visual_annotation: bool,
    pub objects: usize,
    /// Page-space bounds of every text object, masked out when appearances
    /// are compared.
    pub text_boxes: Vec<Rect>,
}

impl PageObservation {
    /// True when a furniture candidate on this page has one of `keys`.
    pub fn has_any(&self, keys: &HashSet<ObjectKey>) -> bool {
        self.coverage
            .furniture_candidates
            .iter()
            .any(|candidate| keys.contains(&candidate.key))
    }
}

/// One page's decision and the measured inputs behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct PageAssessment {
    pub vision: PageVision,
    pub observation: PageObservation,
}

/// What each repeated object looks like on one page (see
/// `FurnitureRenderer::appearances`); empty when it was not rendered.
pub type Appearances = Vec<(ObjectKey, u64)>;

fn furniture_pages(pages: usize) -> usize {
    MIN_FURNITURE_PAGES.max((pages as f64 * MIN_FURNITURE_PAGE_SHARE).ceil() as usize)
}

/// One document's page observations, in page order. Text boxes are kept only
/// while the whole document has at most `MAX_FURNITURE_TEXT_BOXES`.
#[derive(Debug, Clone, Default)]
pub struct DocumentObservations {
    pages: Vec<PageObservation>,
    text_boxes: usize,
    too_much_text: bool,
}

impl DocumentObservations {
    pub fn with_capacity(pages: usize) -> Self {
        Self {
            pages: Vec::with_capacity(pages),
            ..Self::default()
        }
    }

    pub fn push(&mut self, mut page: PageObservation) {
        self.text_boxes = self.text_boxes.saturating_add(page.text_boxes.len());
        if !self.too_much_text && self.text_boxes > MAX_FURNITURE_TEXT_BOXES {
            self.too_much_text = true;
            for earlier in &mut self.pages {
                earlier.text_boxes = Vec::new();
            }
        }
        if self.too_much_text {
            page.text_boxes = Vec::new();
        }
        self.pages.push(page);
    }

    pub fn pages(&self) -> &[PageObservation] {
        &self.pages
    }

    pub fn into_pages(self) -> Vec<PageObservation> {
        self.pages
    }

    /// Keys drawn on enough of the document's pages to be furniture if they
    /// also look the same there. Empty when the text could not be kept.
    pub fn repeated(&self) -> HashSet<ObjectKey> {
        if self.too_much_text {
            return HashSet::new();
        }
        repeated_objects(&self.pages)
    }
}

fn repeated_objects(pages: &[PageObservation]) -> HashSet<ObjectKey> {
    let required = furniture_pages(pages.len());
    let mut pages_with: HashMap<ObjectKey, usize> = HashMap::new();
    for page in pages {
        let keys = page
            .coverage
            .furniture_candidates
            .iter()
            .map(|candidate| candidate.key)
            .collect::<HashSet<_>>();
        for key in keys {
            *pages_with.entry(key).or_default() += 1;
        }
    }
    pages_with
        .into_iter()
        .filter(|(_, count)| *count >= required)
        .map(|(key, _)| key)
        .collect()
}

/// Decides every page of one document, in order. `appearances` holds one
/// entry per page; a graphic is furniture on a page when enough pages draw the
/// same key with the same appearance.
pub fn assess_document(
    pages: Vec<PageObservation>,
    appearances: &[Appearances],
) -> Vec<PageAssessment> {
    let required = furniture_pages(pages.len());
    let per_page = (0..pages.len())
        .map(|index| {
            appearances
                .get(index)
                .map(|page| page.iter().copied().collect::<HashSet<_>>())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let mut pages_with: HashMap<(ObjectKey, u64), usize> = HashMap::new();
    for page in &per_page {
        for seen in page {
            *pages_with.entry(*seen).or_default() += 1;
        }
    }
    let confirmed = per_page
        .iter()
        .map(|page| {
            page.iter()
                .filter(|seen| pages_with[*seen] >= required)
                .map(|(key, _)| *key)
                .collect::<HashSet<_>>()
        })
        .collect::<Vec<_>>();
    pages
        .into_iter()
        .zip(confirmed)
        .map(|(mut observation, furniture)| {
            let coverage = &mut observation.coverage;
            coverage.furniture = Furniture::default();
            for candidate in &coverage.furniture_candidates {
                if !furniture.contains(&candidate.key) {
                    continue;
                }
                match candidate.key.kind {
                    ObjectKind::Raster => coverage.furniture.raster_ratio += candidate.ratio,
                    ObjectKind::Vector => {
                        coverage.furniture.vector_ratio += candidate.ratio;
                        coverage.furniture.vector_count += 1;
                    }
                }
            }
            PageAssessment {
                vision: classify(
                    observation.text,
                    &observation.coverage,
                    observation.visual_annotation,
                ),
                observation,
            }
        })
        .collect()
}

/// Walks one Pdfium page; `assess_document` makes the decision once every page
/// of the document has been observed.
pub fn observe_pdf_page(page: &PdfPage<'_>, extracted_text: Option<&str>) -> PageObservation {
    let mut walker = Walker::default();
    walker.walk(page.objects().iter(), &[], 0);
    let visual_annotation = page.annotations().iter().any(|annotation| {
        // A link's normal appearance stream is what Pdfium paints; the
        // object count reads exactly that stream and is 0 without one.
        annotation_is_painted(
            annotation.annotation_type(),
            !annotation.objects().is_empty(),
        )
    });
    PageObservation {
        text: text_signals(extracted_text),
        coverage: page_coverage(page, walker.primitives),
        visual_annotation,
        objects: walker.objects,
        text_boxes: walker.text_boxes,
    }
}

/// Renders pages for the second pass. The text mask is the union of the text
/// boxes of every page that draws a repeated graphic, so the same pixels are
/// ignored on every page that is compared.
pub struct FurnitureRenderer {
    text_boxes: Vec<Rect>,
    pixel_budget: u64,
    config: PdfRenderConfig,
    /// The mask for the last page geometry, which most pages share.
    mask: Option<(MaskGeometry, TextMask)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MaskGeometry {
    width: usize,
    height: usize,
    to_pixels: Affine,
}

/// Page space to pixel space.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Affine {
    x: [f64; 3],
    y: [f64; 3],
}

impl Affine {
    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.x[0] + self.x[1] * x + self.x[2] * y,
            self.y[0] + self.y[1] * x + self.y[2] * y,
        )
    }

    /// Pixel bounds of a page-space rectangle, clamped to the image.
    fn pixel_box(
        &self,
        rect: &Rect,
        margin: i64,
        width: usize,
        height: usize,
    ) -> Option<[usize; 4]> {
        let corners = [
            self.apply(rect.left, rect.bottom),
            self.apply(rect.left, rect.top),
            self.apply(rect.right, rect.bottom),
            self.apply(rect.right, rect.top),
        ];
        let left = corners
            .iter()
            .map(|corner| corner.0)
            .fold(f64::INFINITY, f64::min);
        let right = corners
            .iter()
            .map(|corner| corner.0)
            .fold(f64::NEG_INFINITY, f64::max);
        let top = corners
            .iter()
            .map(|corner| corner.1)
            .fold(f64::INFINITY, f64::min);
        let bottom = corners
            .iter()
            .map(|corner| corner.1)
            .fold(f64::NEG_INFINITY, f64::max);
        if ![left, right, top, bottom]
            .iter()
            .all(|value| value.is_finite())
        {
            return None;
        }
        let clamp = |value: f64, limit: usize| (value as i64).clamp(0, limit as i64) as usize;
        let x0 = clamp(left.floor() - margin as f64, width);
        let x1 = clamp(right.ceil() + margin as f64, width);
        let y0 = clamp(top.floor() - margin as f64, height);
        let y1 = clamp(bottom.ceil() + margin as f64, height);
        (x0 < x1 && y0 < y1).then_some([x0, y0, x1, y1])
    }
}

impl FurnitureRenderer {
    pub fn new(document: &DocumentObservations, repeated: &HashSet<ObjectKey>) -> Self {
        let text_boxes = document
            .pages()
            .iter()
            .filter(|page| page.has_any(repeated))
            .flat_map(|page| page.text_boxes.iter().copied())
            .collect::<Vec<_>>();
        Self {
            text_boxes,
            pixel_budget: MAX_FURNITURE_RENDER_PIXELS,
            config: PdfRenderConfig::new()
                .scale_page_by_factor(1.0)
                .set_maximum_width(FURNITURE_RENDER_MAX_PIXELS)
                .set_maximum_height(FURNITURE_RENDER_MAX_PIXELS)
                .render_form_data(true)
                .render_annotations(true),
            mask: None,
        }
    }

    /// Takes `cost` from the budget, or fails once it is spent.
    fn spend(&mut self, cost: u64) -> Option<()> {
        self.pixel_budget = self.pixel_budget.checked_sub(cost)?;
        Some(())
    }

    /// Hashes, for every repeated graphic on the page, the masked pixels of
    /// the tiles under its visible bounds. Empty when the page cannot be
    /// rendered or the budget is spent, so nothing on it is furniture.
    pub fn appearances(
        &mut self,
        page: &PdfPage<'_>,
        observation: &PageObservation,
        repeated: &HashSet<ObjectKey>,
    ) -> Appearances {
        self.render_appearances(page, observation, repeated)
            .unwrap_or_default()
    }

    fn render_appearances(
        &mut self,
        page: &PdfPage<'_>,
        observation: &PageObservation,
        repeated: &HashSet<ObjectKey>,
    ) -> Option<Appearances> {
        // At scale 1 a side is its length in points rounded, at least one
        // pixel, and at most the render limit.
        let side = |points: f32| {
            let points = f64::from(points);
            points.is_finite().then(|| {
                points
                    .ceil()
                    .clamp(1.0, f64::from(FURNITURE_RENDER_MAX_PIXELS)) as u64
            })
        };
        let bound = side(page.width().value)? * side(page.height().value)?;
        self.spend(bound)?;
        let image = page
            .render_with_config(&self.config)
            .and_then(|bitmap| bitmap.as_image())
            .ok()?
            .into_rgba8();
        let (width, height) = (image.width() as usize, image.height() as usize);
        if (width as u64).saturating_mul(height as u64) > bound {
            return None;
        }
        let to_pixels = self.affine(page)?;
        let geometry = MaskGeometry {
            width,
            height,
            to_pixels,
        };
        if self
            .mask
            .as_ref()
            .is_none_or(|(cached, _)| *cached != geometry)
        {
            self.spend(self.text_boxes.len() as u64 + bound)?;
            self.mask = Some((
                geometry,
                TextMask::new(&self.text_boxes, &to_pixels, width, height),
            ));
        }
        let mask = &self.mask.as_ref()?.1;
        let mut pixels = image.into_raw();
        for (pixel, masked) in pixels.chunks_exact_mut(4).zip(&mask.masked) {
            if *masked {
                pixel.fill(0);
            }
        }
        let tile = FURNITURE_TILE_PIXELS;
        let tiles_wide = width.div_ceil(tile);
        let tiles_high = height.div_ceil(tile);
        let mut tiles = vec![0u64; tiles_wide * tiles_high];
        for (index, hash) in tiles.iter_mut().enumerate() {
            let (tile_x, tile_y) = (index % tiles_wide, index / tiles_wide);
            let x0 = tile_x * tile;
            let x1 = (x0 + tile).min(width);
            let mut hasher = DefaultHasher::new();
            for row in tile_y * tile..((tile_y + 1) * tile).min(height) {
                hasher.write(&pixels[(row * width + x0) * 4..(row * width + x1) * 4]);
            }
            *hash = hasher.finish();
        }
        let mut appearances = Vec::new();
        for candidate in &observation.coverage.furniture_candidates {
            if !repeated.contains(&candidate.key) {
                continue;
            }
            let Some([x0, y0, x1, y1]) = to_pixels.pixel_box(&candidate.visible, 0, width, height)
            else {
                continue;
            };
            let area = (x1 - x0) * (y1 - y0);
            let unmasked = area - mask.count([x0, y0, x1, y1]);
            if (unmasked as f64) < area as f64 * MIN_FURNITURE_UNMASKED_SHARE {
                continue;
            }
            let (columns, rows) = (x0 / tile..=(x1 - 1) / tile, y0 / tile..=(y1 - 1) / tile);
            self.pixel_budget = self
                .pixel_budget
                .checked_sub((columns.clone().count() * rows.clone().count()) as u64)?;
            let mut hasher = DefaultHasher::new();
            (x0 / tile, y0 / tile, (x1 - 1) / tile, (y1 - 1) / tile).hash(&mut hasher);
            for tile_y in rows {
                for tile_x in columns.clone() {
                    hasher.write_u64(tiles[tile_y * tiles_wide + tile_x]);
                }
            }
            appearances.push((candidate.key, hasher.finish()));
        }
        Some(appearances)
    }

    fn affine(&self, page: &PdfPage<'_>) -> Option<Affine> {
        const SPAN: f32 = 1_000.0;
        let point = |x: f32, y: f32| {
            page.points_to_pixels(PdfPoints::new(x), PdfPoints::new(y), &self.config)
                .ok()
                .map(|(px, py)| (f64::from(px), f64::from(py)))
        };
        let origin = point(0.0, 0.0)?;
        let along_x = point(SPAN, 0.0)?;
        let along_y = point(0.0, SPAN)?;
        let span = f64::from(SPAN);
        Some(Affine {
            x: [
                origin.0,
                (along_x.0 - origin.0) / span,
                (along_y.0 - origin.0) / span,
            ],
            y: [
                origin.1,
                (along_x.1 - origin.1) / span,
                (along_y.1 - origin.1) / span,
            ],
        })
    }
}

/// Pixels under any text box, widened for anti-aliasing, and how many of
/// them lie in any rectangle.
struct TextMask {
    width: usize,
    masked: Vec<bool>,
    /// Masked pixels above and to the left of each pixel corner, one row and
    /// one column wider than the image.
    before: Vec<u32>,
}

impl TextMask {
    /// Each box marks its corners in a difference table, so building the mask
    /// takes one pass over the pixels whatever the boxes' sizes.
    fn new(boxes: &[Rect], to_pixels: &Affine, width: usize, height: usize) -> Self {
        let stride = width + 1;
        let mut edges = vec![0i32; stride * (height + 1)];
        for rect in boxes {
            let Some([x0, y0, x1, y1]) =
                to_pixels.pixel_box(rect, TEXT_MASK_MARGIN_PIXELS, width, height)
            else {
                continue;
            };
            edges[y0 * stride + x0] += 1;
            edges[y0 * stride + x1] -= 1;
            edges[y1 * stride + x0] -= 1;
            edges[y1 * stride + x1] += 1;
        }
        let mut masked = vec![false; width * height];
        let mut before = vec![0u32; stride * (height + 1)];
        let mut covering = vec![0i32; width];
        for row in 0..height {
            let (mut running, mut in_row) = (0i32, 0u32);
            for column in 0..width {
                running += edges[row * stride + column];
                covering[column] += running;
                let is_masked = covering[column] > 0;
                masked[row * width + column] = is_masked;
                in_row += u32::from(is_masked);
                before[(row + 1) * stride + column + 1] =
                    before[row * stride + column + 1] + in_row;
            }
        }
        Self {
            width,
            masked,
            before,
        }
    }

    /// Masked pixels in `[x0, x1) x [y0, y1)`.
    fn count(&self, [x0, y0, x1, y1]: [usize; 4]) -> usize {
        let stride = self.width + 1;
        let at = |x: usize, y: usize| self.before[y * stride + x] as usize;
        (at(x1, y1) + at(x0, y0)) - (at(x1, y0) + at(x0, y1))
    }
}

fn page_coverage(page: &PdfPage<'_>, primitives: Vec<Primitive>) -> Coverage {
    match page_crop(page) {
        Some(crop) => accumulate(primitives, crop),
        None => Coverage {
            unsupported: true,
            ..Coverage::default()
        },
    }
}

fn annotation_is_painted(annotation_type: PdfPageAnnotationType, has_appearance: bool) -> bool {
    match annotation_type {
        // Popups are never painted into the page image.
        PdfPageAnnotationType::Popup => false,
        // Links are usually invisible hotspots, but a link with its own
        // appearance (a button, a drawn frame) is painted like any other.
        PdfPageAnnotationType::Link => has_appearance,
        // Stamps, ink, free text, and form widgets can all be painted; Pdfium
        // also generates appearances for several types that lack one.
        _ => true,
    }
}

fn page_crop(page: &PdfPage<'_>) -> Option<Rect> {
    let boundaries = page.boundaries();
    let bounds = boundaries
        .crop()
        .or_else(|_| boundaries.media())
        .map(|boundary| boundary.bounds)
        .unwrap_or_else(|_| page.page_size());
    let rect = pdf_rect(&bounds);
    (rect.is_finite() && rect.area() > 0.0).then_some(rect)
}

fn pdf_rect(rect: &PdfRect) -> Rect {
    Rect::new(
        f64::from(rect.left().value),
        f64::from(rect.bottom().value),
        f64::from(rect.right().value),
        f64::from(rect.top().value),
    )
}

#[derive(Default)]
struct Walker {
    primitives: Vec<Primitive>,
    objects: usize,
    text_boxes: Vec<Rect>,
}

impl Walker {
    /// `forms` holds the enclosing form XObject matrices, innermost first: an
    /// object inside a form reports its bounds in the form's own space.
    fn walk<'a>(
        &mut self,
        objects: impl Iterator<Item = PdfPageObject<'a>>,
        forms: &[PdfMatrix],
        depth: usize,
    ) {
        for object in objects {
            self.objects += 1;
            if self.objects > MAX_PAGE_OBJECTS {
                self.primitives.push(Primitive::Unsupported);
                return;
            }
            match &object {
                PdfPageObject::Text(_) => {
                    if let Some(rect) = page_space_bounds(&object, forms) {
                        self.text_boxes.push(rect);
                    }
                }
                PdfPageObject::Image(image) => match page_space_bounds(&object, forms) {
                    Some(rect) => self.primitives.push(Primitive::Raster(RasterImage {
                        rect,
                        content: image_content(image),
                    })),
                    None => self.primitives.push(Primitive::Unsupported),
                },
                PdfPageObject::Shading(_) => match page_space_bounds(&object, forms) {
                    Some(rect) => self.primitives.push(Primitive::Vector(VectorShape {
                        rect,
                        filled: true,
                        fill: None,
                        stroked: false,
                        stroke_is_drawing: false,
                        content: None,
                    })),
                    None => self.primitives.push(Primitive::Unsupported),
                },
                PdfPageObject::Path(path) => match path_shape(&object, path, forms) {
                    Some(shape) => self.primitives.push(Primitive::Vector(shape)),
                    None => self.primitives.push(Primitive::Unsupported),
                },
                PdfPageObject::XObjectForm(form) => {
                    if depth >= MAX_FORM_DEPTH {
                        self.primitives.push(Primitive::Unsupported);
                        continue;
                    }
                    let Ok(matrix) = form.matrix() else {
                        self.primitives.push(Primitive::Unsupported);
                        continue;
                    };
                    let mut nested = Vec::with_capacity(forms.len() + 1);
                    nested.push(matrix);
                    nested.extend_from_slice(forms);
                    self.walk(form.iter(), &nested, depth + 1);
                }
                PdfPageObject::Unsupported(_) => self.primitives.push(Primitive::Unsupported),
            }
            if self.objects > MAX_PAGE_OBJECTS {
                return;
            }
        }
    }
}

/// Hash of the encoded image stream; the rendered appearance confirms it.
fn image_content(image: &PdfPageImageObject<'_>) -> Option<u64> {
    let data = image
        .get_raw_image_data()
        .ok()
        .filter(|data| !data.is_empty())?;
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    Some(hasher.finish())
}

fn page_space_bounds(object: &PdfPageObject<'_>, forms: &[PdfMatrix]) -> Option<Rect> {
    let mut quad = object.bounds().ok()?;
    for matrix in forms {
        quad = quad.transform(*matrix);
    }
    let rect = pdf_rect(&quad.to_rect());
    rect.is_finite().then_some(rect)
}

fn path_shape(
    object: &PdfPageObject<'_>,
    path: &PdfPagePathObject<'_>,
    forms: &[PdfMatrix],
) -> Option<VectorShape> {
    let rect = page_space_bounds(object, forms)?;
    let filled = path.fill_mode().ok()? != PdfPathFillMode::None;
    let stroked = path.is_stroked().ok()?;
    let fill = path.fill_color().ok().map(|color| Rgba {
        red: color.red(),
        green: color.green(),
        blue: color.blue(),
        alpha: color.alpha(),
    });
    let stroke_is_drawing = stroked && {
        let rotated = std::iter::once(path.matrix().ok())
            .chain(forms.iter().copied().map(Some))
            .any(|matrix| matrix.is_none_or(|matrix| !preserves_axes(&matrix)));
        rotated || outline_is_drawing(path)
    };
    Some(VectorShape {
        rect,
        filled,
        fill,
        stroked,
        stroke_is_drawing,
        content: path_content(object, path, forms, filled, stroked),
    })
}

/// Hash of the outline, its transforms, and its paint, which groups shapes
/// before the rendered appearance confirms them.
fn path_content(
    object: &PdfPageObject<'_>,
    path: &PdfPagePathObject<'_>,
    forms: &[PdfMatrix],
    filled: bool,
    stroked: bool,
) -> Option<u64> {
    let segments = path.segments();
    if usize::try_from(segments.len()).ok()? > MAX_PATH_SEGMENTS {
        return None;
    }
    let quantize = |value: f32| (f64::from(value) * 1_000.0).round() as i64;
    let mut hasher = DefaultHasher::new();
    for segment in segments.iter() {
        let kind = match segment.segment_type() {
            PdfPathSegmentType::MoveTo => 0u8,
            PdfPathSegmentType::LineTo => 1,
            PdfPathSegmentType::BezierTo => 2,
            PdfPathSegmentType::Unknown => 3,
        };
        (
            kind,
            segment.is_close(),
            quantize(segment.x().value),
            quantize(segment.y().value),
        )
            .hash(&mut hasher);
    }
    for matrix in std::iter::once(path.matrix().ok()?).chain(forms.iter().copied()) {
        [
            matrix.a(),
            matrix.b(),
            matrix.c(),
            matrix.d(),
            matrix.e(),
            matrix.f(),
        ]
        .map(quantize)
        .hash(&mut hasher);
    }
    (filled, stroked).hash(&mut hasher);
    for color in [object.fill_color().ok(), object.stroke_color().ok()]
        .into_iter()
        .flatten()
    {
        [color.red(), color.green(), color.blue(), color.alpha()].hash(&mut hasher);
    }
    if stroked {
        quantize(object.stroke_width().ok()?.value).hash(&mut hasher);
    }
    Some(hasher.finish())
}

/// Scale, translation, and quarter-turn rotations keep horizontal and vertical
/// lines horizontal and vertical.
fn preserves_axes(matrix: &PdfMatrix) -> bool {
    let near_zero = |value: f32| f64::from(value).abs() < MATRIX_EPSILON;
    (near_zero(matrix.b()) && near_zero(matrix.c()))
        || (near_zero(matrix.a()) && near_zero(matrix.d()))
}

/// True when the outline contains a curve or a diagonal line. Segment points
/// are in the path's own space, so this is independent of its transform.
fn outline_is_drawing(path: &PdfPagePathObject<'_>) -> bool {
    let segments = path.segments();
    let count = usize::try_from(segments.len()).unwrap_or(usize::MAX);
    if count > MAX_PATH_SEGMENTS {
        return true;
    }
    let mut current: Option<(f64, f64)> = None;
    let mut subpath_start: Option<(f64, f64)> = None;
    for segment in segments.iter() {
        let point = (f64::from(segment.x().value), f64::from(segment.y().value));
        match segment.segment_type() {
            PdfPathSegmentType::MoveTo => {
                subpath_start = Some(point);
            }
            PdfPathSegmentType::LineTo => {
                if current.is_some_and(|from| is_diagonal(from, point)) {
                    return true;
                }
            }
            PdfPathSegmentType::BezierTo => return true,
            PdfPathSegmentType::Unknown => return true,
        }
        if segment.is_close() {
            if let Some(start) = subpath_start
                && is_diagonal(point, start)
            {
                return true;
            }
            current = subpath_start;
        } else {
            current = Some(point);
        }
    }
    false
}

fn is_diagonal(from: (f64, f64), to: (f64, f64)) -> bool {
    let dx = (to.0 - from.0).abs();
    let dy = (to.1 - from.1).abs();
    let (major, minor) = if dx >= dy { (dx, dy) } else { (dy, dx) };
    major > 0.0 && minor > major * AXIS_ALIGNED_TOLERANCE
}

#[cfg(test)]
mod tests {
    use super::*;

    const LETTER: Rect = Rect {
        left: 0.0,
        bottom: 0.0,
        right: 612.0,
        top: 792.0,
    };

    fn text(valid: usize) -> TextSignals {
        TextSignals {
            valid,
            ..TextSignals::default()
        }
    }

    fn raster(rect: Rect) -> Primitive {
        Primitive::Raster(RasterImage {
            rect,
            content: None,
        })
    }

    fn image(rect: Rect, content: u64) -> Primitive {
        Primitive::Raster(RasterImage {
            rect,
            content: Some(content),
        })
    }

    /// One text page per entry, each drawing the given primitives; every
    /// repeated graphic looks the same on every page.
    fn document(pages: Vec<Vec<Primitive>>) -> Vec<PageVision> {
        document_with(pages, |_, _| Some(0))
    }

    /// Like `document`, with `appearance(page, key)` standing in for the
    /// rendered appearance (`None` when the page was not rendered).
    fn document_with(
        pages: Vec<Vec<Primitive>>,
        appearance: impl Fn(usize, &ObjectKey) -> Option<u64>,
    ) -> Vec<PageVision> {
        let mut observations = DocumentObservations::default();
        for primitives in pages {
            observations.push(PageObservation {
                text: text(200),
                coverage: accumulate(primitives, LETTER),
                visual_annotation: false,
                objects: 0,
                text_boxes: Vec::new(),
            });
        }
        let repeated = observations.repeated();
        let appearances = observations
            .pages()
            .iter()
            .enumerate()
            .map(|(page, observation)| {
                observation
                    .coverage
                    .furniture_candidates
                    .iter()
                    .filter(|candidate| repeated.contains(&candidate.key))
                    .filter_map(|candidate| {
                        appearance(page, &candidate.key).map(|seen| (candidate.key, seen))
                    })
                    .collect()
            })
            .collect::<Vec<Appearances>>();
        assess_document(observations.into_pages(), &appearances)
            .into_iter()
            .map(|assessment| assessment.vision)
            .collect()
    }

    const RASTER: PageVision = PageVision::vision(VisionReason::Raster);
    const VECTOR: PageVision = PageVision::vision(VisionReason::Vector);

    fn filled(rect: Rect, fill: Option<Rgba>) -> Primitive {
        Primitive::Vector(VectorShape {
            rect,
            filled: true,
            fill,
            stroked: false,
            stroke_is_drawing: false,
            content: None,
        })
    }

    /// A dark filled shape with a readable identity.
    fn shape(rect: Rect, content: u64) -> Primitive {
        Primitive::Vector(VectorShape {
            rect,
            filled: true,
            fill: DARK,
            stroked: false,
            stroke_is_drawing: false,
            content: Some(content),
        })
    }

    fn stroked(rect: Rect, stroke_is_drawing: bool) -> Primitive {
        Primitive::Vector(VectorShape {
            rect,
            filled: false,
            fill: None,
            stroked: true,
            stroke_is_drawing,
            content: None,
        })
    }

    fn page_share(share: f64) -> Rect {
        // A square-ish rectangle covering `share` of the letter page.
        let area = LETTER.area() * share;
        let side = area.sqrt();
        Rect::new(100.0, 100.0, 100.0 + side, 100.0 + side)
    }

    fn decide(primitives: Vec<Primitive>) -> PageVision {
        classify(text(200), &accumulate(primitives, LETTER), false)
    }

    const DARK: Option<Rgba> = Some(Rgba {
        red: 40,
        green: 90,
        blue: 160,
        alpha: 255,
    });

    #[test]
    fn plain_text_page_is_text_only() {
        assert_eq!(decide(Vec::new()), PageVision::TEXT);
    }

    #[test]
    fn sparse_and_missing_text_layers_need_vision() {
        let empty = accumulate(Vec::new(), LETTER);
        assert_eq!(
            classify(text(MIN_TEXT_CHARS - 1), &empty, false),
            PageVision::vision(VisionReason::SparseText)
        );
        assert_eq!(
            classify(text_signals(None), &empty, false),
            PageVision::vision(VisionReason::SparseText)
        );
        assert_eq!(
            classify(text(MIN_TEXT_CHARS), &empty, false),
            PageVision::TEXT
        );
    }

    #[test]
    fn whitespace_does_not_count_as_text() {
        let signals = text_signals(Some(" a\r\nb\t c \u{3000}"));
        assert_eq!(signals, text(3));
    }

    #[test]
    fn garbled_text_layer_needs_vision() {
        let empty = accumulate(Vec::new(), LETTER);
        let garbled = format!("{}{}", "a".repeat(90), "\u{FFFD}".repeat(10));
        assert_eq!(
            classify(text_signals(Some(&garbled)), &empty, false),
            PageVision::vision(VisionReason::UnreliableText)
        );
        for unreliable in ['\u{E000}', '\u{F0001}', '\u{0002}', '\u{009F}'] {
            let signals = text_signals(Some(&format!("{}{unreliable}", "a".repeat(10))));
            assert_eq!(signals.unreliable, 1, "{unreliable:?}");
        }
        let mostly_clean = format!("{}{}", "a".repeat(100), "\u{FFFD}".repeat(5));
        assert_eq!(
            classify(text_signals(Some(&mostly_clean)), &empty, false),
            PageVision::TEXT
        );
    }

    #[test]
    fn standalone_symbol_font_bullets_are_neither_text_nor_garbage() {
        // How LibreOffice exports a Wingdings bullet list from a slide deck.
        let bullets =
            "Guards\r\n\u{F07D} Wear eye protection\r\n\u{F07D} Lock out\r\n\u{F07D} Test";
        assert_eq!(
            text_signals(Some(bullets)),
            TextSignals {
                valid: 34,
                unreliable: 0,
                symbols: 3,
            }
        );
        assert_eq!(
            text_signals(Some("\u{F0A7}")),
            TextSignals {
                symbols: 1,
                ..TextSignals::default()
            }
        );
        // Runs of symbol-font characters and ones inside a word are a font
        // without a usable ToUnicode map.
        for garbled in [
            "\u{F041}\u{F042}\u{F043}",
            "a\u{F07D}",
            "\u{F07D}b",
            "\u{F07D}\u{F07D} x",
        ] {
            let signals = text_signals(Some(garbled));
            assert_eq!(signals.symbols, 0, "{garbled:?}");
            assert!(signals.unreliable > 0, "{garbled:?}");
        }
        // Other private-use characters stay unreliable even when standalone.
        assert_eq!(text_signals(Some("\u{E000} x")).unreliable, 1);
        assert_eq!(text_signals(Some("\u{F100} x")).unreliable, 1);
    }

    #[test]
    fn visible_annotations_need_vision() {
        let empty = accumulate(Vec::new(), LETTER);
        assert_eq!(
            classify(text(200), &empty, true),
            PageVision::vision(VisionReason::Annotation)
        );
        assert!(!annotation_is_painted(PdfPageAnnotationType::Link, false));
        assert!(annotation_is_painted(PdfPageAnnotationType::Link, true));
        assert!(!annotation_is_painted(PdfPageAnnotationType::Popup, true));
        for painted in [
            PdfPageAnnotationType::Stamp,
            PdfPageAnnotationType::Ink,
            PdfPageAnnotationType::FreeText,
            PdfPageAnnotationType::Widget,
            PdfPageAnnotationType::Unknown,
        ] {
            assert!(annotation_is_painted(painted, false), "{painted:?}");
        }
    }

    #[test]
    fn raster_coverage_counts_only_images_above_the_logo_floor() {
        assert_eq!(
            decide(vec![raster(page_share(0.031))]),
            PageVision::vision(VisionReason::Raster)
        );
        assert_eq!(decide(vec![raster(page_share(0.029))]), PageVision::TEXT);
        // Ten logos just under the floor never add up to a figure.
        let logos = (0..10)
            .map(|_| raster(page_share(0.004)))
            .collect::<Vec<_>>();
        assert_eq!(decide(logos), PageVision::TEXT);
        // Several small figures above the floor do.
        let figures = (0..4)
            .map(|_| raster(page_share(0.008)))
            .collect::<Vec<_>>();
        assert_eq!(decide(figures), PageVision::vision(VisionReason::Raster));
    }

    #[test]
    fn images_repeated_on_most_pages_are_page_furniture() {
        // A footer bar of a slide master: 7.6% of the page on every slide.
        let bar = || image(Rect::new(0.0, 0.0, 612.0, 60.0), 7);
        let figure = || image(page_share(0.2), 8);
        assert_eq!(
            document(vec![
                vec![bar()],
                vec![bar()],
                vec![bar(), figure()],
                vec![bar()]
            ]),
            vec![PageVision::TEXT, PageVision::TEXT, RASTER, PageVision::TEXT]
        );
        // Without a readable identity nothing can be furniture.
        let unknown = || raster(Rect::new(0.0, 0.0, 612.0, 60.0));
        assert_eq!(
            document(vec![vec![unknown()], vec![unknown()], vec![unknown()]]),
            vec![RASTER; 3]
        );
        // The same image at another position is a different placement.
        let moved = image(Rect::new(0.0, 700.0, 612.0, 760.0), 7);
        assert_eq!(
            document(vec![vec![bar()], vec![bar()], vec![bar()], vec![moved]]),
            vec![PageVision::TEXT, PageVision::TEXT, PageVision::TEXT, RASTER]
        );
    }

    #[test]
    fn furniture_must_also_look_the_same() {
        let bar = || image(Rect::new(0.0, 0.0, 612.0, 60.0), 7);
        let pages = || vec![vec![bar()], vec![bar()], vec![bar()], vec![bar()]];
        // Pages 3-4 paint the same stream differently (a mask, a clip, a
        // palette): two looks on two pages each, neither enough.
        assert_eq!(
            document_with(pages(), |page, _| Some(u64::from(page >= 2))),
            vec![RASTER; 4]
        );
        // Three pages agree; the fourth, whose bar is painted over, keeps it.
        assert_eq!(
            document_with(pages(), |page, _| Some(u64::from(page == 3))),
            vec![PageVision::TEXT, PageVision::TEXT, PageVision::TEXT, RASTER]
        );
        // A page that was not rendered (budget, render error) keeps it too.
        assert_eq!(
            document_with(pages(), |page, _| (page != 0).then_some(0)),
            vec![RASTER, PageVision::TEXT, PageVision::TEXT, PageVision::TEXT]
        );
    }

    #[test]
    fn repeated_shapes_are_furniture_by_area_and_count() {
        // A slide master's diagonal band (8%) and 25 small decorations.
        let template = || {
            let mut shapes = vec![shape(page_share(0.08), 1)];
            shapes.extend((0u32..25).map(|index| {
                let left = 20.0 + 20.0 * f64::from(index);
                shape(
                    Rect::new(left, 700.0, left + 6.0, 706.0),
                    100 + u64::from(index),
                )
            }));
            shapes
        };
        let mut chart = template();
        chart.extend((0..3).map(|index| shape(page_share(0.03), 200 + index)));
        assert_eq!(
            document(vec![template(), template(), chart, template()]),
            vec![PageVision::TEXT, PageVision::TEXT, VECTOR, PageVision::TEXT]
        );
        // Without the document, the template alone needs vision.
        assert_eq!(decide(template()), VECTOR);
        // Shapes without a readable identity always count.
        let unknown = || vec![filled(page_share(0.08), DARK)];
        assert_eq!(
            document(vec![unknown(), unknown(), unknown()]),
            vec![VECTOR; 3]
        );
    }

    #[test]
    fn furniture_needs_enough_pages_and_a_share_of_the_document() {
        let bar = || image(Rect::new(0.0, 0.0, 612.0, 60.0), 7);
        // Two pages are never enough, even in a two-page document.
        assert_eq!(document(vec![vec![bar()], vec![bar()]]), vec![RASTER; 2]);
        // Ten pages need the image on five of them.
        let pages = |repeated: usize| {
            document(
                (0..10)
                    .map(|page| {
                        if page < repeated {
                            vec![bar()]
                        } else {
                            Vec::new()
                        }
                    })
                    .collect(),
            )
        };
        assert_eq!(pages(4)[..4], [RASTER; 4]);
        assert_eq!(pages(5), vec![PageVision::TEXT; 10]);
    }

    #[test]
    fn repeated_full_page_images_are_never_furniture() {
        // A scanned form under different filled-in values on every page.
        let form = || image(LETTER, 9);
        assert_eq!(
            document(vec![vec![form()], vec![form()], vec![form()]]),
            vec![RASTER; 3]
        );
        let banner = || image(page_share(MAX_FURNITURE_IMAGE_RATIO - 0.05), 9);
        assert_eq!(
            document(vec![vec![banner()], vec![banner()], vec![banner()]]),
            vec![PageVision::TEXT; 3]
        );
    }

    #[test]
    fn furniture_candidates_are_bounded_per_page() {
        let images = (0..MAX_FURNITURE_CANDIDATES as u64 + 10)
            .map(|content| image(page_share(0.006), content))
            .collect::<Vec<_>>();
        // Candidates past the bound always count, so they are never subtracted.
        let coverage = accumulate(images, LETTER);
        assert_eq!(
            coverage.furniture_candidates.len(),
            MAX_FURNITURE_CANDIDATES
        );
        let expected = 0.006 * (MAX_FURNITURE_CANDIDATES + 10) as f64;
        assert!((coverage.raster_ratio - expected).abs() < 1e-9);
    }

    #[test]
    fn documents_with_too_much_text_keep_no_boxes_and_no_furniture() {
        let page = |boxes: usize| PageObservation {
            text: text(200),
            coverage: accumulate(vec![image(page_share(0.05), 7)], LETTER),
            visual_annotation: false,
            objects: 0,
            text_boxes: vec![page_share(0.001); boxes],
        };
        let mut document = DocumentObservations::default();
        for _ in 0..3 {
            document.push(page(MAX_FURNITURE_TEXT_BOXES / 3));
        }
        assert_eq!(document.repeated().len(), 1);

        document.push(page(MAX_FURNITURE_TEXT_BOXES / 3));
        document.push(page(1));
        assert!(document.repeated().is_empty());
        assert!(
            document
                .pages()
                .iter()
                .all(|page| page.text_boxes.is_empty())
        );
    }

    #[test]
    fn text_mask_covers_each_widened_box_and_nothing_else() {
        // 72 dpi with the origin at the top left, as Pdfium renders a page.
        let to_pixels = Affine {
            x: [0.0, 1.0, 0.0],
            y: [100.0, 0.0, -1.0],
        };
        let boxes = [
            Rect::new(10.0, 80.0, 20.0, 90.0),
            Rect::new(15.0, 85.0, 30.0, 95.0),
            Rect::new(-50.0, -50.0, 3.0, 3.0),
        ];
        let mask = TextMask::new(&boxes, &to_pixels, 40, 100);
        let expected = |column: usize, row: usize| {
            let inside = |[x0, y0, x1, y1]: [usize; 4]| {
                (x0..x1).contains(&column) && (y0..y1).contains(&row)
            };
            inside([8, 8, 22, 22]) || inside([13, 3, 32, 17]) || inside([0, 95, 5, 100])
        };
        for row in 0..100 {
            for column in 0..40 {
                assert_eq!(
                    mask.masked[row * 40 + column],
                    expected(column, row),
                    "({column}, {row})"
                );
            }
        }
        let counted = |[x0, y0, x1, y1]: [usize; 4]| {
            (y0..y1)
                .flat_map(|row| (x0..x1).map(move |column| (column, row)))
                .filter(|(column, row)| expected(*column, *row))
                .count()
        };
        for area in [
            [0, 0, 40, 100],
            [10, 5, 25, 20],
            [0, 90, 7, 100],
            [30, 30, 40, 40],
        ] {
            assert_eq!(mask.count(area), counted(area), "{area:?}");
        }
    }

    #[test]
    fn full_page_scan_needs_vision() {
        assert_eq!(
            classify(text(0), &accumulate(vec![raster(LETTER)], LETTER), false),
            PageVision::vision(VisionReason::SparseText)
        );
        assert_eq!(
            decide(vec![raster(LETTER)]),
            PageVision::vision(VisionReason::Raster)
        );
    }

    #[test]
    fn coverage_is_clipped_to_the_crop_box() {
        let off_page = Rect::new(1000.0, 1000.0, 1400.0, 1400.0);
        assert_eq!(decide(vec![raster(off_page)]), PageVision::TEXT);
        // A crop box that does not start at the origin.
        let crop = Rect::new(100.0, 100.0, 712.0, 892.0);
        let coverage = accumulate(vec![raster(Rect::new(0.0, 0.0, 200.0, 200.0))], crop);
        let expected = 100.0 * 100.0 / crop.area();
        assert!((coverage.raster_ratio - expected).abs() < 1e-9);
    }

    #[test]
    fn filled_vector_graphics_need_vision_by_area_or_count() {
        assert_eq!(
            decide(vec![filled(page_share(0.051), DARK)]),
            PageVision::vision(VisionReason::Vector)
        );
        assert_eq!(
            decide(vec![filled(page_share(0.049), DARK)]),
            PageVision::TEXT
        );
        let icons = (0..MIN_VECTOR_OBJECTS)
            .map(|_| filled(Rect::new(0.0, 0.0, 6.0, 6.0), DARK))
            .collect::<Vec<_>>();
        assert_eq!(decide(icons), PageVision::vision(VisionReason::Vector));
        let fewer_icons = (0..MIN_VECTOR_OBJECTS - 1)
            .map(|_| filled(Rect::new(0.0, 0.0, 6.0, 6.0), DARK))
            .collect::<Vec<_>>();
        assert_eq!(decide(fewer_icons), PageVision::TEXT);
    }

    #[test]
    fn borders_backgrounds_and_white_fills_never_count() {
        let underline = filled(Rect::new(72.0, 100.0, 540.0, 101.0), DARK);
        let background = filled(LETTER, DARK);
        let white = filled(
            page_share(0.3),
            Some(Rgba {
                red: 255,
                green: 250,
                blue: 245,
                alpha: 255,
            }),
        );
        let transparent = filled(
            page_share(0.3),
            Some(Rgba {
                red: 0,
                green: 0,
                blue: 0,
                alpha: 0,
            }),
        );
        let many_borders = (0..100).map(|_| underline).collect::<Vec<_>>();
        assert_eq!(decide(many_borders), PageVision::TEXT);
        assert_eq!(
            decide(vec![background, white, transparent]),
            PageVision::TEXT
        );
    }

    #[test]
    fn stroked_paths_count_only_when_they_draw_something() {
        // An axis-aligned table grid around most of the page.
        assert_eq!(
            decide(vec![stroked(page_share(0.5), false)]),
            PageVision::TEXT
        );
        // A line chart or a pie outline of the same size.
        assert_eq!(
            decide(vec![stroked(page_share(0.06), true)]),
            PageVision::vision(VisionReason::Vector)
        );
    }

    #[test]
    fn unsupported_objects_and_degenerate_pages_need_vision() {
        assert_eq!(
            decide(vec![Primitive::Unsupported]),
            PageVision::vision(VisionReason::Unsupported)
        );
        let zero_page = Rect::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(
            classify(text(200), &accumulate(Vec::new(), zero_page), false),
            PageVision::vision(VisionReason::Unsupported)
        );
        let infinite = Rect {
            left: 0.0,
            bottom: 0.0,
            right: f64::INFINITY,
            top: 10.0,
        };
        assert_eq!(
            decide(vec![raster(infinite)]),
            PageVision::vision(VisionReason::Unsupported)
        );
    }

    #[test]
    fn diagonal_detection_tolerates_rounding_only() {
        assert!(!is_diagonal((0.0, 0.0), (100.0, 1.0)));
        assert!(is_diagonal((0.0, 0.0), (100.0, 3.0)));
        assert!(!is_diagonal((0.0, 0.0), (0.5, 100.0)));
        assert!(!is_diagonal((5.0, 5.0), (5.0, 5.0)));
    }

    #[test]
    fn quarter_turns_preserve_axes_but_rotations_do_not() {
        let identity = PdfMatrix::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let scaled = PdfMatrix::new(0.5, 0.0, 0.0, 2.0, 10.0, 20.0);
        let quarter = PdfMatrix::new(0.0, 1.0, -1.0, 0.0, 0.0, 0.0);
        let rotated = PdfMatrix::new(0.866, 0.5, -0.5, 0.866, 0.0, 0.0);
        let skewed = PdfMatrix::new(1.0, 0.0, 0.3, 1.0, 0.0, 0.0);
        assert!(preserves_axes(&identity));
        assert!(preserves_axes(&scaled));
        assert!(preserves_axes(&quarter));
        assert!(!preserves_axes(&rotated));
        assert!(!preserves_axes(&skewed));
    }

    #[test]
    fn blank_text_always_falls_back_to_the_image() {
        assert!(page_uses_image(true, Some("text")));
        assert!(!page_uses_image(false, Some("text")));
        assert!(page_uses_image(false, Some("  \n")));
        assert!(page_uses_image(false, None));
    }
}
