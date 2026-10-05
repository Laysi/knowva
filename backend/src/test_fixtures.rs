use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

pub struct FixtureSet {
    pub native_pdf: PathBuf,
    pub scanned_pdf: PathBuf,
    pub docx: PathBuf,
    pub pptx: PathBuf,
    pub xlsx: PathBuf,
    pub markdown: PathBuf,
    pub text: PathBuf,
    pub hybrid_retrieval: PathBuf,
}

pub fn write_all(root: &Path) -> anyhow::Result<FixtureSet> {
    std::fs::create_dir_all(root)?;
    let native_pdf = root.join("native-two-page.pdf");
    let scanned_pdf = root.join("scanned-one-page.pdf");
    let docx = root.join("ordered-two-page.docx");
    let pptx = root.join("ordered-two-slide.pptx");
    let xlsx = root.join("single-sheet.xlsx");
    let markdown = root.join("headings.md");
    let text = root.join("plain.txt");
    let hybrid_retrieval = root.join("hybrid-retrieval.md");

    std::fs::write(&native_pdf, native_pdf_bytes())?;
    std::fs::write(&scanned_pdf, scanned_pdf_bytes())?;
    write_zip(&docx, docx_entries())?;
    write_zip(&pptx, pptx_entries())?;
    write_zip(&xlsx, xlsx_entries())?;
    std::fs::write(
        &markdown,
        "# 第一章\nMarkdown fixture 第一段。\n\n## 第二章\nMarkdown fixture 第二段。\n",
    )?;
    std::fs::write(&text, "TXT fixture 第一段。\n\nTXT fixture 第二段。\n")?;
    std::fs::write(
        &hybrid_retrieval,
        "# 繁中語意詞\n人工智慧助理可以協助企業整理知識；查詢可使用同義縮寫 AI。\n\n\
         ## 精確產品代號\n設備型號 NX-2048-V2 支援 2048 維向量。\n\n\
         ## 重複頁面 A\n此段是刻意重複的檢索內容。\n\n\
         ## 重複頁面 B\n此段是刻意重複的檢索內容。\n",
    )?;

    Ok(FixtureSet {
        native_pdf,
        scanned_pdf,
        docx,
        pptx,
        xlsx,
        markdown,
        text,
        hybrid_retrieval,
    })
}

fn write_zip(path: &Path, entries: Vec<(&'static str, &'static str)>) -> anyhow::Result<()> {
    let mut archive = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .unix_permissions(0o644);
    for (name, content) in entries {
        archive.start_file(name, options)?;
        archive.write_all(content.as_bytes())?;
    }
    let bytes = archive.finish()?.into_inner();
    std::fs::write(path, bytes)?;
    Ok(())
}

fn finish_pdf(objects: Vec<Vec<u8>>) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets = Vec::with_capacity(objects.len());
    for (index, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        writeln!(&mut pdf, "{} 0 obj", index + 1).expect("write to Vec");
        pdf.extend_from_slice(object);
        if !object.ends_with(b"\n") {
            pdf.push(b'\n');
        }
        pdf.extend_from_slice(b"endobj\n");
    }
    let xref_offset = pdf.len();
    writeln!(&mut pdf, "xref\n0 {}", objects.len() + 1).expect("write to Vec");
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in offsets {
        writeln!(&mut pdf, "{offset:010} 00000 n ").expect("write to Vec");
    }
    write!(
        &mut pdf,
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n",
        objects.len() + 1
    )
    .expect("write to Vec");
    pdf
}

fn stream_object(dictionary: &str, data: &[u8]) -> Vec<u8> {
    let mut object = format!("<< {dictionary} /Length {} >>\nstream\n", data.len()).into_bytes();
    object.extend_from_slice(data);
    object.extend_from_slice(b"\nendstream\n");
    object
}

fn native_pdf_bytes() -> Vec<u8> {
    finish_pdf(vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R >>".to_vec(),
        stream_object(
            "",
            b"BT /F1 24 Tf 72 700 Td (Native page one) Tj ET \
              BT /F1 12 Tf 72 670 Td (Quarterly inventory summary for the north warehouse.) Tj ET",
        ),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>".to_vec(),
        stream_object(
            "",
            b"BT /F1 24 Tf 72 700 Td (Native page two) Tj ET \
              BT /F1 12 Tf 72 670 Td (Quarterly inventory summary for the south warehouse.) Tj ET",
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
    ])
}

fn scanned_pdf_bytes() -> Vec<u8> {
    finish_pdf(vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>".to_vec(),
        stream_object("", b"q 612 0 0 792 0 0 cm /Im0 Do Q"),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8",
            &[32, 96, 160],
        ),
    ])
}

/// PDF with a known `needs_vision` decision for each page, in order.
pub struct VisionFixture {
    pub name: &'static str,
    pub path: PathBuf,
    pub needs_vision: Vec<bool>,
}

/// Writes the page-vision calibration fixtures. Every page carries the same
/// sentence-length text layer, so only the drawing decides the outcome.
pub fn write_vision_fixtures(root: &Path) -> anyhow::Result<Vec<VisionFixture>> {
    std::fs::create_dir_all(root)?;
    let pixel = || {
        stream_object(
            "/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8",
            &[32, 96, 160],
        )
    };
    // A 400 x 400 pt image inside a form; the page scales the form down.
    let image_form = || {
        stream_object(
            "/Type /XObject /Subtype /Form /BBox [0 0 400 400] /Resources << /XObject << /Im0 6 0 R >> >>",
            b"q 400 0 0 400 0 0 cm /Im0 Do Q",
        )
    };
    // A 2 x 1 image, so its stream differs from `pixel`.
    let bar = || {
        stream_object(
            "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8",
            &[200, 30, 30, 30, 30, 200],
        )
    };
    // The same stream as `bar` with its colors inverted by /Decode.
    let inverted_bar = || {
        stream_object(
            "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 \
             /Decode [1 0 1 0 1 0]",
            &[200, 30, 30, 30, 30, 200],
        )
    };
    // A stencil mask, painted in whatever fill color is current.
    let stencil = || {
        stream_object(
            "/Type /XObject /Subtype /Image /Width 8 /Height 1 /ImageMask true /BitsPerComponent 1",
            &[0b1010_1010],
        )
    };
    let cases: Vec<(&'static str, Vec<bool>, Vec<u8>)> = vec![
        (
            "text-page",
            vec![false],
            vision_page_pdf(b"", "", "", Vec::new()),
        ),
        (
            "table-grid",
            vec![false],
            vision_page_pdf(
                b"0 0 0 RG 0.75 w 72 300 468 300 re S \
                  72 400 m 540 400 l S 72 500 m 540 500 l S 306 300 m 306 600 l S",
                "",
                "",
                Vec::new(),
            ),
        ),
        (
            "bar-chart",
            vec![true],
            vision_page_pdf(
                b"0.2 0.4 0.8 rg 100 100 60 200 re f 200 100 60 300 re f 300 100 60 150 re f",
                "",
                "",
                Vec::new(),
            ),
        ),
        (
            "line-chart",
            vec![true],
            vision_page_pdf(
                b"0 0 0 RG 1 w 72 100 m 72 400 l S 72 100 m 540 100 l S \
                  0.8 0.1 0.1 RG 2 w 100 150 m 200 300 l 300 220 l 400 380 l 500 260 l S",
                "",
                "",
                Vec::new(),
            ),
        ),
        (
            "background-and-logo",
            vec![false],
            vision_page_pdf(
                b"q 0.95 0.95 1 rg 0 0 612 792 re f Q q 30 0 0 30 520 740 cm /Im0 Do Q",
                "/XObject << /Im0 6 0 R >>",
                "",
                vec![pixel()],
            ),
        ),
        (
            "form-image-half",
            vec![true],
            vision_page_pdf(
                b"q 0.5 0 0 0.5 100 100 cm /Fm0 Do Q",
                "/XObject << /Fm0 7 0 R >>",
                "",
                vec![pixel(), image_form()],
            ),
        ),
        (
            "form-image-tenth",
            vec![false],
            vision_page_pdf(
                b"q 0.1 0 0 0.1 100 100 cm /Fm0 Do Q",
                "/XObject << /Fm0 7 0 R >>",
                "",
                vec![pixel(), image_form()],
            ),
        ),
        (
            "stamp-annotation",
            vec![true],
            vision_page_pdf(
                b"",
                "",
                "/Annots [6 0 R]",
                vec![
                    b"<< /Type /Annot /Subtype /Stamp /Rect [100 100 260 160] /Name /Approved >>"
                        .to_vec(),
                ],
            ),
        ),
        (
            "link-annotation-with-appearance",
            vec![true],
            vision_page_pdf(
                b"",
                "",
                "/Annots [6 0 R]",
                vec![
                    b"<< /Type /Annot /Subtype /Link /Rect [72 100 232 140] /Border [0 0 0] \
                       /A << /S /URI /URI (https://example.com/) >> /AP << /N 7 0 R >> >>"
                        .to_vec(),
                    stream_object(
                        "/Type /XObject /Subtype /Form /BBox [0 0 160 40]",
                        b"0.9 0.2 0.2 rg 0 0 160 40 re f",
                    ),
                ],
            ),
        ),
        (
            "link-annotation",
            vec![false],
            vision_page_pdf(
                b"",
                "",
                "/Annots [6 0 R]",
                vec![
                    b"<< /Type /Annot /Subtype /Link /Rect [72 715 400 735] /Border [0 0 0] \
                       /A << /S /URI /URI (https://example.com/) >> >>"
                        .to_vec(),
                ],
            ),
        ),
        (
            // A slide-master footer bar (7.6% of the page) on every page; only
            // page 3 adds its own figure. Pages 3-4 draw a separate but
            // identical copy of the bar, so identity is content, not object.
            "template-deck",
            vec![false, false, true, false],
            vision_document_pdf(
                &[
                    b"q 612 0 0 60 0 0 cm /Bar Do Q",
                    b"q 612 0 0 60 0 0 cm /Bar Do Q",
                    b"q 612 0 0 60 0 0 cm /Copy Do Q q 300 0 0 300 156 300 cm /Fig Do Q",
                    b"q 612 0 0 60 0 0 cm /Copy Do Q",
                ],
                "/XObject << /Bar 12 0 R /Copy 13 0 R /Fig 14 0 R >>",
                "",
                vec![bar(), bar(), pixel()],
            ),
        ),
        (
            // A full-page image repeated on every page, like a scanned form
            // under different filled-in values, is never page furniture.
            "repeated-page-background",
            vec![true, true, true],
            vision_document_pdf(
                &[b"q 612 0 0 792 0 0 cm /Bg Do Q".as_slice(); 3],
                "/XObject << /Bg 10 0 R >>",
                "",
                vec![bar()],
            ),
        ),
        (
            // A slide master's dark diagonal band (6.2%) on every page; only
            // page 3 adds its own bar chart.
            "vector-template",
            vec![false, false, true, false],
            vision_document_pdf(
                &[
                    b"0.1 0.2 0.4 rg 0 0 m 300 0 l 0 200 l h f",
                    b"0.1 0.2 0.4 rg 0 0 m 300 0 l 0 200 l h f",
                    b"0.1 0.2 0.4 rg 0 0 m 300 0 l 0 200 l h f \
                      0.8 0.2 0.2 rg 350 300 60 200 re f 430 300 60 200 re f 510 300 60 200 re f",
                    b"0.1 0.2 0.4 rg 0 0 m 300 0 l 0 200 l h f",
                ],
                "",
                "",
                Vec::new(),
            ),
        ),
        (
            // The master footer bar on every page, with a note written over
            // it on page 2 only; text is masked when appearances are compared.
            "text-over-template",
            vec![false; 4],
            vision_document_pdf(
                &[
                    b"q 612 0 0 60 0 0 cm /Bar Do Q",
                    b"q 612 0 0 60 0 0 cm /Bar Do Q \
                      BT /F1 12 Tf 72 24 Td (Footer note written over the master bar.) Tj ET",
                    b"q 612 0 0 60 0 0 cm /Bar Do Q",
                    b"q 612 0 0 60 0 0 cm /Bar Do Q",
                ],
                "/XObject << /Bar 12 0 R >>",
                "",
                vec![bar()],
            ),
        ),
        (
            // The same bar clipped to its left half on pages 1-2 and to its
            // right half on pages 3-4.
            "clip-variants",
            vec![true; 4],
            vision_document_pdf(
                &[
                    b"q 0 0 306 60 re W n 612 0 0 60 0 0 cm /Bar Do Q",
                    b"q 0 0 306 60 re W n 612 0 0 60 0 0 cm /Bar Do Q",
                    b"q 306 0 306 60 re W n 612 0 0 60 0 0 cm /Bar Do Q",
                    b"q 306 0 306 60 re W n 612 0 0 60 0 0 cm /Bar Do Q",
                ],
                "/XObject << /Bar 12 0 R >>",
                "",
                vec![bar()],
            ),
        ),
        (
            // The decode variants again, under an invisible OCR-like text
            // layer on page 1 that hides the whole image from the comparison.
            "text-covered-variants",
            vec![true; 4],
            vision_document_pdf(
                &[
                    b"q 300 0 0 60 100 400 cm /Bar Do Q \
                      BT 3 Tr /F1 72 Tf 90 400 Td (MMMMMM) Tj 0 20 Td (MMMMMM) Tj \
                      0 20 Td (MMMMMM) Tj 0 20 Td (MMMMMM) Tj ET",
                    b"q 300 0 0 60 100 400 cm /Bar Do Q",
                    b"q 300 0 0 60 100 400 cm /Inv Do Q",
                    b"q 300 0 0 60 100 400 cm /Inv Do Q",
                ],
                "/XObject << /Bar 12 0 R /Inv 13 0 R >>",
                "",
                vec![bar(), inverted_bar()],
            ),
        ),
        (
            // Every page draws the same stream at the same place, but pages
            // 3-4 invert it with /Decode: two images on two pages each.
            "decode-variants",
            vec![true; 4],
            vision_document_pdf(
                &[
                    b"q 612 0 0 60 0 0 cm /Bar Do Q",
                    b"q 612 0 0 60 0 0 cm /Bar Do Q",
                    b"q 612 0 0 60 0 0 cm /Inv Do Q",
                    b"q 612 0 0 60 0 0 cm /Inv Do Q",
                ],
                "/XObject << /Bar 12 0 R /Inv 13 0 R >>",
                "",
                vec![bar(), inverted_bar()],
            ),
        ),
        (
            // The same image in the same 150 pt square on every page, mirrored
            // on pages 3-4: the bounds match, the drawing does not.
            "flipped-copies",
            vec![true; 4],
            vision_document_pdf(
                &[
                    b"q 150 0 0 150 231 321 cm /Bar Do Q",
                    b"q 150 0 0 150 231 321 cm /Bar Do Q",
                    b"q -150 0 0 150 381 321 cm /Bar Do Q",
                    b"q -150 0 0 150 381 321 cm /Bar Do Q",
                ],
                "/XObject << /Bar 12 0 R >>",
                "",
                vec![bar()],
            ),
        ),
        (
            // The same stream under a different soft mask on pages 3-4.
            "soft-mask-variants",
            vec![true; 4],
            vision_document_pdf(
                &[
                    b"q 612 0 0 60 0 0 cm /A Do Q",
                    b"q 612 0 0 60 0 0 cm /A Do Q",
                    b"q 612 0 0 60 0 0 cm /B Do Q",
                    b"q 612 0 0 60 0 0 cm /B Do Q",
                ],
                "/XObject << /A 12 0 R /B 13 0 R >>",
                "",
                vec![
                    stream_object(
                        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 /SMask 14 0 R",
                        &[200, 30, 30, 30, 30, 200],
                    ),
                    stream_object(
                        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 /SMask 15 0 R",
                        &[200, 30, 30, 30, 30, 200],
                    ),
                    stream_object(
                        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8",
                        &[255, 0],
                    ),
                    stream_object(
                        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8",
                        &[0, 255],
                    ),
                ],
            ),
        ),
        (
            // The same stream under a different explicit stencil /Mask on
            // pages 3-4.
            "explicit-mask-variants",
            vec![true; 4],
            vision_document_pdf(
                &[
                    b"q 612 0 0 60 0 0 cm /A Do Q",
                    b"q 612 0 0 60 0 0 cm /A Do Q",
                    b"q 612 0 0 60 0 0 cm /B Do Q",
                    b"q 612 0 0 60 0 0 cm /B Do Q",
                ],
                "/XObject << /A 12 0 R /B 13 0 R >>",
                "",
                vec![
                    stream_object(
                        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Mask 14 0 R",
                        &[200, 30, 30, 30, 30, 200],
                    ),
                    stream_object(
                        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Mask 15 0 R",
                        &[200, 30, 30, 30, 30, 200],
                    ),
                    stream_object(
                        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ImageMask true /BitsPerComponent 1",
                        &[0b1000_0000],
                    ),
                    stream_object(
                        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ImageMask true /BitsPerComponent 1",
                        &[0b0100_0000],
                    ),
                ],
            ),
        ),
        (
            // One stencil mask on every page, red on pages 1-2 and blue on
            // pages 3-4.
            "stencil-colors",
            vec![true; 4],
            vision_document_pdf(
                &[
                    b"1 0 0 rg q 612 0 0 60 0 0 cm /Mask Do Q 0 g",
                    b"1 0 0 rg q 612 0 0 60 0 0 cm /Mask Do Q 0 g",
                    b"0 0 1 rg q 612 0 0 60 0 0 cm /Mask Do Q 0 g",
                    b"0 0 1 rg q 612 0 0 60 0 0 cm /Mask Do Q 0 g",
                ],
                "/XObject << /Mask 12 0 R >>",
                "",
                vec![stencil()],
            ),
        ),
    ];
    cases
        .into_iter()
        .map(|(name, needs_vision, bytes)| {
            let path = root.join(format!("vision-{name}.pdf"));
            std::fs::write(&path, bytes)?;
            Ok(VisionFixture {
                name,
                path,
                needs_vision,
            })
        })
        .collect()
}

/// Objects 1-5 are fixed (catalog, pages, page, contents, font); `extra_objects`
/// start at object 6.
fn vision_page_pdf(
    drawing: &[u8],
    extra_resources: &str,
    annotations: &str,
    extra_objects: Vec<Vec<u8>>,
) -> Vec<u8> {
    vision_document_pdf(&[drawing], extra_resources, annotations, extra_objects)
}

/// One page per drawing, all sharing `extra_resources` and `annotations`.
/// Objects 1-2 are the catalog and page tree, then each page and its contents,
/// then the font; `extra_objects` start at object `2 * pages + 4`.
fn vision_document_pdf(
    drawings: &[&[u8]],
    extra_resources: &str,
    annotations: &str,
    extra_objects: Vec<Vec<u8>>,
) -> Vec<u8> {
    let font = 2 * drawings.len() + 3;
    let kids = (0..drawings.len())
        .map(|index| format!("{} 0 R", 2 * index + 3))
        .collect::<Vec<_>>()
        .join(" ");
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        format!(
            "<< /Type /Pages /Kids [{kids}] /Count {} >>",
            drawings.len()
        )
        .into_bytes(),
    ];
    for (index, drawing) in drawings.iter().enumerate() {
        let mut content = drawing.to_vec();
        content.extend_from_slice(
            b"\nBT /F1 12 Tf 72 720 Td (Quarterly inventory summary for the north warehouse.) Tj ET",
        );
        objects.push(
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /Font << /F1 {font} 0 R >> {extra_resources} >> /Contents {} 0 R {annotations} >>",
                2 * index + 4
            )
            .into_bytes(),
        );
        objects.push(stream_object("", &content));
    }
    objects.push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec());
    objects.extend(extra_objects);
    finish_pdf(objects)
}

fn package_relationships(target: &'static str) -> &'static str {
    match target {
        "word/document.xml" => {
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"#
        }
        "ppt/presentation.xml" => {
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"#
        }
        "xl/workbook.xml" => {
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#
        }
        _ => unreachable!("fixture target"),
    }
}

fn docx_entries() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "[Content_Types].xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"#,
        ),
        ("_rels/.rels", package_relationships("word/document.xml")),
        (
            "word/document.xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:body>
  <w:p><w:r><w:t>DOCX page one: the maintenance log lists every inspected pump.</w:t></w:r></w:p>
  <w:p><w:r><w:br w:type="page"/></w:r></w:p>
  <w:p><w:r><w:t>DOCX page two: the maintenance log lists every replaced valve.</w:t></w:r></w:p>
  <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
 </w:body>
</w:document>"#,
        ),
    ]
}

fn pptx_entries() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "[Content_Types].xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
 <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
 <Override PartName="/ppt/slides/slide2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
</Types>"#,
        ),
        ("_rels/.rels", package_relationships("ppt/presentation.xml")),
        (
            "ppt/presentation.xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
 <p:sldIdLst><p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/></p:sldIdLst>
 <p:sldSz cx="12192000" cy="6858000" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"#,
        ),
        (
            "ppt/_rels/presentation.xml.rels",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/>
</Relationships>"#,
        ),
        ("ppt/slides/slide1.xml", slide_xml("PPTX slide one")),
        ("ppt/slides/slide2.xml", slide_xml("PPTX slide two")),
    ]
}

fn slide_xml(text: &'static str) -> &'static str {
    match text {
        "PPTX slide one" => SLIDE_ONE,
        "PPTX slide two" => SLIDE_TWO,
        _ => unreachable!("fixture slide"),
    }
}

const SLIDE_ONE: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr><p:sp><p:nvSpPr><p:cNvPr id="2" name="Text 1"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>PPTX slide one: launch checklist for the warehouse team.</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"#;

const SLIDE_TWO: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr><p:sp><p:nvSpPr><p:cNvPr id="2" name="Text 2"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>PPTX slide two: rollout owners and the delivery schedule.</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"#;

fn xlsx_entries() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "[Content_Types].xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
 <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"#,
        ),
        ("_rels/.rels", package_relationships("xl/workbook.xml")),
        (
            "xl/workbook.xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
        ),
        (
            "xl/_rels/workbook.xml.rels",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#,
        ),
        (
            "xl/worksheets/sheet1.xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>XLSX first row: sensor inventory for line A</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>XLSX second row: sensor inventory for line B</t></is></c></row></sheetData><pageSetup orientation="portrait" fitToWidth="1" fitToHeight="1"/></worksheet>"#,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_sources_have_expected_container_shape() -> anyhow::Result<()> {
        let root = std::env::temp_dir().join(format!("knowva-fixtures-{}", uuid::Uuid::new_v4()));
        let fixtures = write_all(&root)?;
        assert!(fixtures.native_pdf.metadata()?.len() > 100);
        assert!(fixtures.scanned_pdf.metadata()?.len() > 100);
        for office in [&fixtures.docx, &fixtures.pptx, &fixtures.xlsx] {
            let archive = zip::ZipArchive::new(std::fs::File::open(office)?)?;
            assert!(archive.len() >= 3);
        }
        assert!(fixtures.markdown.is_file());
        assert!(fixtures.text.is_file());
        let hybrid = std::fs::read_to_string(&fixtures.hybrid_retrieval)?;
        assert!(hybrid.contains("人工智慧"));
        assert!(hybrid.contains("AI"));
        assert!(hybrid.contains("NX-2048-V2"));
        assert_eq!(hybrid.matches("此段是刻意重複的檢索內容").count(), 2);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
