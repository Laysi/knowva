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
        stream_object("", b"BT /F1 24 Tf 72 700 Td (Native page one) Tj ET"),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>".to_vec(),
        stream_object("", b"BT /F1 24 Tf 72 700 Td (Native page two) Tj ET"),
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
  <w:p><w:r><w:t>DOCX page one</w:t></w:r></w:p>
  <w:p><w:r><w:br w:type="page"/></w:r></w:p>
  <w:p><w:r><w:t>DOCX page two</w:t></w:r></w:p>
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
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr><p:sp><p:nvSpPr><p:cNvPr id="2" name="Text 1"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>PPTX slide one</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"#;

const SLIDE_TWO: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr><p:sp><p:nvSpPr><p:cNvPr id="2" name="Text 2"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>PPTX slide two</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"#;

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
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>XLSX first row</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>XLSX second row</t></is></c></row></sheetData><pageSetup orientation="portrait" fitToWidth="1" fitToHeight="1"/></worksheet>"#,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_sources_have_expected_container_shape() -> anyhow::Result<()> {
        let root = std::env::temp_dir().join(format!("km-fixtures-{}", uuid::Uuid::new_v4()));
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
