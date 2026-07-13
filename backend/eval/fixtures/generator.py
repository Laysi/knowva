#!/usr/bin/env python3
"""Build the deterministic, synthetic evaluation dataset without third-party packages."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import sys
import zipfile
import zlib
from html import escape
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PDF_SIZE = (612, 792)
LICENSE = "CC0-1.0"
SOURCE = "Generated synthetic fixture; no external data or production content."

# A single pre-rendered Traditional Chinese bitmap title. It is embedded so that
# regenerating the scanned fixture needs only the Python standard library.
CJK_SCAN_TITLE = (
    183,
    39,
    "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",
)

FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    " ": ("00000",) * 7,
}


class Canvas:
    def __init__(self, width: int, height: int, color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = bytearray(color * (width * height))

    def rectangle(self, x: int, y: int, width: int, height: int, color):
        for row in range(max(0, y), min(self.height, y + height)):
            start = (row * self.width + max(0, x)) * 3
            end = (row * self.width + min(self.width, x + width)) * 3
            self.pixels[start:end] = bytes(color) * ((end - start) // 3)

    def text(self, x: int, y: int, value: str, scale=4, color=(20, 35, 60)):
        cursor = x
        for character in value.upper():
            glyph = FONT.get(character, FONT[" "])
            for row, bits in enumerate(glyph):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        self.rectangle(
                            cursor + column * scale,
                            y + row * scale,
                            scale,
                            scale,
                            color,
                        )
            cursor += 6 * scale

    def bitmap_title(self, x: int, y: int, scale=2, color=(15, 55, 110)):
        width, height, encoded = CJK_SCAN_TITLE
        mask = zlib.decompress(base64.b64decode(encoded))
        if len(mask) != width * height:
            raise ValueError("embedded bitmap has unexpected dimensions")
        for row in range(height):
            for column in range(width):
                gray = mask[row * width + column]
                if gray < 128:
                    self.rectangle(
                        x + column * scale,
                        y + row * scale,
                        scale,
                        scale,
                        color,
                    )

    def raw(self) -> bytes:
        return bytes(self.pixels)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def make_png(canvas: Canvas) -> bytes:
    scanlines = b"".join(
        b"\x00" + canvas.raw()[row * canvas.width * 3 : (row + 1) * canvas.width * 3]
        for row in range(canvas.height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", canvas.width, canvas.height, 8, 2, 0, 0, 0),
        )
        + png_chunk(b"IDAT", zlib.compress(scanlines, 9))
        + png_chunk(b"IEND", b"")
    )


class Pdf:
    def __init__(self):
        self.objects: list[bytes | None] = []

    def reserve(self) -> int:
        self.objects.append(None)
        return len(self.objects)

    def add(self, value: bytes) -> int:
        self.objects.append(value)
        return len(self.objects)

    def set(self, object_id: int, value: bytes):
        self.objects[object_id - 1] = value

    def finish(self, root: int) -> bytes:
        output = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for object_id, value in enumerate(self.objects, 1):
            if value is None:
                raise ValueError(f"PDF object {object_id} was not initialized")
            offsets.append(len(output))
            output.extend(f"{object_id} 0 obj\n".encode())
            output.extend(value)
            output.extend(b"\nendobj\n")
        xref = len(output)
        output.extend(f"xref\n0 {len(offsets)}\n".encode())
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode())
        output.extend(
            (
                f"trailer\n<< /Size {len(offsets)} /Root {root} 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n"
            ).encode()
        )
        return bytes(output)


def pdf_stream(data: bytes, dictionary: bytes = b"") -> bytes:
    return (
        b"<< /Length "
        + str(len(data)).encode()
        + (b" " + dictionary if dictionary else b"")
        + b" >>\nstream\n"
        + data
        + b"\nendstream"
    )


def cjk_hex(value: str) -> bytes:
    return value.encode("utf-16-be").hex().upper().encode()


def native_page_content(lines: list[str], drawing=b"") -> bytes:
    commands = [b"BT /F1 18 Tf 54 735 Td"]
    for index, line in enumerate(lines):
        if index:
            commands.append(b"0 -38 Td")
        commands.append(b"<" + cjk_hex(line) + b"> Tj")
    commands.append(b"ET")
    if drawing:
        commands.append(drawing)
    return b"\n".join(commands)


def make_native_pdf(pages: list[tuple[list[str], bytes]]) -> bytes:
    pdf = Pdf()
    catalog = pdf.reserve()
    pages_object = pdf.reserve()
    to_unicode = pdf.add(
        pdf_stream(
            b"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 beginbfrange
<0000> <FFFF> <0000>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end"""
        )
    )
    descendant = pdf.add(
        b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /MSung-Light "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (CNS1) /Supplement 7 >> "
        b"/DW 1000 >>"
    )
    font = pdf.add(
        (
            f"<< /Type /Font /Subtype /Type0 /BaseFont /MSung-Light "
            f"/Encoding /UniCNS-UCS2-H /DescendantFonts [{descendant} 0 R] "
            f"/ToUnicode {to_unicode} 0 R >>"
        ).encode()
    )
    page_ids = []
    for lines, drawing in pages:
        page = pdf.reserve()
        content = pdf.add(pdf_stream(native_page_content(lines, drawing)))
        pdf.set(
            page,
            (
                f"<< /Type /Page /Parent {pages_object} 0 R "
                f"/MediaBox [0 0 {PDF_SIZE[0]} {PDF_SIZE[1]}] "
                f"/Resources << /Font << /F1 {font} 0 R >> >> "
                f"/Contents {content} 0 R >>"
            ).encode(),
        )
        page_ids.append(page)
    kids = " ".join(f"{page} 0 R" for page in page_ids)
    pdf.set(pages_object, f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode())
    pdf.set(catalog, f"<< /Type /Catalog /Pages {pages_object} 0 R >>".encode())
    return pdf.finish(catalog)


def make_scanned_pdf(canvases: list[Canvas]) -> bytes:
    pdf = Pdf()
    catalog = pdf.reserve()
    pages_object = pdf.reserve()
    page_ids = []
    for canvas in canvases:
        image = pdf.add(
            pdf_stream(
                zlib.compress(canvas.raw(), 9),
                (
                    f"/Type /XObject /Subtype /Image /Width {canvas.width} "
                    f"/Height {canvas.height} /ColorSpace /DeviceRGB "
                    f"/BitsPerComponent 8 /Filter /FlateDecode"
                ).encode(),
            )
        )
        content = pdf.add(pdf_stream(b"q 612 0 0 792 0 0 cm /Im1 Do Q"))
        page = pdf.reserve()
        pdf.set(
            page,
            (
                f"<< /Type /Page /Parent {pages_object} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /XObject << /Im1 {image} 0 R >> >> "
                f"/Contents {content} 0 R >>"
            ).encode(),
        )
        page_ids.append(page)
    kids = " ".join(f"{page} 0 R" for page in page_ids)
    pdf.set(pages_object, f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode())
    pdf.set(catalog, f"<< /Type /Catalog /Pages {pages_object} 0 R >>".encode())
    return pdf.finish(catalog)


def scanned_pages() -> list[Canvas]:
    first = Canvas(900, 1165, (246, 248, 251))
    first.rectangle(0, 0, 900, 180, (222, 232, 244))
    first.bitmap_title(65, 55, 2)
    first.text(90, 300, "NORTH HELMET STOCK: 37", 5)
    first.text(90, 430, "SOUTH HELMET STOCK: 52", 5)
    first.rectangle(90, 570, 650, 8, (55, 105, 165))
    first.text(90, 650, "SYNTHETIC ZH-TW SCAN", 4, (70, 70, 70))

    second = Canvas(900, 1165, (246, 248, 251))
    second.rectangle(0, 0, 900, 180, (229, 238, 226))
    second.bitmap_title(65, 55, 2, (30, 95, 65))
    second.text(75, 310, "LAST INSPECTION: 2026-05-18", 4)
    second.text(75, 445, "NEXT INSPECTION: 2026-08-18", 4)
    second.rectangle(75, 590, 700, 8, (45, 120, 80))
    second.text(75, 670, "IMAGE ONLY - NO TEXT LAYER", 4, (70, 70, 70))
    return [first, second]


def energy_dashboard_png() -> bytes:
    canvas = Canvas(800, 520, (248, 250, 252))
    canvas.rectangle(0, 0, 800, 100, (28, 72, 112))
    canvas.text(50, 30, "ZH-TW ENERGY DASHBOARD", 5, (245, 250, 255))
    canvas.text(60, 150, "SOLAR: 64%", 6, (25, 100, 60))
    canvas.text(60, 250, "WIND: 36%", 6, (45, 95, 155))
    canvas.rectangle(520, 155, 180, 55, (45, 155, 90))
    canvas.rectangle(520, 255, 101, 55, (70, 125, 190))
    canvas.text(60, 380, "GREEN = RENEWABLE", 4, (25, 100, 60))
    return make_png(canvas)


def slide_xml(title: str, lines: list[str]) -> str:
    paragraphs = "".join(
        (
            "<a:p><a:r><a:rPr lang=\"zh-TW\" sz=\"2400\"/>"
            f"<a:t>{escape(line)}</a:t></a:r><a:endParaRPr lang=\"zh-TW\"/></a:p>"
        )
        for line in lines
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
 <p:cSld><p:spTree>
  <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
  <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>
   <a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
  <p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
   <p:spPr><a:xfrm><a:off x="457200" y="274320"/><a:ext cx="8229600" cy="914400"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>
   <p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="zh-TW" sz="3200" b="1"/>
    <a:t>{escape(title)}</a:t></a:r><a:endParaRPr lang="zh-TW"/></a:p></p:txBody>
  </p:sp>
  <p:sp><p:nvSpPr><p:cNvPr id="3" name="Body"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
   <p:spPr><a:xfrm><a:off x="685800" y="1524000"/><a:ext cx="7772400" cy="4114800"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>
   <p:txBody><a:bodyPr/><a:lstStyle/>{paragraphs}</p:txBody>
  </p:sp>
 </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def make_pptx() -> bytes:
    slides = [
        slide_xml("晨星計畫啟動簡報", ["第 1 頁：核准範圍", "產品：知識助理", "狀態：已核准"]),
        slide_xml(
            "發佈窗口",
            ["第 2 頁：正式資料", "日期：2026-11-05", "時間：09:30", "負責人：王雅雯"],
        ),
        slide_xml(
            "回復條件",
            ["第 3 頁：監控", "錯誤率超過 2.5% 即回復", "通知頻道：blue-alert"],
        ),
    ]
    overrides = "".join(
        f'<Override PartName="/ppt/slides/slide{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for index in range(1, len(slides) + 1)
    )
    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
 <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
 <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
 <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
 {overrides}
</Types>"""
    slide_ids = "".join(
        f'<p:sldId id="{255 + index}" r:id="rId{index + 1}"/>'
        for index in range(1, len(slides) + 1)
    )
    presentation = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
 <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
 <p:sldIdLst>{slide_ids}</p:sldIdLst>
 <p:sldSz cx="9144000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""
    presentation_rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    presentation_rels.extend(
        f'<Relationship Id="rId{index + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{index}.xml"/>'
        for index in range(1, len(slides) + 1)
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(presentation_rels)
        + "</Relationships>"
    )
    package = {
        "[Content_Types].xml": content_types,
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>""",
        "ppt/presentation.xml": presentation,
        "ppt/_rels/presentation.xml.rels": relationships,
        "ppt/slideMasters/slideMaster1.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
 <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
 <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
 </p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>
 <p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>""",
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>""",
        "ppt/slideLayouts/slideLayout1.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank">
 <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
 <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
 </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>""",
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>""",
        "ppt/theme/theme1.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Synthetic">
 <a:themeElements><a:clrScheme name="Synthetic"><a:dk1><a:srgbClr val="202020"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F4E79"/></a:dk2><a:lt2><a:srgbClr val="EAF2F8"/></a:lt2><a:accent1><a:srgbClr val="2E75B6"/></a:accent1><a:accent2><a:srgbClr val="70AD47"/></a:accent2><a:accent3><a:srgbClr val="ED7D31"/></a:accent3><a:accent4><a:srgbClr val="A5A5A5"/></a:accent4><a:accent5><a:srgbClr val="5B9BD5"/></a:accent5><a:accent6><a:srgbClr val="FFC000"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme>
 <a:fontScheme name="Synthetic"><a:majorFont><a:latin typeface="Arial"/><a:ea typeface="Microsoft JhengHei"/><a:cs typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/><a:ea typeface="Microsoft JhengHei"/><a:cs typeface="Arial"/></a:minorFont></a:fontScheme>
 <a:fmtScheme name="Synthetic"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements>
</a:theme>""",
    }
    for index, slide in enumerate(slides, 1):
        package[f"ppt/slides/slide{index}.xml"] = slide
        package[f"ppt/slides/_rels/slide{index}.xml.rels"] = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""

    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(package):
            info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, package[name].encode("utf-8"))
    return output.getvalue()


def document_entry(key, path, kind, pages, units, files):
    return {
        "document_key": key,
        "relative_path": path,
        "sha256": hashlib.sha256(files[path]).hexdigest(),
        "license": LICENSE,
        "source": SOURCE,
        "kind": kind,
        "expected": {
            "pages": pages,
            "units": [
                {"unit_id": unit_id, "page": page}
                for unit_id, page in units
            ],
        },
    }


def source(document_key, unit_id, page):
    return {"document_key": document_key, "unit_id": unit_id, "page": page}


def query(
    query_id,
    text,
    scope,
    sources,
    facts,
    answerable=True,
    query_type="fact_lookup",
    difficulty="easy",
    visual=False,
):
    return {
        "query_id": query_id,
        "query": text,
        "document_scope": scope,
        "expected_sources": sources,
        "required_facts": facts,
        "answerable": answerable,
        "type": query_type,
        "difficulty": difficulty,
        "requires_visual": visual,
    }


def build_files() -> dict[str, bytes]:
    north_pdf = make_native_pdf(
        [
            (["北區季度報告", "頁面一：舊專案代號 極光-16", "舊案預算：新臺幣 410 萬元", "狀態：已封存"], b""),
            (["頁面二：目前專案", "專案代號：極光-17", "核准預算：新臺幣 420 萬元", "負責人：林怡君"], b""),
            (["頁面三：審查與期限", "風險審查日期：2026 年 6 月 18 日", "交付期限：2026 年 9 月 30 日"], b""),
        ]
    )
    south_pdf = make_native_pdf(
        [
            (["南區季度報告", "頁面一：舊專案代號 南風-21", "舊案預算：新臺幣 350 萬元", "狀態：已封存"], b""),
            (["頁面二：目前專案", "專案代號：南風-22", "核准預算：新臺幣 360 萬元", "負責人：陳冠宇"], b""),
            (["頁面三：審查與期限", "風險審查日期：2026 年 7 月 2 日", "交付期限：2026 年 10 月 15 日"], b""),
        ]
    )
    chart_one = (
        b"q 0.18 0.45 0.78 rg 110 120 85 180 re f "
        b"0.12 0.62 0.38 rg 245 120 85 270 re f "
        b"0.90 0.48 0.18 rg 380 120 85 225 re f Q"
    )
    chart_two = (
        b"q 0.86 0.92 0.98 rg 80 350 450 55 re f 80 280 450 55 re f "
        b"80 210 450 55 re f 0.1 0.2 0.3 RG 80 210 450 195 re S Q"
    )
    chart_pdf = make_native_pdf(
        [
            (["月度銷售長條圖", "第一季 120；第二季 180；第三季 150", "請依長條高度判斷最高季度"], chart_one),
            (["倉庫庫存表", "A12：42 箱；B07：18 箱；C03：31 箱", "請依表格數值判斷最低庫存"], chart_two),
        ]
    )
    markdown = """# 維運手冊

<!-- unit: chunk-1 -->
## 服務時段

例行維護時段是每週三 22:00 至 23:00（Asia/Taipei）。

<!-- unit: chunk-2 -->
## 升級流程

第一級事件由值班工程師處理；連續 15 分鐘未恢復時，升級給平台主管。
通知代碼為 OPS-AMBER。
""".encode("utf-8")
    text = """[unit:chunk-1]
客服紀錄保留政策
一般客服紀錄保留 180 天；涉及退款爭議的紀錄保留 365 天。

[unit:chunk-2]
匯出規則
資料匯出由隱私窗口核准，核准代碼為 PRIV-EXPORT。
""".encode("utf-8")
    files = {
        "documents/native/north/quarterly-report.pdf": north_pdf,
        "documents/native/south/quarterly-report.pdf": south_pdf,
        "documents/scanned/scanned-warehouse-report.pdf": make_scanned_pdf(scanned_pages()),
        "documents/presentation/launch-plan.pptx": make_pptx(),
        "documents/visual/table-chart.pdf": chart_pdf,
        "documents/image/energy-dashboard.png": energy_dashboard_png(),
        "documents/text/operations.md": markdown,
        "documents/text/support.txt": text,
    }
    documents = [
        document_entry("native-north-quarterly", "documents/native/north/quarterly-report.pdf", "native_pdf", [1, 2, 3], [("page-1", 1), ("page-2", 2), ("page-3", 3)], files),
        document_entry("native-south-quarterly", "documents/native/south/quarterly-report.pdf", "native_pdf", [1, 2, 3], [("page-1", 1), ("page-2", 2), ("page-3", 3)], files),
        document_entry("scanned-warehouse", "documents/scanned/scanned-warehouse-report.pdf", "scanned_pdf", [1, 2], [("page-1", 1), ("page-2", 2)], files),
        document_entry("presentation-launch", "documents/presentation/launch-plan.pptx", "presentation", [1, 2, 3], [("slide-1", 1), ("slide-2", 2), ("slide-3", 3)], files),
        document_entry("visual-table-chart", "documents/visual/table-chart.pdf", "table_chart", [1, 2], [("page-1", 1), ("page-2", 2)], files),
        document_entry("image-energy-dashboard", "documents/image/energy-dashboard.png", "image", [1], [("image-1", 1)], files),
        document_entry("markdown-operations", "documents/text/operations.md", "markdown", [], [("chunk-1", None)], files),
        document_entry("text-support", "documents/text/support.txt", "text", [], [("chunk-1", None)], files),
    ]
    queries = [
        query("native-north-current-project", "北區目前專案的代號、預算與負責人是什麼？", ["native-north-quarterly"], [source("native-north-quarterly", "page-2", 2)], ["專案代號是極光-17", "核准預算是新臺幣 420 萬元", "負責人是林怡君"], difficulty="medium"),
        query("native-north-multi-page", "北區目前專案的代號、風險審查日期與交付期限為何？", ["native-north-quarterly"], [source("native-north-quarterly", "page-2", 2), source("native-north-quarterly", "page-3", 3)], ["專案代號是極光-17", "風險審查日期是 2026 年 6 月 18 日", "交付期限是 2026 年 9 月 30 日"], query_type="multi_source", difficulty="hard"),
        query("native-same-filename-isolation", "南區季度報告中目前專案代號是什麼？", ["native-south-quarterly"], [source("native-south-quarterly", "page-2", 2)], ["專案代號是南風-22"], difficulty="medium"),
        query("native-not-found", "兩份季度報告有列出員工停車位編號嗎？", ["native-north-quarterly", "native-south-quarterly"], [], [], False, "not_found", "easy"),
        query("scanned-helmet-stock", "掃描倉儲報告中，北區與南區安全帽各有幾頂？", ["scanned-warehouse"], [source("scanned-warehouse", "page-1", 1)], ["北區安全帽庫存為 37 頂", "南區安全帽庫存為 52 頂"], query_type="visual_reasoning", difficulty="medium", visual=True),
        query("scanned-not-found", "掃描倉儲報告是否記載手套庫存？", ["scanned-warehouse"], [], [], False, "not_found", "medium", True),
        query("presentation-release-window", "晨星計畫的正式發佈日期、時間與負責人是誰？", ["presentation-launch"], [source("presentation-launch", "slide-2", 2)], ["日期是 2026-11-05", "時間是 09:30", "負責人是王雅雯"], difficulty="medium"),
        query("presentation-not-found", "晨星計畫簡報有提供慶功宴地點嗎？", ["presentation-launch"], [], [], False, "not_found", "easy"),
        query("table-chart-highest-quarter", "銷售長條圖中哪一季最高，數值是多少？", ["visual-table-chart"], [source("visual-table-chart", "page-1", 1)], ["第二季最高", "第二季數值是 180"], query_type="visual_reasoning", difficulty="medium", visual=True),
        query("table-chart-not-found", "圖表文件有列出第四季銷售數值嗎？", ["visual-table-chart"], [], [], False, "not_found", "medium", True),
        query("image-renewable-mix", "能源儀表板中太陽能與風力各占多少？", ["image-energy-dashboard"], [source("image-energy-dashboard", "image-1", 1)], ["太陽能占 64%", "風力占 36%"], query_type="visual_reasoning", difficulty="medium", visual=True),
        query("image-not-found", "能源儀表板有顯示水力發電占比嗎？", ["image-energy-dashboard"], [], [], False, "not_found", "medium", True),
        query("markdown-escalation", "維運手冊規定何時升級事件，通知代碼是什麼？", ["markdown-operations"], [source("markdown-operations", "chunk-1", None)], ["連續 15 分鐘未恢復時升級給平台主管", "通知代碼是 OPS-AMBER"], difficulty="medium"),
        query("markdown-not-found", "維運手冊有說明員工差旅報銷上限嗎？", ["markdown-operations"], [], [], False, "not_found", "easy"),
        query("text-retention", "客服紀錄的一般與退款爭議保留天數各是多少？", ["text-support"], [source("text-support", "chunk-1", None)], ["一般客服紀錄保留 180 天", "退款爭議紀錄保留 365 天"], difficulty="medium"),
        query("text-not-found", "客服政策有列出電話設備採購品牌嗎？", ["text-support"], [], [], False, "not_found", "easy"),
    ]
    manifest = {"version": "1.0", "documents": documents, "queries": queries}
    files["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    return files


def validate_generated(files: dict[str, bytes]):
    manifest = json.loads(files["manifest.json"])
    if manifest["version"] != "1.0":
        raise ValueError("unexpected manifest version")
    document_keys = [document["document_key"] for document in manifest["documents"]]
    query_ids = [query_item["query_id"] for query_item in manifest["queries"]]
    if len(document_keys) != len(set(document_keys)):
        raise ValueError("duplicate document_key")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate query_id")
    documents = {document["document_key"]: document for document in manifest["documents"]}
    for document in manifest["documents"]:
        path = document["relative_path"]
        if hashlib.sha256(files[path]).hexdigest() != document["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        pages = document["expected"]["pages"]
        units = document["expected"]["units"]
        if document["kind"] in {"markdown", "text"}:
            if pages or any(unit["page"] is not None for unit in units):
                raise ValueError(f"unpaged inventory is invalid: {path}")
        else:
            if pages != list(range(1, len(pages) + 1)):
                raise ValueError(f"page inventory is not 1-based: {path}")
        unit_ids = [unit["unit_id"] for unit in units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError(f"duplicate unit_id: {path}")
    for query_item in manifest["queries"]:
        if not query_item["answerable"] and query_item["expected_sources"]:
            raise ValueError(f"negative query has expected sources: {query_item['query_id']}")
        for expected in query_item["expected_sources"]:
            document = documents[expected["document_key"]]
            inventory = {unit["unit_id"]: unit["page"] for unit in document["expected"]["units"]}
            if inventory.get(expected["unit_id"], object()) != expected["page"]:
                raise ValueError(f"source inventory mismatch: {query_item['query_id']}")
    if b"BT" in files["documents/scanned/scanned-warehouse-report.pdf"]:
        raise ValueError("scanned PDF unexpectedly contains a PDF text object")
    with zipfile.ZipFile(BytesIO(files["documents/presentation/launch-plan.pptx"])) as archive:
        if archive.testzip() is not None or "ppt/presentation.xml" not in archive.namelist():
            raise ValueError("PPTX package is invalid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify committed bytes")
    args = parser.parse_args()
    files = build_files()
    validate_generated(files)
    if args.check:
        mismatches = [
            relative_path
            for relative_path, expected in files.items()
            if not (ROOT / relative_path).is_file()
            or (ROOT / relative_path).read_bytes() != expected
        ]
        if mismatches:
            print("out-of-date fixture files: " + ", ".join(mismatches), file=sys.stderr)
            return 1
        print(f"validated {len(files) - 1} documents and manifest.json")
        return 0
    for relative_path, data in files.items():
        destination = ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(f"generated {len(files) - 1} documents and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
