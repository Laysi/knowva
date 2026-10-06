"""Builds rerank candidate sets shaped like Knowva's: per question, the 50 best
pages of a small corpus in rank order, 7 of them carrying their page image and
the rest sent as text, which the backend batches by modality, 8 at a time.

The ranking is BM25 over the pages' text layers, standing in for Knowva's
dense + BM25 RRF; pages are PNG renders at 180 DPI capped at 1600 px, as the
backend stores them.

    uv run --group tools tools/knowva/build_sets.py DOCS_DIR KNOWVA_BENCH_DIR OUT_DIR
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

DPI = 180
MAX_SIDE = 1600
CANDIDATES = 50
IMAGE_PASSAGES = 7
# Pages Knowva sends as images regardless of rank: example_file's only vision
# page (see CLAUDE.md) and every uploaded image.
VISION_PAGES = {("iiot_industry_report.pdf", 7)}

docs_dir, bench_dir, out_dir = (Path(p) for p in sys.argv[1:4])
pool_dir = out_dir / "pool"
pool_dir.mkdir(parents=True, exist_ok=True)


def capped(image):
    scale = MAX_SIDE / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return image


pool = []
for path in sorted(docs_dir.iterdir()):
    if path.suffix == ".pdf":
        pdf = pdfium.PdfDocument(path)
        for index in range(len(pdf)):
            page = pdf[index]
            text = page.get_textpage().get_text_range().replace("\r\n", "\n").strip()
            image = capped(page.render(scale=DPI / 72).to_pil().convert("RGB"))
            pool.append({"doc": path.name, "page": index + 1, "text": text, "image": image,
                         "vision": (path.name, index + 1) in VISION_PAGES})
    elif path.suffix in (".jpg", ".png"):
        pool.append({"doc": path.name, "page": 1, "text": "", "image": capped(Image.open(path).convert("RGB")),
                     "vision": True})

for i, item in enumerate(pool):
    item["id"] = f"{i:03d}"
    (pool_dir / f"{item['id']}.txt").write_text(item["text"])
    item.pop("image").save(pool_dir / f"{item['id']}.png", optimize=False)


def tokens(text):
    text = text.lower()
    words = re.findall(r"[a-z0-9][a-z0-9.\-]*", text)
    cjk = re.findall(r"[㐀-鿿]+", text)
    bigrams = [run[i:i + 2] for run in cjk for i in range(max(1, len(run) - 1))]
    return words + bigrams


docs = [Counter(tokens(item["text"] + " " + item["doc"])) for item in pool]
lengths = [sum(d.values()) for d in docs]
average = sum(lengths) / len(lengths)
df = Counter(term for d in docs for term in d)


def bm25(query, k1=1.2, b=0.75):
    scores = []
    for d, length in zip(docs, lengths):
        score = 0.0
        for term in set(tokens(query)):
            if term in d:
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * d[term] * (k1 + 1) / (d[term] + k1 * (1 - b + b * length / average))
        scores.append(score)
    return scores


in_pool = {item["doc"] for item in pool}
sets = []
for dataset in ("examples.json", "realistic.json"):
    for case in json.loads((bench_dir / dataset).read_text())["cases"]:
        if (case.get("source") or {}).get("filename") not in in_pool:
            continue
        for turn, query in enumerate(case["turns"]):
            scores = bm25(query)
            ranked = sorted(range(len(pool)), key=lambda i: (-scores[i], i))[:CANDIDATES]
            with_image = [i for i in ranked if pool[i]["vision"]][:IMAGE_PASSAGES]
            for i in ranked:
                if len(with_image) >= IMAGE_PASSAGES:
                    break
                if i not in with_image:
                    with_image.append(i)
            sets.append({
                "id": f"{case['id']}#{turn + 1}" if len(case["turns"]) > 1 else case["id"],
                "query": query,
                "source": case["source"],
                "candidates": [{"item": pool[i]["id"], "doc": pool[i]["doc"], "page": pool[i]["page"],
                                "image": i in with_image, "text": bool(pool[i]["text"])} for i in ranked],
            })

(out_dir / "sets.json").write_text(json.dumps(sets, ensure_ascii=False, indent=1))
print(f"{len(pool)} pages, {len(sets)} questions")
for s in sets:
    hit = [n for n, c in enumerate(s["candidates"]) if c["doc"] == s["source"]["filename"]
           and c["page"] in s["source"].get("pages", [c["page"]])]
    print(f"  {s['id']}: source at ranks {hit[:3]}")
