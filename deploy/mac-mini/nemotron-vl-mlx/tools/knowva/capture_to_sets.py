"""Turns the rerank requests capture-proxy.ts recorded into a candidate-set
directory for bench_sets.py: one question per answer, candidates in the order
they were sent (Knowva sends each modality's batches in rank order, which
bench_sets.py batches the same way), plus the logits the service returned.

Knowva sends one answer's batches back to back, but concurrent users interleave
theirs, so a request joins the latest question with the same query unless that
question's previous batch finished more than GAP_SECONDS before it started.

    uv run tools/knowva/capture_to_sets.py CAPTURE_DIR OUT_DIR
"""

import base64
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

GAP_SECONDS = 5.0

capture_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
pool_dir = out_dir / "pool"
pool_dir.mkdir(parents=True, exist_ok=True)

items = {}


def item(text, image):
    """Pool entry for one passage, deduplicated by content."""
    data = base64.b64decode(image.split(",", 1)[1]) if image else b""
    key = hashlib.sha256(text.encode() + b"\0" + data).hexdigest()[:16]
    if key not in items:
        items[key] = True
        (pool_dir / f"{key}.txt").write_bytes(text.encode())
        if data:
            (pool_dir / f"{key}.png").write_bytes(data)
    return key


sets = []
served = []
latest = {}  # query -> (index into sets, when its last batch finished)
for path in sorted(capture_dir.glob("*.json"), key=lambda p: json.loads(p.read_text())["seq"]):
    record = json.loads(path.read_text())
    if record["status"] != 200:
        print(f"skipping {path.name}: HTTP {record['status']}")
        continue
    query = record["request"]["query"]["text"]
    started = datetime.fromisoformat(record["startedAt"].replace("Z", "+00:00")).timestamp()
    index, finished = latest.get(query, (None, 0.0))
    if index is None or started - finished > GAP_SECONDS:
        index = len(sets)
        sets.append({"id": f"q{index + 1:03d}", "query": query, "candidates": []})
        served.append({"id": sets[index]["id"], "seconds": 0.0, "batch_seconds": [], "logits": []})
    latest[query] = (index, started + record["seconds"])
    logits = [None] * len(record["request"]["passages"])
    for ranking in record["response"]["rankings"]:
        logits[ranking["index"]] = ranking["logit"]
    for passage in record["request"]["passages"]:
        text, image = passage.get("text") or "", passage.get("image")
        sets[index]["candidates"].append({"item": item(text, image), "image": bool(image), "text": bool(text)})
    served[index]["logits"].extend(logits)
    served[index]["seconds"] += record["seconds"]
    served[index]["batch_seconds"].append(record["seconds"])

(out_dir / "sets.json").write_text(json.dumps(sets, ensure_ascii=False, indent=1))
# The service's own answers, in compare_runs.py's format.
(out_dir / "served.json").write_text(json.dumps({
    "backend": "service", "device": "-", "dtype": "-", "batching": "captured",
    "peak_device_bytes": 0, "max_rss_bytes": 0, "results": served,
}))
sizes = [len(s["candidates"]) for s in sets]
print(f"{len(sets)} questions, {len(items)} distinct passages, candidates per question {min(sizes)}-{max(sizes)}")
