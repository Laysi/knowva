"""Records what the inference service on 127.0.0.1:9500 returns for a fixed
set of embed and rerank requests, shaped like the backend's, or compares two
recordings for exact equality. Runs on knowva-mac, taking pages from
~/knowva-inference/bench-pages and queries from the deployed realistic.json.

    python tools/knowva/service_outputs.py record OUT.json
    python tools/knowva/service_outputs.py compare BEFORE.json AFTER.json
"""

import base64
import json
import sys
from pathlib import Path

import requests

URL = "http://127.0.0.1:9500"
home = Path.home() / "knowva-inference"


def requests_to_send():
    pages = sorted((home / "bench-pages").glob("p-*.jpg"))[:4]
    texts = [page.with_suffix(".txt").read_bytes().decode().strip() for page in pages]
    images = ["data:image/jpeg;base64," + base64.b64encode(page.read_bytes()).decode() for page in pages]
    cases = json.loads((home / "app/bench/realistic.json").read_text())["cases"]
    queries = [case["turns"][0] for case in cases[:3]]
    embed = [("query", "text", q) for q in queries]
    embed += [("passage", "text", t) for t in texts]
    embed += [("passage", "image", i) for i in images]
    embed += [("passage", "text_image", f'{t}\n<img src="{i}"/>') for t, i in zip(texts, images)]
    rerank = []
    for q in queries:
        rerank.append((q, [{"text": t} for t in texts]))
        rerank.append((q, [{"image": i} for i in images]))
        rerank.append((q, [{"text": t, "image": i} for t, i in zip(texts, images)]))
    return embed, rerank


def record(out):
    embed, rerank = requests_to_send()
    result = {"health": requests.get(f"{URL}/health", timeout=10).json(), "embed": [], "rerank": []}
    for input_type, modality, item in embed:
        response = requests.post(f"{URL}/v1/embeddings", json={
            "model": "nvidia/llama-nemotron-embed-vl-1b-v2", "input": [item],
            "input_type": input_type, "modality": [modality]}, timeout=300)
        response.raise_for_status()
        result["embed"].append({"kind": f"{input_type}/{modality}",
                                "embedding": response.json()["data"][0]["embedding"]})
    for query, passages in rerank:
        response = requests.post(f"{URL}/v1/ranking", json={
            "model": "nvidia/llama-nemotron-rerank-vl-1b-v2", "query": {"text": query},
            "passages": passages, "truncate": "NONE"}, timeout=300)
        response.raise_for_status()
        kind = "+".join(sorted(passages[0]))
        result["rerank"].append({"kind": kind, "rankings": response.json()["rankings"]})
    result["health_after"] = requests.get(f"{URL}/health", timeout=10).json()
    Path(out).write_text(json.dumps(result))
    print(f"{out}: {len(result['embed'])} embeddings, {len(result['rerank'])} rankings")


def compare(before, after):
    a, b = (json.loads(Path(p).read_text()) for p in (before, after))
    differ = 0
    for x, y in zip(a["embed"], b["embed"], strict=True):
        diff = max(abs(u - v) for u, v in zip(x["embedding"], y["embedding"], strict=True))
        differ += diff != 0
        print(f"embed  {x['kind']:20s} dim {len(y['embedding'])} max|diff| {diff:.3g}")
    for x, y in zip(a["rerank"], b["rerank"], strict=True):
        same = x["rankings"] == y["rankings"]
        differ += not same
        logits = [round(r["logit"], 4) for r in y["rankings"]]
        print(f"rerank {x['kind']:20s} {'identical' if same else 'DIFFERENT'} {logits}")
    print(f"{differ} of {len(a['embed']) + len(a['rerank'])} responses differ")
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    if sys.argv[1:2] == ["record"] and len(sys.argv) == 3:
        record(sys.argv[2])
    elif sys.argv[1:2] == ["compare"] and len(sys.argv) == 4:
        compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
