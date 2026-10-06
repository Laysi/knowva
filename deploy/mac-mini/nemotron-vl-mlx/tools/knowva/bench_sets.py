"""Reranks every candidate set from build_sets.py the way Knowva's Mac
inference service would: batches of one modality, at most 8 passages, page
images decoded from PNG per request, a token budget per modality. Records the
logits and the seconds each question's rerank took.

    uv run [--group reference] tools/knowva/bench_sets.py SETS_DIR MODEL_DIR OUT.json \
        --backend torch|mlx|http [--device mps|cpu] [--dtype float16] [--tiles 1] [--url URL]

``http`` sends each batch to a running inference service's /v1/ranking as
Knowva's backend does (PNG data URLs), so the service's own settings apply.
"""

import argparse
import io
import json
import platform
import resource
import time
from pathlib import Path

from PIL import Image

RERANK_MAX_LENGTH = {"text": 8192, "image": 2048, "text_image": 10240}
BATCH_SIZE = 8

parser = argparse.ArgumentParser()
parser.add_argument("sets_dir")
parser.add_argument("model_dir")
parser.add_argument("out")
parser.add_argument("--backend", choices=("torch", "mlx", "http"), required=True)
parser.add_argument("--url", default="http://127.0.0.1:9500")
parser.add_argument("--cache-limit-mb", type=int, default=-1, help="MLX: buffer cache limit, -1 keeps MLX's default")
parser.add_argument("--device", default="mps")
parser.add_argument("--dtype", default="float16")
parser.add_argument("--tiles", type=int, default=1)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--batching", choices=("padded", "unpadded"), default="padded",
                    help="padded: one left-padded batch per request, as Knowva's service does; "
                         "unpadded: each passage at its own length")
parser.add_argument("--warmup", type=int, default=1, help="questions run once before timing")
args = parser.parse_args()

sets_dir = Path(args.sets_dir)
sets = json.loads((sets_dir / "sets.json").read_text())
if args.limit:
    sets = sets[: args.limit]
# Bytes as written: text mode would turn the "\r\n" in extracted pages into "\n".
texts = {p.stem: p.read_bytes().decode() for p in (sets_dir / "pool").glob("*.txt")}
pngs = {p.stem: p.read_bytes() for p in (sets_dir / "pool").glob("*.png")}


def batches(candidates):
    """Knowva's modality_batches: groups in order of first appearance, then
    chunks of BATCH_SIZE, keeping rank order."""
    groups = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(candidate["image"], []).append(index)
    return [indices[i:i + BATCH_SIZE] for indices in groups.values() for i in range(0, len(indices), BATCH_SIZE)]


def passages_of(candidates, indices):
    out = []
    for i in indices:
        candidate = candidates[i]
        image = None
        if candidate["image"]:
            # The service decodes the PNG itself.
            image = pngs[candidate["item"]] if args.backend == "http" else (
                Image.open(io.BytesIO(pngs[candidate["item"]])).convert("RGB"))
        out.append((texts[candidate["item"]] if candidate["text"] else "", image))
    return out


def modality(passages):
    has_image = any(image is not None for _, image in passages)
    has_text = any(text for text, _ in passages)
    return "text_image" if has_image and has_text else "image" if has_image else "text"


if args.backend == "http":
    import base64
    import urllib.request

    def score(query, passages):
        body = {"model": "nvidia/llama-nemotron-rerank-vl-1b-v2", "query": {"text": query}, "truncate": "NONE",
                "passages": []}
        for text, image in passages:
            passage = {"text": text} if text else {}
            if image is not None:
                passage["image"] = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
            body["passages"].append(passage)
        request = urllib.request.Request(f"{args.url}/v1/ranking", data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            rankings = json.load(response)["rankings"]
        logits = [None] * len(passages)
        for ranking in rankings:
            logits[ranking["index"]] = ranking["logit"]
        return logits

    def peak_bytes():
        return 0
elif args.backend == "torch":
    import torch
    from transformers import AutoModelForSequenceClassification, AutoProcessor

    dtype = getattr(torch, args.dtype)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, dtype=dtype, trust_remote_code=True, local_files_only=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).eval().to(args.device)
    processors = {
        m: AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True,
                                         max_input_tiles=args.tiles, use_thumbnail=True, rerank_max_length=n)
        for m, n in RERANK_MAX_LENGTH.items()
    }

    def forward(processor, examples):
        batch = processor.process_queries_documents_crossencoder(examples)
        batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        return model(**batch, return_dict=True).logits.reshape(-1)

    def score(query, passages):
        examples = [{"question": query, "doc_text": text, "doc_image": image if image is not None else ""}
                    for text, image in passages]
        processor = processors[modality(passages)]
        with torch.no_grad():
            if args.batching == "padded":
                logits = forward(processor, examples)
            else:
                logits = torch.cat([forward(processor, [example]) for example in examples])
            return logits.float().cpu().tolist()

    def peak_bytes():
        return torch.mps.driver_allocated_memory() if args.device == "mps" else 0
else:
    import mlx.core as mx

    from nemotron_vl_mlx import Passage, Reranker

    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
    if args.cache_limit_mb >= 0:
        mx.set_cache_limit(args.cache_limit_mb * 2**20)
    reranker = Reranker(args.model_dir, dtype=getattr(mx, args.dtype), max_input_tiles=args.tiles,
                        padded=args.batching == "padded")

    def score(query, passages):
        return reranker(query, [Passage(text=t, image=i) for t, i in passages],
                        RERANK_MAX_LENGTH[modality(passages)]).tolist()

    def peak_bytes():
        return mx.get_peak_memory()

    def cache_bytes():
        return mx.get_cache_memory()


def run(s):
    logits = [None] * len(s["candidates"])
    batch_seconds = []
    began = time.perf_counter()
    for indices in batches(s["candidates"]):
        batch_began = time.perf_counter()
        for i, value in zip(indices, score(s["query"], passages_of(s["candidates"], indices))):
            logits[i] = value
        batch_seconds.append(round(time.perf_counter() - batch_began, 4))
    return logits, time.perf_counter() - began, batch_seconds


for s in sets[: args.warmup]:
    run(s)
results = []
for s in sets:
    logits, seconds, batch_seconds = run(s)
    results.append({"id": s["id"], "seconds": round(seconds, 4), "batch_seconds": batch_seconds, "logits": logits})
    print(f"{s['id']}: {seconds:.2f}s", flush=True)

max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
footprint = None
if platform.system() == "Darwin":
    import os
    import subprocess

    # phys_footprint, which counts GPU buffers; RSS does not.
    footprint = subprocess.run(["top", "-l", "1", "-pid", str(os.getpid()), "-stats", "mem"],
                               capture_output=True, text=True).stdout.split()[-1]
Path(args.out).write_text(json.dumps({
    "backend": args.backend, "device": args.device, "dtype": args.dtype, "tiles": args.tiles,
    "batching": args.batching,
    "machine": platform.platform(), "peak_device_bytes": peak_bytes(),
    "max_rss_bytes": max_rss if platform.system() == "Darwin" else max_rss * 1024,
    "footprint_at_end": footprint,
    "cache_bytes_at_end": cache_bytes() if args.backend == "mlx" else None,
    "cache_limit_mb": args.cache_limit_mb,
    "results": results,
}))
seconds = sorted(r["seconds"] for r in results)
print(f"mean {sum(seconds) / len(seconds):.2f}s median {seconds[len(seconds) // 2]:.2f}s over {len(seconds)} questions; "
      f"footprint {footprint}, peak active {peak_bytes() / 2**30:.2f} GiB"
      + (f", cache {cache_bytes() / 2**30:.2f} GiB" if args.backend == "mlx" else ""))
