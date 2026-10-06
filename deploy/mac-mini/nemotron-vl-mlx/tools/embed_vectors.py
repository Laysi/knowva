"""Embeds a fixed set of queries and pages (text, image, image+text) with the
original sentence-transformers model or the MLX port and saves the vectors,
for compare_embed.py.

    uv run [--group reference] tools/embed_vectors.py MODEL_DIR PAGES_DIR QUESTIONS.json OUT.npz \
        --backend torch|mlx [--device mps] [--dtype float16] [--tiles 6] [--cache-limit-mb 256]
"""

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("pages_dir")
parser.add_argument("questions")
parser.add_argument("out")
parser.add_argument("--backend", choices=("torch", "mlx"), required=True)
parser.add_argument("--device", default="mps")
parser.add_argument("--dtype", default="float16")
parser.add_argument("--tiles", type=int, default=6)
parser.add_argument("--cache-limit-mb", type=int, default=-1, help="MLX: buffer cache limit, -1 keeps MLX's default")
args = parser.parse_args()

pages = sorted(Path(args.pages_dir).glob("p-*.png")) or sorted(Path(args.pages_dir).glob("p-*.jpg"))
texts = [p.with_suffix(".txt").read_text().strip() for p in pages]
images = [Image.open(p).convert("RGB") for p in pages]
queries = [case["turns"][0] for case in json.loads(Path(args.questions).read_text())["cases"]][:20]
kinds = {
    "query": ("query", queries),
    "text": ("passage", texts),
    "image": ("passage", images),
    "text_image": ("passage", [{"image": i, "text": t} for i, t in zip(images, texts)]),
}

if args.backend == "torch":
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_dir, trust_remote_code=True, device="cpu", local_files_only=True,
                                model_kwargs={"dtype": getattr(torch, args.dtype), "attn_implementation": "sdpa"})
    model.processor.max_input_tiles = args.tiles
    model = model.to(args.device)

    def embed(input_type, inputs):
        with torch.inference_mode():
            encode = model.encode_query if input_type == "query" else model.encode_document
            # One input per call, as the backend sends pages.
            return np.stack([np.asarray(encode([x])[0], dtype=np.float32) for x in inputs])

    def peak_bytes():
        return torch.mps.driver_allocated_memory() if args.device == "mps" else 0
else:
    import mlx.core as mx

    from nemotron_vl_mlx import Embedder, Passage

    if args.cache_limit_mb >= 0:
        mx.set_cache_limit(args.cache_limit_mb * 2**20)
    embedder = Embedder(args.model_dir, dtype=getattr(mx, args.dtype), max_input_tiles=args.tiles)

    def embed(input_type, inputs):
        items = [Passage(text=x["text"], image=x["image"]) if isinstance(x, dict) else
                 Passage(image=x) if isinstance(x, Image.Image) else Passage(text=x) for x in inputs]
        return np.concatenate([embedder([item], input_type) for item in items])

    def peak_bytes():
        return mx.get_peak_memory()

out = {}
for kind, (input_type, inputs) in kinds.items():
    embed(input_type, inputs[:1])
    began = time.perf_counter()
    out[kind] = embed(input_type, inputs)
    print(f"{kind}: {len(inputs)} inputs, {(time.perf_counter() - began) / len(inputs):.3f}s each", flush=True)
np.savez(args.out, **out)
footprint = None
if platform.system() == "Darwin":
    # phys_footprint, which counts GPU buffers; RSS does not.
    footprint = subprocess.run(["top", "-l", "1", "-pid", str(os.getpid()), "-stats", "mem"],
                               capture_output=True, text=True).stdout.split()[-1]
print(f"footprint {footprint}, peak {peak_bytes() / 2**30:.2f} GiB")
