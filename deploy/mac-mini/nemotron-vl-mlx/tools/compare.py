"""Runs the MLX reranker on inputs saved by tools/reference.py and reports the
logit differences.

    uv run tools/compare.py MODEL_DIR REF_DIR [--dtype float32|float16|bfloat16] [--cpu]
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_vl_mlx.model import load_reranker

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("ref_dir")
parser.add_argument("--dtype", default="float32")
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
if args.cpu:
    mx.set_default_device(mx.cpu)

model = load_reranker(args.model_dir, dtype=getattr(mx, args.dtype))
worst = 0.0
for path in sorted(Path(args.ref_dir).glob("*.npz")):
    ref = np.load(path)
    pixels = ref["pixel_values"]
    began = time.perf_counter()
    logits = model(
        mx.array(ref["input_ids"]),
        mx.array(ref["attention_mask"]),
        mx.array(pixels) if pixels.size else None,
    )
    mx.eval(logits)
    seconds = time.perf_counter() - began
    logits = np.array(logits)
    diff = np.abs(logits - ref["logits"]).max()
    worst = max(worst, diff)
    same_order = np.array_equal(np.argsort(-logits), np.argsort(-ref["logits"]))
    print(f"{path.stem}: {seconds:.2f}s max|diff|={diff:.5f} same order={same_order}")
    print(f"  mlx={np.round(logits, 4).tolist()}\n  ref={np.round(ref['logits'], 4).tolist()}")
print(f"worst max|diff|={worst:.5f}")
