"""Measures fp16/bf16 matmul throughput on this machine for MLX and PyTorch
MPS, at a large square shape and at the reranker's own shapes.

    uv run --group reference tools/matmul_peak.py
"""

import time

import mlx.core as mx
import torch

SHAPES = [(4096, 4096, 4096), (32 * 600, 2048, 8192), (600, 2048, 8192), (600, 8192, 2048), (8 * 900, 2048, 2048)]


def timed(fn, sync, repeat=20):
    for _ in range(3):
        fn()
    sync()
    began = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync()
    return (time.perf_counter() - began) / repeat


for m, k, n in SHAPES:
    flops = 2 * m * k * n
    row = [f"{m}x{k}x{n}"]
    for dtype in ("float16", "bfloat16"):
        a = mx.random.normal((m, k)).astype(getattr(mx, dtype))
        b = mx.random.normal((k, n)).astype(getattr(mx, dtype))
        mx.eval(a, b)
        out = []
        seconds = timed(lambda: out.append(a @ b) or mx.eval(out[-1]), lambda: None)
        out.clear()
        row.append(f"mlx {dtype} {flops / seconds / 1e12:.1f}")
        ta = torch.randn(m, k, device="mps", dtype=getattr(torch, dtype))
        tb = torch.randn(k, n, device="mps", dtype=getattr(torch, dtype))
        seconds = timed(lambda: ta @ tb, torch.mps.synchronize)
        row.append(f"mps {dtype} {flops / seconds / 1e12:.1f}")
    print("  ".join(row), "TFLOPS", flush=True)
