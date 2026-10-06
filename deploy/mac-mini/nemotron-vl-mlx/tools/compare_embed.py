"""Compares embed_vectors.py runs with a reference run: cosine similarity per
input kind, and whether each query retrieves the same top pages. "mixed"
pairs the run's query vectors with the reference's page vectors, as when only
the query side changes and stored page vectors are kept.

    uv run tools/compare_embed.py REF.npz RUN.npz [RUN.npz ...]
"""

import sys

import numpy as np

TOP_K = 5


def unit(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def top(queries, pages):
    # Dense retrieval orders by cosine similarity.
    return [list(np.argsort(-row, kind="stable")[:TOP_K]) for row in unit(queries) @ unit(pages).T]


ref = np.load(sys.argv[1])
for path in sys.argv[2:]:
    run = np.load(path)
    print(path)
    for kind in ref.files:
        cosine = (unit(ref[kind]) * unit(run[kind])).sum(1)
        print(f"  {kind:10s} n={len(cosine):2d} cosine min {cosine.min():.6f} mean {cosine.mean():.6f}")
    for kind in (k for k in ref.files if k != "query"):
        expected = top(ref["query"], ref[kind])
        for label, got in (("run", top(run["query"], run[kind])), ("mixed", top(run["query"], ref[kind]))):
            sets = sum(set(a) != set(b) for a, b in zip(expected, got))
            orders = sum(a != b for a, b in zip(expected, got))
            print(f"  top-{TOP_K} {kind:10s} {label:5s}: set differs {sets}/{len(expected)}, order differs {orders}")
