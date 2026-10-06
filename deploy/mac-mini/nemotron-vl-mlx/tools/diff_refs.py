"""Compares the logits of two tools/reference.py output directories.

    uv run tools/diff_refs.py A_DIR B_DIR
"""

import sys
from pathlib import Path

import numpy as np

a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
worst = 0.0
for a_path in sorted(a_dir.glob("*.npz")):
    a, b = np.load(a_path), np.load(b_dir / a_path.name)
    assert np.array_equal(a["input_ids"], b["input_ids"]), a_path.name
    diff = np.abs(a["logits"] - b["logits"]).max()
    worst = max(worst, diff)
    print(f"{a_path.stem}: max|diff|={diff:.5f}")
print(f"worst max|diff|={worst:.5f}")
