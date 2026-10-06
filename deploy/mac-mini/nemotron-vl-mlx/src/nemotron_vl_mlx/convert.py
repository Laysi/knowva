"""Converts an original llama-nemotron-rerank-vl-1b-v2 or
llama-nemotron-embed-vl-1b-v2 snapshot into the layout this package loads
directly: MLX parameter names, the patch convolution channels-last, and the
SigLIP attention-pooling head, which the models never use, dropped. Values
keep the checkpoint's bfloat16, so nothing is rounded; loading casts them to
the requested dtype as it does for the original.

    python -m nemotron_vl_mlx.convert SRC_DIR DST_DIR
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import mlx.core as mx

from .model import CONVERTED_FORMAT, ModelConfig, sanitize

# What the port reads (config.json, tokenizer.json), plus the tokenizer files
# other tools expect beside them.
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def convert(src: str | Path, dst: str | Path) -> None:
    src, dst = Path(src), Path(dst)
    # Refuses a configuration the port does not implement before writing anything.
    ModelConfig.from_dir(src)
    dst.mkdir(parents=True, exist_ok=True)
    weights = sanitize(mx.load(str(src / "model.safetensors")))
    mx.save_safetensors(str(dst / "model.safetensors"), weights, metadata={"format": CONVERTED_FORMAT})
    for name in FILES:
        shutil.copy2(src / name, dst / name)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    convert(sys.argv[1], sys.argv[2])
