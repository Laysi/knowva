"""Checks that nemotron_vl_mlx.processing produces exactly the inputs the
model's own processor does: token ids, attention mask and pixel values.

    uv run --group reference tools/check_processing.py MODEL_DIR PAGES_DIR
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from nemotron_vl_mlx.processing import Passage, RerankProcessor

model_dir, pages_dir = Path(sys.argv[1]), Path(sys.argv[2])
pages = sorted(pages_dir.glob("p-*.jpg"))
texts = [p.with_suffix(".txt").read_text().strip() for p in pages]
images = [Image.open(p).convert("RGB") for p in pages]
query = "IoT4Industry D.1 文件的作者與 QA 負責人是誰？"

cases = [
    ("text", 8192, [Passage(text=t) for t in texts[:8]]),
    ("image", 2048, [Passage(image=i) for i in images[:4]]),
    ("text_image", 10240, [Passage(text=t, image=i) for t, i in zip(texts[4:8], images[4:8])]),
    ("mixed", 10240, [Passage(text=texts[0]), Passage(text=texts[1], image=images[1]), Passage(image=images[2])]),
    ("truncated", 64, [Passage(text=texts[3]), Passage(text="short")]),
    ("literal-image", 8192, [Passage(text="see <image> here and <image> there"), Passage(text="<img>x</img>")]),
    ("wide-image", 2048, [Passage(image=images[0].rotate(90, expand=True)), Passage(image=images[1].resize((300, 90)))]),
]

failed = False
for tiles in (1, 6):
    ours = RerankProcessor(model_dir, max_input_tiles=tiles)
    theirs_by_length = {}
    for name, max_length, passages in cases:
        if max_length not in theirs_by_length:
            theirs_by_length[max_length] = AutoProcessor.from_pretrained(
                model_dir, trust_remote_code=True, local_files_only=True,
                max_input_tiles=tiles, use_thumbnail=True, rerank_max_length=max_length,
            )
        theirs = theirs_by_length[max_length].process_queries_documents_crossencoder(
            [{"question": query, "doc_text": p.text, "doc_image": p.image if p.image is not None else ""}
             for p in passages]
        )
        mine = ours(query, passages, max_length)
        ids_equal = np.array_equal(theirs["input_ids"].numpy(), mine.input_ids)
        mask_equal = np.array_equal(theirs["attention_mask"].numpy(), mine.attention_mask)
        if theirs["pixel_values"] is None:
            pixels_equal = mine.pixel_values is None
            n_tiles = 0
        else:
            reference = theirs["pixel_values"].to(torch.float32).permute(0, 2, 3, 1).numpy()
            pixels_equal = mine.pixel_values is not None and np.array_equal(reference, mine.pixel_values)
            n_tiles = reference.shape[0]
        ok = ids_equal and mask_equal and pixels_equal
        failed |= not ok
        print(f"tiles={tiles} {name:14s} shape={tuple(mine.input_ids.shape)} tiles={n_tiles} "
              f"ids={ids_equal} mask={mask_equal} pixels={pixels_equal}")

sys.exit(1 if failed else 0)
