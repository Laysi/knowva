"""Checks that nemotron_vl_mlx's EmbedProcessor builds exactly the inputs
sentence-transformers gives llama-nemotron-embed-vl-1b-v2 through encode_query and
encode_document: token ids (padding removed) and pixel values.

    uv run --group reference tools/check_embed_processing.py MODEL_DIR PAGES_DIR
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from nemotron_vl_mlx.processing import EmbedProcessor, Passage

model_dir, pages_dir = Path(sys.argv[1]), Path(sys.argv[2])
pages = sorted(pages_dir.glob("p-*.jpg"))
texts = [p.with_suffix(".txt").read_text().strip() for p in pages]
images = [Image.open(p).convert("RGB") for p in pages]
model = SentenceTransformer(str(model_dir), trust_remote_code=True, device="cpu", local_files_only=True,
                            model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"})
model.processor.max_input_tiles = 6

captured = []


class Stop(Exception):
    pass


def forward(*args, **kwargs):
    captured.append(kwargs)
    raise Stop


model[0].auto_model.forward = forward
ours = EmbedProcessor(model_dir, max_input_tiles=6)
long_text = " ".join(texts) * 3
cases = {
    "query": ("query", ["IoT4Industry D.1 文件的作者與 QA 負責人是誰？"]),
    "query-long": ("query", [long_text]),
    "text": ("passage", texts[:4]),
    "text-long": ("passage", [long_text, texts[0]]),
    "text-literal-image": ("passage", ["see <image> here", "<img>x</img>"]),
    "image": ("passage", [images[0], images[6]]),
    "image-wide": ("passage", [images[1].rotate(90, expand=True), images[2].resize((300, 90))]),
    "image-text": ("passage", [{"image": images[3], "text": texts[3]}, {"image": images[4], "text": texts[4]}]),
    "image-text-long": ("passage", [{"image": images[5], "text": long_text}]),
}

failed = False
for name, (input_type, inputs) in cases.items():
    captured.clear()
    try:
        (model.encode_query if input_type == "query" else model.encode_document)(inputs, batch_size=len(inputs))
    except Stop:
        pass
    kwargs = captured[0]
    # Sentence Transformers sorts a batch by length; match rows by content.
    theirs = sorted(tuple(ids[mask.bool()].tolist()) for ids, mask in zip(kwargs["input_ids"], kwargs["attention_mask"]))
    items = [Passage(text=i["text"], image=i["image"]) if isinstance(i, dict) else
             Passage(image=i) if isinstance(i, Image.Image) else Passage(text=i) for i in inputs]
    sequences, pixels = ours(items, input_type)
    mine = sorted(tuple(s.tolist()) for s in sequences)
    ids_equal = theirs == mine
    reference = kwargs.get("pixel_values")
    if reference is None:
        pixels_equal = pixels is None
    else:
        # Same sort applies to tiles, so compare as multisets of tiles.
        ref = reference.to(torch.float32).permute(0, 2, 3, 1).numpy()
        pixels_equal = pixels is not None and ref.shape == pixels.shape and (
            sorted(t.tobytes() for t in ref) == sorted(t.tobytes() for t in pixels))
    ok = ids_equal and pixels_equal
    failed |= not ok
    print(f"{name:20s} lengths={[len(s) for s in sequences]} tiles={0 if pixels is None else len(pixels)} "
          f"ids={ids_equal} pixels={pixels_equal}")
sys.exit(1 if failed else 0)
