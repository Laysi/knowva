"""Runs the original PyTorch reranker on tools/cases.py and saves inputs and
logits for tools/compare.py.

    uv run --group reference tools/reference.py MODEL_DIR PAGES_DIR OUT_DIR [--device cpu|mps] [--dtype float32|float16] [--tiles N]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from cases import MAX_LENGTH, QUERY, load_cases  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("pages_dir")
parser.add_argument("out_dir")
parser.add_argument("--device", default="cpu")
parser.add_argument("--dtype", default="float32")
parser.add_argument("--tiles", type=int, default=1)
args = parser.parse_args()

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_dir, dtype=getattr(torch, args.dtype), trust_remote_code=True,
    local_files_only=True, attn_implementation="sdpa", low_cpu_mem_usage=True,
).eval().to(args.device)

for name, (modality, passages) in load_cases(args.pages_dir).items():
    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True,
        max_input_tiles=args.tiles, use_thumbnail=True, rerank_max_length=MAX_LENGTH[modality],
    )
    batch = processor.process_queries_documents_crossencoder(
        [{"question": QUERY, "doc_text": p.text, "doc_image": p.image if p.image is not None else ""}
         for p in passages]
    )
    began = time.perf_counter()
    with torch.no_grad():
        logits = model(**{k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()},
                       return_dict=True).logits.squeeze(-1).float().cpu().numpy()
    seconds = time.perf_counter() - began
    pixels = batch["pixel_values"]
    np.savez(
        out / f"{name}.npz",
        input_ids=batch["input_ids"].numpy(),
        attention_mask=batch["attention_mask"].numpy(),
        pixel_values=pixels.float().permute(0, 2, 3, 1).numpy() if pixels is not None else np.zeros((0,)),
        logits=logits,
    )
    print(f"{name}: shape={tuple(batch['input_ids'].shape)} {seconds:.2f}s logits={np.round(logits, 4).tolist()}", flush=True)
