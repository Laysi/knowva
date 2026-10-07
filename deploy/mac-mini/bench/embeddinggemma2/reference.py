"""PyTorch float32 reference for a sample of the corpus and the questions,
through Sentence Transformers exactly as the model card documents it, to check
embed_corpus.py's MLX inputs against.

  uv run python reference.py models/embeddinggemma-2-hf out/reference.json
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).parent / "data"
# Text units, text-only pages, page image + text, and image-only uploads.
SAMPLE = [1, 21, 22, 352, 390, 7, 283, 393, 351, 19, 396]

model_path, out = sys.argv[1:]
units = {unit["id"]: unit for unit in map(json.loads, (DATA / "corpus.jsonl").open())}
questions = json.loads((DATA / "queries.json").read_text())[:8]

model = SentenceTransformer(model_path, device="cpu", model_kwargs={"torch_dtype": torch.float32})
result = {"units": {}, "queries": {}}
for unit_id in SAMPLE:
    unit = units[unit_id]
    text = (unit["content"] or "").strip()
    if not unit["uses_image"]:
        vector = model.encode(unit["content"], prompt_name="Document")
    else:
        image = Image.open(DATA / "images" / unit["image_path"]).convert("RGB")
        if text:
            vector = model.encode({"text": text, "image": image}, prompt_name="Document")
        else:
            # Prefixes apply to text only.
            vector = model.encode(image)
    result["units"][unit_id] = vector.tolist()
for question in questions:
    result["queries"][question["id"]] = {
        "search": model.encode(question["query"], prompt_name="SearchQuery").tolist(),
        "qa": model.encode(question["query"], prompt_name="QuestionAnswering").tolist(),
    }
Path(out).write_text(json.dumps(result))
print(f"{len(result['units'])} units, {len(result['queries'])} queries")
