"""Does the embed VL model load and run on MPS? Mirrors nim_shim.py's setup."""
import os, time, torch
from PIL import Image

MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
DEVICE = os.environ.get("DEV", "mps")
DTYPE = getattr(torch, os.environ.get("DT", "float16"))
TILES = int(os.environ.get("TILES", "2"))

from sentence_transformers import SentenceTransformer

t = time.time()
model = SentenceTransformer(MODEL, trust_remote_code=True, device=DEVICE,
                            model_kwargs={"torch_dtype": DTYPE, "attn_implementation": "sdpa"})
model.processor.max_input_tiles = TILES
print(f"load {time.time()-t:.1f}s  device={DEVICE} dtype={DTYPE}", flush=True)

with torch.inference_mode():
    t = time.time(); q = model.encode_query(["這份報告提到哪些工業物聯網的重點?"])
    print(f"query embed {time.time()-t:.2f}s dim={len(q[0])}", flush=True)

    page = Image.new("RGB", (612, 792), "white")  # one letter page at 72 dpi
    t = time.time(); d = model.encode_document([{"image": page, "text": "工業物聯網產業報告"}])
    print(f"image+text embed {time.time()-t:.2f}s dim={len(d[0])}", flush=True)

    t = time.time(); _ = model.encode_query(["資安的部分呢?"])
    print(f"query embed (warm) {time.time()-t:.2f}s", flush=True)
print("EMBED_OK", flush=True)
