"""Rerank VL model on MPS. Loads on CPU in float16 first, then moves.

device_map="mps" stalled on an 8 GB machine: accelerate dispatches shard by shard
while the config carries torch_dtype=null, so the intermediate is float32. Loading
to CPU at an explicit dtype and then moving separates "does it fit" from "does MPS
run it" — two different failures that look the same from outside.
"""
import os, resource, time, torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoProcessor

MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
DEVICE = os.environ.get("DEV", "mps")
DTYPE = getattr(torch, os.environ.get("DT", "float16"))
TILES = int(os.environ.get("TILES", "1"))
MODALITY = os.environ.get("MODALITY", "text")
N = int(os.environ.get("N", "5"))
MAX_LENGTH = {"text": 8192, "image": 2048, "text_image": 10240}

def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1073741824

t = time.time()
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, dtype=DTYPE, trust_remote_code=True,
    attn_implementation="sdpa", low_cpu_mem_usage=True).eval()
print(f"cpu load {time.time()-t:.1f}s peak_rss={rss():.2f}GB", flush=True)

t = time.time()
model = model.to(DEVICE)
print(f"move to {DEVICE} {time.time()-t:.1f}s peak_rss={rss():.2f}GB", flush=True)

query = "這份報告提到哪些工業物聯網的重點?"
texts = ["工業物聯網導入的三個階段與投資回收期",
         "感測器網路與邊緣運算架構",
         "員工餐廳本週菜單",
         "資安風險與存取控制建議",
         "附錄：參考文獻清單"][:N]
page = Image.new("RGB", (612, 792), "white") if MODALITY != "text" else ""
examples = [{"question": query, "doc_text": x, "doc_image": page} for x in texts]

processor = AutoProcessor.from_pretrained(
    MODEL, trust_remote_code=True, max_input_tiles=TILES,
    use_thumbnail=True, rerank_max_length=MAX_LENGTH[MODALITY])
t = time.time()
with torch.no_grad():
    batch = processor.process_queries_documents_crossencoder(examples)
    batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
    logits = model(**batch, return_dict=True).logits.squeeze(-1)
elapsed = time.time() - t
scores = logits.reshape(-1).float().tolist()
print(f"{MODALITY}: {elapsed:.2f}s / {len(examples)} passages "
      f"= {elapsed/len(examples):.2f}s each, peak_rss={rss():.2f}GB", flush=True)
print(f"  ranking: {sorted(range(len(scores)), key=lambda i: -scores[i])}", flush=True)
print(f"  scores: {[round(s,2) for s in scores]}", flush=True)
print("RERANK_OK", flush=True)
