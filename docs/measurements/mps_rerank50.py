"""Is RERANK_CANDIDATE_TOP_K=50 affordable on MPS?

knowva cut it to 4 because cost is linear in candidates and a page cost seconds
there. Passages here are the length of a real report page's text layer, not the
short strings an earlier probe used, and they go through in batches of
RERANK_BATCH_SIZE the way the backend sends them.
"""
import os, resource, time, torch
from transformers import AutoModelForSequenceClassification, AutoProcessor

MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
DEVICE = os.environ.get("DEV", "mps")
DTYPE = getattr(torch, os.environ.get("DT", "float16"))
N = int(os.environ.get("N", "50"))
BATCH = int(os.environ.get("BATCH", "4"))
CHARS = int(os.environ.get("CHARS", "2000"))

def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1073741824

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, dtype=DTYPE, trust_remote_code=True,
    attn_implementation="sdpa", low_cpu_mem_usage=True).eval().to(DEVICE)
print(f"loaded on {DEVICE} dtype={DTYPE} peak_rss={rss():.2f}GB", flush=True)

query = "這份報告提到哪些工業物聯網的重點?"
topics = ["工業物聯網導入的三個階段與投資回收期評估",
          "感測器網路佈署與邊緣運算架構設計",
          "廠區資安風險評估與存取控制建議",
          "預測性維護的資料收集與模型訓練流程",
          "通訊協定選擇：OPC UA、MQTT 與 Modbus 比較"]
filler = ("本節說明實際導入過程中觀察到的狀況，包含設備連線方式、資料收集頻率、"
          "以及與現有系統整合時遇到的限制。案例來自製造業現場的實際部署紀錄。")
passages = []
for i in range(N):
    topic = topics[i % len(topics)]
    body = (filler * 20)[: CHARS - len(topic)]
    passages.append(f"{topic}\n{body}")
print(f"{N} passages, {len(passages[0])} chars each, batch={BATCH}", flush=True)

processor = AutoProcessor.from_pretrained(
    MODEL, trust_remote_code=True, max_input_tiles=1,
    use_thumbnail=True, rerank_max_length=8192)

scores = []
start = time.time()
for offset in range(0, N, BATCH):
    chunk = passages[offset : offset + BATCH]
    examples = [{"question": query, "doc_text": p, "doc_image": ""} for p in chunk]
    t = time.time()
    with torch.no_grad():
        batch = processor.process_queries_documents_crossencoder(examples)
        tokens = batch["input_ids"].shape[-1] if "input_ids" in batch else -1
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(**batch, return_dict=True).logits.squeeze(-1)
    scores += logits.reshape(-1).float().tolist()
    print(f"  batch {offset//BATCH + 1}: {time.time()-t:.2f}s "
          f"({len(chunk)} passages, {tokens} tokens) rss={rss():.2f}GB", flush=True)

total = time.time() - start
print(f"TOTAL {total:.1f}s for {N} passages = {total/N:.2f}s each", flush=True)
print(f"top5: {sorted(range(len(scores)), key=lambda i: -scores[i])[:5]}", flush=True)
print("RERANK50_OK", flush=True)
