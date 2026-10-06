"""NIM-shaped embed/rerank service for the Mac mini appliance, with chat passed
through to Ollama.

This is docs/measurements/nim_shim.py moved from the CPU host to Apple Silicon.
Both Nemotron VL models run on MPS in float16, which docs/measurements §4
verified as op-complete and ranking-correct on an M1. Either can instead run
on MLX through nemotron-vl-mlx, a port that reads the same snapshots; with
both on MLX the service never imports PyTorch.

    KM_DEVICE           mps (default) | cpu, for the models on PyTorch
    KM_DTYPE            float16 (default) | bfloat16 | float32
    KM_EMBED_PATH       local snapshot of nvidia/llama-nemotron-embed-vl-1b-v2
    KM_RERANK_PATH      local snapshot of nvidia/llama-nemotron-rerank-vl-1b-v2
    KM_MAX_TILES        image tiles per page for embedding, 6 = model default
    KM_RERANK_TILES     tiles for rerank only, defaults to KM_MAX_TILES
    KM_RERANK_INPUT     both (default) | text | image: what of a page to rank
    KM_EMBED_BACKEND    torch (default) | mlx: what runs embed, at KM_DTYPE
    KM_RERANK_BACKEND   torch (default) | mlx: what runs rerank, at KM_DTYPE
    KM_MLX_CACHE_MB     MLX buffer cache limit, default 256
    KM_IDLE_SECS        seconds a model stays loaded after its last request,
                        default 600; negative keeps both loaded for good
    KM_HOST, KM_PORT    listen address, default 127.0.0.1:9500
    KM_OLLAMA_URL       upstream chat endpoint
    KM_REASONING_EFFORT none (default) disables Gemma 4's thinking
    KM_CHAT_IMAGE_MAX_SIDE  longest side of page images forwarded to Gemma,
                        0 (default) forwards them as sent
"""

import base64
import gc
import io
import os
import re
import threading
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel

DEVICE = os.environ.get("KM_DEVICE", "mps")
DTYPE_NAME = os.environ.get("KM_DTYPE", "float16")
EMBED_PATH = os.environ["KM_EMBED_PATH"]
RERANK_PATH = os.environ["KM_RERANK_PATH"]
MAX_TILES = int(os.environ.get("KM_MAX_TILES", "6"))
# Rerank runs per query while embeddings are computed once, so rerank pays for
# tiles on every question. Kept separate so lowering it does not make new
# embeddings geometrically inconsistent with the ones already stored.
RERANK_TILES = int(os.environ.get("KM_RERANK_TILES", MAX_TILES))
# Unloading after every question was measured and lost: with Gemma resident
# the weights fall out of the file cache, so each reload costs 4-5 s per model
# against the 4-5 s of Gemma prefill that the freed memory buys back. The
# window therefore spans a conversation, keeping both models warm while
# someone is asking and returning their ~6 GB once nobody is.
IDLE_SECS = float(os.environ.get("KM_IDLE_SECS", "600"))
# The backend sends a page as its text layer plus its image. Rerank cost is
# per token and runs on every query, so a page carrying both can be ranked on
# one of them: `text` drops the image, `image` drops the text, and a passage
# that only has one keeps it either way.
RERANK_INPUT = os.environ.get("KM_RERANK_INPUT", "both")
if RERANK_INPUT not in ("both", "text", "image"):
    raise SystemExit(f"KM_RERANK_INPUT must be both, text or image, not {RERANK_INPUT!r}")
# Token budgets the model card sets per modality.
RERANK_MAX_LENGTH = {"text": 8192, "image": 2048, "text_image": 10240}
# MLX ranks a 50-candidate question about 1.5x faster than MPS
# (docs/measurements §8.14), mostly because it runs each passage at its own
# length instead of padding the batch.
RERANK_BACKEND = os.environ.get("KM_RERANK_BACKEND", "torch")
EMBED_BACKEND = os.environ.get("KM_EMBED_BACKEND", "torch")
BACKENDS = {"embed": EMBED_BACKEND, "rerank": RERANK_BACKEND}
for kind, backend in BACKENDS.items():
    if backend not in ("torch", "mlx"):
        raise SystemExit(f"KM_{kind.upper()}_BACKEND must be torch or mlx, not {backend!r}")
if "torch" in BACKENDS.values():
    import torch

    DTYPE = getattr(torch, DTYPE_NAME)
# MLX keeps freed buffers for reuse with no practical limit by default; every
# passage has its own length, so the cache grew to 13.8 GiB over 36 questions.
# 256 MB cost 2% of rerank time against an unlimited cache (§8.14).
MLX_CACHE_MB = int(os.environ.get("KM_MLX_CACHE_MB", "256"))
OLLAMA_URL = os.environ.get("KM_OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions")
REASONING_EFFORT = os.environ.get("KM_REASONING_EFFORT", "none")
# Gemma 4 spends image tokens in proportion to pixels: a 1600 px page (the
# backend's ceiling) costs ~860 tokens and two thirds of the time to first
# token. Downscaling here keeps that a setting of this machine rather than a
# backend release. The output is deterministic, so a follow-up that resends
# the same pages still reuses Ollama's prompt cache.
CHAT_IMAGE_MAX_SIDE = int(os.environ.get("KM_CHAT_IMAGE_MAX_SIDE", "0"))
# Same quality the backend encodes its context images with.
CHAT_IMAGE_JPEG_QUALITY = 80
# `text\n<img src="data:image/png;base64,..."/>` is how the backend packs one
# multimodal input into a single string.
TEXT_IMAGE = re.compile(r'^(?P<text>.*)\n<img src="(?P<url>data:image/[^"]+)"/>$', re.DOTALL)

app = FastAPI()
# One GPU, one forward pass at a time; MPS gains nothing from interleaving.
_lock = threading.Lock()
_models = {}
_processors = {}
_last_used = {}


def _release(kind):
    """Drops a model and hands its memory back to the system, which Gemma in
    Ollama shares: freeing the tensors alone leaves them in MPS's cache."""
    if _models.pop(kind, None) is None:
        return
    gc.collect()
    if BACKENDS[kind] == "mlx":
        import mlx.core as mx

        mx.clear_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def _after_request(kind):
    _last_used[kind] = time.monotonic()


def _reap_idle():
    """Releases models idle for IDLE_SECS. Runs under the request lock, so a
    model is never dropped while a request is using it."""
    while True:
        time.sleep(0.5)
        with _lock:
            now = time.monotonic()
            for kind in [kind for kind in _models if now - _last_used.get(kind, now) >= IDLE_SECS]:
                _release(kind)


if IDLE_SECS >= 0:
    threading.Thread(target=_reap_idle, name="reap-idle", daemon=True).start()


def _embed_model():
    if "embed" not in _models:
        from sentence_transformers import SentenceTransformer

        # Loaded on the CPU at an explicit dtype and moved afterwards: letting
        # the loader place shards on MPS directly materialised float32 and
        # stalled on the 8 GB M1 (docs/measurements §4).
        model = SentenceTransformer(
            EMBED_PATH,
            trust_remote_code=True,
            device="cpu",
            local_files_only=True,
            model_kwargs={"dtype": DTYPE, "attn_implementation": "sdpa"},
        )
        model.processor.max_input_tiles = MAX_TILES
        _models["embed"] = model.to(DEVICE)
    return _models["embed"]


def _rerank_model():
    if "rerank" not in _models:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            RERANK_PATH,
            dtype=DTYPE,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval()
        _models["rerank"] = model.to(DEVICE)
    return _models["rerank"]


def _mlx_embedder():
    if "embed" not in _models:
        import mlx.core as mx
        from nemotron_vl_mlx import Embedder

        mx.set_cache_limit(MLX_CACHE_MB * 2**20)
        _models["embed"] = Embedder(EMBED_PATH, dtype=getattr(mx, DTYPE_NAME), max_input_tiles=MAX_TILES)
    return _models["embed"]


def _mlx_reranker():
    if "rerank" not in _models:
        import mlx.core as mx
        from nemotron_vl_mlx import Reranker

        mx.set_cache_limit(MLX_CACHE_MB * 2**20)
        _models["rerank"] = Reranker(
            RERANK_PATH, dtype=getattr(mx, DTYPE_NAME), max_input_tiles=RERANK_TILES, use_thumbnail=True
        )
    return _models["rerank"]


def _rerank_processor(modality):
    """The processor carries the modality's token budget, so one per modality."""
    if modality not in _processors:
        from transformers import AutoProcessor

        _processors[modality] = AutoProcessor.from_pretrained(
            RERANK_PATH,
            trust_remote_code=True,
            local_files_only=True,
            max_input_tiles=RERANK_TILES,
            use_thumbnail=True,
            rerank_max_length=RERANK_MAX_LENGTH[modality],
        )
    return _processors[modality]


def _image(data_url):
    try:
        payload = data_url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"invalid image data URL: {error}")


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str]
    input_type: str = "passage"
    modality: list[str] = ["text"]
    encoding_format: str = "float"
    embedding_type: str = "float"
    truncate: str = "NONE"


class RerankQuery(BaseModel):
    text: str


class RerankPassage(BaseModel):
    text: str | None = None
    image: str | None = None


class RerankRequest(BaseModel):
    model: str
    query: RerankQuery
    passages: list[RerankPassage]
    truncate: str = "NONE"


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "dtype": DTYPE_NAME, "loaded": sorted(_models),
            "idle_secs": IDLE_SECS, "tiles": MAX_TILES, "rerank_tiles": RERANK_TILES,
            "rerank_input": RERANK_INPUT, "embed_backend": EMBED_BACKEND, "rerank_backend": RERANK_BACKEND,
            "chat_image_max_side": CHAT_IMAGE_MAX_SIDE}


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="input must not be empty")
    modality = request.modality[0] if request.modality else "text"

    documents = []
    for item in request.input:
        if modality == "text":
            documents.append(item)
        elif modality == "image":
            documents.append(_image(item))
        elif modality == "text_image":
            match = TEXT_IMAGE.match(item)
            if not match:
                raise HTTPException(status_code=400, detail="text_image input is malformed")
            documents.append(
                {"image": _image(match.group("url")), "text": match.group("text").strip()}
            )
        else:
            raise HTTPException(status_code=400, detail=f"unsupported modality: {modality}")

    with _lock:
        try:
            if EMBED_BACKEND == "mlx":
                from nemotron_vl_mlx import Passage

                vectors = _mlx_embedder()(
                    [Passage(text=d) if isinstance(d, str) else Passage(image=d) if isinstance(d, Image.Image)
                     else Passage(text=d["text"], image=d["image"]) for d in documents],
                    "query" if request.input_type == "query" else "passage",
                )
            else:
                with torch.inference_mode():
                    model = _embed_model()
                    if request.input_type == "query":
                        vectors = model.encode_query(documents)
                    else:
                        vectors = model.encode_document(documents)
        finally:
            _after_request("embed")

    return {
        "data": [
            {"index": index, "embedding": [float(value) for value in vector], "object": "embedding"}
            for index, vector in enumerate(vectors)
        ],
        "model": request.model,
        "object": "list",
    }


@app.post("/v1/ranking")
def ranking(request: RerankRequest):
    if not request.passages:
        raise HTTPException(status_code=400, detail="passages must not be empty")

    passages = request.passages
    if RERANK_INPUT == "text":
        passages = [p.model_copy(update={"image": None}) if p.text else p for p in passages]
    elif RERANK_INPUT == "image":
        passages = [p.model_copy(update={"text": None}) if p.image else p for p in passages]
    has_image = any(passage.image for passage in passages)
    has_text = any(passage.text for passage in passages)
    modality = "text_image" if has_image and has_text else "image" if has_image else "text"
    examples = [
        {
            "question": request.query.text,
            "doc_text": passage.text or "",
            "doc_image": _image(passage.image) if passage.image else "",
        }
        for passage in passages
    ]

    with _lock:
        try:
            if RERANK_BACKEND == "mlx":
                from nemotron_vl_mlx import Passage

                scores = _mlx_reranker()(
                    request.query.text,
                    [Passage(text=e["doc_text"], image=e["doc_image"] or None) for e in examples],
                    RERANK_MAX_LENGTH[modality],
                ).tolist()
            else:
                with torch.no_grad():
                    model = _rerank_model()
                    batch = _rerank_processor(modality).process_queries_documents_crossencoder(examples)
                    batch = {
                        key: value.to(DEVICE) if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    }
                    logits = model(**batch, return_dict=True).logits.squeeze(-1)
                    scores = logits.reshape(-1).float().cpu().tolist()
        finally:
            _after_request("rerank")

    return {"rankings": [{"index": index, "logit": score} for index, score in enumerate(scores)]}


def _shrink(url):
    if not url.startswith("data:image/"):
        return url
    image = _image(url)
    if max(image.size) <= CHAT_IMAGE_MAX_SIDE:
        return url
    image.thumbnail((CHAT_IMAGE_MAX_SIDE, CHAT_IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=CHAT_IMAGE_JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def _shrink_images(body):
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_url = part.get("image_url") or {}
                if isinstance(image_url.get("url"), str):
                    image_url["url"] = _shrink(image_url["url"])


@app.post("/v1/chat/completions")
async def chat(request: Request):
    """Pass-through to Ollama. It edits two things: the thinking switch Ollama
    ignores in every other form (`think: false` on this endpoint does nothing),
    and, when configured, the size of the page images."""
    body = await request.json()
    body["reasoning_effort"] = REASONING_EFFORT
    if CHAT_IMAGE_MAX_SIDE > 0:
        await run_in_threadpool(_shrink_images, body)
    client = httpx.AsyncClient(timeout=None)
    upstream = client.build_request("POST", OLLAMA_URL, json=body)

    if not body.get("stream"):
        try:
            response = await client.send(upstream)
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/json"),
            )
        finally:
            await client.aclose()

    response = await client.send(upstream, stream=True)

    async def relay():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        relay(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "text/event-stream"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("KM_HOST", "127.0.0.1"),
                port=int(os.environ.get("KM_PORT", "9500")))
