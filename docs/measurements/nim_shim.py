"""NIM-shaped HTTP front end for the two local Nemotron VL models.

The backend speaks the NVIDIA NIM embedding/reranking contract; the models speak
Python. This translates one into the other. It also passes chat through to
Ollama, whose only purpose here is to switch Gemma 4's thinking off: thinking
costs 107s before the first visible character on this CPU, against 1.5s without.

    KM_DTYPE            bfloat16 (default) | float32
    KM_MAX_TILES        image tiles per page, 6 = model default, lower = faster on CPU
    KM_RERANK_TILES     tiles for rerank only, defaults to KM_MAX_TILES
    KM_RERANK_TEXT_ONLY rank passages that have text on their text alone
    KM_RERANK_DTYPE     dtype for rerank only, defaults to KM_DTYPE
    KM_EVICT            drop the unused model to save memory, at ~15s per swap
    KM_PIN_EMBED        keep the embed model loaded, drop only rerank after use
    KM_THREADS          torch CPU threads, default 4
    KM_PORT             listen port, default 9500
    KM_OLLAMA_URL       upstream chat endpoint
    KM_REASONING_EFFORT none (default) disables Gemma 4's thinking
"""

import base64
import gc
import io
import os
import re
import threading

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
DTYPE = getattr(torch, os.environ.get("KM_DTYPE", "bfloat16"))
MAX_TILES = int(os.environ.get("KM_MAX_TILES", "6"))
# Rerank runs per query while embeddings are computed once, so rerank pays for
# tiles on every question. Kept separate so lowering it does not make new
# embeddings geometrically inconsistent with the ones already stored.
RERANK_TILES = int(os.environ.get("KM_RERANK_TILES", MAX_TILES))
# Reranking a page as image+text costs ~25s here against ~3s as text, and the
# model ranks text natively. With this on, a passage that carries text is ranked
# on its text; a page with no text layer still goes through as an image.
RERANK_TEXT_ONLY = os.environ.get("KM_RERANK_TEXT_ONLY", "false").lower() == "true"
# Reranking text is cheap enough to afford the slower dtype, and half the memory
# is what lets both models stay resident instead of swapping on every query.
RERANK_DTYPE = getattr(torch, os.environ.get("KM_RERANK_DTYPE", os.environ.get("KM_DTYPE", "bfloat16")))
EVICT = os.environ.get("KM_EVICT", "true").lower() == "true"
# Every turn calls embed first, then rerank, then chat. Evicting both ways costs
# two ~15s loads per turn; pinning the embed model and dropping only the rerank
# model after it answers costs one, and the peak (embed + rerank, then embed +
# Gemma) still fits.
PIN_EMBED = os.environ.get("KM_PIN_EMBED", "true").lower() == "true"
# Token budgets the model card sets per modality.
RERANK_MAX_LENGTH = {"text": 8192, "image": 2048, "text_image": 10240}
OLLAMA_URL = os.environ.get("KM_OLLAMA_URL", "http://localhost:11434/v1/chat/completions")
REASONING_EFFORT = os.environ.get("KM_REASONING_EFFORT", "none")
# `text\n<img src="data:image/png;base64,..."/>` is how the backend packs one
# multimodal input into a single string.
TEXT_IMAGE = re.compile(r'^(?P<text>.*)\n<img src="(?P<url>data:image/[^"]+)"/>$', re.DOTALL)

torch.set_num_threads(int(os.environ.get("KM_THREADS", "4")))

app = FastAPI()
# ponytail: one global lock; four CPU cores fit one forward pass at a time anyway.
_lock = threading.Lock()
_models = {}
_processors = {}


def _evict(keep):
    """Both models in float32 plus Gemma 4 in Ollama exceed this machine's 23 GB.
    Evicting costs ~15s of reloading on every query, so it is only worth it when
    they genuinely do not fit: with the rerank model in bfloat16 they do."""
    if not EVICT:
        return
    for kind in [name for name in _models if name != keep]:
        if kind == "embed" and PIN_EMBED:
            continue
        del _models[kind]
        gc.collect()


def _embed_model():
    if "embed" not in _models:
        _evict("embed")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            EMBED_MODEL,
            trust_remote_code=True,
            device="cpu",
            model_kwargs={"torch_dtype": DTYPE, "attn_implementation": "sdpa"},
        )
        model.processor.max_input_tiles = MAX_TILES
        _models["embed"] = model
    return _models["embed"]


def _rerank_model():
    if "rerank" not in _models:
        _evict("rerank")
        from transformers import AutoModelForSequenceClassification

        _models["rerank"] = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL,
            torch_dtype=RERANK_DTYPE,
            trust_remote_code=True,
            attn_implementation="sdpa",
            device_map="cpu",
        ).eval()
    return _models["rerank"]


def _rerank_processor(modality):
    """The processor carries the modality's token budget, so one per modality."""
    if modality not in _processors:
        from transformers import AutoProcessor

        _processors[modality] = AutoProcessor.from_pretrained(
            RERANK_MODEL,
            trust_remote_code=True,
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
    return {"status": "ok", "loaded": sorted(_models), "dtype": str(DTYPE),
            "tiles": MAX_TILES, "rerank_tiles": RERANK_TILES,
            "rerank_text_only": RERANK_TEXT_ONLY,
            "rerank_dtype": str(RERANK_DTYPE), "evict": EVICT, "pin_embed": PIN_EMBED}


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

    with _lock, torch.inference_mode():
        model = _embed_model()
        if request.input_type == "query":
            vectors = model.encode_query(documents)
        else:
            vectors = model.encode_document(documents)

    return {
        "data": [
            {"index": index, "embedding": [float(value) for value in vector], "object": "embedding"}
            for index, vector in enumerate(vectors)
        ],
        "model": request.model,
        "object": "list",
    }


@app.post("/v1/reranking")
def reranking(request: RerankRequest):
    if not request.passages:
        raise HTTPException(status_code=400, detail="passages must not be empty")

    passages = request.passages
    if RERANK_TEXT_ONLY:
        passages = [
            passage.model_copy(update={"image": None}) if passage.text else passage
            for passage in passages
        ]
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

    with _lock, torch.no_grad():
        model = _rerank_model()
        batch = _rerank_processor(modality).process_queries_documents_crossencoder(examples)
        logits = model(**batch, return_dict=True).logits.squeeze(-1)

    scores = logits.reshape(-1).float().tolist()
    if EVICT and PIN_EMBED:
        # Freed here rather than on the next embed call, so the chat model has
        # the memory while it answers.
        with _lock:
            _models.pop("rerank", None)
            gc.collect()
    return {"rankings": [{"index": index, "logit": score} for index, score in enumerate(scores)]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    """Pass-through to Ollama; the only edit is the thinking switch it ignores in
    every other form (`think: false` on this endpoint does nothing)."""
    body = await request.json()
    body["reasoning_effort"] = REASONING_EFFORT
    client = httpx.AsyncClient(timeout=None)
    upstream = client.build_request("POST", OLLAMA_URL, json=body)

    if not body.get("stream"):
        try:
            response = await client.send(upstream)
            return JSONResponse(content=response.json(), status_code=response.status_code)
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

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("KM_PORT", "9500")))
