"""Embeds Knowva's corpus and question set on MLX, one input per forward pass
(Knowva's ingestion sends one page per request), and records per-input latency
and the process's physical footprint.

  eg2:      uv run python embed_corpus.py eg2 --model models/embeddinggemma-2-bf16 --budget 280 --out out/eg2-bf16-280.json
  nemotron: ~/knowva-inference/app/inference/.venv/bin/python embed_corpus.py nemotron --out out/nemotron.json

The nemotron backend loads the production inference service's own
nemotron_vl_mlx with the service's settings (float16, 6 tiles, 256 MB cache).
Inputs come from data/ (see run_embeds.sh).
"""

import argparse
import ctypes
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

DATA = Path(__file__).parent / "data"
DOC_PROMPT = "title: none | text: "
QUERY_PROMPTS = {
    "search": "task: search result | query: ",
    "qa": "task: question answering | query: ",
}
NEMOTRON_PATH = Path.home() / "knowva-inference/models/llama-nemotron-embed-vl-1b-v2-mlx"

_libproc = ctypes.CDLL("libproc.dylib")


def footprint_gib():
    """Current and lifetime-maximum phys_footprint (rusage_info_v4), which,
    unlike RSS, counts Metal buffers."""
    info = (ctypes.c_uint64 * 40)()
    if _libproc.proc_pid_rusage(os.getpid(), 4, info) != 0:
        raise OSError("proc_pid_rusage failed")
    # Two words of ri_uuid, then ri_phys_footprint is field 7 and
    # ri_lifetime_max_phys_footprint field 28.
    return round(info[2 + 7] / 2**30, 3), round(info[2 + 28] / 2**30, 3)


def unit_kind(unit):
    if not unit["uses_image"]:
        return "text"
    return "image_text" if (unit["content"] or "").strip() else "image"


def page_image(unit):
    return Image.open(DATA / "images" / unit["image_path"]).convert("RGB")


class Eg2:
    def __init__(self, model_path, budget, audio=True):
        if audio:
            from mlx_vlm import load

            self.model, self.processor = load(model_path)
        else:
            # Knowva rejects audio and video uploads, so the 300M-parameter
            # audio tower can be left out (the model card's selective loading).
            from mlx_vlm.embedding_loader import load_embedding_model
            from mlx_vlm.utils import load_config
            from transformers import AutoProcessor

            config = load_config(Path(model_path))
            config["audio_config"] = None
            self.model = load_embedding_model(Path(model_path), config=config)
            self.processor = AutoProcessor.from_pretrained(model_path)
        self.budget = budget

    def _forward(self, inputs):
        out = self.model(**{key: mx.array(value) for key, value in inputs.items()}).text_embeds
        mx.eval(out)
        return out[0].tolist()

    def text(self, text):
        inputs = self.processor.tokenizer([text], return_tensors="np")
        return inputs, inputs["input_ids"].shape[1]

    def unit(self, unit):
        kind = unit_kind(unit)
        if kind == "text":
            return self.text(DOC_PROMPT + unit["content"])
        content = [{"type": "image", "image": page_image(unit)}]
        conversation = [{"role": "user", "content": content}]
        if kind == "image_text":
            # Prefixes apply to text only; the page text comes first, as in
            # Knowva's Nemotron text_image input.
            content.insert(0, {"type": "text", "text": unit["content"].strip()})
            conversation.insert(0, {"role": "system", "content": DOC_PROMPT})
        inputs = self.processor.apply_chat_template(
            [conversation], tokenize=True, return_dict=True, return_tensors="np", max_soft_tokens=self.budget
        )
        return inputs, inputs["input_ids"].shape[1]

    def queries(self, query):
        return {name: self.text(prompt + query) for name, prompt in QUERY_PROMPTS.items()}

    forward = _forward


class Nemotron:
    def __init__(self):
        from nemotron_vl_mlx import Embedder, Passage

        self.passage = Passage
        self.embedder = Embedder(str(NEMOTRON_PATH), dtype=mx.float16, max_input_tiles=6)

    def unit(self, unit):
        kind = unit_kind(unit)
        if kind == "text":
            passage = self.passage(text=unit["content"])
        elif kind == "image_text":
            passage = self.passage(text=unit["content"].strip(), image=page_image(unit))
        else:
            passage = self.passage(image=page_image(unit))
        return ("passage", passage), None

    def queries(self, query):
        return {"query": (("query", self.passage(text=query)), None)}

    def forward(self, inputs):
        input_type, passage = inputs
        return [float(value) for value in self.embedder([passage], input_type)[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["eg2", "nemotron"])
    parser.add_argument("--model")
    parser.add_argument("--budget", type=int, default=280)
    parser.add_argument("--cache-mb", type=int, default=256)
    parser.add_argument("--no-audio", action="store_true", help="eg2: load without the audio tower")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    units = [json.loads(line) for line in (DATA / "corpus.jsonl").open()]
    questions = json.loads((DATA / "queries.json").read_text())

    mx.set_cache_limit(args.cache_mb * 2**20)
    began = time.perf_counter()
    embedder = Eg2(args.model, args.budget, audio=not args.no_audio) if args.backend == "eg2" else Nemotron()
    load_s = time.perf_counter() - began
    after_load = footprint_gib()

    # Warm up kernels on one input of each kind; not timed.
    for kind in ("text", "image_text", "image"):
        sample = next(unit for unit in units if unit_kind(unit) == kind)
        embedder.forward(embedder.unit(sample)[0])

    vectors, timings = {}, []
    for unit in units:
        started = time.perf_counter()
        inputs, tokens = embedder.unit(unit)
        prepared = time.perf_counter()
        vectors[unit["id"]] = embedder.forward(inputs)
        done = time.perf_counter()
        timings.append({"id": unit["id"], "kind": unit_kind(unit), "tokens": tokens,
                        "prep_s": round(prepared - started, 4), "forward_s": round(done - prepared, 4)})

    # Timed like the units: input preparation (tokenizing) plus the forward pass.
    query_vectors, query_seconds = {}, []
    for question in questions:
        query_vectors[question["id"]] = {}
        started = time.perf_counter()
        prepared = embedder.queries(question["query"])
        prepare_s = (time.perf_counter() - started) / len(prepared)
        for name, (inputs, _) in prepared.items():
            started = time.perf_counter()
            query_vectors[question["id"]][name] = embedder.forward(inputs)
            query_seconds.append(prepare_s + time.perf_counter() - started)

    summary = {}
    for kind in ("text", "image_text", "image"):
        rows = [row for row in timings if row["kind"] == kind]
        if rows:
            total = [row["prep_s"] + row["forward_s"] for row in rows]
            summary[kind] = {"count": len(rows), "mean_s": round(statistics.mean(total), 4),
                             "median_s": round(statistics.median(total), 4), "max_s": round(max(total), 4),
                             "sum_s": round(sum(total), 2)}
    meta = {
        "backend": args.backend, "model": args.model or str(NEMOTRON_PATH),
        **({"budget": args.budget, "audio_tower": not args.no_audio} if args.backend == "eg2" else {}),
        "cache_mb": args.cache_mb, "load_s": round(load_s, 2), "footprint_after_load_gib": after_load[0],
        "footprint_end_gib": footprint_gib()[0], "footprint_lifetime_max_gib": footprint_gib()[1],
        "mlx_peak_gib": round(mx.get_peak_memory() / 2**30, 3), "units": summary,
        "query_mean_s": round(statistics.mean(query_seconds), 4), "mlx_version": mx.__version__,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "units": vectors, "queries": query_vectors,
                                          "timings": timings}))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
