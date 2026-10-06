# nemotron-vl-mlx

MLX port of NVIDIA's `llama-nemotron-rerank-vl-1b-v2` and
`llama-nemotron-embed-vl-1b-v2`. The Mac mini inference service runs them on
it with `KM_RERANK_BACKEND=mlx` and `KM_EMBED_BACKEND=mlx`
(`../inference/server.py`), where it is a path dependency.

It is published as github.com/Laysi/llama-nemotron-vl-mlx, with the weights
on Hugging Face as `Laysi-ai/llama-nemotron-{rerank,embed}-vl-1b-v2-mlx`
(`../setup/fetch-models.ts` downloads them). `src/` and `tools/*.py` here
match it (it also has `tools/prepare_hub.py` and the model cards);
`tools/knowva/` holds the Knowva measurements.

- The loader reads either the converted weights or NVIDIA's original
  snapshots; `python -m nemotron_vl_mlx.convert SRC DST` converts one.
- `processing.py` reproduces the models' transformers processors without
  PyTorch: identical token ids, attention mask and pixel values
  (`tools/check_processing.py`, `tools/check_embed_processing.py`).
- `model.py` matches PyTorch float32 to 1e-5 in logits with
  `MLX_ENABLE_TF32=0`; MLX uses TF32 for float32 matmuls by default on M5/M6.
- `Reranker` runs each passage at its own length rather than as a padded
  batch. MLX's buffer cache has no practical limit by default; the service
  caps it (`KM_MLX_CACHE_MB`).
- `Embedder` gives what sentence-transformers gives for the embed model:
  `query: ` and `passage: ` prompts, 4096-token truncation, the unnormalised
  mean of the last hidden state; `tools/embed_vectors.py` and
  `tools/compare_embed.py` compare it with the original.
- Measurements: `docs/measurements/README.md` §8.14. `tools/` compares the
  port with the PyTorch original (`reference.py`, `compare.py`) and replays
  Knowva-shaped candidate sets (`tools/knowva/`).

Development: `proto install`, then
`proto exec uv -- uv sync --all-groups --python ~/.proto/tools/python/3.12.15/bin/python`;
the `reference` group brings the PyTorch original, `tools` brings pypdfium2.

Weights: NVIDIA Open Model License. NVIDIA's model code, which this follows:
Apache 2.0. Built with Llama (Llama 3.2 Community License).
