# nemotron-vl-mlx

MLX port of NVIDIA's `llama-nemotron-rerank-vl-1b-v2`, loaded straight from the
original `model.safetensors`. The Mac mini inference service runs rerank on it
with `KM_RERANK_BACKEND=mlx` (`../inference/server.py`), where it is a path
dependency; it lives here until it is published on its own (#47).

- `processing.py` reproduces the model's transformers processor without PyTorch:
  identical token ids, attention mask and pixel values
  (`tools/check_processing.py`).
- `model.py` matches PyTorch float32 to 1e-5 in logits with
  `MLX_ENABLE_TF32=0`; MLX uses TF32 for float32 matmuls by default on M5/M6.
- `Reranker` runs each passage at its own length rather than as a padded
  batch. MLX's buffer cache has no practical limit by default; the service
  caps it (`KM_MLX_CACHE_MB`).
- Measurements: `docs/measurements/README.md` §8.14. `tools/` compares the
  port with the PyTorch original (`reference.py`, `compare.py`) and replays
  Knowva-shaped candidate sets (`tools/knowva/`).

Development: `proto install`, then
`proto exec uv -- uv sync --all-groups --python ~/.proto/tools/python/3.12.15/bin/python`;
the `reference` group brings the PyTorch original, `tools` brings pypdfium2.

Weights: NVIDIA Open Model License. NVIDIA's model code, which this follows:
Apache 2.0. Built with Llama (Llama 3.2 Community License).
