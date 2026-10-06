"""Embeddings for queries and passages (text, image, or both)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from .model import load_embedder
from .processing import IMG_CONTEXT_ID, TOKENS_PER_TILE, EmbedProcessor, Passage


class Embedder:
    def __init__(self, model_dir: str | Path, dtype=mx.float16, max_input_tiles: int = 6,
                 use_thumbnail: bool = True):
        self.model = load_embedder(model_dir, dtype=dtype)
        self.processor = EmbedProcessor(model_dir, max_input_tiles=max_input_tiles, use_thumbnail=use_thumbnail)

    def __call__(self, items: Sequence[Passage], input_type: str = "passage") -> np.ndarray:
        """(len(items), 2048) float32: the mean of the last hidden state over
        every token, as sentence-transformers' mean pooling gives, unnormalised.
        ``input_type`` is ``query`` or ``passage``."""
        if input_type not in ("query", "passage"):
            raise ValueError(f"input_type must be query or passage, not {input_type!r}")
        sequences, pixels = self.processor(items, input_type)
        tiles = 0 if pixels is None else pixels.shape[0]
        image_tokens = sum(int(np.count_nonzero(ids == IMG_CONTEXT_ID)) for ids in sequences)
        if image_tokens != tiles * TOKENS_PER_TILE:
            raise ValueError(f"{image_tokens} image tokens for {tiles} tiles; truncation cut into an image")
        pooled = self.model.pooled_unpadded(sequences, None if pixels is None else mx.array(pixels))
        return np.array(pooled)
