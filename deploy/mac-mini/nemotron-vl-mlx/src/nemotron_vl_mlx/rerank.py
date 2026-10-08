"""One call from a query and its passages to relevance logits."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from .model import load_reranker
from .processing import IMG_CONTEXT_ID, RERANK_MAX_LENGTH, TOKENS_PER_TILE, Passage, RerankProcessor


class Reranker:
    def __init__(self, model_dir: str | Path, dtype=mx.float16, max_input_tiles: int = 6,
                 use_thumbnail: bool = True, padded: bool = False):
        """``padded`` runs a request as one left-padded batch, as the original
        does; otherwise each passage runs at its own length, which skips the
        padding tokens and changes the logits only by rounding."""
        self.padded = padded
        self.model = load_reranker(model_dir, dtype=dtype)
        self.processor = RerankProcessor(model_dir, max_input_tiles=max_input_tiles,
                                         use_thumbnail=use_thumbnail)

    def __call__(self, query: str, passages: Sequence[Passage],
                 max_length: int = RERANK_MAX_LENGTH) -> np.ndarray:
        """Logits in passage order. ``max_length`` is the token budget per
        query-passage pair; see ``processing.RERANK_MAX_LENGTH``."""
        batch = self.processor(query, passages, max_length)
        tiles = 0 if batch.pixel_values is None else batch.pixel_values.shape[0]
        image_tokens = int((batch.input_ids == IMG_CONTEXT_ID).sum())
        if image_tokens != tiles * TOKENS_PER_TILE:
            # Truncation cut into an image; the original would misalign the
            # image features with their positions.
            raise ValueError(f"{image_tokens} image tokens for {tiles} tiles; raise max_length")
        pixels = None if batch.pixel_values is None else mx.array(batch.pixel_values)
        if self.padded:
            logits = self.model(mx.array(batch.input_ids), mx.array(batch.attention_mask), pixels)
        else:
            sequences = [ids[mask.astype(bool)] for ids, mask in zip(batch.input_ids, batch.attention_mask)]
            logits = self.model.score_unpadded(sequences, pixels)
        return np.array(logits.astype(mx.float32))
