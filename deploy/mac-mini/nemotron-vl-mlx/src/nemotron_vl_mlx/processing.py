"""Input preparation for the Nemotron VL reranker without PyTorch.

Reproduces ``LlamaNemotronVLRerankProcessor.process_queries_documents_crossencoder``
from the model's remote code: same prompt, same token ids, same padding and
truncation, and the same pixel values, including the bfloat16 rounding the
original applies before the model casts them to its own dtype.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from tokenizers import Tokenizer

IMAGE_SIZE = 512
TOKENS_PER_TILE = 256
IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<IMG_CONTEXT>"
IMG_CONTEXT_ID = 128258
PAD_ID = 128004
PAD_TOKEN = "<|finetune_right_pad_id|>"


@dataclasses.dataclass
class Passage:
    text: str = ""
    image: Image.Image | None = None


@dataclasses.dataclass
class Batch:
    # (batch, length), left padded.
    input_ids: np.ndarray
    attention_mask: np.ndarray
    # (tiles, 512, 512, 3) float32 holding bfloat16-representable values, in
    # passage order; None when no passage has an image.
    pixel_values: np.ndarray | None


def round_to_bfloat16(values: np.ndarray) -> np.ndarray:
    """Rounds float32 values to the nearest bfloat16 (ties to even), as
    ``tensor.to(torch.bfloat16)`` does, and keeps them as float32."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + (((bits >> 16) & 1) + 0x7FFF)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_factor = float("-inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        area_ratio = (ratio[0] * ratio[1] * image_size * image_size) / area
        factor = min(area_ratio, 0.6) * min(
            target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio
        )
        if factor > best_factor:
            best_factor = factor
            best_ratio = ratio
    return best_ratio


def dynamic_tiles(image, max_num, image_size=IMAGE_SIZE, use_thumbnail=True, min_num=1):
    """Splits an image into up to ``max_num`` square tiles, plus a thumbnail of
    the whole image when there is more than one. A single tile is the whole
    image squashed to a square."""
    width, height = image.size
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    ratio = _closest_aspect_ratio(width / height, target_ratios, width, height, image_size)
    target_width = image_size * ratio[0]
    target_height = image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    columns = target_width // image_size

    resized = image.resize((target_width, target_height))
    tiles = []
    for i in range(blocks):
        left = (i % columns) * image_size
        top = (i // columns) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def tile_pixels(tile: Image.Image, image_size=IMAGE_SIZE) -> np.ndarray:
    """One tile as (H, W, 3) values in [-1, 1]: the torchvision transform
    (RGB, bicubic resize, ToTensor, SigLIP normalisation) in channels-last."""
    if tile.mode != "RGB":
        tile = tile.convert("RGB")
    tile = tile.resize((image_size, image_size), Image.Resampling.BICUBIC)
    pixels = np.asarray(tile, dtype=np.uint8).astype(np.float32) / np.float32(255)
    return (pixels - np.float32(0.5)) / np.float32(0.5)


# The tokenizer's model_max_length, which the original truncates to unless
# told otherwise. NVIDIA's model card suggests 2048 for image passages, 8192
# for text and 10240 for image plus text.
RERANK_MAX_LENGTH = 10240


class RerankProcessor:
    def __init__(self, model_dir: str | Path, max_input_tiles: int = 6, use_thumbnail: bool = True):
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.max_input_tiles = max_input_tiles
        self.use_thumbnail = use_thumbnail

    def prompt(self, query: str, passage: Passage) -> tuple[str, list[Image.Image]]:
        content = f"question:{query} \n \n passage:{passage.text}"
        tiles = []
        if passage.image is not None:
            tiles = dynamic_tiles(
                passage.image, self.max_input_tiles, use_thumbnail=self.use_thumbnail
            )
            content = "<image> " + content
            image_tokens = IMG_START + IMG_CONTEXT * (TOKENS_PER_TILE * len(tiles)) + IMG_END
            content = content.replace("<image>", image_tokens, 1)
        else:
            # The original removes the first literal "<image>" even when the
            # passage text is what contains it.
            content = content.replace("<image>", "", 1)
        return content, tiles

    def __call__(self, query: str, passages: Sequence[Passage],
                 max_length: int = RERANK_MAX_LENGTH) -> Batch:
        prompts = []
        pixel_values = []
        for passage in passages:
            prompt, tiles = self.prompt(query, passage)
            prompts.append(prompt)
            pixel_values.extend(tile_pixels(tile) for tile in tiles)

        tokenizer = self.tokenizer
        tokenizer.enable_truncation(max_length, strategy="longest_first", direction="right")
        tokenizer.enable_padding(direction="left", pad_id=PAD_ID, pad_token=PAD_TOKEN)
        encodings = tokenizer.encode_batch(prompts, add_special_tokens=True)
        input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.array(
            [encoding.attention_mask for encoding in encodings], dtype=np.int64
        )

        pixels = None
        if pixel_values:
            pixels = round_to_bfloat16(np.stack(pixel_values))
        return Batch(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixels)


# What sentence-transformers sends llama-nemotron-embed-vl-1b-v2: its prompts
# for text, the processor's own "passage: " prefix for images, and the
# processor's passage budget (p_max_length) for every input, queries included.
QUERY_PROMPT = "query: "
PASSAGE_PROMPT = "passage: "
EMBED_MAX_LENGTH = 4096


class EmbedProcessor:
    def __init__(self, model_dir: str | Path, max_input_tiles: int = 6, use_thumbnail: bool = True,
                 max_length: int = EMBED_MAX_LENGTH):
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length, strategy="longest_first", direction="right")
        self.tokenizer.no_padding()
        self.max_input_tiles = max_input_tiles
        self.use_thumbnail = use_thumbnail

    def prompt(self, item: Passage, input_type: str) -> tuple[str, list[Image.Image]]:
        if item.image is None:
            return (QUERY_PROMPT if input_type == "query" else PASSAGE_PROMPT) + item.text, []
        if input_type == "query":
            raise ValueError("queries are text only")
        tiles = dynamic_tiles(item.image, self.max_input_tiles, use_thumbnail=self.use_thumbnail)
        content = PASSAGE_PROMPT + "<image> " + item.text
        image_tokens = IMG_START + IMG_CONTEXT * (TOKENS_PER_TILE * len(tiles)) + IMG_END
        return content.replace("<image>", image_tokens, 1), tiles

    def __call__(self, items: Sequence[Passage], input_type: str) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Token ids per input (unpadded) and the image tiles of all inputs in
        input order, as (tiles, 512, 512, 3) bfloat16-representable float32."""
        prompts = []
        pixel_values = []
        for item in items:
            prompt, tiles = self.prompt(item, input_type)
            prompts.append(prompt)
            pixel_values.extend(tile_pixels(tile) for tile in tiles)
        encodings = self.tokenizer.encode_batch(prompts, add_special_tokens=True)
        sequences = [np.array(encoding.ids, dtype=np.int64) for encoding in encodings]
        pixels = round_to_bfloat16(np.stack(pixel_values)) if pixel_values else None
        return sequences, pixels
