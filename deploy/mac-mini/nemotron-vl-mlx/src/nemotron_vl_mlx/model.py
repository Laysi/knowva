"""MLX implementation of NVIDIA's Llama Nemotron VL models.

A SigLIP 2 vision encoder, a pixel-shuffle MLP projector and a bidirectional
Llama 3.2 1B, loaded straight from the original ``model.safetensors``. The
embedding model mean-pools the last hidden state; the reranker scores that
mean with one linear layer.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .processing import IMG_CONTEXT_ID


@dataclasses.dataclass
class VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    patch_size: int = 16
    image_size: int = 512
    layer_norm_eps: float = 1e-6


@dataclasses.dataclass
class TextConfig:
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    vocab_size: int = 128267
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling: dict | None = None


@dataclasses.dataclass
class ModelConfig:
    vision: VisionConfig
    text: TextConfig
    downsample_ratio: float = 0.5
    temperature: float = 1.0

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        vision = raw["vision_config"]
        text = raw["llm_config"]
        if raw.get("select_layer", -1) != -1 or raw.get("ps_version") != "v2":
            raise ValueError("only select_layer -1 with pixel shuffle v2 is implemented")
        if text.get("pooling", "avg") != "avg":
            raise ValueError("only average pooling is implemented")
        return cls(
            vision=VisionConfig(**{f.name: vision[f.name] for f in dataclasses.fields(VisionConfig)}),
            text=TextConfig(**{f.name: text[f.name] for f in dataclasses.fields(TextConfig)}),
            downsample_ratio=raw["downsample_ratio"],
            temperature=text.get("temperature", 1.0),
        )


# ---------------------------------------------------------------- vision ----


class SiglipAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dims = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.scale = (dims // self.num_heads) ** -0.5
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x):
        batch, length, dims = x.shape

        def heads(t):
            return t.reshape(batch, length, self.num_heads, -1).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x)), scale=self.scale
        )
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, dims))


class SiglipMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x):
        # gelu_pytorch_tanh
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class SiglipEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]


class SiglipEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            3, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.position_embedding = nn.Embedding(
            (config.image_size // config.patch_size) ** 2, config.hidden_size
        )

    def __call__(self, pixel_values):
        # (tiles, H, W, 3) -> (tiles, H/16 * W/16, hidden), patches row-major.
        patches = self.patch_embedding(pixel_values)
        tiles, rows, cols, dims = patches.shape
        return patches.reshape(tiles, rows * cols, dims) + self.position_embedding.weight


class SiglipVisionTransformer(nn.Module):
    """The vision tower up to ``post_layernorm``; the attention-pooling head
    in the checkpoint is not used by the Nemotron VL models."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embeddings = SiglipEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, pixel_values):
        x = self.embeddings(pixel_values)
        for layer in self.encoder.layers:
            x = layer(x)
        return self.post_layernorm(x)


class SiglipVisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)

    def __call__(self, pixel_values):
        return self.vision_model(pixel_values)


class Projector(nn.Module):
    """``mlp1``: LayerNorm, Linear, GELU (erf), Linear."""

    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.layers = [nn.LayerNorm(in_dims, eps=1e-5), nn.Linear(in_dims, out_dims), nn.Linear(out_dims, out_dims)]

    def __call__(self, x):
        norm, fc1, fc2 = self.layers
        return fc2(nn.gelu(fc1(norm(x))))


def pixel_shuffle(x, scale: float):
    """Pixel shuffle v2 of the original, (n, w, h, c) -> (n, w*s, h*s, c/s^2)."""
    n, w, h, c = x.shape
    x = x.reshape(n, w, int(h * scale), int(c / scale))
    x = x.transpose(0, 2, 1, 3)
    x = x.reshape(n, int(h * scale), int(w * scale), int(c / (scale * scale)))
    return x.transpose(0, 2, 1, 3)


# ------------------------------------------------------------------ text ----


def llama3_inv_freq(config: TextConfig) -> np.ndarray:
    """The rotary inverse frequencies transformers computes for ``llama3``
    scaling, in float32."""
    dim = config.head_dim
    inv_freq = 1.0 / (
        np.float32(config.rope_theta) ** (np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / dim)
    )
    inv_freq = inv_freq.astype(np.float32)
    scaling = config.rope_scaling
    if not scaling:
        return inv_freq
    if scaling.get("rope_type") != "llama3":
        raise ValueError(f"unsupported rope scaling {scaling.get('rope_type')!r}")
    factor = scaling["factor"]
    low_freq_factor = scaling["low_freq_factor"]
    high_freq_factor = scaling["high_freq_factor"]
    old_context_len = scaling["original_max_position_embeddings"]
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = (2 * math.pi / inv_freq).astype(np.float32)
    inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq).astype(np.float32)
    smooth = ((old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)).astype(np.float32)
    smoothed = ((1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama).astype(np.float32)
    medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    return np.where(medium, smoothed, inv_freq_llama).astype(np.float32)


class LlamaAttention(nn.Module):
    def __init__(self, config: TextConfig, rope_freqs: mx.array):
        super().__init__()
        dims = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dims, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dims, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dims, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dims, bias=False)
        self._rope_freqs = rope_freqs

    def __call__(self, x, mask):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch, length, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch, length, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        # Positions are 0..length-1 across the padded batch, as in the
        # original; rotary attention only sees their differences.
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=None, scale=1.0, offset=0, freqs=self._rope_freqs)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=None, scale=1.0, offset=0, freqs=self._rope_freqs)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class LlamaMLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, rope_freqs: mx.array):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, rope_freqs)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def __call__(self, x, mask):
        x = x + self.self_attn(self.input_layernorm(x), mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaBidirectionalModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        # mx.fast.rope takes periods: the reciprocal of the inverse frequencies.
        rope_freqs = mx.array(1.0 / llama3_inv_freq(config))
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LlamaDecoderLayer(config, rope_freqs) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, embeds, attention_mask):
        # Bidirectional: every token attends to every non-padding token.
        mask = None
        if attention_mask is not None:
            mask = attention_mask.astype(mx.bool_)[:, None, None, :]
        x = embeds
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ------------------------------------------------------------- top level ----


class LlamaNemotronVLModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionModel(config.vision)
        self.language_model = LlamaBidirectionalModel(config.text)
        scale = int(1 / config.downsample_ratio)
        self.mlp1 = Projector(config.vision.hidden_size * scale * scale, config.text.hidden_size)

    def extract_feature(self, pixel_values):
        """(tiles, 512, 512, 3) -> (tiles, 256, text hidden)."""
        x = self.vision_model(pixel_values)
        tiles, n, c = x.shape
        side = int(n**0.5)
        x = pixel_shuffle(x.reshape(tiles, side, side, c), self.config.downsample_ratio)
        x = x.reshape(tiles, -1, x.shape[-1])
        return self.mlp1(x)

    def image_tokens(self, pixel_values):
        """(tiles, 512, 512, 3) -> (tiles * 256, text hidden), in tile order."""
        dtype = self.language_model.embed_tokens.weight.dtype
        features = self.extract_feature(pixel_values.astype(dtype))
        return features.reshape(-1, features.shape[-1])

    def embed(self, input_ids, image_tokens=None):
        """Token embeddings with each ``<IMG_CONTEXT>`` replaced, in order, by
        the next of ``image_tokens``."""
        embeds = self.language_model.embed_tokens(input_ids)
        if image_tokens is None:
            return embeds
        batch, length, dims = embeds.shape
        is_image = (input_ids == IMG_CONTEXT_ID).reshape(-1)
        order = mx.maximum(mx.cumsum(is_image.astype(mx.int32)) - 1, 0)
        flat = mx.where(is_image[:, None], image_tokens[order], embeds.reshape(-1, dims))
        return flat.reshape(batch, length, dims)

    def __call__(self, input_ids, attention_mask, pixel_values=None):
        """Last hidden state, after the final norm."""
        image_tokens = None if pixel_values is None else self.image_tokens(pixel_values)
        return self.language_model(self.embed(input_ids, image_tokens), attention_mask)

    def pooled_unpadded(self, sequences, pixel_values=None):
        """Mean of the last hidden state per sequence, in float32, for
        sequences of different lengths, each run on its own so no token is
        padding. The image tiles of all sequences are encoded together, in
        sequence order."""
        image_tokens = None if pixel_values is None else self.image_tokens(pixel_values)
        pooled = []
        offset = 0
        for ids in sequences:
            count = int(np.count_nonzero(np.asarray(ids) == IMG_CONTEXT_ID))
            ids = mx.array(ids)[None]
            tokens = None
            if count:
                tokens = image_tokens[offset:offset + count]
                offset += count
            hidden = self.language_model(self.embed(ids, tokens), None)
            pooled.append(hidden.astype(mx.float32).mean(axis=1))
        return mx.concatenate(pooled, axis=0)


def mean_pool(hidden, attention_mask):
    """Average over non-padding tokens, accumulated in float32."""
    mask = attention_mask.astype(mx.float32)[..., None]
    return (hidden.astype(mx.float32) * mask).sum(axis=1) / mask.sum(axis=1)


class LlamaNemotronVLReranker(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = LlamaNemotronVLModel(config)
        self.score = nn.Linear(config.text.hidden_size, 1, bias=False)

    def head(self, pooled):
        # The original keeps the score head in float32.
        logits = pooled @ self.score.weight.astype(mx.float32).T
        return (logits / self.config.temperature).reshape(-1)

    def __call__(self, input_ids, attention_mask, pixel_values=None):
        """Relevance logits, shape (batch,), for a left-padded batch."""
        hidden = self.model(input_ids, attention_mask, pixel_values)
        return self.head(mean_pool(hidden, attention_mask))

    def score_unpadded(self, sequences, pixel_values=None):
        """Relevance logits for sequences of different lengths; see
        ``LlamaNemotronVLModel.pooled_unpadded``."""
        return self.head(self.model.pooled_unpadded(sequences, pixel_values))


# safetensors metadata "format" of checkpoints that convert.py wrote.
CONVERTED_FORMAT = "nemotron-vl-mlx"


def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Maps checkpoint names and layouts onto the modules above."""
    out = {}
    for name, value in weights.items():
        if ".vision_model.head." in f".{name}":
            continue
        if name.endswith("patch_embedding.weight"):
            # PyTorch (out, in, kh, kw) -> MLX (out, kh, kw, in).
            value = value.transpose(0, 2, 3, 1)
        for torch_index, mlx_index in (("mlp1.0.", "mlp1.layers.0."), ("mlp1.1.", "mlp1.layers.1."), ("mlp1.3.", "mlp1.layers.2.")):
            name = name.replace(torch_index, mlx_index)
        out[name] = value
    return out


def _load(module, model_dir: Path, dtype) -> None:
    weights, metadata = mx.load(str(model_dir / "model.safetensors"), return_metadata=True)
    # convert.py's output is already in this layout; sanitize would transpose
    # the patch convolution a second time.
    if metadata.get("format") != CONVERTED_FORMAT:
        weights = sanitize(weights)
    # Converted and evaluated one tensor at a time, dropping each checkpoint
    # tensor as soon as it is copied, so loading never holds the model twice.
    converted = []
    for name in list(weights):
        value = weights.pop(name).astype(mx.float32 if name == "score.weight" else dtype)
        mx.eval(value)
        converted.append((name, value))
    module.load_weights(converted, strict=True)
    module.eval()


def load_reranker(model_dir: str | Path, dtype=mx.float16) -> LlamaNemotronVLReranker:
    """llama-nemotron-rerank-vl-1b-v2."""
    model_dir = Path(model_dir)
    model = LlamaNemotronVLReranker(ModelConfig.from_dir(model_dir))
    _load(model, model_dir, dtype)
    return model


def load_embedder(model_dir: str | Path, dtype=mx.float16) -> LlamaNemotronVLModel:
    """llama-nemotron-embed-vl-1b-v2, whose checkpoint is the bare model."""
    model_dir = Path(model_dir)
    model = LlamaNemotronVLModel(ModelConfig.from_dir(model_dir))
    _load(model, model_dir, dtype)
    return model
