from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm

from .pixelrepa_weights import PIXELREPA_PRESET_CONFIGS, remap_legacy_state_dict


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = {len(t.shape) for t in tensors}
    if len(shape_lens) != 1:
        raise ValueError("tensors must all have the same number of dimensions")
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*(list(t.shape) for t in tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    if not all(len(set(t[1])) <= 2 for t in expandable_dims):
        raise ValueError("invalid dimensions for broadcastable concatenation")
    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*(t[1] for t in expanded_dims)))
    tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.view(*x.shape[:-2], -1)


class PixelREPARotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        theta=10000,
        num_cls_token=0,
    ):
        super().__init__()
        if custom_freqs is not None:
            freqs = custom_freqs
        else:
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()
            _, dim_freq = cos_img.shape
            cos_pad = torch.ones(num_cls_token, dim_freq, dtype=cos_img.dtype)
            sin_pad = torch.zeros(num_cls_token, dim_freq, dtype=sin_img.dtype)
            self.register_buffer("freqs_cos", torch.cat([cos_pad, cos_img], dim=0), persistent=False)
            self.register_buffer("freqs_sin", torch.cat([sin_pad, sin_img], dim=0), persistent=False)
        else:
            self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]), persistent=False)
            self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]), persistent=False)

    def forward(self, tensor):
        seq_len = tensor.shape[1]
        freqs_cos = self.freqs_cos[:seq_len].to(device=tensor.device, dtype=tensor.dtype)
        freqs_sin = self.freqs_sin[:seq_len].to(device=tensor.device, dtype=tensor.dtype)
        return tensor * freqs_cos[:, None, :] + rotate_half(tensor) * freqs_sin[:, None, :]


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PixelREPAPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class PixelREPATimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=None):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if dtype is not None:
            t_freq = t_freq.to(dtype=dtype)
        return self.mlp(t_freq)


class PixelREPALabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        return self.embedding_table(labels)


class PixelREPAAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q = rope(q)
            k = rope(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        dropout_p = self.attn_drop if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PixelREPASwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class PixelREPABlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        self.attn = PixelREPAAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            eps=eps,
        )
        self.norm2 = RMSNorm(hidden_size, eps=eps)
        self.mlp = PixelREPASwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.act = nn.SiLU()
        self.adaLN_modulation = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def forward(self, x, c, feat_rope=None):
        c = self.act(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be divisible by 2, but got {embed_dim}")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be divisible by 2, but got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class PixelREPATransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        num_classes: int = 1000,
        bottleneck_dim: int = 128,
        in_context_len: int = 32,
        in_context_start: int = 4,
        norm_eps: float = 1e-6,
        model_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
    ):
        super().__init__()
        if num_class_embeds is not None:
            num_classes = int(num_class_embeds)
        if model_type in PIXELREPA_PRESET_CONFIGS:
            preset = PIXELREPA_PRESET_CONFIGS[model_type]
            sample_size = int(preset["sample_size"])
            patch_size = int(preset["patch_size"])
            hidden_size = int(preset["hidden_size"])
            num_layers = int(preset["num_layers"])
            num_attention_heads = int(preset["num_attention_heads"])
            bottleneck_dim = int(preset["bottleneck_dim"])
            in_context_len = int(preset["in_context_len"])
            in_context_start = int(preset["in_context_start"])
            if attention_dropout == 0.0:
                attention_dropout = float(preset["attention_dropout"])
            if dropout == 0.0:
                dropout = float(preset["dropout"])

        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.norm_eps = norm_eps
        self.gradient_checkpointing = False

        self.t_embedder = PixelREPATimestepEmbedder(hidden_size)
        self.y_embedder = PixelREPALabelEmbedder(num_classes, hidden_size)
        self.x_embedder = PixelREPAPatchEmbed(
            img_size=sample_size,
            patch_size=patch_size,
            in_chans=in_channels,
            pca_dim=bottleneck_dim,
            embed_dim=hidden_size,
            bias=True,
        )

        num_patches = self.x_embedder.num_patches
        pos_embed = get_2d_sincos_pos_embed(hidden_size, int(num_patches**0.5))
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=True)

        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size))

        half_head_dim = hidden_size // num_attention_heads // 2
        hw_seq_len = sample_size // patch_size
        self.feat_rope = PixelREPARotaryEmbedding(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0)
        self.feat_rope_incontext = PixelREPARotaryEmbedding(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len,
        )

        self.blocks = nn.ModuleList(
            [
                PixelREPABlock(
                    hidden_size,
                    num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attention_dropout if (num_layers // 4 * 3 > i >= num_layers // 4) else 0.0,
                    proj_drop=dropout if (num_layers // 4 * 3 > i >= num_layers // 4) else 0.0,
                    eps=norm_eps,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_final = RMSNorm(hidden_size, eps=norm_eps)
        self.linear_final = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)
        self.act_final = nn.SiLU()
        self.adaLN_modulation_final = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.LongTensor,
        class_labels: torch.LongTensor,
        return_dict: bool = True,
    ):
        timestep = torch.as_tensor(timestep, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.repeat(sample.shape[0])
        else:
            timestep = timestep.reshape(-1)
            if timestep.shape[0] == 1 and sample.shape[0] > 1:
                timestep = timestep.repeat(sample.shape[0])

        t_emb = self.t_embedder(timestep, dtype=sample.dtype)
        y_emb = self.y_embedder(class_labels).to(dtype=sample.dtype)
        c = t_emb + y_emb

        x = self.x_embedder(sample)
        x = x + self.pos_embed.to(x.dtype)

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb.to(in_context_tokens.dtype)
                x = torch.cat([in_context_tokens, x], dim=1)

            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, c, rope, use_reentrant=False)
            else:
                x = block(x, c, feat_rope=rope)

        if self.in_context_len > 0:
            x = x[:, self.in_context_len :]

        c = self.act_final(c)
        shift, scale = self.adaLN_modulation_final(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear_final(x)

        height = width = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], height, width, self.patch_size, self.patch_size, self.out_channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        output = x.reshape(shape=(x.shape[0], self.out_channels, height * self.patch_size, width * self.patch_size))

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pixelrepa_checkpoint(
        cls,
        checkpoint_path: str,
        weights: Literal["model", "ema1", "ema2"] = "ema1",
        map_location: str = "cpu",
        strict: bool = True,
    ) -> Tuple["PixelREPATransformer2DModel", Dict[str, object]]:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if "args" not in checkpoint:
            raise ValueError("Checkpoint is missing 'args', cannot infer PixelREPA architecture config.")

        ckpt_args = checkpoint["args"]
        if isinstance(ckpt_args, argparse.Namespace):
            args_dict = vars(ckpt_args)
        elif isinstance(ckpt_args, Mapping):
            args_dict = dict(ckpt_args)
        else:
            raise TypeError(f"Unsupported checkpoint args type: {type(ckpt_args)}")

        model_type = args_dict.get("model") or args_dict.get("model_name") or args_dict.get("model_type")
        if model_type not in PIXELREPA_PRESET_CONFIGS:
            raise ValueError(f"Unknown PixelREPA preset '{model_type}'.")

        config = dict(PIXELREPA_PRESET_CONFIGS[model_type])
        config["num_classes"] = int(args_dict.get("class_num") or args_dict.get("num_classes") or 1000)
        config["model_type"] = model_type
        config["attention_dropout"] = float(
            args_dict.get("attn_dropout", args_dict.get("attention_dropout", config["attention_dropout"]))
        )
        config["dropout"] = float(args_dict.get("proj_dropout", args_dict.get("dropout", config["dropout"])))
        if args_dict.get("img_size") is not None:
            config["sample_size"] = int(args_dict["img_size"])

        model = cls(**config)

        if weights == "model":
            if "model" not in checkpoint:
                raise ValueError("Checkpoint key 'model' not found.")
            raw_state = {k: v for k, v in checkpoint["model"].items() if k.startswith("net.")}
        else:
            key = f"model_{weights}"
            if key not in checkpoint:
                raise ValueError(f"Checkpoint key '{key}' not found. Available keys: {list(checkpoint.keys())}")
            raw_state = checkpoint[key]

        state_dict = remap_legacy_state_dict(raw_state)
        model.load_state_dict(state_dict, strict=strict)

        metadata = {
            "checkpoint_path": checkpoint_path,
            "weights": weights,
            "epoch": checkpoint.get("epoch"),
            "model_type": model_type,
            "source_args": checkpoint.get("args"),
        }
        return model, metadata

    def to_pixelrepa_checkpoint(
        self,
        ema_mode: Literal["none", "copy_to_both"] = "copy_to_both",
        prefix: str = "net.",
        ema_prefix: str = "",
    ) -> Dict[str, object]:
        base_state: Dict[str, torch.Tensor] = {}
        ema_state: Dict[str, torch.Tensor] = {}
        for key, value in self.state_dict().items():
            legacy_key = key
            if legacy_key.startswith("norm_final"):
                legacy_key = legacy_key.replace("norm_final", "final_layer.norm_final", 1)
            if legacy_key.startswith("linear_final"):
                legacy_key = legacy_key.replace("linear_final", "final_layer.linear", 1)
            if legacy_key.startswith("adaLN_modulation_final"):
                legacy_key = legacy_key.replace("adaLN_modulation_final", "final_layer.adaLN_modulation", 1)
            legacy_key = legacy_key.replace(".adaLN_modulation.", ".adaLN_modulation.1.")
            base_state[f"{prefix}{legacy_key}"] = value.detach().cpu()
            ema_state[f"{ema_prefix}{legacy_key}"] = value.detach().cpu()

        checkpoint = {"model": base_state}
        if ema_mode == "copy_to_both":
            checkpoint["model_ema1"] = {k: v.clone() for k, v in ema_state.items()}
            checkpoint["model_ema2"] = {k: v.clone() for k, v in ema_state.items()}
        elif ema_mode != "none":
            raise ValueError(f"Unsupported ema_mode='{ema_mode}'.")
        return checkpoint

    @property
    def net(self):
        return self


PixelREPAModel = PixelREPATransformer2DModel
