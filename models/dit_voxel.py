"""
References:
    - DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/unet3d.py
    - Latte: https://github.com/Vchitect/Latte/blob/main/models/latte.py
"""
import contextlib
import warnings
from dataclasses import dataclass
from math import prod
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import utils3d.torch.transforms as utils3d_t
from einops import rearrange
from timm.models.vision_transformer import Mlp
from torch import nn

from models.attention import MultiHeadRMSNorm
from models.embeddings import (ActionCameraEmbedder, TimestepEmbedder, VoxelPatchEmbedder,
                               RotaryEmbedding, apply_rotary_emb, PixelPatchEmbedder,
                               AbsolutePositionEmbedder, camera_to_raymap, raymap_from_view_projection)
from utils.camera_util import camera_params_to_matrices, project_voxel_local_to_worldcam


@contextlib.contextmanager
def allclose_accept_scalar_zero():
    """Patch torch.allclose to accept scalar zero as close to any tensor."""
    _orig_allclose = torch.allclose

    def _patched_allclose(a, b, *args, **kwargs):
        if not torch.is_tensor(b):
            b = torch.zeros_like(a)
        return _orig_allclose(a, b, *args, **kwargs)

    torch.allclose = _patched_allclose
    try:
        yield
    finally:
        torch.allclose = _orig_allclose


@dataclass
class VoxelDitDenoiserArgs:
    """Configuration for DiT Denoiser."""

    input_x : int = 12
    input_y : int = 12
    input_z : int = 12
    patch_size : int = 2
    in_channels : int = 48
    hidden_size : int = 1024
    depth : int = 12
    num_heads : int = 16
    mlp_ratio : int = 4
    pixel_cond_shape : Tuple[int, ...] = (16, 36, 64)
    pixel_cond_patch_size : int = 2
    cross_cond_hidden_size : int = 1024
    action_cond_dim : int = 23
    camera_cond_dim : int = 10
    camera_pos_scaling_factor : float = 1.0
    timestep_scaling_factor: float = 1000.0
    joint_action_camera_emb : bool = True
    context_window_size : int = 8
    camera_use_ape: bool = True # Use absolute position embedder for camera condition
    voxel_use_ape: bool = True # Use absolute position embedder for spatial blocks
    pixel_use_raymap: bool = True # Use camera raymap embedder when injecting the pixel condition
    legacy_pixel_use_raymap: bool = False # defaults to true for old checkpoints (Run5/6)
    voxel_rotary_max_freq : int = 512
    pixel_rotary_max_freq : int = 256
    qk_rms_norm : bool = True

    @property
    def name(self) -> str:
        return "VoxelDiT"


def modulate(x, shift, scale):
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // scale.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def gate(x, g):
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


@torch.compiler.disable
def _kv_cache_rope_offset(rotary_emb, axial_freqs, global_start_idx, spatial_sequence_shape=None):
    """Compute RoPE position offset for KV cache mode. Disabled from torch.compile."""
    idx = global_start_idx.item() if isinstance(global_start_idx, torch.Tensor) else global_start_idx
    if spatial_sequence_shape is not None:
        warnings.warn("_kv_cache_rope_offset with spatial_sequence_shape doubles spatial frequencies — only safe with temporal-only RoPE configs")
        encoded_offset = rotary_emb.get_axial_freqs(1, *spatial_sequence_shape, offset=idx).reshape(1, prod(spatial_sequence_shape), -1)
    else:
        encoded_offset = rotary_emb.get_axial_freqs(1, offset=idx)
    return axial_freqs + encoded_offset


@torch.compiler.disable
def _kv_cache_update(kv_cache, k, v, global_start_idx, T_new):
    """Update KV cache with new entries. Disabled from torch.compile to avoid recompilation."""
    idx = global_start_idx.item() if isinstance(global_start_idx, torch.Tensor) else global_start_idx
    global_end_idx = idx + T_new
    g_end = kv_cache["global_end_index"].item() if isinstance(kv_cache["global_end_index"], torch.Tensor) else kv_cache["global_end_index"]
    window_moved = global_end_idx > g_end

    cache_size = kv_cache["k"].shape[2]
    l_end = kv_cache["local_end_index"].item() if isinstance(kv_cache["local_end_index"], torch.Tensor) else kv_cache["local_end_index"]
    local_end_idx = l_end

    if not window_moved:
        local_end_idx = max(0, local_end_idx - T_new)

    if window_moved and local_end_idx + T_new > cache_size:
        n_remove = local_end_idx + T_new - cache_size
        n_keep = local_end_idx - n_remove
        kv_cache["k"][:, :, :n_keep] = kv_cache["k"][:, :, n_remove:local_end_idx].clone()
        kv_cache["v"][:, :, :n_keep] = kv_cache["v"][:, :, n_remove:local_end_idx].clone()
        local_end_idx -= n_remove

    kv_cache["k"][:, :, local_end_idx:local_end_idx+T_new] = k
    kv_cache["v"][:, :, local_end_idx:local_end_idx+T_new] = v

    if isinstance(kv_cache["global_end_index"], torch.Tensor):
        kv_cache["global_end_index"].fill_(global_end_idx)
        kv_cache["local_end_index"].fill_(local_end_idx + T_new)
    else:
        kv_cache["global_end_index"] = global_end_idx
        kv_cache["local_end_index"] = local_end_idx + T_new

    return kv_cache["k"][:, :, :local_end_idx+T_new], kv_cache["v"][:, :, :local_end_idx+T_new]


class VoxelTemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.is_causal = is_causal

        self.qk_rms_norm = qk_rms_norm
        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)

    def forward(self, x: torch.Tensor, kv_cache=None, global_start_idx=0, prefill=False):
        B, T, S, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T S (h d) -> (B S) h T d", h=self.heads)
        k = rearrange(k, "B T S (h d) -> (B S) h T d", h=self.heads)
        v = rearrange(v, "B T S (h d) -> (B S) h T d", h=self.heads)

        axial_freqs = self.rotary_emb.axial_freqs[:T]

        # When using KV cache, offset RoPE to encode global position
        if kv_cache is not None:
            axial_freqs = _kv_cache_rope_offset(self.rotary_emb, axial_freqs, global_start_idx)

        # Correct order: QK RMSNorm first, then RoPE
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        q = apply_rotary_emb(axial_freqs, q)
        k = apply_rotary_emb(axial_freqs, k)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # KV cache management (in compile-disabled helper to avoid dynamo guards)
        if kv_cache is not None:
            k, v = _kv_cache_update(kv_cache, k, v, global_start_idx, T)

        # During prefill (T>1 with KV cache), causal mask is needed.
        # During single-frame decode, cache only contains past frames so no mask needed.
        use_causal = self.is_causal and (kv_cache is None or prefill)
        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=use_causal)

        x = rearrange(x, "(B S) h T d -> B T S (h d)", B=B, S=S)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x

class VoxelSpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb

        self.qk_rms_norm = qk_rms_norm
        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)

    def forward(self, x: torch.Tensor):
        B, T, S, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T S (h d) -> (B T) h S d", h=self.heads)
        k = rearrange(k, "B T S (h d) -> (B T) h S d", h=self.heads)
        v = rearrange(v, "B T S (h d) -> (B T) h S d", h=self.heads)

        # Correct order: QK RMSNorm first, then RoPE
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        if self.rotary_emb:
            freqs = self.rotary_emb.axial_freqs
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        x = rearrange(x, "(B T) h S d -> B T S (h d)", B=B)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x

class VoxelPixelCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_channels: int,
        heads: int,
        dim_head: int,
        rotary_emb: Optional[RotaryEmbedding] = None,
        rotary_cond_emb: Optional[RotaryEmbedding] = None,
        is_causal: bool = False,
        qk_rms_norm: bool = False,
        context_window_size: int = 32
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.cond_channels = cond_channels
        self.rotary_emb = rotary_emb
        self.rotary_cond_emb = rotary_cond_emb
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(cond_channels, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)
        self.is_causal = is_causal
        self.qk_rms_norm = qk_rms_norm
        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
        if self.is_causal:
            main_seq_context = context_window_size
            cond_seq_context = context_window_size
            attn_mask = torch.ones(main_seq_context, cond_seq_context, dtype=torch.bool).tril(
                diagonal=cond_seq_context - main_seq_context)
            attn_mask = attn_mask[:, None, :, None]  # (T, 1, Tc, 1)
            self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, kv_cache=None, global_start_idx=0, prefill=False):
        B, T, S, D = x.shape
        _, Tc, Sc, Dc = cond.shape

        q = self.to_q(x)
        k, v = self.to_kv(cond).chunk(2, dim=-1)

        # full spatio-temporal attn
        q = rearrange(q, "B T S (h d) -> B h T S d", h=self.heads)
        k = rearrange(k, "B Tc Sc (h d) -> B h Tc Sc d", h=self.heads)
        v = rearrange(v, "B Tc Sc (h d) -> B h Tc Sc d", h=self.heads)

        # Correct order: QK RMSNorm first, then RoPE
        if self.qk_rms_norm:
            q = rearrange(q, "B h T S d -> B h (T S) d", B=B, T=T, S=S)
            k = rearrange(k, "B h Tc Sc d -> B h (Tc Sc) d", B=B, Tc=Tc, Sc=Sc)
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
            q = rearrange(q, "B h (T S) d -> B h T S d", B=B, T=T, S=S)
            k = rearrange(k, "B h (Tc Sc) d -> B h Tc Sc d", B=B, Tc=Tc, Sc=Sc)

        # apply rotary embeddings
        if self.rotary_emb:
            freqs = self.rotary_emb.axial_freqs[:T]
            cond_freqs = self.rotary_cond_emb.axial_freqs[:Tc]

            if kv_cache is not None:
                freqs = _kv_cache_rope_offset(self.rotary_emb, freqs, global_start_idx, self.rotary_emb.spatial_sequence_shape)
                cond_freqs = _kv_cache_rope_offset(self.rotary_cond_emb, cond_freqs, global_start_idx, self.rotary_cond_emb.spatial_sequence_shape)

            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(cond_freqs, k)

        # KV cache management (in compile-disabled helper to avoid dynamo guards)
        if kv_cache is not None:
            k, v = _kv_cache_update(kv_cache, k, v, global_start_idx, Tc)

            Tc_cached = k.shape[2]
            # Flatten cached K, V for attention
            k = rearrange(k, "B h Tc Sc d -> B h (Tc Sc) d")
            v = rearrange(v, "B h Tc Sc d -> B h (Tc Sc) d")

            # Q: flatten spatial
            q = rearrange(q, "B h T S d -> B h (T S) d", B=B, T=T, S=S)

            # During prefill (T>1), causal mask is needed.
            # During single-frame decode, cache only contains past + current frames.
            if prefill and self.is_causal:
                attn_mask = self.attn_mask[:T, :, :Tc_cached, :]
                attn_mask = attn_mask.expand(T, S, Tc_cached, Sc)
                attn_mask = rearrange(attn_mask, "T Sq Tc Sk -> (T Sq) (Tc Sk)", Sq=S, Sk=Sc)
            else:
                attn_mask = None
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_mask)

        else:
            # prepare for attn
            q = rearrange(q, "B h T S d -> B h (T S) d", B=B, T=T, S=S)
            k = rearrange(k, "B h Tc Sc d -> B h (Tc Sc) d", B=B, Tc=Tc, Sc=Sc)
            v = rearrange(v, "B h Tc Sc d -> B h (Tc Sc) d", B=B, Tc=Tc, Sc=Sc)

            if self.is_causal:
                attn_mask = self.attn_mask[:T, :, :Tc, :]  # (T, 1, Tc, 1)
                attn_mask = attn_mask.expand(T, S, Tc, Sc)      # (T, S_q, Tc, S_k)
                attn_mask = rearrange(attn_mask, "T S_q Tc S_k -> (T S_q) (Tc S_k)", S_q=S, S_k=Sc)
            else:
                attn_mask = None

            x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_mask)

        x = rearrange(x, "B h (T S) d -> B T S (h d)", B=B, T=T, S=S)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x


class SpatioTemporalCrossVoxelDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        cross_cond_channels,
        num_heads,
        mlp_ratio=4.0,
        is_causal=True,
        qk_rms_norm=False,
        context_window_size=32,
        spatial_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
        cross_voxel_emb: Optional[RotaryEmbedding] = None,
        cross_pixel_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.is_causal = is_causal
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.s_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_attn = VoxelSpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_emb,
            qk_rms_norm=qk_rms_norm,
        )
        self.s_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.s_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.t_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_attn = VoxelTemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            is_causal=is_causal,
            rotary_emb=temporal_rotary_emb,
            qk_rms_norm=qk_rms_norm,
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.cross_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = VoxelPixelCrossAttention(
            hidden_size,
            cross_cond_channels,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            is_causal=is_causal,
            rotary_emb=cross_voxel_emb,
            rotary_cond_emb=cross_pixel_emb,
            qk_rms_norm=qk_rms_norm,
            context_window_size=context_window_size

        )
        self.cross_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.cross_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, cross_cond, temporal_kv_cache=None, cross_kv_cache=None, global_start_idx=0, prefill=False):
        B, T, S, D = x.shape

        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa), kv_cache=temporal_kv_cache, global_start_idx=global_start_idx, prefill=prefill), t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        # cross attention block
        cross_shift_msa, cross_scale_msa, cross_gate_msa, cross_shift_mlp, cross_scale_mlp, cross_gate_mlp = self.cross_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.cross_attn(modulate(self.cross_norm1(x), cross_shift_msa, cross_scale_msa), cross_cond, kv_cache=cross_kv_cache, global_start_idx=global_start_idx, prefill=prefill), cross_gate_msa)
        x = x + gate(self.cross_mlp(modulate(self.cross_norm2(x), cross_shift_mlp, cross_scale_mlp)), cross_gate_mlp)

        return x

class VoxelFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class VoxelDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_x=16,
        input_y=16,
        input_z=16,
        patch_size=2,
        in_channels=8,
        hidden_size=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        pixel_cond_shape=(16, 18, 32), # (C, H, W) of the pixel-based conditioning input
        pixel_cond_patch_size=2,
        cross_cond_hidden_size=1024,
        action_cond_dim=23,
        camera_cond_dim=8,
        camera_pos_scaling_factor=1.0,
        timestep_scaling_factor=1000.0,
        joint_action_camera_emb: bool = False,
        context_window_size=4,
        camera_use_ape=False,
        voxel_use_ape=False,
        pixel_use_raymap=False,
        legacy_pixel_use_raymap=True,
        voxel_rotary_max_freq=1024,
        pixel_rotary_max_freq=256,
        qk_rms_norm=True,
    ):
        super().__init__()
        self.vox_grid_size = (input_x, input_y, input_z)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.timestep_scaling_factor = timestep_scaling_factor
        self.context_window_size = context_window_size
        self.batch_size = None
        self.temporal_kv_caches = []
        self.cross_kv_caches = []

        self.x_embedder = VoxelPatchEmbedder(input_x, input_y, input_z, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        vox_x, vox_y, vox_z = self.x_embedder.grid_size
        self.cross_cond_embedder = PixelPatchEmbedder(pixel_cond_shape[1], pixel_cond_shape[2], pixel_cond_patch_size, pixel_cond_shape[0], cross_cond_hidden_size, flatten=False)

        # position embedders
        self.temporal_rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            cache_if_possible=True,
            temporal_context_length=context_window_size,
        )
        self.camera_use_ape = camera_use_ape
        self.voxel_use_ape = voxel_use_ape
        self.pixel_use_raymap = pixel_use_raymap
        self.legacy_pixel_use_raymap = legacy_pixel_use_raymap

        if camera_use_ape:
            assert camera_cond_dim == 10, "--camera_use_ape only supported for [rot_6d, xyz, fov] representation"
            camera_pos_scaling_factor = 1.0  # ignore input scaling factor when using APE
            self.cam_embedding_layer_rot = AbsolutePositionEmbedder(
                channels=255,
                in_channels=6,
                pos_range=4
            )
            self.cam_embedding_layer_xyz = AbsolutePositionEmbedder(
                channels=256,
                in_channels=3,
                pos_range=4/(input_x*3) # xyz pos range is in +/- 2/vox_orig_size and vox_orig_size = input_x*3
            )
            camera_cond_dim = 255 + 256 + 1  # new camera cond dim after APE
        self.camera_pos_scaling_factor = camera_pos_scaling_factor
        if voxel_use_ape:
            assert vox_x == vox_y == vox_z, "--voxel_use_ape only supported for uniform grid sizes"
            # sinusoidal spatial pos enc
            pos_embedder = AbsolutePositionEmbedder(
                channels=hidden_size,
                in_channels=3,
                pos_range=2
            )
            coords = torch.meshgrid(*[torch.arange(res) for res in [vox_x] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("voxel_spatial_emb", pos_emb, persistent=False)
            # temporal rope
            self.cross_voxel_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            cache_if_possible=True,
            temporal_context_length=context_window_size,
            )
            freqs = self.cross_voxel_emb.axial_freqs.unsqueeze(1)
            self.cross_voxel_emb.reset_axial_freqs(freqs)
        else:
            self.voxel_spatial_emb = RotaryEmbedding(
                dim=hidden_size // num_heads // 3,
                freqs_for="voxel",
                max_freq=voxel_rotary_max_freq,
                cache_if_possible=True,
                spatial_sequence_shape=(vox_x, vox_y, vox_z)
            )
            
            self.cross_voxel_emb = RotaryEmbedding(
                dim=hidden_size // num_heads // 4,
                freqs_for="voxeltime",
                max_freq=voxel_rotary_max_freq,
                cache_if_possible=True,
                spatial_sequence_shape=(vox_x, vox_y, vox_z),
                temporal_context_length=context_window_size,
            )

        if pixel_use_raymap:
            # Store pixel latent resolution for raymap generation
            self.pixel_latent_h = pixel_cond_shape[1]
            self.pixel_latent_w = pixel_cond_shape[2]
            # Use temporal-only RoPE since raymap provides spatial info
            self.cross_pixel_emb = RotaryEmbedding(
                dim=hidden_size // num_heads,
                cache_if_possible=True,
                temporal_context_length=context_window_size,
            )
            freqs = self.cross_pixel_emb.axial_freqs.unsqueeze(1)
            self.cross_pixel_emb.reset_axial_freqs(freqs)
            # Update cross_cond_embedder to accept raymap channels (6) concatenated with pixel latents
            self.cross_cond_embedder = PixelPatchEmbedder(
                pixel_cond_shape[1], pixel_cond_shape[2], pixel_cond_patch_size,
                pixel_cond_shape[0] + 6,  # +6 for raymap (origin xyz + direction xyz)
                cross_cond_hidden_size, flatten=False
            )
        else:
            pix_h, pix_w = self.cross_cond_embedder.grid_size
            self.cross_pixel_emb = RotaryEmbedding(
                dim=hidden_size // num_heads // 3,
                freqs_for="pixeltime",
                max_freq=pixel_rotary_max_freq,
                cache_if_possible=True,
                spatial_sequence_shape=(pix_h, pix_w),
                temporal_context_length=context_window_size,
            )
        if joint_action_camera_emb and action_cond_dim > 0 and camera_cond_dim > 0:
            self.action_camera_embedder = ActionCameraEmbedder(action_cond_dim, camera_cond_dim, hidden_size)
            self.action_embedder = None
            self.camera_embedder = None
        else:
            self.action_embedder = nn.Linear(action_cond_dim, hidden_size) if action_cond_dim > 0 else nn.Identity()
            self.camera_embedder = nn.Linear(camera_cond_dim, hidden_size) if camera_cond_dim > 0 else nn.Identity()
            self.action_camera_embedder = None

        self.blocks = nn.ModuleList(
            [
                SpatioTemporalCrossVoxelDiTBlock(
                    hidden_size,
                    cross_cond_hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    qk_rms_norm=qk_rms_norm,
                    context_window_size=context_window_size,
                    spatial_emb=self.voxel_spatial_emb if not voxel_use_ape else None,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                    cross_voxel_emb=self.cross_voxel_emb,
                    cross_pixel_emb=self.cross_pixel_emb,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = VoxelFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        w = self.cross_cond_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.cross_cond_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize joint action-camera embedding MLP:
        if self.action_camera_embedder:
            nn.init.normal_(self.action_camera_embedder.action_mlp[0].weight, std=0.06)
            nn.init.normal_(self.action_camera_embedder.camera_mlp[0].weight, std=0.06)
            nn.init.normal_(self.action_camera_embedder.joint_mlp[0].weight, std=0.06)
            nn.init.normal_(self.action_camera_embedder.joint_mlp[2].weight, std=0.06)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.cross_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.cross_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _initialize_kv_caches(self):
        vox_x, vox_y, vox_z = self.x_embedder.grid_size
        S_vox = vox_x * vox_y * vox_z
        pix_h, pix_w = self.cross_cond_embedder.grid_size
        S_pix = pix_h * pix_w
        head_dim = self.hidden_size // self.num_heads
        device = self.x_embedder.proj.weight.device
        dtype = self.x_embedder.proj.weight.dtype

        self.temporal_kv_caches = []
        self.cross_kv_caches = []
        for _ in range(self.depth):
            # Temporal self-attention cache: (B*S_vox, heads, ctx_win, head_dim)
            temporal_shape = (self.batch_size * S_vox, self.num_heads, self.context_window_size, head_dim)
            self.temporal_kv_caches.append({
                "k": torch.zeros(temporal_shape, device=device, dtype=dtype),
                "v": torch.zeros(temporal_shape, device=device, dtype=dtype),
                "global_end_index": torch.tensor(0, dtype=torch.long, device=device),
                "local_end_index": torch.tensor(0, dtype=torch.long, device=device),
            })
            # Cross-attention cache: (B, heads, ctx_win, S_pix, head_dim)
            cross_shape = (self.batch_size, self.num_heads, self.context_window_size, S_pix, head_dim)
            self.cross_kv_caches.append({
                "k": torch.zeros(cross_shape, device=device, dtype=dtype),
                "v": torch.zeros(cross_shape, device=device, dtype=dtype),
                "global_end_index": torch.tensor(0, dtype=torch.long, device=device),
                "local_end_index": torch.tensor(0, dtype=torch.long, device=device),
            })

    def _reset_kv_cache_indices(self):
        """Reset KV cache indices to zero without reallocating buffers.

        Called before a prefill pass so the entire current window is
        written fresh into the cache.
        """
        for cache in self.temporal_kv_caches:
            cache["global_end_index"].fill_(0)
            cache["local_end_index"].fill_(0)
        for cache in self.cross_kv_caches:
            cache["global_end_index"].fill_(0)
            cache["local_end_index"].fill_(0)

    def unpatchify(self, x):
        """
        x: (N, X, Y, Z, patch_size**3 * C)
        vox: (N, C, X * patch_size, Y * patch_size, Z * patch_size)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        X, Y, Z = x.shape[1:4]

        x = x.reshape(shape=(x.shape[0], X, Y, Z, p, p, p, c))
        x = torch.einsum("nxyzpqrc->ncxpyqzr", x)
        vox = x.reshape(shape=(x.shape[0], c, X * p, Y * p, Z * p))
        return vox

    def forward(self, x, t, external_cond=None, global_start_idx=None, prefill=False):
        """
        Forward pass of DiT.
        x: (B, T, C, X, Y, Z) tensor of spatial inputs (voxels or latent representations of voxels)
        t: (B, T,) tensor of diffusion timesteps
        prefill: if True, enables causal masking with KV cache for multi-frame prefill passes
        """
        if global_start_idx is None:
            global_start_idx = 0

        B, T, C, X, Y, Z = x.shape
        cross_cond = None

        # add spatial embeddings
        x = rearrange(x, "b t c x y z -> (b t) c x y z")
        x = self.x_embedder(x)  # (B*T, C, X, Y, Z) -> (B*T, X/2, Y/2, Z/2, D) , C = 8, D = d_model
        # restore shape
        x = rearrange(x, "(b t) x y z d -> b t (x y z) d", t=T)
        if self.voxel_use_ape:
            x = x + self.voxel_spatial_emb[None, None]
        # embed noise steps
        t = rearrange(t, "b t -> (b t)")
        c = self.t_embedder(self.timestep_scaling_factor * t)  # (N, D)
        c = rearrange(c, "(b t) d -> b t d", t=T)
        if external_cond:
            external_cond = {k: v.clone() for k, v in external_cond.items()}
            cross_cond = external_cond['pixel_latents']
            if self.pixel_use_raymap:
                with torch.no_grad():
                    camera_for_raymap = external_cond['camera'].clone()  # (B, T, 8) or (B, T, 10)
                    B_cam, T_cam = camera_for_raymap.shape[:2]
                    
                    # old - broken method for backcompatibility
                    if self.legacy_pixel_use_raymap:
                        camera_flat = rearrange(camera_for_raymap, "b t d -> (b t) d")
                        view, proj = camera_params_to_matrices(
                            camera_flat,
                            image_width=self.pixel_latent_w,
                            image_height=self.pixel_latent_h
                        )
                        # Convert to camera-to-world for raymap generation
                        extrinsics = utils3d_t.view_to_extrinsics(view)  # world-to-camera
                        camtoworlds = torch.inverse(extrinsics)  # camera-to-world
                        with allclose_accept_scalar_zero():
                            intrinsics = utils3d_t.perspective_to_intrinsics(proj)  # (B*T, 3, 3)
                        raymap = camera_to_raymap(
                            Ks=intrinsics,
                            camtoworlds=camtoworlds,
                            height=self.pixel_latent_h,
                            width=self.pixel_latent_w
                        )
                    
                    # new - expects voxel_local and OpenGL format view/proj
                    else:
                        camera_for_raymap = project_voxel_local_to_worldcam(camera_for_raymap) 
                        camera_for_raymap = rearrange(camera_for_raymap, "b t d -> (b t) d")
                        view, proj = camera_params_to_matrices(camera_for_raymap, image_width=self.pixel_latent_w, image_height=self.pixel_latent_h)
                        raymap = raymap_from_view_projection(view, proj, width=self.pixel_latent_w, height=self.pixel_latent_h)
                    
                    raymap = rearrange(raymap, "(b t) h w c -> b t c h w", b=B_cam, t=T_cam)
                    cross_cond = torch.cat([cross_cond, raymap], dim=2)  # (B, T, C+6, H, W)

            camera_for_ada = external_cond['camera'].clone()
            if self.camera_use_ape:
                camera_for_ada = rearrange(camera_for_ada, "b t d -> (b t) d")
                cam_rot, cam_xyz, cam_fov = camera_for_ada.split([6, 3, 1], dim=-1)
                cam_rot = self.cam_embedding_layer_rot(cam_rot)
                cam_xyz = self.cam_embedding_layer_xyz(cam_xyz)
                camera_for_ada = torch.cat((cam_rot, cam_xyz, cam_fov), dim=-1)
                camera_for_ada = rearrange(camera_for_ada, "(b t) d -> b t d", b=B, t=T)
            else:
                camera_for_ada[..., 4:7] = camera_for_ada[..., 4:7] * self.camera_pos_scaling_factor
            if self.action_camera_embedder:
                c += self.action_camera_embedder(external_cond['action'], camera_for_ada)
            else:
                c += self.action_embedder(external_cond['action']) + self.camera_embedder(camera_for_ada)
            # embed cross cond
            cross_cond = rearrange(cross_cond, "b t c h w -> (b t) c h w")
            cross_cond = self.cross_cond_embedder(cross_cond)  # (B*T, C, H, W ) -> (B*T, H/2, W/2, D)
            cross_cond = rearrange(cross_cond, "(b t) h w d -> b t (h w) d", t=T)
        # Convert to tensor for torch.compile compatibility (avoids dynamo guards on int values)
        global_start_idx_t = torch.tensor(global_start_idx, device=x.device)
        for i, block in enumerate(self.blocks):
            temporal_cache = self.temporal_kv_caches[i] if self.temporal_kv_caches else None
            cross_cache = self.cross_kv_caches[i] if self.cross_kv_caches else None
            x = block(x, c, cross_cond, temporal_kv_cache=temporal_cache, cross_kv_cache=cross_cache, global_start_idx=global_start_idx_t, prefill=prefill)
        Xp, Yp, Zp = self.x_embedder.grid_size
        x = rearrange(x, "b t (x y z) d -> b t x y z d", x=Xp, y=Yp, z=Zp)
        x = self.final_layer(x, c)  # (B, T, X, Y, Z, patch_size ** 3 * out_channels)
        # unpatchify
        x = rearrange(x, "b t x y z d -> (b t) x y z d")
        x = self.unpatchify(x)  # (N, out_channels, X, Y, Z)
        x = rearrange(x, "(b t) c x y z -> b t c x y z", t=T)

        return x
