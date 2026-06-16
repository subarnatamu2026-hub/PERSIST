import math
from dataclasses import dataclass
from functools import partial
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Mlp
from torch import autocast

from models.attention import MultiHeadRMSNorm
from models.embeddings import AbsolutePositionEmbedder, RotaryEmbedding, apply_rotary_emb, ActionCameraEmbedder, \
    get_spherical_harmonics
from utils.camera_util import pitch_yaw_to_cam_dir, cam_dir_to_6d, camera_to_xyz_pry_fov, angular_diff_deg


@dataclass
class CameraTransformerArgs:
    """Configuration for CameraTransformer."""

    cam_translation_input_rep: Literal["extrinsics", "voxel_local"] = 'voxel_local'
    cam_translation_output_rep: Literal["extrinsics", "voxel_local", "delta_voxel_local"] = 'voxel_local'
    cam_rotation_input_rep: Literal["6d", "cam_dir"] = '6d'
    cam_rotation_output_rep: Literal["6d", "cam_dir", "delta_angle", "delta_angle_cat"] = '6d'
    cam_fov_output_rep: Literal["absolute", "delta"] = 'absolute'
    cam_rotation_input_rep_range: int = 2
    cam_rotation_emb_channels: int = 256
    cam_translation_emb_channels: int = 256
    hidden_size: int = 1024
    depth : int = 12
    num_heads : int = 16
    mlp_ratio : int = 4
    action_cond_dim : int = 23
    voxel_cond_shape : Tuple[int, ...] = (4, 4, 4)
    voxel_vocab_size: int = 2138
    context_window_size : int = 8
    qk_rms_norm : bool = True

    @property
    def name(self) -> str:
        return "CameraTransformer"

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


class TemporalAxialAttention(nn.Module):
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

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T (h d) -> B h T d", h=self.heads)
        k = rearrange(k, "B T (h d) -> B h T d", h=self.heads)
        v = rearrange(v, "B T (h d) -> B h T d", h=self.heads)

        # Correct order: QK RMSNorm first, then RoPE
        # legacy_qk_norm=True preserves old (incorrect) behavior for backward compatibility
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        q = apply_rotary_emb(self.rotary_emb.axial_freqs[:T], q)
        k = apply_rotary_emb(self.rotary_emb.axial_freqs[:T], k)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=self.is_causal)

        x = rearrange(x, "B h T d -> B T (h d)")
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x


class TemporalBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        is_causal=True,
        qk_rms_norm=False,
        context_window_size=8,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.is_causal = is_causal
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.t_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_attn = TemporalAxialAttention(
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

    def forward(self, x, c):
        B, T, D = x.shape

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        return x


class CameraFinalLayer(nn.Module):
    """
    The final layer of DiT for camera prediction.
    """

    def __init__(self, hidden_size, out_channels, num_layers=1):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        net = [nn.Linear(hidden_size, hidden_size, bias=True), nn.SiLU()] * (num_layers - 1)
        net.append(nn.Linear(hidden_size, out_channels, bias=True))
        self.net = nn.Sequential(*net)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1) #note: no gate
        x = modulate(self.norm_final(x), shift, scale)
        x = self.net(x)
        return x


class CameraTransformer(nn.Module):
    def __init__(
        self,
        cam_translation_input_rep: Literal["extrinsics", "voxel_local"] = 'voxel_local',
        cam_translation_output_rep: Literal["extrinsics", "voxel_local", "delta_voxel_local"] = 'voxel_local',
        cam_rotation_input_rep: Literal["6d", "cam_dir"] = 'cam_dir',
        cam_rotation_output_rep: Literal["6d", "cam_dir", "delta_angle", "delta_angle_cat"] = 'cam_dir',
        cam_fov_output_rep: Literal["absolute", "delta"] = 'absolute',
        cam_rotation_input_rep_range: int = 2,
        cam_rotation_emb_channels: int = 256,
        cam_translation_emb_channels: int = 256,
        hidden_size: int = 1024,
        depth : int = 12,
        num_heads : int = 16,
        mlp_ratio : int = 4,
        action_cond_dim : int = 23,
        voxel_cond_shape : Tuple[int, ...] = (4, 4, 4),
        voxel_vocab_size : int = 2138,
        voxel_embedding_dim : int = 32,
        context_window_size : int = 8,
        qk_rms_norm : bool = True,
    ):
        super().__init__()
        self.cam_translation_input_rep = cam_translation_input_rep
        self.cam_translation_output_rep = cam_translation_output_rep
        self.cam_rotation_input_rep = cam_rotation_input_rep
        self.cam_rotation_input_rep_range = cam_rotation_input_rep_range
        self.cam_translation_emb_channels = cam_translation_emb_channels
        self.cam_rotation_emb_channels = cam_rotation_emb_channels
        self.cam_rotation_output_rep = cam_rotation_output_rep
        self.cam_fov_output_rep = cam_fov_output_rep
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.action_cond_dim = action_cond_dim
        self.voxel_cond_shape = voxel_cond_shape
        self.context_window_size = context_window_size
        self.qk_rms_norm = qk_rms_norm

        self.setup_camera_input_layer()

        self.voxel_embedder = nn.Sequential(
            nn.Embedding(voxel_vocab_size, voxel_embedding_dim),
            nn.LayerNorm(voxel_embedding_dim, elementwise_affine=False, eps=1e-6)
        )
        self.ada_embedder = ActionCameraEmbedder(
            action_cond_dim,
            math.prod(voxel_cond_shape)*voxel_embedding_dim,
            hidden_size,
        )

        self.temporal_rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            cache_if_possible=True,
            temporal_context_length=context_window_size,
        )
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(
                    hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    qk_rms_norm=qk_rms_norm,
                    context_window_size=context_window_size,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )

        self.setup_camera_output_layer()
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.output_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.output_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output_layer.net[-1].weight, 0)
        nn.init.constant_(self.output_layer.net[-1].bias, 0)

    def setup_camera_input_layer(self):

        if self.cam_rotation_input_rep == '6d':
            self.cam_rot_in_dim = 6
            self.embedding_layer_rot = AbsolutePositionEmbedder(
                channels=self.cam_rotation_emb_channels,
                in_channels=6,
                pos_range=self.cam_rotation_input_rep_range
            )
            cam_rot_dim_post_emb = self.cam_rotation_emb_channels
        elif self.cam_rotation_input_rep == 'cam_dir':
            self.cam_rot_in_dim = 3
            self.embedding_layer_rot = partial(get_spherical_harmonics, levels=3)
            cam_rot_dim_post_emb = (3 + 1)**2
        else:
            raise NotImplementedError(f"cam_rotation_input_rep: {self.cam_rotation_input_rep} not implemented")

        if self.cam_translation_input_rep == 'extrinsics':
            self.embedding_layer_tr = AbsolutePositionEmbedder(
                channels=self.cam_translation_emb_channels,
                in_channels=3,
                pos_range=4/48
            )
            cam_tr_dim_post_emb = self.cam_translation_emb_channels
        elif self.cam_translation_input_rep == 'voxel_local':
            self.embedding_layer_tr = AbsolutePositionEmbedder(
                channels=self.cam_translation_emb_channels,
                in_channels=3,
                pos_range=4/48
            )
            cam_tr_dim_post_emb = self.cam_translation_emb_channels
        else:
            raise NotImplementedError(f"cam_translation_input_rep: {self.cam_translation_input_rep} not implemented")

        input_dim = self.cam_rot_in_dim + 3 + 1 #rot + translation(3) + fov(1)
        input_dim_post_emb = cam_rot_dim_post_emb + cam_tr_dim_post_emb + 1
        self.input_layer = nn.Linear(input_dim_post_emb, self.hidden_size)

    def setup_camera_output_layer(self):
        if self.cam_rotation_output_rep == '6d':
            self.cam_rot_out_dim = 6
        elif self.cam_rotation_output_rep == 'cam_dir':
            self.cam_rot_out_dim = 3
        elif self.cam_rotation_output_rep == 'delta_angle':
            self.cam_rot_out_dim = 2
        elif self.cam_rotation_output_rep == 'delta_angle_cat':
            self.cam_rot_out_dim = 2 * 3 #3 categories (-/0/+ delta angle)
        else:
            raise NotImplementedError(f"cam_rotation_output_rep: {self.cam_rotation_output_rep} not implemented")
        self.output_dim = self.cam_rot_out_dim + 3 + 1
        self.output_layer = CameraFinalLayer(self.hidden_size, self.output_dim, num_layers=2)

    def forward(self, x, action, voxel):
        B, T = x.shape[:2]

        x = rearrange(x, "b t d -> (b t) d")
        x_r, x_tr, x_f = x.split([self.cam_rot_in_dim, 3, 1], dim=-1)
        x_r = self.embedding_layer_rot(x_r)
        x_tr = self.embedding_layer_tr(x_tr)
        x = torch.cat((x_r, x_tr, x_f), dim=-1)
        x = rearrange(x, "(b t) d -> b t d", b=B, t=T)
        x = self.input_layer(x)

        voxel = self.voxel_embedder(voxel)
        voxel = rearrange(voxel, "b t x y z d -> b t (x y z d)")
        c = self.ada_embedder(action, voxel)

        for block in self.blocks:
            x = block(x, c)

        x = self.output_layer(x, c)
        if self.cam_rotation_output_rep == 'cam_dir':
            x_r, x_tr, x_f = x.split([self.cam_rot_out_dim, 3, 1], dim=-1)
            x_r = F.normalize(x_r, p=2, dim=-1)
            x = torch.cat((x_r, x_tr, x_f), dim=-1)

        return x

    @autocast("cuda", enabled=False)
    def predict_next_camera(self, x, action, voxel):
        B, T, D = x.shape
        x_next = self.forward(x, action, voxel)
        if self.cam_translation_input_rep == self.cam_translation_output_rep and self.cam_rotation_input_rep == self.cam_rotation_output_rep:
            return x_next

        assert self.cam_rotation_input_rep == "6d" # only support 6d input for now
        assert self.cam_translation_input_rep == "voxel_local"

        assert self.cam_rotation_output_rep in ["delta_angle", "delta_angle_cat"]
        assert self.cam_translation_output_rep in ["voxel_local", "delta_voxel_local"]
        assert self.cam_fov_output_rep == "delta"

        x = rearrange(x, "b t d -> (b t) d")
        x_next = rearrange(x_next, "b t d -> (b t) d")

        xyz, pry, fov = camera_to_xyz_pry_fov(x, in_degrees=True, rotation_rep=self.cam_rotation_input_rep)
        fov = fov.unsqueeze(-1)
        pitch, _, yaw = pry.unbind(dim=-1)
        delta_pred_py, pred_xyz, delta_pred_fov = x_next.split([self.cam_rot_out_dim, 3, 1], dim=-1)
        if self.cam_rotation_output_rep == "delta_angle":
            delta_pitch, delta_yaw = delta_pred_py.unbind(dim=-1)
        elif self.cam_rotation_output_rep == "delta_angle_cat":
            GAIN_PITCH, GAIN_YAW = 3.6, 6.4
            delta_pred_py = rearrange(delta_pred_py, '... (r c) -> ... r c', r=2, c=3)
            delta_pred_py = torch.argmax(delta_pred_py, dim=-1)
            delta_pitch = torch.zeros_like(delta_pred_py[..., 0], dtype=torch.float32)
            delta_yaw = torch.zeros_like(delta_pred_py[..., 0], dtype=torch.float32)
            delta_pitch[delta_pred_py[...,0] == 0] = -GAIN_PITCH
            delta_pitch[delta_pred_py[..., 0] == 2] = GAIN_PITCH
            delta_yaw[delta_pred_py[..., 1] == 0] = -GAIN_YAW
            delta_yaw[delta_pred_py[..., 1] == 2] = GAIN_YAW
        pitch = (pitch + delta_pitch).clamp(0+1e-8, 180-1e-8) # avoid discontinuity
        yaw = angular_diff_deg(yaw, -delta_yaw)
        cam_dir = pitch_yaw_to_cam_dir(pitch, yaw, in_degrees=True)
        cam_rot_6d = cam_dir_to_6d(cam_dir)

        if self.cam_translation_output_rep == "delta_voxel_local":
            xyz = xyz + pred_xyz
        elif self.cam_translation_output_rep == "voxel_local":
            xyz = pred_xyz
        fov = torch.deg2rad(fov + delta_pred_fov)

        x_next_converted = torch.cat([cam_rot_6d, xyz, fov], dim=-1)
        x_next_converted = rearrange(x_next_converted, "(b t) d -> b t d", b=B, t=T)
        return x_next_converted
