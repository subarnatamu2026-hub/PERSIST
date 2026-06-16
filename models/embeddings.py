from __future__ import annotations

from math import pi, log, prod
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from loguru import logger
from timm.layers.helpers import to_2tuple, to_3tuple
from torch import nn, einsum, broadcast_tensors, Tensor
from torch.amp import autocast
from torch.nn import Module

from utils.camera_util import rotation_6d_to_matrix


class ActionCameraEmbedder(nn.Module):
    """
    Embeds actions (multi-hot vector) and cameras (pose[3f], quat[4f], fov[1f]) into a joint vector representation.
    """

    def __init__(
            self,
            action_size,
            camera_size,
            hidden_size,
    ):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.Linear(action_size, hidden_size // 2, bias=True),  # hidden_size is diffusion model hidden size
            nn.SiLU(),
        )
        self.camera_mlp = nn.Sequential(
            nn.Linear(camera_size, hidden_size // 2, bias=True),  # hidden_size is diffusion model hidden size
            nn.SiLU(),
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),  # hidden_size is diffusion model hidden size
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, action, camera):
        action_emb = self.action_mlp(action)
        camera_emb = self.camera_mlp(camera)
        joint_emb = torch.cat([action_emb, camera_emb], dim=-1)
        joint_emb = self.joint_mlp(joint_emb)
        return joint_emb

# Timestep embedder from https://github.com/etched-ai/open-oasis
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),  # hidden_size is diffusion model hidden size
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        half = frequency_embedding_size // 2
        freqs = torch.exp(-log(max_period) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        
        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class PixelPatchEmbedder(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_height=256,
        img_width=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = (img_height, img_width)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, random_sample=False):
        B, C, H, W = x.shape
        assert random_sample or (H == self.img_size[0] and W == self.img_size[1]), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "B C H W -> B (H W) C")
        else:
            x = rearrange(x, "B C H W -> B H W C")
        x = self.norm(x)
        return x

class DepthPatchEmbedder(nn.Module):
    """1D Depth Layer to Patch Embedding

    Compresses depth layers via Conv1d, analogous to how PixelPatchEmbedder
    compresses spatial patches via Conv2d.

    Concatenates depth embeddings in channel dimension for depth-awareness.
    """

    def __init__(
        self,
        num_layers=64,
        patch_size=4,
        stride=4,
        in_chans=8,
        out_chans=16,
        use_depth_pos_enc=True,
        num_freqs=16,
        max_period=10000,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.use_depth_pos_enc = use_depth_pos_enc

        half = num_freqs // 2
        freqs = torch.exp(-log(max_period) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.num_freqs = num_freqs
        logger.info(f"DepthPatchEmbedder out_chan: {out_chans}, freqs: {num_freqs}")

        conv_in_chans = in_chans + num_freqs if use_depth_pos_enc else in_chans
        self.proj = nn.Conv1d(conv_in_chans, out_chans,
                             kernel_size=patch_size,
                             stride=stride)
        # print("DepthPatchEmbedder: conv1d out_chans", out_chans)
        # print("DepthPatchEmbedder: conv1d stride", stride)
        # print("DepthPatchEmbedder: conv1d kernel_size", patch_size)
    
    def pos_enc(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        
        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.num_freqs % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x, depth=None):
        """
        Args:
            x: (B, H, W, L, C) - depth-peeled raster features
            depth: (B, H, W, L) - depth values per layer
        Returns:
            (B, H, W, L', C_out) - depth-downsampled features
        """
        B, H, W, L, C = x.shape
        # Concatenate depth embeddings in channel dimension
        if self.use_depth_pos_enc and depth is not None:
            depth_flat = rearrange(depth, 'b h w l -> (b h w l)')
            depth_emb = self.pos_enc(depth_flat * 1000)
            depth_emb = rearrange(depth_emb, '(b h w l) c -> b h w l c',
                                 b=B, h=H, w=W, l=L)
            x = torch.cat([x, depth_emb], dim=-1)

        # Reshape for Conv1d: (B*H*W, C_in, L)
        x = rearrange(x, 'b h w l c -> (b h w) c l')

        # Compress depth dimension
        x = self.proj(x)

        # Reshape to (B, H, W, L', C_out)
        x = rearrange(x, '(b h w) c l -> b h w l c', b=B, h=H, w=W)

        return x

class VoxelPatchEmbedder(nn.Module):
    """3D Voxel to Patch Embedding"""

    def __init__(
        self,
        dim_x=16,
        dim_y=16,
        dim_z=16,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        vox_size = (dim_x, dim_y, dim_z)
        self.voxel_size = vox_size
        self.patch_size = patch_size
        self.grid_size = (vox_size[0] // patch_size[0], vox_size[1] // patch_size[1], vox_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, random_sample=False):
        B, C, X, Y, Z = x.shape
        assert random_sample or (X == self.voxel_size[0] and Y == self.voxel_size[1] and Z == self.voxel_size[2]), \
            (f"Input voxel size ({X}*{Y}*{Z}) doesn't match model "
             f"({self.voxel_size[0]}*{self.voxel_size[1]}*{self.voxel_size[2]}).")
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "B C X Y Z -> B (X Y Z) C")
        else:
            x = rearrange(x, "B C X Y Z -> B X Y Z C")
        x = self.norm(x)
        return x


# helper functions

"""
Adapted from https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
and https://github.com/etched-ai/open-oasis
"""
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# rotary embedding helper functions


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0):
    dtype = t.dtype

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)

    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)


# classes

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for: Literal["lang", "pixel", "voxel", "constant", "pixeltime", "voxeltime"] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=False,
        spatial_sequence_shape=None,
        temporal_context_length=None,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "voxel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "pixeltime":
            time_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "voxeltime":
            time_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()

        # dummy for device
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

        if learned_freq:
            self.freqs = nn.Parameter(freqs, requires_grad=True)
        else:
            self.register_buffer("freqs", freqs, persistent=False)
            if cache_if_possible:
                if freqs_for in ["pixel", "voxel"] and spatial_sequence_shape is not None:
                    spatial_seq_len = prod(spatial_sequence_shape)
                    axial_freqs = self.get_axial_freqs(*spatial_sequence_shape).reshape(spatial_seq_len, -1) # (S, n_freqs)
                    self.register_buffer("axial_freqs", axial_freqs, persistent=False)
                elif freqs_for in ["lang"] and temporal_context_length is not None:
                    axial_freqs = self.get_axial_freqs(temporal_context_length) # (T, nfreqs)
                    self.register_buffer("axial_freqs", axial_freqs, persistent=False)

        if freqs_for in ["pixeltime", "voxeltime"]:
            if learned_freq:
                self.time_freqs = nn.Parameter(time_freqs, requires_grad=True)
            else:
                self.register_buffer("time_freqs", time_freqs, persistent=False)
                if cache_if_possible and spatial_sequence_shape is not None and temporal_context_length is not None:
                    spatial_seq_len = prod(spatial_sequence_shape)
                    axial_freqs = self.get_axial_freqs(
                        temporal_context_length, *spatial_sequence_shape).reshape(temporal_context_length, spatial_seq_len, -1) #(T, S, n_freqs)
                    self.register_buffer("axial_freqs", axial_freqs, persistent=False)

        self.learned_freq = learned_freq
        self.spatial_sequence_shape = spatial_sequence_shape

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer("scale", scale, persistent=False)

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def reset_axial_freqs(self, freqs):
        self.register_buffer("axial_freqs", freqs, persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, freqs, seq_dim=None, offset=0, scale=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(scale), "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        seq_freqs = self.forward(seq, freqs, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            seq_freqs = rearrange(seq_freqs, "n d -> n 1 d")

        return apply_rotary_emb(seq_freqs, t, scale=default(scale, 1.0))

    def rotate_queries_and_keys(self, q, k, freqs, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        seq_freqs = self.forward(seq, freqs, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            seq_freqs = rearrange(seq_freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = apply_rotary_emb(seq_freqs, q, scale=scale)
        rotated_k = apply_rotary_emb(seq_freqs, k, scale=scale**-1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(self, t: Tensor, seq_len: int | None = None, offset=0):
        assert self.use_xpos

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = repeat(scale, "n d -> n (d r)", r=2)

        return scale

    def get_axial_freqs(self, *dims, offset=0):
        # pixel: (H, W), voxel: (X, Y, Z), lang: (L,), constant: (L,), pixeltime: (L, H, W), voxeltime: (L, X, Y, Z)
        match self.freqs_for:
            case "lang" | "constant":
                assert len(dims) == 1
                spatial_ndim = 0
            case "pixel":
                assert len(dims) == 2
                spatial_ndim = 2
            case "voxel":
                assert len(dims) == 3
                spatial_ndim = 3
            case "pixeltime":
                assert len(dims) == 3
                spatial_ndim = 2
            case "voxeltime":
                assert len(dims) == 4
                spatial_ndim = 3
            case _:
                raise ValueError(f"Invalid freqs_for {self.freqs_for}")

        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            # only allow pixel freqs for last two dimensions
            set_spatial_freq = spatial_ndim and ind >= len(dims) - spatial_ndim
            if set_spatial_freq:
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device) + offset

            if self.freqs_for in ["pixeltime", "voxeltime"] and not set_spatial_freq:
                seq_freqs = self.forward(pos, self.time_freqs, seq_len=dim)
            else:
                seq_freqs = self.forward(pos, self.freqs, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(seq_freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, freqs: Tensor, seq_len=None, offset=None):

        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        return freqs

# from https://github.com/microsoft/TRELLIS
class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, channels: int, in_channels: int = 3, pos_range: float = 2):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = (1/pos_range) + (1/pos_range) * (15 ** self.freqs) 

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed

def raymap_from_view_projection(
    view: torch.Tensor,        # (..., 4, 4)
    projection: torch.Tensor,  # (..., 4, 4)
    height: int,
    width: int,
):
    """
    Construct ray origins and directions from OpenGL view + projection matrices. 

    Args:
        view: (..., 4, 4) OpenGL view matrix (world -> camera)
        projection: (..., 4, 4) OpenGL projection matrix
        height, width: image resolution

    Returns:
        (..., H, W, 6) raymap = [origin(3), direction(3)]
    """
    device, dtype = view.device, view.dtype

    # Inverses
    inv_view = torch.linalg.inv(view)
    inv_proj = torch.linalg.inv(projection)

    # NDC grid
    x, y = torch.meshgrid(
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        indexing="xy",
    )

    # OpenGL NDC: z = -1 is near plane
    ndc = torch.stack([x, y, -torch.ones_like(x), torch.ones_like(x)], dim=-1)
    # Unproject to camera space
    cam = torch.einsum("...ij,hwj->...hwi", inv_proj, ndc)
    cam = cam[..., :3] / cam[..., 3:4]
    dirs_cam = F.normalize(cam, dim=-1) #cam space

    R = inv_view[..., :3, :3]
    dirs_world = torch.einsum("...ij,...hwj->...hwi", R, dirs_cam) #to world space

    origin = inv_view[..., :3, 3]
    origins = origin[..., None, None, :].expand_as(dirs_world)

    return torch.cat([origins, dirs_world], dim=-1)


def camera_to_raymap(
    Ks: Tensor,
    camtoworlds: Tensor | None = None,
    height: int = None,
    width: int = None,
    downscale: int = 1,
    include_ups: bool = False,
    cam_pos: Tensor | None = None,
    rotation_6d: Tensor | None = None,
):
    """Construct the raymap from the camera intrinsics and extrinsics.

    OLD - TO BE REMOVED

    Note: This function expects OpenCV camera coordinates.

    Args:
        Ks: The camera intrinsics tensor with shape (..., 3, 3).
        camtoworlds: The camera extrinsics tensor with shape (..., 4, 4).
            Optional if cam_pos and rotation_6d are provided.
        height: The height of original image corresponding to intrinsics.
        width: The width of original image corresponding to intrinsics.
        downscale: Downscale factor for the raymap.
        include_ups: Whether to include the up direction in the raymap.
        cam_pos: Camera position in world coordinates with shape (..., 3).
            Alternative to camtoworlds. Must be provided with rotation_6d.
        rotation_6d: 6D continuous rotation representation with shape (..., 6).
            Alternative to camtoworlds. Must be provided with cam_pos.

    Returns:
        The raymap tensor with shape (..., H, W, 6).
    """
    assert Ks.shape[-2:] == (3, 3), "Expected Ks to have shape (..., 3, 3)."
    assert width % downscale == 0, "Expected width to be divisible by downscale."
    assert height % downscale == 0, "Expected height to be divisible by downscale."

    # Validate input format
    if camtoworlds is not None:
        assert cam_pos is None and rotation_6d is None, (
            "Cannot provide both camtoworlds and (cam_pos, rotation_6d). "
            "Use either camtoworlds OR (cam_pos + rotation_6d)."
        )
        assert camtoworlds.shape[-2:] == (4, 4), (
            "Expected camtoworlds to have shape (..., 4, 4)."
        )
    elif cam_pos is not None and rotation_6d is not None:
        # Construct camtoworlds from cam_pos and rotation_6d
        assert cam_pos.shape[-1] == 3, "Expected cam_pos to have shape (..., 3)."
        assert rotation_6d.shape[-1] == 6, "Expected rotation_6d to have shape (..., 6)."

        # Convert 6D rotation to rotation matrix
        rotation_matrix = rotation_6d_to_matrix(rotation_6d)  # [..., 3, 3]

        # Construct 4x4 transformation matrix
        batch_shape = rotation_matrix.shape[:-2]
        camtoworlds = torch.zeros(*batch_shape, 4, 4, device=Ks.device, dtype=Ks.dtype)
        camtoworlds[..., :3, :3] = rotation_matrix
        camtoworlds[..., :3, 3] = cam_pos
        camtoworlds[..., 3, 3] = 1.0
    else:
        raise ValueError(
            "Must provide either camtoworlds OR both (cam_pos and rotation_6d)."
        )

    # Downscale the intrinsics.
    Ks = torch.stack(
        [
            Ks[..., 0, :] / downscale,
            Ks[..., 1, :] / downscale,
            Ks[..., 2, :],
        ],
        dim=-2,
    )  # [..., 3, 3]
    width //= downscale
    height //= downscale

    # Construct pixel coordinates
    x, y = torch.meshgrid(
        torch.arange(width, device=Ks.device),
        torch.arange(height, device=Ks.device),
        indexing="xy",
    )  # [H, W]
    coords = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1)  # [H, W, 3]

    # To camera coordinates [..., H, W, 3]
    dirs = torch.einsum("...ij,...hwj->...hwi", Ks.float().inverse().to(Ks.dtype), coords)

    # To world coordinates [..., H, W, 3]
    dirs = torch.einsum("...ij,...hwj->...hwi", camtoworlds[..., :3, :3], dirs)
    dirs = F.normalize(dirs, p=2, dim=-1)

    # Camera origin in world coordinates [..., H, W, 3]
    origins = torch.broadcast_to(camtoworlds[..., None, None, :3, -1], dirs.shape)

    if include_ups:
        # Extract the up direction (second column)
        ups = torch.broadcast_to(camtoworlds[..., None, None, :3, 1], dirs.shape)
        ups = F.normalize(ups, p=2, dim=-1)
        return torch.cat([origins, dirs, ups], dim=-1)
    else:
        return torch.cat([origins, dirs], dim=-1)  # [..., H, W, 6]


# Spherical Harmonics functions for view dir encoding. From Neuralangelo



SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def get_spherical_harmonics(dirs, levels=3):

    """Evaluate spherical harmonics bases at unit directions.
    Args:
        dirs: (..., 3) tensor of unit directions.
        levels: int, number of SH levels to evaluate (up to 5). 
    Returns:
        (..., (levels + 1) ** 2) tensor of spherical harmonics evaluated at, deault is 3 levels.
        """
    # Evaluate spherical harmonics bases at unit directions, without taking linear combination.
    vals = torch.empty((*dirs.shape[:-1], (levels + 1) ** 2), device=dirs.device)
    vals[..., 0] = SH_C0
    if levels >= 1:
        x, y, z = dirs.unbind(-1)
        vals[..., 1] = -SH_C1 * y
        vals[..., 2] = SH_C1 * z
        vals[..., 3] = -SH_C1 * x
    if levels >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        vals[..., 4] = SH_C2[0] * xy
        vals[..., 5] = SH_C2[1] * yz
        vals[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
        vals[..., 7] = SH_C2[3] * xz
        vals[..., 8] = SH_C2[4] * (xx - yy)
    if levels >= 3:
        vals[..., 9] = SH_C3[0] * y * (3 * xx - yy)
        vals[..., 10] = SH_C3[1] * xy * z
        vals[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
        vals[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
        vals[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
        vals[..., 14] = SH_C3[5] * z * (xx - yy)
        vals[..., 15] = SH_C3[6] * x * (xx - 3 * yy)
    if levels >= 4:
        vals[..., 16] = SH_C4[0] * xy * (xx - yy)
        vals[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
        vals[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
        vals[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
        vals[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
        vals[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
        vals[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
        vals[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
        vals[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    if levels >= 5:
        raise NotImplementedError
    return vals
