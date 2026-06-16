from dataclasses import dataclass, field
from typing import *


# Architecture hyperparameters for the persist_S pipeline (the "S" variant).
# Checkpoints are NOT stored alongside this folder: every model weight is passed
# explicitly as a script argument at load time (see scripts/run_inference.py).
# The "XL" variant reuses these configs with a couple of overrides applied at load
# time (voxel-denoiser patch_size 1 and sampler noise_cond 0.0); see
# `pipelines.pipeline.pipeline_variant_overrides`.

### Voxel VAE

@dataclass
class ResNetEncoderArgs:
    """Configuration for ResNet encoder."""

    in_channels: int = 2138
    input_x: int = 48
    input_y: int = 48
    input_z: int = 48
    channels: Tuple[int, ...] = (32, 128, 512)
    latent_channels: int = 48
    num_res_blocks: int = 2
    num_res_blocks_middle: int = 2
    mem_efficient_one_hot: bool = True

    @property
    def name(self) -> str:
        return "ResNet3dEncoder"

@dataclass
class ResNetDecoderArgs:
    """Configuration for ResNet decoder."""

    out_channels: int = 2138
    channels: Tuple[int, ...] = (512, 128, 32)
    latent_channels: int = 48
    num_res_blocks: int = 2
    num_res_blocks_middle: int = 2

    @property
    def name(self) -> str:
        return "ResNet3dDecoder"

### Pixel VAE

@dataclass
class ViTVaeArgs:
    """Configuration for ViT-based VAE models."""

    latent_dim: int = 16
    input_height: int = 360
    input_width: int = 640
    patch_size: int = 10
    enc_dim: int = 1024
    enc_depth: int = 6
    enc_heads: int = 16
    dec_dim: int = 1024
    dec_depth: int = 12
    dec_heads: int = 16
    mlp_ratio: float = 4.0
    use_variational: bool = True
    qk_rms_norm : bool = True

    @property
    def name(self) -> str:
        return "ViTVae"

### Camera Model
@dataclass
class CameraTransformerArgs:
    """Configuration for CameraTransformer."""

    cam_translation_input_rep: Literal["extrinsics", "voxel_local"] = 'voxel_local'
    cam_translation_output_rep: Literal["extrinsics", "voxel_local", "delta_voxel_local"] = 'voxel_local'
    cam_rotation_input_rep: Literal["6d", "cam_dir"] = '6d'
    cam_rotation_output_rep: Literal["6d", "cam_dir", "delta_angle", "delta_angle_cat"] = 'delta_angle'
    cam_fov_output_rep: Literal["absolute", "delta"] = 'delta'
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

#### Voxel Denoiser
@dataclass
class VoxelDitDenoiserArgs:
    """Configuration for DiT Denoiser.

    The "XL" variant overrides ``patch_size`` from 2 to 1 at load time.
    """

    input_x : int = 12 #non-default
    input_y : int = 12 #non-default
    input_z : int = 12 #non-default
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
    camera_cond_dim : int = 10 # non-default
    camera_pos_scaling_factor : float = 1.0 # non-default
    timestep_scaling_factor: float = 1000.0
    joint_action_camera_emb : bool = True
    context_window_size : int = 8
    camera_use_ape: bool = True # non-default
    voxel_use_ape: bool = True # non-default
    pixel_use_raymap: bool = True # non-default
    voxel_rotary_max_freq : int = 512 # non-default
    pixel_rotary_max_freq : int = 256
    qk_rms_norm : bool = True
    legacy_qk_norm : bool = False

    @property
    def name(self) -> str:
        return "VoxelDiT"

### Pixel denoiser
@dataclass
class RasterizedVoxelDepthTransformerArgs:
    feature_dim: int = 32
    heads: int = 4
    mlp_ratio: float = 4.0
    output_dim: Optional[int] = None
    frequency_embedding_size: int = 32
    max_period: int = 10000
    num_layers: int = 192
    vol_render_mode: bool = False
    qk_rms_norm: bool = True
    is_causal_depth: bool = True #fixme inconsistent!!
    flash_attn: bool = False
    old: bool = False

    @property
    def name(self) -> str:
        return "RasterizedVoxelDepthTransformer"

@dataclass
class FrameDepthStackPixelDitDenoiserArgs:

    """Configuration for DiT Denoiser."""
    input_h : int = 36
    input_w : int = 64
    patch_size : int = 2
    in_channels : int = 16
    hidden_size : int = 1024
    depth : int = 12
    num_heads : int = 16
    mlp_ratio : int = 4.0
    raster_cond_shape : Tuple[int, ...] = (32, 36, 64)
    raster_cond_patch_size : int = 2
    action_cond_dim : int = 23
    context_window_size : int = 16 #changed
    rotary_max_freq : int = 256
    voxel_dim : int = 48
    num_freqs: int = 8
    qk_rms_norm : bool = True
    flash_attn : bool = False
    detach_raster_grad : bool = False
    learned_padding : bool = False
    dropout_p_pixel : float = 0.0
    depth_patch_size : int = 6
    depth_stride_size : int = 4
    depth_output_dim : int = 16
    aggregation_config: RasterizedVoxelDepthTransformerArgs = field(
        default_factory=RasterizedVoxelDepthTransformerArgs
    )

    @property
    def name(self) -> str:
        return "FrameDepthStackPixelDiT"

@dataclass
class VoxelClassEmbedderArgs:
    """Configuration for VoxelClassEmbedder."""
    num_embeddings : int = 2138
    embedding_dim : int = 32

    @property
    def name(self) -> str:
        return "VoxelClassEmbedder"
