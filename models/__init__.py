from torch import nn

from .camera_transformer import CameraTransformer
from .dit_pixel import FrameDepthStackPixelDiT
from .dit_voxel import VoxelDiT
from .vae_pixel import ViTVae
from .vae_voxel import ResNet3dEncoder, ResNet3dDecoder

VoxelClassEmbedder = nn.Embedding
__all__ = [
    "ResNet3dEncoder",
    "ResNet3dDecoder",
    "ViTVae",
    "CameraTransformer",
    "VoxelDiT",
    "FrameDepthStackPixelDiT",
    "VoxelClassEmbedder"
]
