from dataclasses import dataclass
from functools import partial
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def mem_efficient_one_hot_fn(ids, num_classes):
    id_shape = ids.shape
    ids = ids.view(-1, 1)
    out_tensor = torch.zeros(ids.shape[0], num_classes, dtype=torch.int16, device=ids.device)
    out_tensor.scatter_(1, ids, 1)
    out_tensor = out_tensor.reshape(*id_shape, num_classes)
    return out_tensor

class ChannelLayerNorm(nn.LayerNorm):
    """
    A LayerNorm layer that normalizes over the channel dimension.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x

def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    3D pixel shuffle.
    """
    B, C, H, W, D = x.shape
    C_ = C // scale_factor**3
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, C_, H*scale_factor, W*scale_factor, D*scale_factor)
    return x


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    Return a normalization layer.
    """
    if norm_type == "group":
        return nn.GroupNorm(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm(*args, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
    ):
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif mode == "avgpool":
            assert in_channels == out_channels, "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels*8, 3, padding=1)
        elif mode == "nearest":
            assert in_channels == out_channels, "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")

@dataclass
class ResNetEncoderArgs:
    """Configuration for ResNet encoder."""

    in_channels: int = 2138
    input_x: int = 64
    input_y: int = 64
    input_z: int = 64
    channels: Tuple[int, ...] = (32, 128, 512)
    latent_channels: int = 48
    num_res_blocks: int = 2
    num_res_blocks_middle: int = 2
    mem_efficient_one_hot: bool = True

    @property
    def name(self) -> str:
        return "ResNet3dEncoder"

class ResNet3dEncoder(nn.Module):
    """
    3D ResNet Encoder.
    
    Args:
        in_channels (int): Channels of the input.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the encoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
    """
    def __init__(
        self,
        input_x: int,
        input_y: int,
        input_z: int,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        mem_efficient_one_hot: bool = True,
    ):
        super().__init__()
        self.inpux_x = input_x
        self.input_y = input_y
        self.input_z = input_z
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.one_hot_fn = partial(mem_efficient_one_hot_fn, num_classes=self.in_channels) if mem_efficient_one_hot \
            else partial(F.one_hot, num_classes=self.in_channels)

        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    DownsampleBlock3d(ch, channels[i+1])
                )
        
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1])
            for _ in range(num_res_blocks_middle)
        ])

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels*2, 3, padding=1)
        )

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False) -> torch.Tensor:
        x = self.one_hot_fn(x).float()
        x = rearrange(x, "b x y z c -> b c x y z")
        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x)
        x = self.middle_block(x)

        x = self.out_layer(x)

        mean, logvar = x.chunk(2, dim=1)

        if sample_posterior:
            std = torch.exp(0.5 * logvar) #always fp32
            z = mean + std * torch.randn_like(std) #always fp32
        else:
            z = mean
            
        if return_raw:
            return z, mean, logvar
        return z
        
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

class ResNet3dDecoder(nn.Module):
    """
    3d ResNet Decoder.
    
    Args:
        out_channels (int): Channels of the output.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
    """ 
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0])
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock3d(ch, channels[i+1])
                )

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1)
        )

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
                
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)

        h = self.out_layer(h)
        h = rearrange(h, "b c x y z -> b x y z c")
        return h