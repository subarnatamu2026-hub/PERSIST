from dataclasses import dataclass
from typing import Literal


@dataclass
class OpenStatic4LArgs:
    """Arguments for the OpenStatic4L environment."""

    # Dataset configuration
    dataset_name: str = "openstatic4L"
    env_id: str = "OpenWorldDataset-v0"
    ep_timesteps: int = 15
    num_levels: int = 4
    actions: Literal["mouse_look"] = "mouse_look"

    # Environment parameters
    seed: int = 0
    init_frames: int = 200
    voxel_obs_rx: int = 32
    voxel_obs_ry: int = 32
    voxel_obs_rz: int = 32
    resolution: int = 518
    fov: int = 90
    fps_max: int = 30
    pmul: int = 1


@dataclass
class OpenStatic100kLArgs:
    """Arguments for the OpenStatic100kL environment."""

    # Dataset configuration
    dataset_name: str = "openstatic100kL"
    env_id: str = "OpenWorldDataset-v0"
    ep_timesteps: int = 150
    num_levels: int = 100_000
    actions: Literal["mouse_look"] = "mouse_look"

    # Environment parameters
    seed: int = 0
    init_frames: int = 200
    voxel_obs_rx: int = 32
    voxel_obs_ry: int = 32
    voxel_obs_rz: int = 32
    resolution: int = 518
    fov: int = 90
    fps_max: int = 30
    pmul: int = 1
