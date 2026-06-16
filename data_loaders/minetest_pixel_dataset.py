import os

import numpy as np
import torch
import torchvision.io as io

from . import StandardDatasetBase


class MinetestPixel(StandardDatasetBase):
    """
    Minetest pixel/image dataset for training VAEs and diffusion models.

    This dataset loads RGB images from the raw Minetest environment data,


    Args:
        root (str): path to the dataset directory
        resolution (int): expected resolution of input images (default: 256)
        min_level_quality_score (float): minimum level quality score (default: 1.0)
        split (str): split name ('train' or 'val')
    """

    def __init__(
            self,
            root: str,
            min_level_quality_score: float = 1.0,
            split: str = 'train',
    ):

        self.min_level_quality_score = min_level_quality_score
        self.split = split

        super().__init__(root)

    def filter_metadata(self, metadata, instances=None):
        stats = {}
        metadata = metadata[metadata['preprocessed']]
        stats['Available samples'] = len(metadata)
        metadata = metadata[metadata['level_quality_score'] >= self.min_level_quality_score]
        stats[f'Level quality score >= {self.min_level_quality_score}'] = len(metadata)
        
        if self.split == "train":
            metadata = metadata[metadata["validation_set"] == False]
        elif self.split == "val":
            metadata = metadata[metadata["validation_set"] == True]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        stats['In split'] = len(metadata)
        return metadata, stats

    def get_instance(self, instance, instance_idx):
        """Get a single instance (image) from the dataset."""
        image, sampled_ts = self._get_image_from_raw(instance)
        
        return {
            'image': image,
            'instance_idx': instance_idx,
            'timestep_idx': sampled_ts,
        }

    def _get_image_from_raw(self, instance):
        """Load image from raw .npz data files."""
        # Get the local path from metadata
        local_path = self.metadata.loc[instance]['local_path']

        if os.path.exists(os.path.join(self.root, local_path, 'rgb.mp4')):
            mp4_path = os.path.join(self.root, local_path, 'rgb.mp4')
            obs_rgb, _, _ = io.read_video(mp4_path, pts_unit='sec')  # video: (T, H, W, 3), uint8
        else:
            raw_data_path = os.path.join(self.root, local_path, 'data.npz')
            # Load the raw data
            data = np.load(raw_data_path, allow_pickle=True)
            # Extract RGB observations
            obs_rgb = torch.from_numpy(data['obs_rgb'])  # Shape: (n_timesteps, height, width, 3)

        n_timesteps = len(obs_rgb)
        # Sample a random timestep
        sampled_ts = np.random.randint(n_timesteps)
        image_data = obs_rgb[sampled_ts]  # Shape: (height, width, 3)
        
        # Convert numpy array to tensor: (H, W, C) -> (C, H, W)
        image = image_data.permute(2, 0, 1).float()
        
        # Normalize from [0, 255] to [-1, 1] for VAE training
        image = (image / 255.0) * 2.0 - 1.0
        
        return image, sampled_ts

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader."""
        return {
            'image': torch.stack([b['image'] for b in batch]),
            'instance_idx': torch.tensor([b['instance_idx'] for b in batch]),
            'timestep_idx': torch.tensor([b['timestep_idx'] for b in batch]),
        }