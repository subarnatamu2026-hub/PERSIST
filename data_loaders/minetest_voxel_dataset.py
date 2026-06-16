# fmt: off
import json
import os

import numpy as np
import torch

from . import StandardDatasetBase


class MinetestVoxel(StandardDatasetBase):
    """
    SparseFeat2MtRender dataset.

    Args:
        roots (str): paths to the dataset
        resolution (int): resolution of the data
        min_level_quality_score (float): minimum level quality score
        split (str): split name (train, val)
        load_light_classes (bool): whether to load light classes (not implemented yet)
    """

    def __init__(
            self,
            root: str,
            resolution: int = 64,
            num_instances: int = None,
            min_level_quality_score: float = 1.0,
            split: str = 'train',
            instances: str = None,
            load_light_classes: bool = False,
    ):

        self.resolution = resolution
        self.num_instances = num_instances if not instances else None
        self.min_level_quality_score = min_level_quality_score if instances is None else -1.0
        self.split = split if instances is None else instances

        super().__init__(root)

        with open(os.path.join(root, 'mt_voxel_classdict.json')) as f:
            self.voxel_classdict = json.load(f)
        self.load_config = {
            'node_classes': True,
            'light_classes': load_light_classes,
        }
        if self.load_config['light_classes']:
            raise NotImplementedError("Loading light classes is not implemented yet.")

        self.vocab_size = {
            'node_classes': len(self.voxel_classdict['node_classes']),
            'light_classes': len(self.voxel_classdict['light_classes']) if self.load_config['light_classes'] else None,
        }

    def filter_metadata(self, metadata, instances=None):
        stats = {}
        metadata = metadata[metadata[f'voxelized']]
        stats['Available samples'] = len(metadata)
        metadata = metadata[metadata['level_quality_score'] >= self.min_level_quality_score]
        stats[f'Level quality score >= {self.min_level_quality_score}'] = len(metadata)

        if self.split == "train":
            metadata = metadata[metadata["validation_set"] == False]
        elif self.split == "val":
            metadata = metadata[metadata["validation_set"] == True]
        else:
            instances = self.split.split(',')
            metadata = metadata[metadata['sha256'].isin(instances)]

        if self.num_instances is not None:
            metadata = metadata.iloc[:self.num_instances]

        stats['In split'] = len(metadata)
        return metadata, stats

    def get_instance(self, instance, instance_idx):
        voxel_classes, sampled_ts = self._get_voxel_classes(instance)
        return {
            'voxel_classes': voxel_classes,
            'instance_idx': instance_idx,
            'timestep_idx': sampled_ts,
        }

    def _get_voxel_classes(self, instance):
        voxel_classes_path = os.path.join(self.root, 'voxel_classes', f'{instance}.npz')
        voxel_classes = np.load(voxel_classes_path)['node_classes']
        sampled_ts = np.random.randint(len(voxel_classes))
        voxel_classes = voxel_classes[sampled_ts]
        voxel_classes = torch.from_numpy(voxel_classes)
        return voxel_classes, sampled_ts

    @staticmethod
    def collate_fn(batch):
        return {
            'voxel_classes': torch.stack([b['voxel_classes'] for b in batch]),
            'instance_idx': torch.tensor([b['instance_idx'] for b in batch]),
            'timestep_idx': torch.tensor([b['timestep_idx'] for b in batch]),
        }

class MinetestVoxelActionVCenterEpisode(MinetestVoxel):
    """
    Minetest Voxel Action Center Episode dataset.

    Args:

    """

    def __init__(
            self,
            root: str,
            resolution: int = 64,
            num_instances: int = None,
            min_level_quality_score: float = 1.0,
            split: str = 'train',
            instances: str = None,
    ):
        super().__init__(root, resolution, num_instances, min_level_quality_score, split, instances)

    def get_instance(self, instance, instance_idx):
        voxel_classes = self._get_voxel_classes(instance)
        raw_data = self._get_raw_data(instance)
        action = raw_data['action']
        voxel_center = raw_data['obs_voxel_center']

        return {
            'voxel_classes': torch.from_numpy(voxel_classes),
            'action': torch.from_numpy(action).long(),

            # Metadata
            'voxel_center': torch.from_numpy(voxel_center).long(),
            'instance_idx': instance_idx,
        }

    def _get_voxel_classes(self, instance):
        voxel_classes_path = os.path.join(self.root, 'voxel_classes', f'{instance}.npz')
        if os.path.exists(voxel_classes_path):
            voxel_classes = np.load(voxel_classes_path)['node_classes']
        else:
            raise FileNotFoundError(f"Voxel classes not found: {voxel_classes_path}")
        return voxel_classes

    def _get_raw_data(self, instance):
        # Get the local path from metadata
        local_path = self.metadata.loc[instance]['local_path']
        raw_data_path = os.path.join(self.root, local_path, 'data.npz')

        data = np.load(raw_data_path, allow_pickle=True)
        return data

    @staticmethod
    def collate_fn(batch):
        return {
            'voxel_classes': torch.stack([(b['voxel_classes']) for b in batch]),
            'action': torch.stack([b['action'] for b in batch]),
            'voxel_center': torch.stack([(b['voxel_center']) for b in batch]),
            'instance_idx': torch.tensor([b['instance_idx'] for b in batch]),
        }