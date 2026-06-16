import os
import time
from typing import *

import numpy as np
import torch
import torchvision.io as io
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation as R

from utils.camera_util import matrix_to_rotation_6d, compute_dpitch_dyaw_from_cam_dir
from . import StandardDatasetBase


class MinetestLatentCameraAction(StandardDatasetBase):
    """
    Minetest camera dataset for training voxel diffusion models.

    This dataset loads pre-computed latent voxel and pixel representations as well as intrinsics, extrinsics, and actions
    from raw Minetest data. It converts camera parameters to a (quaternion,translation,FOV) representation.

    The latent voxel context window can be different from the rest of the data context windows

    Args:
        root (str): path to the dataset directory
        min_level_quality_score (float): minimum level quality score (default: 1.0)
        split (str): split name ('train' or 'val')
    """

    def __init__(
            self,
            root: str,
            num_instances: int = None,
            min_level_quality_score: float = 1.0,
            split: str = 'train',
            clip_len: int = -1,
            sample_from_start_of_episode: bool = False,
            voxel_offset_ts: int = 0, # Will sample voxels voxel_offset_ts backwards from sampled timestep
            pixel_offset_ts: int = 0, # Will sample pixels pixel_offset_ts backwards from sampled timestep
            camera_offset_ts: int = 0, # Will sample camera camera_offset_ts backwards from sampled timestep
            cam_translation_representation: Literal["extrinsics", "voxel_local"] = "extrinsics",
            cam_rotation_representation: Literal["quaternion", "6d", "cam_dir"] = 'quaternion',
            sample_voxel_importance_mask: bool = False,
            sample_cropped_voxel: bool = False,
            sample_raw_images: bool = False,
            sample_voxel_classes: bool = False,
            no_latents: bool = False, #disable loading latents entirely
            normalize_latents: bool = True, # Whether to normalize latents using pre-computed stats
            instances: str = None, # if specified limits dataset to these instances (comma-separated sha256 hashes)
            remove_bad_frames: bool = False,
            bad_frames_config: dict = None,
    ):

        self.num_instances = num_instances if not instances else None
        self.min_level_quality_score = min_level_quality_score if instances is None else -1.0
        self.split = split if instances is None else instances
        self.clip_len = clip_len
        self.sample_from_start_of_episode = sample_from_start_of_episode
        self.voxel_offset_ts = voxel_offset_ts
        self.pixel_offset_ts = pixel_offset_ts
        self.camera_offset_ts = camera_offset_ts
        self.cam_translation_representation = cam_translation_representation
        self.cam_rotation_representation = cam_rotation_representation
        self.sample_voxel_importance_mask = sample_voxel_importance_mask
        self.sample_cropped_voxel = sample_cropped_voxel
        self.sample_raw_images = sample_raw_images
        self.sample_voxel_classes = sample_voxel_classes
        self.no_latents = no_latents
        self.normalize_latents = normalize_latents

        # defines heuristics to remove bad/anomalous data
        if remove_bad_frames:
            if bad_frames_config is None:
                self.bad_frames_config = {
                    "fps": True,                    # filters out clips with unstable fps
                    "dcam_xyz": True,               # filters out clips with anomalous camera positions changes
                    "dcam_pitchyaw": True,          # filters out clips with anomalous camera orientation changes
                    "voxel_importance_mask": True   # filters out clips with aliased (partial) voxel grid updates
                }
            else:
                self.bad_frames_config = bad_frames_config

            if self.bad_frames_config["voxel_importance_mask"] and not self.sample_voxel_importance_mask:
                raise ValueError("To filter bad frames using voxel importance mask, sample_voxel_importance_mask must be True.")
        else:
            self.bad_frames_config = None

        super().__init__(root, instances=instances)

        self.normalization = {
            'voxel_latent': None,
            'pixel_latent': None,
        }
        normalize_latents = False if no_latents else normalize_latents
        if normalize_latents:
            assert os.path.exists(os.path.join(self.root, 'latent_stats.npz')), \
            f"Latent stats file not found: {os.path.join(self.root, 'latents_stats.npz')}."
            latents_stats = np.load(os.path.join(self.root, 'latent_stats.npz'))
            self.normalization['voxel_latent'] = {
                'mean': torch.from_numpy(latents_stats['voxel_mean']),
                'std': torch.from_numpy(latents_stats['voxel_std']),
            }
            logger.info(f"Loaded voxel latent normalization stats from {os.path.join(self.root, 'latent_stats.npz')}")
            self.normalization['pixel_latent'] = {
                'mean': torch.from_numpy(latents_stats['pixel_mean']),
                'std': torch.from_numpy(latents_stats['pixel_std']),
            }
            logger.info(f"Loaded pixel latent normalization stats from {os.path.join(self.root, 'latent_stats.npz')}")

    def filter_metadata(self, metadata, instances=None):
        stats = {}
        metadata = metadata[metadata['pixel_latent_generated'] & metadata['voxel_latent_generated']]
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

    def filter_bad_frames(self, rawdata, voxel_importance_mask, min_clip_len):

        def player_is_swimming(cam_pos_g):
            T = len(cam_pos_g)
            dcam_pos_g = cam_pos_g[1:] - cam_pos_g[:-1]
            start_swim = np.argwhere(dcam_pos_g[...,2] < -0.8).squeeze(-1) + 1
            end_swim = np.argwhere(dcam_pos_g[...,2] > 0.8).squeeze(-1) + 1
            if len(start_swim) > len(end_swim):
                end_swim = np.append(end_swim, T-1)
            if len(start_swim) != len(end_swim): # if we can't detect swim start/ end we filter out all swimming ts
                return np.zeros((T,), dtype=bool)

            # Build mask via prefix sums: mark +1 at starts, -1 at ends, then cumsum > 0
            diff = np.zeros(T, dtype=np.int32)
            np.add.at(diff, start_swim,  1)
            np.add.at(diff, end_swim,   -1)
            swim_mask = (np.cumsum(diff[:-1]) > 0)
            swim_mask = np.concatenate([np.array([False], dtype=bool), swim_mask])
            return swim_mask

        def remove_short_clips(mask, min_clip_len):

            padded = np.concatenate(([False], mask, [False]))
            diff = np.diff(padded.astype(int))

            starts = np.where(diff == 1)[0]
            ends   = np.where(diff == -1)[0]
            lengths = ends - starts

            # Build output mask using a difference array
            diff_out = np.zeros(len(mask) + 1, dtype=int)

            valid = lengths >= min_clip_len
            np.add.at(diff_out, starts[valid],  1)
            np.add.at(diff_out, ends[valid],   -1)

            return np.cumsum(diff_out[:-1]) > 0

        def drop_first_n_true(mask, n):
            """
            For each contiguous True run, set the first n entries to False.
            Keeps entries from (start + n) to end.
            Args:
                mask: (T,) boolean array
                n: number of initial True frames to drop per run
            Returns:
                (T,) boolean array
            """
            mask = np.asarray(mask, dtype=bool)
            T = mask.size
            if n <= 0 or T == 0:
                return mask.copy()
            # Find run boundaries
            padded = np.concatenate(([False], mask, [False]))
            diff = np.diff(padded.astype(np.int8))
            starts = np.where(diff == 1)[0]  # inclusive
            ends = np.where(diff == -1)[0]  # exclusive
            # New kept intervals: [starts + n, ends)
            keep_starts = starts + n
            keep_ends = ends
            # Clamp starts so intervals with keep_start >= end become empty
            valid = keep_starts < keep_ends
            keep_starts = keep_starts[valid]
            keep_ends = keep_ends[valid]
            # Reconstruct mask via difference array + cumsum
            diff_out = np.zeros(T + 1, dtype=np.int32)
            np.add.at(diff_out, keep_starts, 1)
            np.add.at(diff_out, keep_ends, -1)
            return (np.cumsum(diff_out[:-1]) > 0)

        assert self.bad_frames_config is not None

        _debug = False
        if _debug:
            t_start = time.time()

        good_frame_masks = {}
        # filters out clips with unstable fps reported by game engine
        if self.bad_frames_config["fps"]:
            min_fps, max_fps = 22, 30 #hardcoded
            fps = 1/rawdata['dt_minetest']
            good_frames_fps = (fps >= min_fps) & (fps <= max_fps)
            good_frame_masks['fps'] = good_frames_fps
        # filters out clips with anomalous camera positions changes reported by game engine
        if self.bad_frames_config["dcam_xyz"]:
            min_xy, max_xy = 0, 1
            min_z, max_z = 1.42, 2.6
            min_z_swim, max_z_swim = 0.57, 1.6
            voxel_info = self.dataset_params["voxel_info"]
            vox_grid_dim = voxel_info["dims"]
            # assert len(vox_grid_dim) == 3, "voxels must have the XYZ shape"
            assert vox_grid_dim[0] == vox_grid_dim[1] == vox_grid_dim[2], "Only cubic grids are supported"
            grid_size = vox_grid_dim[0]

            cam_pos_l = rawdata['cam_pos_local'] * grid_size
            cam_pos_g = rawdata['cam_pos']

            good_frames_xy = (cam_pos_l[...,0] >= min_xy) & (cam_pos_l[...,0] <= max_xy) & (cam_pos_l[...,1] >= min_xy) & (cam_pos_l[...,1] <= max_xy)
            swim_mask = player_is_swimming(cam_pos_g)
            good_frames_z = (~swim_mask) & (cam_pos_l[...,2] >= min_z) & (cam_pos_l[...,2] <= max_z)
            good_frames_z_swim = swim_mask & (cam_pos_l[...,2] >= min_z_swim) & (cam_pos_l[...,2] <= max_z_swim)
            good_frames_z = good_frames_z | good_frames_z_swim
            good_frame_masks['dcam_xyz'] = good_frames_xy & good_frames_z
        # filters out clips with anomalous camera orientation changes reported by game engine
        if self.bad_frames_config["dcam_pitchyaw"]:
            mag_pitch = 3.6
            mag_yaw = 6.4
            eps = 0.1

            cam_dir = rawdata['cam_dir']
            dpitch, dyaw = compute_dpitch_dyaw_from_cam_dir(torch.from_numpy(cam_dir))
            dpitch = np.concatenate([[0], dpitch])
            dyaw = np.concatenate([[0], dyaw])
            good_frames_pitch = (
                    (np.abs(dpitch) >= (mag_pitch - eps)) & (np.abs(dpitch) <= (mag_pitch + eps)) |
                    (np.abs(dpitch) >= 0) & (np.abs(dpitch) <= eps)
            )
            good_frames_yaw = (
                    (np.abs(dyaw) >= (mag_yaw - eps)) & (np.abs(dyaw) <= (mag_yaw + eps)) |
                    (np.abs(dyaw) >= 0) & (np.abs(dyaw) <= eps)
            )
            good_frame_masks['dcam_pitchyaw'] = good_frames_pitch & good_frames_yaw
        # filters out clips with aliased (partial) voxel grid updates across two timesteps
        if self.bad_frames_config["voxel_importance_mask"] and voxel_importance_mask is not None:
            if len(voxel_importance_mask) > 0:
                max_vox_per_ts = 40 # hardcoded
                ti, xi, yi, zi = torch.from_numpy(voxel_importance_mask).T.int()
                unique_tis, ti_count = torch.unique(ti, return_counts=True)
                invalid_tis = unique_tis[ti_count > max_vox_per_ts].tolist()
                good_frames_vox_imp_mask = np.ones_like(good_frame_masks['dcam_xyz'], dtype=bool) #assumes dcam_xyz is always computed
                good_frames_vox_imp_mask[invalid_tis] = False
                good_frame_masks['voxel_importance_mask'] = good_frames_vox_imp_mask

        if not good_frame_masks:
            return None

        combined_mask = np.stack(list(good_frame_masks.values())).all(axis=0)
        combined_mask_no_short_clips = remove_short_clips(combined_mask, min_clip_len)

        if _debug:
            t_exec = time.time() - t_start
            T = len(rawdata['cam_pos'])
            for key, val in good_frame_masks.items():
                num_bad_frames = (~val).sum()
                n_frac = num_bad_frames / T * 100
                print(f"{key}: {num_bad_frames} ({n_frac}%) bad frames")
            num_bad_frames = (~combined_mask).sum()
            n_frac = num_bad_frames / T * 100
            print(f"combined: {num_bad_frames} ({n_frac}%) bad frames")
            num_good_frames = combined_mask_no_short_clips.sum()
            n_frac = num_good_frames / T * 100
            print(f"remaining good frames: {num_good_frames} ({n_frac}%) - min_clip_len: {min_clip_len}")
            print(f"filtering took {t_exec}s")
            print("===============================")

        # for each valid sequence, remove drop first n-1 frames as we sample clips w.r.t their last frame.
        combined_mask_no_short_clips = drop_first_n_true(combined_mask_no_short_clips, n=min_clip_len - 1)
        return combined_mask_no_short_clips

    def sample_ts(self, min_t, max_t, raw_data, voxel_importance_mask, instance):
        # note: clip is indexed by its last frame + 1

        if self.bad_frames_config:
            good_frames = self.filter_bad_frames(raw_data, voxel_importance_mask, min_t)
            if good_frames.sum() == 0:
                raise ValueError(f"No valid clip for instance {instance}. Another instance will be sampled.")
        else:
            good_frames = np.ones((max_t,), dtype=bool)
            good_frames[:min_t] = False

        t_end_excluded = np.random.choice(np.flatnonzero(good_frames)) + 1

        return t_end_excluded

    def get_instance(self, instance, instance_idx):
        raw_data = self._get_raw_data(instance)
        action = raw_data['action']  # (clip_len,)
        camera = self._extract_camera_representation(raw_data)
        imp_mask = None
        if self.sample_voxel_importance_mask:
            imp_mask = self._get_voxel_importance_mask(instance) # [sparse], (N, 4) [timestep_id, x_id, y_id, z_id]

        n_timesteps = len(action)
        max_offset = max(self.voxel_offset_ts, self.pixel_offset_ts, self.camera_offset_ts)
        if self.clip_len == -1:
            clip_len = n_timesteps - max_offset # sample largest clip possible
        elif n_timesteps < self.clip_len + max_offset:
            raise ValueError(f"Not enough timesteps! Instance {instance}: n_timesteps={n_timesteps}, "
                             f"clip_len={self.clip_len}"
                             f", max_offset={max_offset}")
        else:
            clip_len = self.clip_len

        min_t = clip_len + max_offset
        if self.sample_from_start_of_episode:
            sampled_ts = min_t
        else:
            sampled_ts = self.sample_ts(min_t, n_timesteps, raw_data, imp_mask, instance)

        # extract necessary info from raw_data
        context_start = sampled_ts - clip_len

        if self.no_latents:
            voxel_latent = torch.empty(size=(clip_len,), dtype=torch.float32)
            pixel_latent = torch.empty(size=(clip_len,), dtype=torch.float32)
        else:
            voxel_latent = self._get_voxel_latent(instance)
            pixel_latent = self._get_pixel_latent(instance)
            # sample timestep, then convert to tensor
            voxel_latent = voxel_latent[context_start - self.voxel_offset_ts:sampled_ts - self.voxel_offset_ts]
            if isinstance(voxel_latent, np.ndarray):
                voxel_latent = torch.from_numpy(voxel_latent)
            voxel_latent = voxel_latent.float()

            pixel_latent = pixel_latent[context_start - self.pixel_offset_ts:sampled_ts - self.pixel_offset_ts]
            if isinstance(pixel_latent, np.ndarray):
                pixel_latent = torch.from_numpy(pixel_latent)
            pixel_latent = pixel_latent.float()
            pixel_latent = rearrange(pixel_latent, 't h w c -> t c h w')  # (clip_len, C, H, W)

        if self.sample_raw_images:
            if 'obs_rgb' not in raw_data:
                raw_images = self._get_rgb_from_mp4(instance)
            else:
                raw_images = torch.from_numpy(raw_data['obs_rgb'])
            raw_images = raw_images[context_start - self.pixel_offset_ts:sampled_ts - self.pixel_offset_ts]
            raw_images = rearrange(raw_images, 't h w c -> t c h w').float() / 255.0 * 2.0 - 1.0

        if self.sample_voxel_classes:
            voxel_classes = self._get_voxel_classes(instance)
            voxel_classes = torch.from_numpy(voxel_classes[context_start - self.voxel_offset_ts:sampled_ts - self.voxel_offset_ts]).long()  # (T, X, Y, Z)

        action = action[context_start:sampled_ts]
        action = torch.from_numpy(action).float()  # (clip_len, action_dim)

        camera = camera[context_start - self.camera_offset_ts :sampled_ts - self.camera_offset_ts]
        camera = torch.from_numpy(camera).float()

        if self.normalization['voxel_latent'] is not None:
            voxel_latent = ((voxel_latent - self.normalization['voxel_latent']['mean'].view(1, -1, 1, 1, 1).to(voxel_latent))
                                / self.normalization['voxel_latent']['std'].view(1, -1, 1, 1, 1).to(voxel_latent))
        if self.normalization['pixel_latent'] is not None:
            pixel_latent = ((pixel_latent - self.normalization['pixel_latent']['mean'].view(1, -1, 1, 1).to(pixel_latent))
                         / self.normalization['pixel_latent']['std'].view(1, -1, 1, 1).to(pixel_latent))


        result = {
            # Context sequences
            'camera': camera,  # (clip_len, cam_dim)
            'action': action,  # (clip_len, action_dim)
            'pixel': pixel_latent,  # (clip_len, Cp, H, W)
            'voxel': voxel_latent,  # (clip_len, Cv, X, Y, Z)

            # Metadata
            'instance_idx': instance_idx, # int
            'timestep_idx': context_start, # int
        }

        if self.sample_voxel_importance_mask:
            # filter out any timesteps not in [context_start, sampled_ts)
            valid_ids = np.argwhere((imp_mask[:,0]>=context_start - self.voxel_offset_ts) & (imp_mask[:,0]<sampled_ts - self.voxel_offset_ts)).squeeze(-1)
            result['voxel_importance_mask'] = imp_mask[valid_ids]
            result['voxel_importance_mask'][:,0] -= (context_start + self.voxel_offset_ts) # align mask timesteps with start of clip

        if self.sample_cropped_voxel:
            cropped_voxel = self._get_cropped_voxel(instance)
            cropped_voxel = cropped_voxel[context_start - self.voxel_offset_ts: sampled_ts - self.voxel_offset_ts]
            result['cropped_voxel'] = torch.from_numpy(cropped_voxel).long()

        if self.sample_raw_images:
            result['raw_images'] = raw_images
        if self.sample_voxel_classes:
            result['voxel_classes'] = voxel_classes

        return result

    def _get_pixel_latent(self, instance):
        pixel_latents_path = os.path.join(self.root, 'pixel_latents', f'{instance}.safetensors')
        if os.path.exists(pixel_latents_path):
            pixel_latent = load_file(pixel_latents_path)['mean']
        else:
            pixel_latents_path = os.path.join(self.root, 'pixel_latents', f'{instance}.npz')
            if os.path.exists(pixel_latents_path):
                pixel_latent = np.load(pixel_latents_path)['mean']
            else:
                raise FileNotFoundError(f"Pixel latents not found: {pixel_latents_path}")
        return pixel_latent

    def _get_voxel_latent(self, instance):
        voxel_latents_path = os.path.join(self.root, 'voxel_latents', f'{instance}.safetensors')
        if os.path.exists(voxel_latents_path):
            voxel_latent = load_file(voxel_latents_path)['mean']
        else:
            voxel_latents_path = os.path.join(self.root, 'voxel_latents', f'{instance}.npz')
            if os.path.exists(voxel_latents_path):
                voxel_latent = np.load(voxel_latents_path)['mean']
            else:
                raise FileNotFoundError(f"Voxel latents not found: {voxel_latents_path}")
        return voxel_latent

    def _get_raw_data(self, instance):
        # Get the local path from metadata
        local_path = self.metadata.loc[instance]['local_path']
        raw_data_path = os.path.join(self.root, local_path, 'data.npz')

        data = np.load(raw_data_path, allow_pickle=True)
        return data

    def _get_rgb_from_mp4(self, instance):
        local_path = self.metadata.loc[instance]['local_path']
        mp4_path = os.path.join(self.root, local_path, 'rgb.mp4')
        video, _, _ = io.read_video(mp4_path, pts_unit='sec') # video: (T, H, W, 3), uint8
        return video

    def _extract_camera_representation(self, data):

        fov_x = data['fov_x']
        if (self.cam_translation_representation in ["extrinsics"]) or (
            self.cam_rotation_representation in ["quaternion", "6d"]
        ):
            extrinsics = data['extrinsics_local']  # Shape: (T, 4, 4)
        else:
            extrinsics = None

        # Convert extrinsics matrices to rotations + translation (vectorized)
        # Extract all rotation matrices and translations at once
        if self.cam_translation_representation == "extrinsics":
            camera_translations = extrinsics[:, :3, 3]  # (T, 3)
        elif self.cam_translation_representation == "voxel_local":
            camera_translations = data['cam_pos_local'] # (T, 3)

        if self.cam_rotation_representation == "quaternion":
            rotation_matrices = extrinsics[:, :3, :3]  # (T, 3, 3)
            # Vectorized quaternion conversion using scipy
            rotations = R.from_matrix(rotation_matrices)  # Handle all timesteps at once
            camera_rotations = rotations.as_quat()  # (T, 4) in (x, y, z, w) format
        elif self.cam_rotation_representation == "6d":
            rotation_matrices = extrinsics[:, :3, :3]  # (T, 3, 3)
            camera_rotations = matrix_to_rotation_6d(rotation_matrices)  # (T, 6)
        elif self.cam_rotation_representation == "cam_dir":
            camera_rotations = data['cam_dir']
            norm = np.linalg.norm(camera_rotations, axis=-1, keepdims=True)
            camera_rotations = camera_rotations / np.clip(norm, a_min=1e-8, a_max=None)
        else:
            raise ValueError(f"Unknown camera rotation representation: {self.cam_rotation_representation}")

        # Combine camera context into sequence of camera states
        cameras = np.concatenate([
            camera_rotations,
            camera_translations,
            np.expand_dims(fov_x, -1)
        ], axis=-1)  # (T, N)

        return cameras

    def _get_voxel_importance_mask(self, instance):
        path = os.path.join(self.root, 'importance_masks', f'{instance}.npy')
        if os.path.exists(path):
            masks = np.load(path)
        else:
            path = os.path.join(self.root, 'voxel_classes', f'{instance}.npz')
            masks = np.load(path)['importance_masks']
        return masks

    def _get_cropped_voxel(self, instance):
        path = os.path.join(self.root, 'cropped_voxel_classes', f'{instance}.npy')
        if os.path.exists(path):
            cropped_voxel = np.load(path)
        else:
            path = os.path.join(self.root, 'voxel_classes', f'{instance}.npz')
            cropped_voxel = np.load(path)['cropped_voxel']
        return cropped_voxel

    def _get_voxel_classes(self, instance):
        voxel_classes_path = os.path.join(self.root, 'voxel_classes', f'{instance}.npz')
        if not os.path.exists(voxel_classes_path):
            raise FileNotFoundError(f"Voxel classes not found: {voxel_classes_path}")
        voxel_classes = np.load(voxel_classes_path)['node_classes']
        return voxel_classes

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for timestep-based DataLoader."""
        result = {

            'camera': torch.stack([b['camera'] for b in batch]),  # (B, clip_len, 8)
            'action': torch.stack([b['action'] for b in batch]),  # (B, clip_len, 18)
            'pixel': torch.stack([b['pixel'] for b in batch]),  # (clip_len, H, W, Cp)
            'voxel': torch.stack([b['voxel'] for b in batch]),  # (clip_len, X, Y, Z, Cv)

            # Metadata (episode_idx is string sha256, keep as list)
            'instance_idx': torch.tensor([b['instance_idx'] for b in batch]),
            'timestep_idx': torch.tensor([b['timestep_idx'] for b in batch]),
        }

        # given that importance_masks is a sparse encoding, batchify by setting first dim as the instance id in batch
        if 'voxel_importance_mask' in batch[0]:
            batched_vals = []
            for i, b in enumerate(batch):
                batched_vals.append(torch.cat([
                    torch.full((b['voxel_importance_mask'].shape[0], 1), i, dtype=torch.int16),
                    torch.from_numpy(b['voxel_importance_mask']).to(torch.int16),
                ], dim=-1))
            result['voxel_importance_mask'] = torch.cat(batched_vals) # [sparse] (N, 5) [instance_id, timestep_id, x_id, y_id, z_id]

        if 'cropped_voxel' in batch[0]:
            result['cropped_voxel'] = torch.stack([b['cropped_voxel'] for b in batch])

        if 'raw_images' in batch[0]:
            result['raw_images'] = torch.stack([b['raw_images'] for b in batch])

        if 'voxel_classes' in batch[0]:
            result['voxel_classes'] = torch.stack([b['voxel_classes'] for b in batch])

        return result

    @property
    def raw_pixel_obs_width(self):
        return self.dataset_params["info"]["script_args"]["resolution_w"]

    @property
    def raw_pixel_obs_height(self):
        return self.dataset_params["info"]["script_args"]["resolution_h"]

    @property
    def uncompressed_voxel_resolution(self):
        return self.dataset_params["voxel_info"]["dims"][:3]


class MinetestLatentCameraActionEval(MinetestLatentCameraAction):

    def __init__(
            self,
            root: str,
            num_instances: int = None,
            clip_len: int = -1,
            sample_from_start_of_episode: bool = True,
            instances: str = None, # if specified limits dataset to these instances (comma-separated sha256 hashes)
            **kwargs
    ):
        if len(kwargs) > 0:
            logger.warning(f"MinetestLatentCameraActionEval ignore these arguments: {kwargs.keys()}")

        super().__init__(
            root=root,
            clip_len=clip_len,
            sample_from_start_of_episode=sample_from_start_of_episode,
            num_instances=num_instances,
            instances=instances,
            min_level_quality_score=1.0,
            split='val',
            voxel_offset_ts=0,
            pixel_offset_ts=0,
            camera_offset_ts=0,
            cam_translation_representation="voxel_local",
            cam_rotation_representation='6d',
            sample_voxel_importance_mask=False,
            sample_cropped_voxel=False,
            sample_raw_images=True,
            sample_voxel_classes=True,
            no_latents=True,
            normalize_latents=False,
            remove_bad_frames=False,
            bad_frames_config=None
        )

    def filter_metadata(self, metadata, instances=None):
        stats = {}
        metadata = metadata[metadata["validation_set"] == True]
        stats['Available samples'] = len(metadata)
        metadata = metadata[metadata['level_quality_score'] >= self.min_level_quality_score]
        stats[f'Level quality score >= {self.min_level_quality_score}'] = len(metadata)

        if self.split is not None and self.split not in ["train", "val"]:
            instances = self.split.split(',')
            metadata = metadata[metadata['sha256'].isin(instances)]

        if self.num_instances is not None:
            metadata = metadata.iloc[:self.num_instances]

        stats['In split'] = len(metadata)
        return metadata, stats

import random
from utils.camera_util import minetest_cam_to_worldmem_cam

class MinetestWorldMem(MinetestLatentCameraAction):

    def __init__(
            self,
            memory_condition_length=8,
            n_frames=8,
            training_dropout=0.1,
            pos_range=8,
            angle_range=110,
            add_timestamp_embedding=True,
            causal_frame=False,
            sample_more_event=False,
            **kwargs
    ):
        # default parameters of WorldMem
        self.memory_condition_length = memory_condition_length
        self.n_frames = n_frames
        self.training_dropout = training_dropout
        self.pos_range = pos_range
        self.angle_range = angle_range
        self.add_timestamp_embedding = add_timestamp_embedding
        self.causal_frame = causal_frame
        self.sample_more_event = sample_more_event

        super().__init__(**kwargs)

    def filter_metadata(self, metadata, instances=None):
        stats = {}
        if not self.no_latents:
            # to ensure we use same training data for every baseline
            metadata = metadata[metadata['pixel_latent_generated'] & metadata['voxel_latent_generated']]
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

    def _extract_camera_representation(self, data):

        fov_x = data['fov_x']
        focal_length = 1 / (2 * np.tan(fov_x / 2))
        cam_pos = torch.from_numpy(data['cam_pos'])
        cam_dir = torch.from_numpy(data['cam_dir'])
        poses = minetest_cam_to_worldmem_cam(cam_pos, cam_dir).numpy() # x,y,z,pitch,yaw

        return poses, focal_length

    def get_instance(self, instance, instance_idx):
        raw_data = self._get_raw_data(instance)
        actions_pool = raw_data['action']  # (clip_len,)
        poses_pool, focal_lengths_pool = self._extract_camera_representation(raw_data)
        n_timesteps = len(actions_pool)

        imp_mask = None
        if n_timesteps < self.n_frames:
            raise ValueError(f"Not enough timesteps! Instance {instance}: n_timesteps={n_timesteps}, "
                             f"clip_len={self.n_frames}")

        if self.sample_from_start_of_episode:
            sampled_ts = n_frames = n_timesteps
            frame_idx = 0
        else:
            sampled_ts = self.sample_ts(self.n_frames, n_timesteps, raw_data, imp_mask, instance)
            frame_idx = sampled_ts - self.n_frames
            n_frames = self.n_frames

        # extract necessary info from raw_data
        actions = np.copy(actions_pool[frame_idx : frame_idx + n_frames])
        poses = np.copy(poses_pool[frame_idx : frame_idx + n_frames])
        focal_lengths = np.copy(focal_lengths_pool[frame_idx : frame_idx + n_frames])
        if self.no_latents:
            pixel_latent_pool = torch.empty(size=(n_frames,), dtype=torch.float32)
        else:
            pixel_latent_pool = self._get_pixel_latent(instance)
        pixel_latent = pixel_latent_pool[frame_idx:sampled_ts]

        # === 4. Normalize poses relative to current segment ===
        def normalize_pose(pose, ref_pose):
            pose[:, :3] -= ref_pose[:1, :3]
            pose[:, -1] = -pose[:, -1]
            pose[:, 3:] %= 360
            return pose

        poses_pool = normalize_pose(poses_pool, poses)
        poses = normalize_pose(poses, poses)

        # === 5. Sample memory frames for training ===
        if not self.sample_from_start_of_episode and self.memory_condition_length > 0:
            use_memory = random.uniform(0, 1) > self.training_dropout

            if use_memory:
                # Compute pose distance between current and candidate frames
                dis = np.abs(poses[:, None] - poses_pool[None, :])
                dis[..., 3:][dis[..., 3:] > 180] = 360 - dis[..., 3:][dis[..., 3:] > 180]

                spatial_match = (dis[..., :3] <= self.pos_range).sum(-1) >= 3
                angular_match = (dis[..., 3:] <= self.angle_range).sum(-1) >= 2
                not_exact_match = ((dis[..., :3] > 0).sum(-1) >= 1) | ((dis[..., 3:] > 0).sum(-1) >= 1)

                valid_index = (spatial_match & angular_match & not_exact_match).sum(0)

                # Exclude future if causality and timestamp are enabled
                # note self.causal_frame always false in original codebase so this is never triggered
                if self.add_timestamp_embedding and self.causal_frame and (actions_pool[:frame_idx, 24] == 1).sum() > 0:
                    valid_index[frame_idx:] = 0

                # Select indices satisfying condition
                mask = valid_index >= 1
                mask[0] = False
                candidate_indices = np.argwhere(mask)

                # Backup candidates with weaker conditions
                mask2 = valid_index >= 0
                mask2[0] = False

                count = min(self.memory_condition_length, candidate_indices.shape[0])
                selected = candidate_indices[np.random.choice(candidate_indices.shape[0], count, replace=True)][:, 0]

                if count < self.memory_condition_length:
                    extra = np.argwhere(mask2)
                    extra = extra[np.random.choice(extra.shape[0], self.memory_condition_length - count, replace=True)][:, 0]
                    selected = np.concatenate([selected, extra])

                # Prioritize event-trigger frames if applicable
                if self.sample_more_event and random.uniform(0, 1) > 0.3:
                    event_idx = torch.nonzero(actions_pool[:frame_idx, 24] == 1)[:, 0]
                    if len(event_idx) > self.memory_condition_length // 2:
                        event_idx = event_idx[-self.memory_condition_length // 2:]
                    if len(event_idx) > 0:
                        selected[-len(event_idx):] = event_idx + 4

            else:
                selected = np.full(self.memory_condition_length, random.randint(0, frame_idx))

            pixel_latent = torch.cat([pixel_latent, pixel_latent_pool[selected]], dim=0)
            actions = np.concatenate([actions, actions_pool[selected]], axis=0)
            poses = np.concatenate([poses, poses_pool[selected]], axis=0)
            focal_lengths = np.concatenate([focal_lengths, focal_lengths_pool[selected]], axis=0)
            timestamp = np.concatenate([np.arange(frame_idx, frame_idx + n_frames), selected])
        else:
            timestamp = np.arange(n_frames)

        if self.sample_raw_images:
            if 'obs_rgb' not in raw_data:
                raw_images = self._get_rgb_from_mp4(instance)
            else:
                raw_images = torch.from_numpy(raw_data['obs_rgb'])
            raw_images = raw_images[frame_idx:sampled_ts]
            raw_images = rearrange(raw_images, 't h w c -> t c h w').float() / 255.0 * 2.0 - 1.0

        if self.sample_voxel_classes:
            voxel_classes = self._get_voxel_classes(instance)
            voxel_classes = torch.from_numpy(voxel_classes[frame_idx:sampled_ts]).long()  # (T, X, Y, Z)

        if not self.no_latents:
            if isinstance(pixel_latent, np.ndarray):
                pixel_latent = torch.from_numpy(pixel_latent)
            pixel_latent = pixel_latent.float()
            pixel_latent = rearrange(pixel_latent, 't h w c -> t c h w')  # (clip_len, C, H, W)
            if self.normalization['pixel_latent'] is not None:
                pixel_latent = ((pixel_latent - self.normalization['pixel_latent']['mean'].view(1, -1, 1, 1).to(pixel_latent))
                                / self.normalization['pixel_latent']['std'].view(1, -1, 1, 1).to(pixel_latent))
        actions = torch.from_numpy(actions).float()  # (clip_len, action_dim)
        poses = torch.from_numpy(poses).float()
        focal_lengths = torch.from_numpy(focal_lengths).float()
        timestamp = torch.from_numpy(timestamp).long()

        result = {
            # Context sequences
            'pose': poses,  # (T , 5)
            'focal_length': focal_lengths, # (T,)
            'action': actions,  # (T, action_dim)
            'pixel': pixel_latent,  # (T, Cp, H, W)
            'ts': timestamp, # (T,)

            # Metadata
            'instance_idx': instance_idx, # int
            'timestep_idx': frame_idx, # int
        }

        if self.sample_raw_images:
            result['raw_images'] = raw_images
        if self.sample_voxel_classes:
            result['voxel_classes'] = voxel_classes

        return result

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for timestep-based DataLoader."""
        result = {

            'pose': torch.stack([b['pose'] for b in batch]),  # (B, T, 5)
            'focal_length': torch.stack([b['focal_length'] for b in batch]),  # (B, T)
            'action': torch.stack([b['action'] for b in batch]),  # (B, T, 23)
            'pixel': torch.stack([b['pixel'] for b in batch]),  # (clip_len, H, W, Cp)
            'ts': torch.stack([b['ts'] for b in batch]),

            # Metadata (episode_idx is string sha256, keep as list)
            'instance_idx': torch.tensor([b['instance_idx'] for b in batch]),
            'timestep_idx': torch.tensor([b['timestep_idx'] for b in batch]),
        }

        if 'raw_images' in batch[0]:
            result['raw_images'] = torch.stack([b['raw_images'] for b in batch])

        if 'voxel_classes' in batch[0]:
            result['voxel_classes'] = torch.stack([b['voxel_classes'] for b in batch])

        return result