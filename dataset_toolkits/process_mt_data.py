import contextlib
import itertools
import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import tyro
from einops import rearrange
from loguru import logger
from tqdm import tqdm

from utils.camera_util import angular_diff_deg
from utils.camera_util import compute_cam_pose_local, compute_extrinsics


def get_data(local_path):
    data_path = output_dir / local_path / "data.npz"
    # slow and RAM hungry, consider the commented line below instead if RAM is an issue
    with np.load(data_path) as f:
        # copy everything into a dict
        data = {k: f[k].copy() for k in f.files}
    # data = np.load(data_path)
    return data


def check_data_validity(data):
    # Initialize dictionary to track which conditions were triggered
    triggered_conditions = {
        "early_termination": False,
        "moving_player": False,
        "corrupted_data": False,
        "framerate_drop": False,
        "frozen_voxel": False,
    }

    # Check each condition and update the dictionary
    if args.check_early_termination and (
            np.any(data["termination_flag"]) or np.any(data["truncation_flag"])
    ):
        triggered_conditions["early_termination"] = True

    if args.check_moving_player and (
            np.any(data["player_vel"] != 0)
            or not np.allclose(data["player_pos"], data["player_pos"][0])
    ):
        triggered_conditions["moving_player"] = True

    if args.check_corrupted_data and np.any(data["obs_voxel_mt"] > 8192):
        triggered_conditions["corrupted_data"] = True

    if args.check_framerate_drop and np.any(data["dt_minetest"] > 1.5 * data["dt_minetest"].mean()):
        triggered_conditions["framerate_drop"] = True

    if args.check_frozen_voxel:
        min_static_frames = 2  # minimum number of consecutive frames with the same voxel center to consider as frozen
        max_n_vox_shift = 2  # max number of voxel the voxel center can move between two timesteps
        max_shift = math.ceil(math.sqrt(3) * max_n_vox_shift)
        obs_voxel_center = data['obs_voxel_center']
        # check for repetitions in obs_voxel_center over time:
        if np.any(np.linalg.norm(obs_voxel_center[1:] - obs_voxel_center[:-1], axis=-1) < 1e-5):
            ts_repeated = np.where(np.linalg.norm(obs_voxel_center[1:] - obs_voxel_center[:-1], axis=-1) < 1e-5)[0]
            # separate into non consecutive repetitions
            ts_seq_start = ts_repeated[np.insert(np.diff(ts_repeated) > 1, 0, True)]
            ts_seq_end = ts_repeated[np.insert(np.diff(ts_repeated) > 1, -1, True)] + 1
            ts_seq_len = ts_seq_end - ts_seq_start
            long_seq_end_ts = ts_seq_end[ts_seq_len > min_static_frames]
            long_seq_next_ts = np.clip(long_seq_end_ts + 1, a_min=0, a_max=len(obs_voxel_center) - 1)
            pos_change_ts = np.where(
                np.linalg.norm(obs_voxel_center[long_seq_next_ts] - obs_voxel_center[long_seq_end_ts],
                               axis=-1) > max_shift)[0]
            # check 1 : voxel centre shifts more than {max_shift} after being static for over {min_static_frames} frames
            # check 2: instance ends with a static sequence of over {min_static_frames} frames
            if len(pos_change_ts) > 0 or long_seq_end_ts[-1] == (len(obs_voxel_center) - 1):
                triggered_conditions["frozen_voxel"] = True

    # Return both the validity check result and the triggered conditions
    is_valid = not any(triggered_conditions.values())
    return is_valid, triggered_conditions


def get_cropped_voxel_mt(obs_voxel_mt, crop_args, output_vox_grid):
    # Only load 'obs_voxel_mt' from the .npz file
    assert obs_voxel_mt.ndim == 5, "MT voxel data must have 5 dimensions (T, X, Y, Z, C)"
    assert obs_voxel_mt.shape[-1] == 2, "MT voxel data must have 2 channels"

    mt_vox_shape_xyz = obs_voxel_mt.shape[1:4]
    assert all([d >= output_vox_grid for d in mt_vox_shape_xyz]), (
        "MT voxel array must be as large or larger than the output voxel grid"
    )
    # Crop the voxel grid around each edge to the desired size
    obs_voxel_mt = obs_voxel_mt[
        :,
        crop_args["left"][0]: crop_args["right"][0],
        crop_args["left"][1]: crop_args["right"][1],
        crop_args["left"][2]: crop_args["right"][2],
    ]
    return obs_voxel_mt


def get_mt_voxel_classdict(unique_nodes, ignore_nodes):
    mt_voxel_classdict = {
        "node_classes": [],
    }
    class_id = 0
    for node in unique_nodes:
        if node[0] in ignore_nodes:
            continue
        # Convert NumPy types to Python native types
        node_tuple = tuple(int(x) if hasattr(x, "item") else x for x in node)
        mt_voxel_classdict["node_classes"].append(node_tuple)
        class_id += 1

    return mt_voxel_classdict


def preprocess_mt_data(
        metadata, output_dir, max_workers=1, desc="Preprocessing MTdata..."
) -> Tuple[pd.DataFrame, dict]:
    # load dataset params
    with open(output_dir / "dataset_params.json") as f:
        dataset_params = json.load(f)
    voxel_info = dataset_params["voxel_info"]
    crop_args, ignore_nodes, output_vox_grid = get_preprocessing_params(
        voxel_info, args.save_as_sparse
    )

    # load metadata
    metadata = metadata.to_dict("records")

    # processing objects
    with (
        ThreadPoolExecutor(max_workers=max_workers) as executor,
        tqdm(total=len(metadata), desc=desc) as pbar,
    ):

        def worker(metadatum):
            local_path = metadatum["local_path"]
            sha256 = metadatum["sha256"]
            valid, unique_node = False, np.array([])
            triggered_conditions = None
            try:
                data = get_data(local_path)
                update_data = False
                if args.compute_cam_pos_local and "cam_pos_local" not in data:
                    cam_pos_local = compute_cam_pose_local(torch.from_numpy(data['cam_pos']).unsqueeze(0),
                                                           torch.from_numpy(data['obs_voxel_center']).unsqueeze(0),
                                                           dataset_params)
                    data["cam_pos_local"] = cam_pos_local.squeeze(0).numpy()
                    update_data = True
                if args.apply_fix_cam_dir_extrinsics:
                    cam_dir = data["cam_dir"]
                    cam_dir, fix_applied = apply_cam_dir_fix(cam_dir)
                    if fix_applied:
                        extrinsics_local = compute_extrinsics(
                            torch.from_numpy(data['cam_pos_local']).to(torch.float64),
                            torch.from_numpy(cam_dir).to(torch.float64)
                        ).to(torch.float32).numpy()
                        extrinsics_global = compute_extrinsics(
                            torch.from_numpy(data['cam_pos']).to(torch.float64),
                            torch.from_numpy(cam_dir).to(torch.float64)
                        ).to(torch.float32).numpy()
                        data["cam_dir"] = cam_dir
                        data["extrinsics_local"] = extrinsics_local
                        data["extrinsics_global"] = extrinsics_global
                        update_data = True

                if update_data:
                    np.savez_compressed(output_dir / local_path / "data.npz", **data)
                valid, triggered_conditions = check_data_validity(data)
                if valid and not args.check_conditions_only:
                    voxel_mt = get_cropped_voxel_mt(
                        data["obs_voxel_mt"], crop_args, output_vox_grid
                    )
                    unique_node = check_unique(voxel_mt)
                pbar.update()
            except Exception as e:
                logger.info(
                    f"Error processing object {sha256}: {e}"
                )  # TODO: should we have a separate flag for errors?
                pbar.update()

            return sha256, valid, unique_node, triggered_conditions

        futures = executor.map(worker, metadata)
        executor.shutdown(wait=True)

    sha256s, valids, unique_nodes, triggered_conditions_list = zip(*futures)
    valids = np.array(valids)
    level_quality_scores = np.where(valids, 10, 0)
    if not args.check_conditions_only:
        unique_nodes = [x for i, x in enumerate(unique_nodes) if valids[i]]
        unique_nodes = check_unique(np.concatenate(unique_nodes))

        mt_voxel_classdict = get_mt_voxel_classdict(unique_nodes, ignore_nodes)
    else:
        mt_voxel_classdict = None

    records = []
    for sha256, level_score, conditions in zip(
            sha256s, level_quality_scores, triggered_conditions_list
    ):
        records.append(
            {
                "sha256": sha256,
                "preprocessed": True,
                "level_quality_score": level_score,
                "level_validity_conditions": conditions,
            }
        )

    return pd.DataFrame.from_records(records), mt_voxel_classdict


def apply_cam_dir_fix(cam_dir, eps=1e-8, max_yaw_delta_deg=10):
    def compute_dyaw_from_cam_dir(cam_dir):
        f = cam_dir / cam_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        yaw = torch.rad2deg(torch.atan2(f[..., 1], f[..., 0]))  # depends on your axis convention
        dyaw = angular_diff_deg(yaw[..., 1:], yaw[..., :-1])
        return dyaw

    cam_dir = torch.from_numpy(cam_dir)
    dyaw = compute_dyaw_from_cam_dir(cam_dir)

    fix_applied = False
    if dyaw.abs().max() > max_yaw_delta_deg:
        vals = cam_dir[(cam_dir[..., 2] > 0.1 - eps) | (cam_dir[..., 2] < -0.1 + eps)]
        vals[..., 0] = -vals[..., 0]
        vals[..., 1] = -vals[..., 1]
        cam_dir[(cam_dir[..., 2] > 0.1 - eps) | (cam_dir[..., 2] < -0.1 + eps)] = vals
        dyaw = compute_dyaw_from_cam_dir(cam_dir)
        assert dyaw.abs().max() < max_yaw_delta_deg, f"cam_dir/yaw anomalies still present after applying cam_dir fix"
        fix_applied = True

    return cam_dir.numpy(), fix_applied


def check_unique(voxel_mt):
    """Set-based implementation"""
    nodes = voxel_mt[..., 0].ravel()
    params = voxel_mt[..., 1].ravel()

    # Use sets for faster unique finding
    unique_node_param_pairs = set(zip(nodes, params))

    # Convert back to numpy array
    unique_node_param_pairs = np.asarray(list(unique_node_param_pairs))
    return unique_node_param_pairs


def get_preprocessing_params(voxel_info, remove_empty_nodes=False):
    vox_grid_dim = voxel_info["dims"]
    assert len(vox_grid_dim) == 3, "voxels must have the XYZ shape"
    assert vox_grid_dim[0] == vox_grid_dim[1] == vox_grid_dim[2], "Only cubic grids are supported"
    grid_size = vox_grid_dim[0]
    vox_preprocessing = voxel_info["active_voxel_preprocessing"]
    if remove_empty_nodes and "empty_space_node_ids" in vox_preprocessing:
        ignore_nodes = vox_preprocessing["empty_space_node_ids"]
    else:
        ignore_nodes = []
    assert (
            vox_preprocessing["crop"]
            and not vox_preprocessing["pad"]
            and not vox_preprocessing["scale"]
    ), "Voxel preprocessing must crop and not resize"
    crop_args = vox_preprocessing["crop"]
    return crop_args, ignore_nodes, grid_size


def apply_voxel_classes(metadata, output_dir, max_workers=1, compute_masks=False, desc="Voxelizing",
                        time_batch_size=64, gpu_max_concurrency=1) -> pd.DataFrame:
    # load metadata
    metadata = metadata.to_dict("records")

    # Bound how many worker threads run the GPU-heavy importance-mask step at once.
    gpu_semaphore = threading.Semaphore(gpu_max_concurrency) if compute_masks else None

    # Pre-warm einops' lazy per-type backend registration single-threaded; resolving it
    # concurrently inside worker threads can race and raise "Tensor type unknown to einops".
    rearrange(np.zeros((1, 1), dtype=np.int16), "a b -> b a")
    rearrange(torch.zeros(1, 1), "a b -> b a")

    # load dataset params
    with open(output_dir / "dataset_params.json") as f:
        dataset_params = json.load(f)
    voxel_info = dataset_params["voxel_info"]

    crop_args, ignore_nodes, output_vox_grid = get_preprocessing_params(voxel_info)

    with open(output_dir / "mt_voxel_classdict.json") as f:
        mt_voxel_classdict = json.load(f)
    node2class = {tuple(v): int(k) for k, v in mt_voxel_classdict["node_classes"].items()}

    # processing objects
    (output_dir / "voxel_classes").mkdir(exist_ok=True)
    # if args.save_as_sparse:
    #     (output_dir / "voxel_coords").mkdir(exist_ok=True)
    with (
        ThreadPoolExecutor(max_workers=max_workers) as executor,
        tqdm(total=len(metadata), desc=desc) as pbar,
    ):

        def worker(metadatum):
            sha256 = metadatum["sha256"]
            voxelized = False
            level_quality_score = metadatum["level_quality_score"]
            try:
                local_path = metadatum["local_path"]
                valid = metadatum["level_quality_score"] == 10
                if valid:
                    with np.load(output_dir / local_path / "data.npz") as f:
                        voxel_mt = f["obs_voxel_mt"]
                    voxel_mt = get_cropped_voxel_mt(voxel_mt, crop_args, output_vox_grid)
                    record = _apply_voxel_classes(
                        voxel_mt, sha256, output_dir, ignore_nodes, output_vox_grid, node2class, compute_masks,
                        time_batch_size=time_batch_size, gpu_semaphore=gpu_semaphore
                    )
                    if record is not None:
                        voxelized = True
                    else:
                        # Contains (node_id, node_param) pairs absent from the classdict -> invalid.
                        level_quality_score = 0
                        logger.warning(
                            f"Level {sha256} contains voxel (node_id, node_param) pairs not present "
                            f"in mt_voxel_classdict.json; marking it invalid."
                        )
                pbar.update()
            except Exception as e:
                logger.error(f"Error processing object {sha256}: {e}")
                pbar.update()
            finally:
                return sha256, voxelized, level_quality_score

        futures = list(executor.map(worker, metadata))
        executor.shutdown(wait=True)

    if futures:
        sha256s, voxelizeds, scores = zip(*futures)
    else:
        sha256s, voxelizeds, scores = (), (), ()
    return pd.DataFrame(
        {"sha256": list(sha256s), "voxelized": list(voxelizeds), "level_quality_score": list(scores)}
    )


def _apply_voxel_classes(mt_voxel, sha256, output_dir, ignore_nodes, output_vox_grid, node2class, compute_masks,
                         time_batch_size=64, gpu_semaphore=None):
    mt_voxel_shape = mt_voxel.shape
    assert len(mt_voxel_shape) == 5, "MT voxel data must have 4 dimensions (T, X, Y, Z, C)"
    assert mt_voxel_shape[-1] == 2, "MT voxel data must have 2 channels"
    mt_nodes = mt_voxel[..., 0]  # Only take the first channel of the MT voxel data

    # if args.save_as_sparse:
    #     node_mask = np.ones(mt_nodes.shape, dtype=bool)
    #     for node in ignore_nodes:
    #         node_mask &= mt_nodes != node
    #     mt_voxel = mt_voxel[node_mask]
    #     voxel_grid = np.argwhere(node_mask)  # L x 4
    #     vertices = (voxel_grid + 0.5) / output_vox_grid - 0.5
    #     vertices[:, 0] = voxel_grid[:, 0]
    #     np.savez_compressed(os.path.join(output_dir, "voxel_coords", f"{sha256}.npz"), vertices)
    # else:
    #     # flatten the voxel array except for the last dimension
    #     mt_voxel = rearrange(mt_voxel, "T X Y Z C -> (T X Y Z) C")
    # Plain numpy reshape (equivalent to the einops flatten) -- einops' lazy backend
    # resolution can race across worker threads and raise "Tensor type unknown to einops".
    mt_voxel = mt_voxel.reshape(-1, mt_voxel.shape[-1])

    # voxel_classes shape = L
    node_classes, valid = compute_voxel_classes(mt_voxel, node2class)
    if not valid:
        # Level contains (node_id, node_param) pairs absent from mt_voxel_classdict.json;
        # voxelizing it would corrupt the class encoding, so skip it and flag it invalid.
        return None
    voxel_classes = node_classes.reshape(mt_voxel_shape[:-1])
    voxel_classes = {"node_classes": voxel_classes}
    if compute_masks:
        # Serialize the GPU-heavy importance-mask computation so concurrent worker
        # threads don't pile their allocations onto the GPU at the same time.
        gpu_ctx = gpu_semaphore if gpu_semaphore is not None else contextlib.nullcontext()
        with gpu_ctx:
            importance_masks = compute_importance_masks(
                voxel_classes["node_classes"], translation_radius=2, time_batch_size=time_batch_size
            )
        voxel_classes["importance_masks"] = importance_masks
    np.savez_compressed(os.path.join(output_dir, "voxel_classes", f"{sha256}.npz"), **voxel_classes)

    return {
        "sha256": sha256,
        "voxelized": True,
    }


def compute_importance_masks(voxel_classes, translation_radius=2, time_batch_size=64):
    """Compute importance masks based on translation invariant dynamic changes between frames.

    For each consecutive frame pair, find the integer spatial translation (within
    +/- translation_radius on each axis) that minimises the number of changed voxels, and
    record the coordinates that still differ under that best translation.

    Each translation is applied with torch.roll -- the gather-based formulation
    (arange(X) - dx) % X is exactly a cyclic shift -- so we iterate over the B translations
    instead of materialising a [B, T, X, Y, Z] index grid. Peak memory therefore scales with
    T * X * Y * Z (one rolled frame stack at a time) rather than B * T * X * Y * Z, which keeps
    larger voxel grids (e.g. 64^3) within GPU memory. The frame-pair (time) axis is still
    processed in chunks of `time_batch_size` as an extra memory bound; results are
    bit-for-bit identical regardless of chunk size.
    """
    assert voxel_classes.ndim == 4, (
        f"voxel_classes must have shape (T, X, Y, Z), got {voxel_classes.shape}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shifts = [i for i in range(-translation_radius, translation_radius + 1)]
    translations = list(itertools.product(shifts, repeat=3))

    T, X, Y, Z = voxel_classes.shape
    R = translation_radius
    voxel_classes = torch.from_numpy(voxel_classes).to(device=device)  # (T, X, Y, Z)
    offset = torch.tensor([1, R, R, R]).to(dtype=torch.int16, device=device)

    chunk_coords = []
    for start in range(0, T - 1, time_batch_size):
        end = min(start + time_batch_size, T - 1)

        vcls_current_ts = voxel_classes[start:end]  # (tb, X, Y, Z)
        vcls_next_ts = voxel_classes[start + 1 : end + 1, R : X - R, R : Y - R, R : Z - R]

        # Per-timestep running best: fewest changed voxels over translations tried so far.
        # Iterating translations in itertools.product order with a strict `<` update keeps
        # the first translation achieving the minimum -- matching torch.argmin's tie-break.
        best_count = None  # (tb,)
        best_mask = None   # (tb, x', y', z')
        for dx, dy, dz in translations:
            rolled = torch.roll(vcls_current_ts, shifts=(dx, dy, dz), dims=(1, 2, 3))
            rolled = rolled[:, R : X - R, R : Y - R, R : Z - R]
            mask = rolled != vcls_next_ts  # (tb, x', y', z')
            count = mask.sum(dim=[1, 2, 3])  # (tb,)
            if best_count is None:
                best_count = count
                best_mask = mask
            else:
                improve = count < best_count
                best_count = torch.where(improve, count, best_count)
                best_mask[improve] = mask[improve]

        coords = torch.argwhere(best_mask).to(torch.int16)
        coords += offset  # offset for time index (+1) and spatial cropping (+R)
        coords[:, 0] += start  # shift local frame-pair index to its global position
        chunk_coords.append(coords.cpu())

    return torch.cat(chunk_coords, dim=0).numpy().astype(np.int16)


def compute_voxel_classes(mt_voxel, node2class):
    """Map (node_id, node_param) voxels to class ids via a lookup table.

    Returns (node_classes, valid). `valid` is False if mt_voxel contains any
    (node_id, node_param) pair absent from node2class -- including pairs whose
    indices fall outside the lookup table. Undefined pairs map to class id -1.
    """

    # Create lookup tables if memory usage is reasonable
    max_node = max(k[0] for k in node2class.keys())
    max_param = max(k[1] for k in node2class.keys())

    node_lookup = np.full((max_node + 1, max_param + 1), -1, dtype=np.int16)
    for (node, param), class_id in node2class.items():
        node_lookup[node, param] = class_id

    nodes = mt_voxel[:, 0]
    params = mt_voxel[:, 1]
    # Pairs whose node/param index exceeds the lookup table are undefined by
    # definition; clip the gather index to a valid slot and flag them as misses
    # so we get -1 instead of an out-of-bounds error. Both these and in-bounds
    # misses (already -1 in the table) are detected by the single np.any below.
    in_bounds = (nodes <= max_node) & (params <= max_param)
    node_classes = node_lookup[np.where(in_bounds, nodes, 0), np.where(in_bounds, params, 0)]
    node_classes = np.where(in_bounds, node_classes, np.int16(-1)).astype(np.int16)

    valid = not bool(np.any(node_classes == -1))
    return node_classes, valid

def compute_cropped_voxels(metadata: pd.DataFrame, output_dir: Path, max_workers: int = 1, desc: str = "Cropping voxels"):
    """
    Load ALREADY COMPUTED voxel classes from output_dir/voxel_classes/{sha256}.npz,
    compute cropped voxel via _compute_cropped_voxels(), then update the same .npz
    to include entry 'cropped_voxel' while keeping all existing entries untouched.
    """
    voxel_dir = output_dir / "voxel_classes"
    if not voxel_dir.exists():
        raise ValueError(f"{voxel_dir} not found; run voxelization stage first.")

    # worklist
    records = metadata.to_dict("records")

    def _update_npz_in_place(npz_path: Path, new_key: str, new_value: np.ndarray) -> None:
        # Load existing entries
        with np.load(npz_path) as f:
            data = {k: f[k].copy() for k in f.files}

        # Update / add only the requested key
        data[new_key] = new_value

        # IMPORTANT: temp file must end with ".npz" or savez will add it implicitly
        tmp_path = npz_path.with_name(npz_path.name + ".tmp.npz")

        np.savez_compressed(tmp_path, **data)
        os.replace(tmp_path, npz_path)

    def worker(metadatum):
        sha256 = metadatum["sha256"]

        # Optional gating consistent with rest of file:
        # - Only operate on valid voxelized objects (if those columns exist)
        if "preprocessed" in metadatum and metadatum["preprocessed"] is not True:
            return
        if "voxelized" in metadatum and metadatum["voxelized"] is not True:
            return

        npz_path = voxel_dir / f"{sha256}.npz"
        if not npz_path.exists():
            logger.warning(f"Missing voxel_classes file for {sha256}: {npz_path}")
            return

        # If not overwriting and already present, skip
        if not args.overwrite:
            try:
                with np.load(npz_path) as f:
                    if "cropped_voxel" in f.files:
                        return
            except Exception as e:
                logger.error(f"Error reading {npz_path}: {e}")
                return

        try:
            with np.load(npz_path) as f:
                if "node_classes" not in f.files:
                    logger.error(f"'node_classes' missing in {npz_path}")
                    return
                voxel_classes = f["node_classes"]  # expected shape (T, X, Y, Z)

            cropped_voxel = _compute_cropped_voxels(
                voxel_classes,
                cropped_voxel_shape=args.cropped_voxel_shape
            ).astype(np.int16, copy=False)

            _update_npz_in_place(npz_path, "cropped_voxel", cropped_voxel)

        except Exception as e:
            logger.error(f"Error processing {sha256}: {e}")

    # parallel pattern consistent with rest of file
    try:
        with (
            ThreadPoolExecutor(max_workers=max_workers) as executor,
            tqdm(total=len(records), desc=desc) as pbar,
        ):
            def wrapped_worker(m):
                worker(m)
                pbar.update()

            executor.map(wrapped_worker, records)
            executor.shutdown(wait=True)
    except Exception as e:
        logger.error(f"Error happened during cropped voxel processing: {e}")

def _compute_cropped_voxels(voxel_classes, cropped_voxel_shape=(4, 4, 4)):
    vx_shape = np.array(voxel_classes.shape[1:])
    crop = (vx_shape - np.array(cropped_voxel_shape)) // 2
    vx_cropped = voxel_classes[
        ..., crop[0]:vx_shape[0] - crop[0], crop[1]:vx_shape[1] - crop[1], crop[2]:vx_shape[2] - crop[2]]
    return vx_cropped


@dataclass
class Args:
    """Command line arguments for the program. This script should be ran in two stages:
    1. Stage 1: Pass --create_voxel_class_dict to crop the voxels, filter out bad levels and generate the data to create the voxel dictionary
    (mt_voxel_classdict.json).
    2. Stage 2: Run without the flag to assign the voxel classes to the voxel data."""

    output_dir: str
    """Directory to save the metadata"""
    rank: int = 0
    """Process rank"""
    world_size: int = 1
    """Total number of processes"""
    max_workers: Optional[int] = None
    """Maximum number of worker processes"""
    overwrite: bool = False
    """Whether to overwrite files when they already exist"""
    save_as_sparse: bool = False
    """Whether to encode the voxel data as sparse arrays (voxel_coords, voxel_classes). Empty nodes will be removed."""
    debug: bool = False
    """Whether to run in debug mode (no multiprocessing)"""
    create_voxel_class_dict: bool = False
    """Filters out bad levels and generates the voxel class dictionary (mt_voxel_classdict.json) and stops.
    The --check_* flags are relevant only for this mode."""
    check_conditions_only: bool = False
    """Only checks the validity conditions and does not generate the voxel class dictionary."""
    check_early_termination: bool = True
    """Whether to check for early termination"""
    check_moving_player: bool = False
    """Whether to check for moving player"""
    check_corrupted_data: bool = True
    """Whether to check for corrupted data"""
    check_framerate_drop: bool = True
    """Whether to check for framerate drop"""
    check_frozen_voxel: bool = True
    """Whether to check for freezes in voxel observations"""
    compute_cam_pos_local: bool = True
    """Whether to compute camera positions in local frame"""
    compute_cropped_voxels: bool = False
    """Whether to compute cropped voxels in local frame"""
    cropped_voxel_shape: Tuple[int, int, int] = (4, 4, 4)
    """Shape of the cropped voxels (from player position voxel)"""
    apply_fix_cam_dir_extrinsics: bool = False
    """Whether to apply known bug-fix to cam_dir and extrinsics"""
    voxel_importance_time_batch_size: int = 64
    """Number of frame-pairs processed per chunk in the GPU importance-mask computation.
    Lower this if voxelization runs out of GPU memory; raise it for more throughput."""
    voxel_gpu_max_concurrency: int = 1
    """Maximum number of worker threads allowed to run the GPU importance-mask step
    concurrently. Bounds peak GPU memory when --max_workers is large."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.debug:
        from utils import MockThreadPoolExecutor as ThreadPoolExecutor

    # get file list
    output_dir = Path(args.output_dir)
    if not (output_dir / "metadata.csv").exists():
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(output_dir / "metadata.csv")

    if args.compute_cropped_voxels:
        metadata = metadata[metadata["preprocessed"] == True]
        compute_cropped_voxels(metadata, output_dir, max_workers=args.max_workers)
        sys.exit(0)

    assert not (args.create_voxel_class_dict and args.check_conditions_only), "Either pass --create_voxel_class_dict or --check_conditions_only, not both"

    stage = 1 if args.create_voxel_class_dict or args.check_conditions_only else 2
    # print the arguments to terminal using loguru
    logger.info(f"Running process_mt_data.py Stage {stage}.")
    logger.info(f"Arguments: {args}")

    if args.check_conditions_only:
        if not args.overwrite:  # we will only check already validated levels
            if "preprocessed" in metadata.columns:
                metadata = metadata[metadata["preprocessed"] == True]
        else:
            if "preprocessed" in metadata.columns:
                n_processed = len(metadata[metadata["preprocessed"] == True])
                logger.info(f"Found {n_processed} already preprocessed objects. Will overwrite.")
    elif args.create_voxel_class_dict:
        if not args.overwrite:
            if "preprocessed" in metadata.columns:
                metadata = metadata[metadata["preprocessed"] == False]
        elif "preprocessed" in metadata.columns:
            n_processed = len(metadata[metadata["preprocessed"] == True])
            logger.info(f"Found {n_processed} already preprocessed objects. Will overwrite.")

    # runs stage 1 of script
    if args.create_voxel_class_dict or args.check_conditions_only:
        start = len(metadata) * args.rank // args.world_size
        end = len(metadata) * (args.rank + 1) // args.world_size
        metadata = metadata[start:end]
        logger.info(f"Running {len(metadata)} objects...")

        records, mt_voxel_classdict = preprocess_mt_data(
            metadata, output_dir, max_workers=args.max_workers
        )
        records.to_csv(output_dir / f"preprocessed_{args.rank}.csv", index=False)
        if mt_voxel_classdict is not None:
            with open(output_dir / f"mt_voxel_classdict_{args.rank}.json", "w") as f:
                json.dump(mt_voxel_classdict, f)

    # runs stage 2 of script
    else:
        # filter out objects that are already processed
        metadata = metadata[metadata["preprocessed"] == True]
        if not args.overwrite:
            if "voxelized" in metadata.columns:
                metadata = metadata[metadata["voxelized"] == False]
        elif "voxelized" in metadata.columns:
            n_voxelized = len(metadata[metadata["voxelized"] == True])
            logger.info(f"Found {n_voxelized} already voxelized objects. Will overwrite.")

        start = len(metadata) * args.rank // args.world_size
        end = len(metadata) * (args.rank + 1) // args.world_size
        metadata = metadata[start:end]
        logger.info(f"Running {len(metadata)} objects...")

        # process objects
        records = apply_voxel_classes(
            metadata, output_dir, max_workers=args.max_workers, compute_masks=True,
            time_batch_size=args.voxel_importance_time_batch_size,
            gpu_max_concurrency=args.voxel_gpu_max_concurrency,
        )
        records.to_csv(output_dir / f"voxelized_{args.rank}.csv", index=False)
