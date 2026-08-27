"""Voxel identity and tracking helpers.

The Craftium -> PERSIST pipeline stores voxels as a dense, agent-centered grid
(`obs_voxel_mt` / `voxel_classes`). A voxel is addressed only by its *local grid
index* ``[i, j, k]`` within that cube, and the cube re-centers on the agent every
timestep, so the same index refers to a different physical voxel from frame to
frame. There is no per-voxel unique id anywhere in the data.

A stable identity does exist implicitly, though: the voxel's absolute world
coordinate. The dataset stores the anchor needed to recover it -- ``obs_voxel_center``
(the world coordinate of the grid's center cell, per timestep) -- together with the
grid's ``origin_idx`` (from ``dataset_params.json``). Because one Minetest node is one
world unit, the world coordinate is:

    world_id = obs_voxel_center[t] + (local_idx - origin_idx)

which is a permanent, unique key for the voxel at that world location. Inverting it
gives the voxel's local index in any other frame:

    local_idx(t) = world_id - obs_voxel_center[t] + origin_idx

Tracking a voxel across a movement (or across leaving and re-entering the window) is
just evaluating that inverse each frame and checking whether the index is in range.

Notes / caveats:
  * This is *location* identity, not *object* identity: if a block is dug and another
    placed at the same coordinate, ``world_id`` is reused. Compare the stored
    ``(node_id, node_param)`` at that id over time to detect content changes.
  * On recorded (ground-truth) data this is exact. Under model generation there is no
    ``obs_voxel_center``; you must reconstruct it from the predicted camera, and content
    consistency on re-entry is not guaranteed by the model.
  * All stored spatial data is in the ENU frame (see ``dataset_params['voxel_info']``).

See ``docs/coordinate_frames.md`` for the full data.npz frame breakdown.
"""

from __future__ import annotations

import json
import os

import numpy as np

__all__ = [
    "load_grid_params",
    "local_to_world_id",
    "world_id_to_local",
    "in_grid",
    "window_shift",
    "track_voxel",
]


def load_grid_params(dataset_root, grid="processed"):
    """Read ``origin_idx`` and grid ``dims`` from a dataset's ``dataset_params.json``.

    Args:
        dataset_root: path to the dataset directory containing ``dataset_params.json``.
        grid: which grid the local indices refer to.
            ``"raw"``       -> the uncropped ``obs_voxel_mt`` grid (``minetest_voxel_info``).
            ``"processed"`` -> the cropped ``voxel_classes`` grid (``voxel_info``).

    Returns:
        (origin_idx, dims): int64 arrays of shape (3,). ``origin_idx`` is the grid index
        of the cell that ``obs_voxel_center`` refers to; ``dims`` is the grid size per axis.
    """
    with open(os.path.join(dataset_root, "dataset_params.json")) as f:
        params = json.load(f)
    if grid == "raw":
        info = params["minetest_voxel_info"]
    elif grid == "processed":
        info = params["voxel_info"]
    else:
        raise ValueError(f"grid must be 'raw' or 'processed', got {grid!r}")
    origin_idx = np.asarray(info["origin_idx"], dtype=np.int64)
    dims = np.asarray(info["dims"][:3], dtype=np.int64)
    return origin_idx, dims


def local_to_world_id(local_idx, voxel_center, origin_idx):
    """Map a local grid index to its absolute world-coordinate id.

    Args:
        local_idx: (..., 3) integer grid indices ``[i, j, k]``.
        voxel_center: (..., 3) ``obs_voxel_center`` for the same timestep(s).
        origin_idx: (3,) grid index of the center cell (from ``load_grid_params``).

    Returns:
        (..., 3) int64 world coordinate(s). Stable across timesteps for a fixed voxel.
    """
    local_idx = np.asarray(local_idx)
    voxel_center = np.asarray(voxel_center, dtype=np.float64)
    origin_idx = np.asarray(origin_idx)
    world = voxel_center + (local_idx - origin_idx)
    return np.rint(world).astype(np.int64)


def world_id_to_local(world_id, voxel_center, origin_idx):
    """Map a world-coordinate id back to its local grid index at a given timestep.

    The returned index may be outside the grid (voxel not in the window); use
    :func:`in_grid` to test.

    Args:
        world_id: (..., 3) world coordinate(s) from :func:`local_to_world_id`.
        voxel_center: (..., 3) ``obs_voxel_center`` for the target timestep(s).
        origin_idx: (3,) grid index of the center cell.

    Returns:
        (..., 3) int64 local grid index/indices.
    """
    world_id = np.asarray(world_id, dtype=np.float64)
    voxel_center = np.asarray(voxel_center, dtype=np.float64)
    origin_idx = np.asarray(origin_idx)
    local = world_id - voxel_center + origin_idx
    return np.rint(local).astype(np.int64)


def in_grid(local_idx, dims):
    """Whether local index/indices fall inside a grid of the given dims.

    Args:
        local_idx: (..., 3) integer grid indices.
        dims: (3,) grid size per axis.

    Returns:
        Boolean scalar for a single index, or a (...,) boolean array for a batch.
    """
    local_idx = np.asarray(local_idx)
    dims = np.asarray(dims)
    inside = np.all((local_idx >= 0) & (local_idx < dims), axis=-1)
    return bool(inside) if inside.ndim == 0 else inside


def window_shift(voxel_centers, t0, t1):
    """Integer window displacement ``Delta = C[t1] - C[t0]`` between two frames.

    A voxel's index moves by ``-Delta`` between ``t0`` and ``t1``
    (``new_index = old_index - Delta``).
    """
    voxel_centers = np.asarray(voxel_centers, dtype=np.float64)
    return np.rint(voxel_centers[t1] - voxel_centers[t0]).astype(np.int64)


def track_voxel(voxel_centers, origin_idx, dims, t0, index0):
    """Track one voxel across every timestep of an episode.

    Args:
        voxel_centers: (T, 3) ``obs_voxel_center`` for the whole episode.
        origin_idx: (3,) grid index of the center cell.
        dims: (3,) grid size per axis.
        t0: timestep at which ``index0`` is observed.
        index0: (3,) local grid index of the voxel at ``t0``.

    Returns:
        world_id: (3,) int64 permanent world-coordinate id of the voxel.
        indices: (T, 3) int64 local grid index of the voxel at each timestep
            (may lie outside the grid when off-window).
        visible: (T,) bool, True where the voxel is inside the grid.
    """
    voxel_centers = np.asarray(voxel_centers, dtype=np.float64)
    origin_idx = np.asarray(origin_idx)
    dims = np.asarray(dims)

    world_id = local_to_world_id(index0, voxel_centers[t0], origin_idx)
    # Broadcast the inverse map over all timesteps at once.
    indices = np.rint(world_id[None, :] - voxel_centers + origin_idx).astype(np.int64)
    visible = np.all((indices >= 0) & (indices < dims), axis=-1)
    return world_id, indices, visible


def _selfcheck():
    """Validate the round-trip and tracking math on synthetic data (no dataset needed)."""
    rng = np.random.default_rng(0)
    origin_idx = np.array([24, 24, 24])
    dims = np.array([48, 48, 48])

    # Synthetic agent path: drift the window far enough on the x-axis that a tracked
    # voxel leaves the grid, then bring it back so re-entry is exercised too.
    T = 40
    drift = np.concatenate([np.full(20, 2), np.full(20, -2)])  # +x for 20 steps, then -x
    voxel_centers = np.zeros((T, 3), dtype=np.int64)
    voxel_centers[:] = np.array([100, 100, 100])
    voxel_centers[:, 0] = 100 + np.cumsum(drift)
    voxel_centers[:, 1:] += rng.integers(-1, 2, size=(T, 2))  # mild wander on y,z

    # 1) round-trip: local -> world -> local is the identity, at any timestep.
    for t in range(T):
        idx = rng.integers(0, dims)
        wid = local_to_world_id(idx, voxel_centers[t], origin_idx)
        back = world_id_to_local(wid, voxel_centers[t], origin_idx)
        assert np.array_equal(back, idx), (t, idx, wid, back)

    # 2) shift identity: new_index == old_index - Delta.
    idx0 = np.array([20, 24, 28])
    wid = local_to_world_id(idx0, voxel_centers[0], origin_idx)
    for t in range(1, T):
        delta = window_shift(voxel_centers, 0, t)
        expected = idx0 - delta
        got = world_id_to_local(wid, voxel_centers[t], origin_idx)
        assert np.array_equal(got, expected), (t, expected, got)

    # 3) track_voxel agrees with the per-frame inverse and marks visibility correctly.
    world_id, indices, visible = track_voxel(voxel_centers, origin_idx, dims, t0=0, index0=idx0)
    assert np.array_equal(world_id, wid)
    for t in range(T):
        assert np.array_equal(indices[t], world_id_to_local(wid, voxel_centers[t], origin_idx))
        assert bool(visible[t]) == in_grid(indices[t], dims)

    # 4) the id is constant wherever the voxel is visible.
    vis_ids = local_to_world_id(indices[visible], voxel_centers[visible], origin_idx)
    assert np.all(vis_ids == world_id[None, :])

    print(f"voxel_id self-check passed: T={T}, visible {int(visible.sum())}/{T} frames, world_id={world_id.tolist()}")


if __name__ == "__main__":
    _selfcheck()
