# Coordinate frames and voxel identity

This note documents the coordinate systems in the raw `data.npz` produced by
`dataset_toolkits/generate_raw_data.py`, and how to derive a stable per-voxel
identity from them.

## TL;DR

- The voxel observation is a dense, **agent-centered** grid. A voxel is addressed
  only by its local grid index `[i, j, k]`; the grid re-centers on the agent every
  timestep, so a fixed index refers to a *different* physical voxel from frame to
  frame. **There is no per-voxel unique id stored anywhere.**
- A voxel's true identity is its **absolute world coordinate**, which is recoverable
  because `data.npz` stores `obs_voxel_center` (the world coordinate of the grid's
  center cell) and `dataset_params.json` stores the grid `origin_idx`.
- One Minetest node = one world unit, so world coordinates are in voxel units and the
  mapping is exact integers:

  ```
  world_id       = obs_voxel_center[t] + (local_idx - origin_idx)     # local -> id
  local_idx(t)   = world_id - obs_voxel_center[t] + origin_idx        # id -> local at t
  ```

- Helpers implementing this live in `utils/voxel_id.py`.

## The three frames in data.npz

Everything spatial is stored in the **ENU** axis convention
(`dataset_params['voxel_info']['xyz'] == "ENU"`). Minetest uses a different internal
axis order (NUE); the `enu_to_nue` / `NueToEnuVoxelObs` conversions are applied on the
way *in*, so by the time data reaches `data.npz` it is consistently ENU.

### World frame (absolute Minetest, ENU, units = voxels)

| Field | Shape | Meaning |
|---|---|---|
| `obs_voxel_center` | (T, 3) | World coordinate of the grid's center cell (= agent). **The anchor.** |
| `player_pos` | (T, 3) | Player absolute world position |
| `player_vel` | (T, 3) | Velocity along world axes (units/sec) |
| `cam_pos` | (T, 3) | Camera absolute world position |
| `cam_dir` | (T, 3) | Camera forward direction (unit vector; orientation, not a position) |
| `extrinsics_global` | (T, 4, 4) | World->camera transform, world frame |

### Grid frame

Two sub-forms:

| Field | Shape | Meaning |
|---|---|---|
| `obs_voxel_mt` | (T, D, D, D, 2) | The cube. **Indexed** by grid `[i, j, k]`; each cell's **value** is `(node_id, node_param)` -- material properties, not coordinates. |
| `cam_pos_local`* | (T, 3) | Camera position inside the grid, divided by grid size (~[0, 1]) |
| `extrinsics_local` | (T, 4, 4) | World->camera transform built in the grid-normalized frame |

Integer grid indices `[i, j, k]` run over `[0, D)` per axis. `cam_pos_local` and
`extrinsics_local` use a *normalized* grid frame (position divided by grid size).

\* `cam_pos_local` is added during preprocessing (`process_mt_data.py`, on by default),
not in the very first raw dump.

### Frame-independent (angles / intrinsics / non-spatial)

| Field | Shape | Meaning |
|---|---|---|
| `player_pitch` | (T,) | Look up/down angle (deg) |
| `player_yaw` | (T,) | Look left/right angle (deg) |
| `fov_x`, `fov_y` | (T,) | Field-of-view angles |
| `intrinsics` | (T, 3, 3) | Camera intrinsic matrix (image space) |
| `timestep_craftium` | (T,) | Step counter |
| `dt_minetest` | (T,) | Seconds elapsed per step |
| `action` | (T, ...) | Action taken (multihot) |
| `reward` | (T,) | Reward signal (if present) |
| `termination_flag`, `truncation_flag` | (T,) | Episode-end booleans |

## Why `obs_voxel_center` is a world coordinate

Two independent confirmations from the codebase:

1. `dataset_params['minetest_voxel_info']` declares `"origin": "agent"` and
   `"xyz": "ENU"` -- the grid origin is the agent, in world axes.
2. `compute_cam_pose_local` (in both `utils/camera_util.py` and
   `dataset_toolkits/generate_raw_data.py`) computes
   `cam_pos_local = (cam_pos - obs_voxel_center) / voxel_grid_size`. It subtracts
   `obs_voxel_center` from the world-space `cam_pos` (the same `cam_pos` used to build
   `extrinsics_global`). The subtraction is only meaningful if both are in the same
   frame, so `obs_voxel_center` is in world coordinates.

## The `origin_idx` to use

`local_idx = 0..D-1`, and the cell that corresponds to `obs_voxel_center` is
`origin_idx`, read from `dataset_params.json`:

- Raw `obs_voxel_mt` grid: `minetest_voxel_info.origin_idx` with `minetest_voxel_info.dims`.
- Processed `voxel_classes` grid: `voxel_info.origin_idx` with `voxel_info.dims`.

The pipeline uses a right-only crop (`crop.left == 0`), so the origin cell keeps the
same index in both grids; only the valid index range differs.

## Tracking a voxel (including leaving and re-entering the window)

Because `world_id` does not depend on the window, tracking across a movement -- or
across the voxel leaving and later re-entering the grid -- is just evaluating the
inverse map each frame and checking whether the index is in range:

```python
from utils.voxel_id import load_grid_params, track_voxel
import numpy as np

data = np.load("path/to/data.npz")
origin_idx, dims = load_grid_params("path/to/dataset_root", grid="raw")  # obs_voxel_mt is the raw grid

world_id, indices, visible = track_voxel(
    voxel_centers=data["obs_voxel_center"],
    origin_idx=origin_idx,
    dims=dims,
    t0=0,
    index0=(24, 24, 24),
)
# world_id : (3,)   permanent id of the voxel
# indices  : (T, 3) its local index each frame (may be out of range when off-window)
# visible  : (T,)   True where it is inside the grid; re-entry shows up as True again
```

The window displacement between two frames is `Delta = obs_voxel_center[t1] -
obs_voxel_center[t0]`, and a voxel's index moves by `-Delta` (`new_index =
old_index - Delta`).

## Caveats

- **Location identity, not object identity.** `world_id` identifies "the voxel at this
  world spot." If a block is dug and another placed at the same coordinate, the id is
  reused; compare the stored `(node_id, node_param)` at that id over time to detect
  content changes. Following *the same block* through dig-and-replace requires emitting
  a real per-voxel id from Craftium (an engine-side change), which is not derivable
  from `obs_voxel_center`.
- **Exact on recorded data; approximate under generation.** During model rollouts there
  is no `obs_voxel_center`; it must be reconstructed from the predicted camera, and the
  model does not guarantee content consistency when a voxel re-enters view.
