# All information about data

## Data folder structure

Best viewed on a wide screen. Files or folders labeled with `[POST]` are generated during preprocessing of the raw data.

```
opendynamic10kL/
├── level_seeds.txt                         # all level seeds in the dataset
├── dataset_params.json                     # params for generating the dataset
├── raw/                                    # raw data folder
|   └── OpenWorldDataset-v0/
|       ├── 1000095647/                     # a specific level (seed as folder name)
|       |   ├── data.npz                    # data file: see next section for detailed descriptions
|       |   ├── data_dynamic.npz            # dynamic-agent (mobs/animals) data, if collected
|       |   ├── rgb.mp4                     # obs video
|       |   ├── sha256.txt                  # [POST] hash of the raw data
|       |   └── level_metadata.json         # metadata for this level
|       └── ...
├── metadata.csv                            # [POST] Record the preprocessing information of each data
├── mt_voxel_classdict.json                 # [POST] defines mapping node class -> (node_id, node_param) of Minetest raw voxel data
├── preprocessing_statistics.txt            # [POST] Summary of preprocessing stats
├── statistics.txt                          # [POST] Data stats
├── voxel_classes/                          # [POST] voxel data re-encoded as classes defined in mt_voxel_classdict.json
|   ├── <hash>.npz                          # [POST] shape (T, X, Y, Z)
|   └── ...
├── voxel_latents
|   ├── <hash>.safetensors                  # (or <hash>.npz if encoded with --save_as npz)
|   |   ├── Key: mean                       # [POST] shape (T, C, x, y, z)
|   |   ├── Key: timesteps                  # [POST] length of this datum
|   |   └── Key: encoding_type              # [POST] deterministic or stochastic
|   └── ...
├── pixel_latents
|   ├── <hash>.safetensors                  # (or <hash>.npz if encoded with --save_as npz)
|   |   ├── Key: mean                       # [POST] shape (T, h, w, C)
|   |   ├── Key: timesteps                  # [POST] length of this datum
|   |   └── Key: encoding_type              # [POST] deterministic or stochastic
|   └── ...
└── merged_records/                         # [POST] For logging purposes only. merged records aggregated from past `build_metadata.py` runs
```

## mt_voxel_classdict.json dictionary: reuse vs. regenerate

`mt_voxel_classdict.json` maps each `(node_id, node_param)` pair of the raw
Minetest data to a unique voxel class id. It defines the class encoding used by `voxel_classes/` and, downstream, 
the input and output spaces of the 3D-VAE. When building a dataset (`build_dataset.sh`) you choose one of two modes:

* **(a) Reuse a pre-specified dictionary** (`--default-classdict <path>`): the file is copied into
  the dataset and reused as-is. Class ids stay identical to the data/checkpoints that dictionary
  came from, so **previously generated data and pre-trained models remain compatible**. Any level
  containing `(node_id, node_param)` pairs absent from the dictionary is **filtered out (marked
  invalid, `level_quality_score = 0`)** during voxelization. We provide the dictionary used for our pre-trained 
  models in `dataset_toolkits/default_mt_voxel_classdict.json`.

* **(b) Generate a fresh dictionary** (default, no flag): a new `mt_voxel_classdict.json` is built
  from the nodes found in this dataset's valid levels. This maximises node coverage for the dataset,
  but the resulting class ids differ from any previous dictionary, **breaking compatibility with
  previously generated data and any model trained against the old mapping**.

Use (a) to extend or rebuild data for existing checkpoints; use (b) only when starting a new model
family where the class encoding is allowed to change.

## Raw data content
The `data.npz` file for a specific level contains
* `timestep_craftium`: shape `(T,)`
* `dt_minetest`: shape `(T,)`
* `player_pos`: shape `(T, 3)`
* `player_vel`: shape `(T, 3)`
* `player_pitch`: shape `(T,)`
* `player_yaw`: shape `(T,)`
* `cam_pos`: shape `(T, 3)`
* `cam_dir`: shape `(T, 3)`
* `fov_x`: shape `(T,)`
* `fov_y`: shape `(T,)`
* `obs_voxel_center`: shape `(T, 3)`
* `obs_voxel_mt`: shape `(T, D, D, D, 2) [contains (node_id, node_param) pairs of each voxel]`
* `intrinsics`: shape `(T, 3, 3)`
* `extrinsics_local`: shape `(T, 4, 4)`
* `extrinsics_global`: shape `(T, 4, 4)`
* `action`: shape `(T,)`
* `reward`: shape `(T,)`
* `termination_flag`: shape `(T,)`
* `truncation_flag`: shape `(T,)`

## Dynamic-agent data content (`data_dynamic.npz`)

When `generate_raw_data.py` is run with `--collect_dynamic_data` (default on),
Craftium spawns `--num_dynamic_agents` dynamic agents (mobs/animals; currently
`--dynamic_agent_entity mobs_mc:sheep`) around the player and logs their state
each frame. The result is saved next to `data.npz` as `data_dynamic.npz`, aligned
to the same `T` collected frames (aligned by matching `player_pos`). Positions
and velocities use the same **ENU** convention as `player_pos` in `data.npz`.
Agents keep a stable slot index `n` in `[0, N)`; when an agent is missing/dead in
a frame its `dyn_present` entry is 0 and its other values are 0.

Note on what `dyn_pos` means: mobs are **entities** (a rigged mesh + a collision
box), not voxels — they never appear in `obs_voxel_mt`. `dyn_pos` is the entity
origin, i.e. the **bottom-center (ground-contact) point** of the agent, not its
head/tail/leg. Use `dyn_yaw`/`dyn_rotation` + the collision box for the body extent.

Per-frame state:
* `dyn_present`: shape `(T, N)` `int8` — 1 if agent `n` exists at frame `t`
* `dyn_pos`: shape `(T, N, 3)` — agent world position (ENU), bottom-center
* `dyn_vel`: shape `(T, N, 3)` — agent velocity (ENU)
* `dyn_yaw`: shape `(T, N)` — agent yaw in radians (about the up axis)
* `dyn_rotation`: shape `(T, N, 3)` — full rotation `(pitch, yaw, roll)` in radians, **Minetest axes** (not ENU-reindexed)
* `dyn_hp`: shape `(T, N)` — agent health points
* `dyn_rel_pos`: shape `(T, N, 3)` — agent position relative to the player (ENU)
* `dyn_sheared`: shape `(T, N)` `int8` — mob state (sheep: sheared or not)
* `dyn_baby`: shape `(T, N)` `int8` — mob state (baby/adult)
* `dyn_color`: shape `(T, N)` object — wool/dye color where applicable
* `dyn_frame_time`: shape `(T,)` — Minetest gametime of each frame

Articulation / motion (which animation is playing + explicitly-posed bones; the
full per-bone skeleton is NOT available from the engine):
* `dyn_anim_range`: shape `(T, N, 2)` — active animation clip frame range `[start, end]` (indicates stand/walk/run/eat…)
* `dyn_anim_speed`: shape `(T, N)` — animation playback speed
* `dyn_bone_names`: shape `(B,)` object — names of bones that were explicitly posed (e.g. `head`); union across the episode
* `dyn_bone_rot`: shape `(T, N, B, 3)` — per-frame rotation (radians) of each posed bone (e.g. sheep head swivel)
* `dyn_bone_present`: shape `(T, N, B)` `int8` — 1 where that bone has an override that frame
* `dyn_bone_rot_units`: scalar — `"radians"`

Bounding-box ground truth (ENU world coordinates):
* `dyn_collisionbox`: shape `(T, N, 6)` — collision box `(x1,y1,z1,x2,y2,z2)` relative to `dyn_pos`, **Minetest axes**
* `dyn_obb_corners`: shape `(T, N, 8, 3)` — 8 corners of the yaw-oriented box (ENU world). Corner bit order: bit0→x2 else x1, bit1→y2 else y1, bit2→z2 else z1 (Minetest axes, before ENU reindex)
* `dyn_aabb_min` / `dyn_aabb_max`: shape `(T, N, 3)` — axis-aligned world box (ENU)

Static per-slot metadata (for drawing the mesh body later):
* `dyn_names`: shape `(N,)` — entity name of each agent slot
* `dyn_mesh`: shape `(N,)` object — model file name (e.g. `mobs_mc_sheep.b3d`)
* `dyn_textures`: shape `(N,)` object — list of texture file names per slot
* `dyn_visual`: shape `(N,)` object — visual type (e.g. `mesh`)
* `dyn_visual_size`: shape `(N, 3)` — model scale
* `dyn_collisionbox_static`: shape `(N, 6)` — collision box at spawn, Minetest axes
* `dyn_num_agents`: scalar `N`
* `dyn_entity_name`: scalar — the spawned entity id
* `dyn_align_offset`: scalar — offset of the collected window within the raw log
* `dyn_align_rmse`: scalar — RMSE of the `player_pos` alignment (sanity check)

> The mesh/texture files themselves live in the VoxeLibre game assets
> (`.../mods/ENTITIES/mobs_mc/models` and `.../textures`); `dyn_mesh`/`dyn_textures`
> name them so you can load and pose the model using `dyn_pos` + `dyn_rotation` +
> `dyn_visual_size`.

Player body (the player's **kinematics** — pos/vel/pitch/yaw/cam — are in
`data.npz`; these fields add the player's mesh/box ground truth to match the agents):
* `dyn_player_present`: shape `(T,)` `int8`
* `dyn_player_pos`: shape `(T, 3)` — player world position (ENU), bottom-center (feet)
* `dyn_player_rotation`: shape `(T, 3)` — `(pitch, yaw, roll)` radians, Minetest axes
* `dyn_player_collisionbox`: shape `(T, 6)` — collision box relative to pos, Minetest axes
* `dyn_player_obb_corners`: shape `(T, 8, 3)` — yaw-oriented box corners (ENU world)
* `dyn_player_aabb_min` / `dyn_player_aabb_max`: shape `(T, 3)` — axis-aligned world box (ENU)
* `dyn_player_anim_range`: shape `(T, 2)` — active animation clip frame range (stand/walk/mine…)
* `dyn_player_anim_speed`: shape `(T,)` — animation playback speed
* `dyn_player_bone_names`: shape `(Bp,)` object — posed player bones (e.g. head/arm controls)
* `dyn_player_bone_rot`: shape `(T, Bp, 3)` — per-frame bone rotation (radians)
* `dyn_player_bone_present`: shape `(T, Bp)` `int8`
* `dyn_player_mesh`: scalar object — player model file (e.g. `mcl_armor_character.b3d`)
* `dyn_player_textures`: object — list of player textures (skin/armor)
* `dyn_player_visual`: scalar object — visual type
* `dyn_player_visual_size`: shape `(3,)` — model scale
* `dyn_player_collisionbox_static`: shape `(6,)` — collision box at capture, Minetest axes
