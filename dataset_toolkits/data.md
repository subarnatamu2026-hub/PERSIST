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
