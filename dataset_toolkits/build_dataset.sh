#!/bin/bash

# Usage: build_dataset.sh <dataset_directory> [batch_size] [--default-classdict <path>]
#
# Node-class dictionary (mt_voxel_classdict.json) -- choose ONE of two modes:
#
#   (a) --default-classdict <path>  (reuse an existing dictionary)
#       Copies <path> into the dataset as mt_voxel_classdict.json and reuses it. The
#       node-class mapping stays identical to
#       the data that dictionary came from, so previously generated data and
#       pre-trained models REMAIN COMPATIBLE with the new dataset. Levels containing (node_id, node_param) pairs
#       absent from the dictionary are filtered out (marked invalid) during voxelization.
#       The `--create_voxel_class_dict` step is skipped; `--check_conditions_only` is run
#       instead so metadata.csv still gets its `preprocessed` / `level_quality_score` fields.
#
#   (b) no flag  (default behavior, generate a fresh dictionary)
#       Builds a new mt_voxel_classdict.json from the nodes found in this dataset's valid
#       levels. Maximises node coverage for this dataset, but the class ids differ from any
#       previous dictionary, which BREAKS COMPATIBILITY with previously generated data and any
#       model trained against the old mapping.
#
# See dataset_toolkits/data.md ("Node-class dictionary: reuse vs. regenerate") for details.

# Parse optional --default-classdict flag, leaving positional args intact.
DEFAULT_CLASSDICT=""
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --default-classdict)
            DEFAULT_CLASSDICT="$2"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

# Check if output directory is provided
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <dataset_directory> [batch_size] [--default-classdict <path>]"
    exit 1
fi

OUTPUT_DIR=$1
BATCH_SIZE=${2:-32}  # Use second argument if provided, otherwise default to 32 - not used now but keeping it there for later

# Initialize environment
source .venv/bin/activate

bash dataset_toolkits/check_missing_data.sh $OUTPUT_DIR -d

python dataset_toolkits/build_metadata.py --output_dir $OUTPUT_DIR --from_file

if [ -n "$DEFAULT_CLASSDICT" ]; then
    # Mode (a): reuse a pre-specified node-class dictionary (see header).
    echo "Using pre-specified mt_voxel_classdict.json from $DEFAULT_CLASSDICT"
    cp "$DEFAULT_CLASSDICT" "$OUTPUT_DIR/mt_voxel_classdict.json"
    # Run validity checks only (no dictionary generation) so metadata.csv gets populated.
    python dataset_toolkits/process_mt_data.py --output_dir $OUTPUT_DIR --check_conditions_only --overwrite
else
    # Mode (b): generate a fresh node-class dictionary from this dataset (see header).
    python dataset_toolkits/process_mt_data.py --output_dir $OUTPUT_DIR --create_voxel_class_dict
fi

python dataset_toolkits/build_metadata.py --output_dir $OUTPUT_DIR
python dataset_toolkits/process_mt_data.py --output_dir $OUTPUT_DIR

python dataset_toolkits/build_metadata.py --output_dir $OUTPUT_DIR --from_file --set_validation_samples vae
