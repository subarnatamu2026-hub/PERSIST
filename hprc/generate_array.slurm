#!/bin/bash
# Slurm JOB ARRAY: parallel Craftium dataset generation. Each array task is one
# data-parallel worker (--rank) over the SAME dataset's level seeds, so N tasks
# generate N disjoint chunks concurrently. Fully detached (survives logout).
#
# Pass DATASET and WORLD_SIZE via --export; entities are read from the split
# files (seen for train/eval1, unseen for eval2) so no comma-laden value goes
# through --export. account/partition come from submit_all.sh or the sbatch CLI.
#
#SBATCH --job-name=craftium-gen
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
### #SBATCH --account=YOUR_ACCOUNT
### #SBATCH --partition=YOUR_PARTITION

set -e
mkdir -p logs

SIF="${SIF:-$SCRATCH/craftium.sif}"
WORLD_SIZE="${WORLD_SIZE:?set WORLD_SIZE (must equal the array size)}"
DATASET="${DATASET:?set DATASET}"
FRAMES="${FRAMES:-600}"
ENVID="${ENVID:-OpenWorldCreative-v0}"

# Entities: eval2 draws from the unseen pool, everything else from seen.
if [ -n "${ENTITIES_FILE:-}" ]; then
  ENTS_FILE="$ENTITIES_FILE"
elif [ "$DATASET" = "eval2" ]; then
  ENTS_FILE="datasets/agent_split/unseen.txt"
else
  ENTS_FILE="datasets/agent_split/seen.txt"
fi

RANK="${SLURM_ARRAY_TASK_ID}"
PORT=$((30000 + RANK * 4))   # unique Minetest port per worker (Xvfb display is 99+rank in code)

singularity exec --bind /scratch "$SIF" bash -lc "
  set -e
  cd '$SCRATCH/PERSIST' && source .venv/bin/activate
  ENTS=\$(cat '$ENTS_FILE')
  uv run python dataset_toolkits/generate_raw_data.py \
    --dataset_dir datasets --dataset_name '$DATASET' --env_id '$ENVID' \
    --disable_commit_check --ep_timesteps $FRAMES \
    --world_size $WORLD_SIZE --rank $RANK --mt_port $PORT \
    --dynamic_agent_entities \"\$ENTS\"
"
