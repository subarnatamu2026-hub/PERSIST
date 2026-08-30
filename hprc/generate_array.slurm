#!/bin/bash
# Slurm JOB ARRAY: parallel Craftium dataset generation. Each array task is one
# data-parallel worker (--rank) over the SAME dataset's level seeds, so N tasks
# generate N disjoint chunks concurrently.
#
# Fill in --account / --partition for your allocation, then submit ONE array per
# dataset (train/eval1/eval2), passing settings via --export. Example at bottom.
#
#SBATCH --job-name=craftium-gen
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --array=0-31
### #SBATCH --account=YOUR_ACCOUNT
### #SBATCH --partition=YOUR_PARTITION

set -e
mkdir -p logs

# --- cluster-specific: load Apptainer (check `module spider apptainer`) ---
module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true

SIF="${SIF:-$SCRATCH/craftium.sif}"
WORLD_SIZE="${WORLD_SIZE:-32}"          # must equal the array size (0..N-1)
DATASET="${DATASET:?set DATASET}"
ENTITIES="${ENTITIES:?set ENTITIES}"
FRAMES="${FRAMES:-600}"
ENVID="${ENVID:-OpenWorldCreative-v0}"

RANK="${SLURM_ARRAY_TASK_ID}"
# Unique Minetest server port per worker so co-located tasks don't collide
# (the Xvfb display is already per-rank in the code: 99+rank).
PORT=$((30000 + RANK * 4))

apptainer exec --bind /scratch "$SIF" bash -lc "
  set -e
  cd '$SCRATCH/PERSIST'
  source .venv/bin/activate
  export UV_CACHE_DIR='$SCRATCH/.uv-cache' HF_HOME='$SCRATCH/.hf-cache'
  uv run python dataset_toolkits/generate_raw_data.py \
    --dataset_dir datasets --dataset_name '$DATASET' --env_id '$ENVID' \
    --disable_commit_check --ep_timesteps $FRAMES \
    --world_size $WORLD_SIZE --rank $RANK --mt_port $PORT \
    --dynamic_agent_entities '$ENTITIES'
"

# --- how to submit (run these from $SCRATCH/PERSIST after --init, see README) ---
# SEEN=$(cat datasets/agent_split/seen.txt); UNSEEN=$(cat datasets/agent_split/unseen.txt)
# sbatch --array=0-31 --export=ALL,WORLD_SIZE=32,DATASET=train_1k,ENTITIES="$SEEN"  hprc/generate_array.slurm
# sbatch --array=0-15 --export=ALL,WORLD_SIZE=16,DATASET=eval1,ENTITIES="$SEEN"     hprc/generate_array.slurm
# sbatch --array=0-7  --export=ALL,WORLD_SIZE=8, DATASET=eval2,ENTITIES="$UNSEEN"   hprc/generate_array.slurm
