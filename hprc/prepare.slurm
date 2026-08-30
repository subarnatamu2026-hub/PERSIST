#!/bin/bash
# Batch job: compute the seen/unseen split and initialize the three datasets'
# level seeds (light; no Minetest launched). Depends on setup finishing first.
#SBATCH --job-name=craftium-prepare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
### #SBATCH --account=YOUR_ACCOUNT
### #SBATCH --partition=YOUR_PARTITION

set -e
mkdir -p logs
cd "$SCRATCH/PERSIST"
source hprc/container.sh
FRAMES="${FRAMES:-600}"
TRAIN_N="${TRAIN_N:-1000}"; EVAL1_N="${EVAL1_N:-140}"; EVAL2_N="${EVAL2_N:-60}"

ctr_exec "
  set -e
  cd '$SCRATCH/PERSIST' && source .venv/bin/activate
  python tools/agent_splits.py
  SEEN=\$(cat datasets/agent_split/seen.txt)
  UNSEEN=\$(cat datasets/agent_split/unseen.txt)
  init () {  # name nlevels seed entities
    uv run python dataset_toolkits/generate_raw_data.py --dataset_dir datasets \
      --dataset_name \"\$1\" --env_id OpenWorldCreative-v0 --ep_timesteps $FRAMES \
      --seed \"\$3\" --init --overwrite_init --num_levels \"\$2\" \
      --dynamic_agent_entities \"\$4\"
  }
  init train_1k $TRAIN_N 1 \"\$SEEN\"
  init eval1     $EVAL1_N 2 \"\$SEEN\"
  init eval2     $EVAL2_N 3 \"\$UNSEEN\"
"
