#!/usr/bin/env bash
# Build the full dataset suite:
#   - train  : 1000 levels, agents drawn from the SEEN (70%) pool
#   - eval1   : 140 levels,  agents from the SEEN pool     (in-distribution)
#   - eval2   : 60 levels,   agents from the UNSEEN (30%) pool (zero-shot)
# Each level uses a distinct seed -> distinct terrain; the three groups use
# different base seeds so their terrains don't overlap. 600 frames each.
set -e
cd ~/PERSIST
source .venv/bin/activate

FRAMES=600
ENV=OpenWorldCreative-v0

TRAIN_NAME=train_1k;  TRAIN_N=1000; TRAIN_SEED=1
EVAL1_NAME=eval1;     EVAL1_N=140;  EVAL1_SEED=2
EVAL2_NAME=eval2;     EVAL2_N=60;   EVAL2_SEED=3

# 1) Deterministic 70/30 split of the agent pool (writes datasets/agent_split/*.txt)
python tools/agent_splits.py --seen_ratio 0.7 --seed 0 --out_dir datasets/agent_split
SEEN=$(cat datasets/agent_split/seen.txt)
UNSEEN=$(cat datasets/agent_split/unseen.txt)
echo "SEEN  pool -> train + eval1"
echo "UNSEEN pool -> eval2 (zero-shot)"

run_group () {  # name  num_levels  seed  entities
  local NAME=$1 N=$2 SEED=$3 ENTS=$4
  echo "==================================================================="
  echo "==> $NAME : $N levels, seed=$SEED, $FRAMES frames"
  echo "==================================================================="
  uv run python dataset_toolkits/generate_raw_data.py \
    --dataset_dir datasets --dataset_name "$NAME" --env_id "$ENV" \
    --ep_timesteps "$FRAMES" --seed "$SEED" --init --num_levels "$N" \
    --dynamic_agent_entities "$ENTS"
  uv run python dataset_toolkits/generate_raw_data.py \
    --dataset_dir datasets --dataset_name "$NAME" --env_id "$ENV" \
    --disable_commit_check --ep_timesteps "$FRAMES" \
    --dynamic_agent_entities "$ENTS"
}

run_group "$TRAIN_NAME" "$TRAIN_N" "$TRAIN_SEED" "$SEEN"
run_group "$EVAL1_NAME" "$EVAL1_N" "$EVAL1_SEED" "$SEEN"
run_group "$EVAL2_NAME" "$EVAL2_N" "$EVAL2_SEED" "$UNSEEN"

echo "==> ALL DONE. Datasets: datasets/$TRAIN_NAME, datasets/$EVAL1_NAME, datasets/$EVAL2_NAME"
