#!/usr/bin/env bash
# Quick 3-level smoke test: generate a few levels so you can eyeball the code
# changes (player movement speed, water-only guard, mob spawning) BEFORE kicking
# off a full 100/28/12 run with build_datasets.sh.
#
#   cd ~/PERSIST && ./smoke_test.sh                 # 3 levels, 600 frames
#   FRAMES=200 N=3 ./smoke_test.sh                  # faster: 200 frames
#   NAME=smoke2 SEED=7 ./smoke_test.sh              # different terrain
#
# Output:
#   datasets/<NAME>/raw/OpenWorldCreative-v0/*/rgb.mp4          (watch these)
#   datasets/<NAME>/raw/OpenWorldCreative-v0/*/data.npz         (player+camera)
#   datasets/<NAME>/raw/OpenWorldCreative-v0/*/data_dynamic.npz (mob ground truth)
set -e
cd ~/PERSIST
source .venv/bin/activate

FRAMES="${FRAMES:-600}"
ENV=OpenWorldCreative-v0
NAME="${NAME:-smoketest}"
N="${N:-3}"
SEED="${SEED:-1}"

echo "==> Smoke test: $N levels, $FRAMES frames, seed=$SEED, name=$NAME"

# Agent pool (seen 70%) so mobs are drawn from the normal set.
python tools/agent_splits.py --seen_ratio 0.7 --seed 0 --out_dir datasets/agent_split
ENTS=$(cat datasets/agent_split/seen.txt)

# Phase 1: init the dataset (level seeds + params).
uv run python dataset_toolkits/generate_raw_data.py \
  --dataset_dir datasets --dataset_name "$NAME" --env_id "$ENV" \
  --ep_timesteps "$FRAMES" --seed "$SEED" --init --overwrite_init --num_levels "$N" \
  --dynamic_agent_entities "$ENTS"

# Phase 2: render + record. --overwrite_leveldata re-renders any existing levels
# so you never watch a stale video from a previous code version.
uv run python dataset_toolkits/generate_raw_data.py \
  --dataset_dir datasets --dataset_name "$NAME" --env_id "$ENV" \
  --disable_commit_check --ep_timesteps "$FRAMES" --overwrite_leveldata \
  --dynamic_agent_entities "$ENTS"

echo "==================================================================="
echo "Done. Watch the videos:"
echo "  datasets/$NAME/raw/$ENV/*/rgb.mp4"
echo "Mob-in-frame coverage:"
python tools/check_seen.py "datasets/$NAME/raw/$ENV/*" || true
