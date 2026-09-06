#!/usr/bin/env bash
# Build the full dataset suite:
#   - train  : 100 levels, agents drawn from the SEEN (70%) pool
#   - eval1   : 28 levels,  agents from the SEEN pool     (in-distribution)
#   - eval2   : 12 levels,  agents from the UNSEEN (30%) pool (zero-shot)
# Each level uses a distinct seed -> distinct terrain; the three groups use
# different base seeds so their terrains don't overlap. 600 frames each.
set -e
cd ~/PERSIST
source .venv/bin/activate

FRAMES="${FRAMES:-600}"
ENV=OpenWorldCreative-v0

# Sizes are overridable via env vars for quick tests, e.g.:
#   TRAIN_N=4 EVAL1_N=2 EVAL2_N=2 ./build_datasets.sh
TRAIN_NAME="${TRAIN_NAME:-train}";  TRAIN_N="${TRAIN_N:-100}"; TRAIN_SEED="${TRAIN_SEED:-1}"
EVAL1_NAME="${EVAL1_NAME:-eval1}";  EVAL1_N="${EVAL1_N:-28}";  EVAL1_SEED="${EVAL1_SEED:-2}"
EVAL2_NAME="${EVAL2_NAME:-eval2}";  EVAL2_N="${EVAL2_N:-12}";  EVAL2_SEED="${EVAL2_SEED:-3}"

# 1) Deterministic 70/30 split of the agent pool (writes datasets/agent_split/*.txt)
python tools/agent_splits.py --seen_ratio 0.7 --seed 0 --out_dir datasets/agent_split
SEEN=$(cat datasets/agent_split/seen.txt)
UNSEEN=$(cat datasets/agent_split/unseen.txt)
echo "SEEN  pool -> train + eval1"
echo "UNSEEN pool -> eval2 (zero-shot)"

# Set OVERWRITE=1 to regenerate levels that already exist on disk (otherwise
# existing levels are skipped). Needed after a code change so stale videos from
# an earlier run are re-rendered instead of kept.
OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" = "1" ]; then OVERWRITE_FLAG="--overwrite_leveldata"; fi

run_group () {  # name  num_levels  seed  entities
  local NAME=$1 N=$2 SEED=$3 ENTS=$4
  echo "==================================================================="
  echo "==> $NAME : $N levels, seed=$SEED, $FRAMES frames  OVERWRITE=${OVERWRITE:-0}"
  echo "==================================================================="
  uv run python dataset_toolkits/generate_raw_data.py \
    --dataset_dir datasets --dataset_name "$NAME" --env_id "$ENV" \
    --ep_timesteps "$FRAMES" --seed "$SEED" --init --overwrite_init --num_levels "$N" \
    --dynamic_agent_entities "$ENTS"
  uv run python dataset_toolkits/generate_raw_data.py \
    --dataset_dir datasets --dataset_name "$NAME" --env_id "$ENV" \
    --disable_commit_check --ep_timesteps "$FRAMES" $OVERWRITE_FLAG \
    --dynamic_agent_entities "$ENTS"
}

run_group "$TRAIN_NAME" "$TRAIN_N" "$TRAIN_SEED" "$SEEN"
run_group "$EVAL1_NAME" "$EVAL1_N" "$EVAL1_SEED" "$SEEN"
run_group "$EVAL2_NAME" "$EVAL2_N" "$EVAL2_SEED" "$UNSEEN"

# Record the seeds ACTUALLY used, per dataset. Each generated level folder is named
# by the seed that produced it, so this captures reseeds too (a level that hit water
# and was regenerated ends up under a different seed than the planned level_seeds.txt).
MANIFEST="datasets/seeds_used.csv"
echo "dataset,seed" > "$MANIFEST"
for NAME in "$TRAIN_NAME" "$EVAL1_NAME" "$EVAL2_NAME"; do
  d="datasets/$NAME/raw/$ENV"
  [ -d "$d" ] || continue
  for lvl in "$d"/*/; do
    [ -d "$lvl" ] || continue
    echo "$NAME,$(basename "$lvl")" >> "$MANIFEST"
  done
done
echo "==> Seeds actually used written to $MANIFEST ($(($(wc -l < "$MANIFEST") - 1)) levels)"
echo "    (planned seeds per dataset are also in datasets/<name>/level_seeds.txt)"

echo "==> ALL DONE. Datasets: datasets/$TRAIN_NAME, datasets/$EVAL1_NAME, datasets/$EVAL2_NAME"
