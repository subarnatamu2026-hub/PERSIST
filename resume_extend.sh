#!/usr/bin/env bash
# Resume + grow the dataset WITHOUT losing already-generated levels.
#
# For each dataset it rebuilds level_seeds.txt as:
#     (seeds already completed on disk)  +  (new random seeds up to the target)
# so every finished level is guaranteed to be skipped, and only the missing
# levels are generated. Then it runs the generation phase (no --init, no
# --overwrite_leveldata, so existing levels are preserved).
#
# Targets (override via env):  TRAIN_N=200  EVAL1_N=56  EVAL2_N=24   (eval 80 @ 70/30)
#
# Usage (detached, survives closing the terminal):
#   cd ~/PERSIST
#   nohup bash -c 'source .venv/bin/activate && ./resume_extend.sh' > resume.log 2>&1 &
set -e
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true

FRAMES="${FRAMES:-600}"
ENV=OpenWorldCreative-v0
TRAIN_N="${TRAIN_N:-200}"
EVAL1_N="${EVAL1_N:-56}"
EVAL2_N="${EVAL2_N:-24}"

SEEN=$(cat datasets/agent_split/seen.txt)
UNSEEN=$(cat datasets/agent_split/unseen.txt)

# 1) Drop any interrupted level folders (no data.npz) so they get regenerated.
for NAME in train eval1 eval2; do
  D="datasets/$NAME/raw/$ENV"
  [ -d "$D" ] && find "$D" -mindepth 1 -maxdepth 1 -type d \
      '!' -exec test -e '{}/data.npz' ';' -print -exec rm -rf '{}' ';' || true
done

# 2) Rebuild level_seeds.txt = completed-on-disk seeds + new randoms up to target.
extend_seeds () {  # name target
  python - "$1" "$2" "$ENV" <<'PY'
import sys, os, numpy as np
name, target, env = sys.argv[1], int(sys.argv[2]), sys.argv[3]
raw = f"datasets/{name}/raw/{env}"
done = []
if os.path.isdir(raw):
    for d in sorted(os.listdir(raw)):
        p = os.path.join(raw, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "data.npz")):
            try: done.append(int(d))
            except ValueError: pass
seeds, seen = [], set()
for s in done:                      # keep every completed level first
    if s not in seen: seen.add(s); seeds.append(s)
while len(seeds) < target:          # top up with fresh, non-colliding seeds
    for s in np.random.randint(0, 2**31, target - len(seeds)).tolist():
        if s not in seen: seen.add(s); seeds.append(s)
os.makedirs(f"datasets/{name}", exist_ok=True)
np.savetxt(f"datasets/{name}/level_seeds.txt", np.array(seeds, dtype=np.int64), fmt="%d")
print(f"[{name}] completed={len(done)}  target={target}  seeds_file={len(seeds)}")
PY
}
extend_seeds train "$TRAIN_N"
extend_seeds eval1 "$EVAL1_N"
extend_seeds eval2 "$EVAL2_N"

# 3) Generate the missing levels (existing ones are skipped automatically).
gen () {  # name entities
  echo "=================================================================="
  echo "==> $1 : target seeds, $FRAMES frames (existing skipped)"
  echo "=================================================================="
  uv run python dataset_toolkits/generate_raw_data.py \
    --dataset_dir datasets --dataset_name "$1" --env_id "$ENV" \
    --disable_commit_check --ep_timesteps "$FRAMES" \
    --dynamic_agent_entities "$2"
}
gen train "$SEEN"
gen eval1 "$SEEN"
gen eval2 "$UNSEEN"

echo "==> ALL DONE (resume_extend)"
