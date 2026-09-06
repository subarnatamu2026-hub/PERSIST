#!/bin/bash
# =====================================================================
# Self-contained Grace (TAMU HPRC) job: build venv (if needed) -> split +
# init train/eval1/eval2 -> generate ALL levels in parallel on ONE node,
# using every core. One `sbatch` submission; detached (survives logout).
#
# Uses Singularity (Grace's container tool) + $SCRATCH/craftium.sif. No
# Charliecloud / FASTER stuff. Submit with your account + a CPU partition:
#
#   cd $SCRATCH/PERSIST
#   sbatch --account=<your_account> --partition=medium hprc/grace_generate.slurm
#
# Optional overrides (env via --export):
#   sbatch --account=A --partition=medium \
#     --export=ALL,TRAIN_N=150,EVAL1_N=45,EVAL2_N=19,WORKERS=24 \
#     hprc/grace_generate.slurm
# =====================================================================
#SBATCH --job-name=craftium-grace
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
### Set these on the command line (recommended) or uncomment and edit:
### #SBATCH --account=YOUR_ACCOUNT
### #SBATCH --partition=medium

set -e
cd "$SCRATCH/PERSIST"
mkdir -p logs

SIF="${SIF:-$SCRATCH/craftium.sif}"
ENV="${ENVID:-OpenWorldCreative-v0}"
FRAMES="${FRAMES:-600}"
TRAIN_N="${TRAIN_N:-150}"; EVAL1_N="${EVAL1_N:-45}"; EVAL2_N="${EVAL2_N:-19}"
# One worker per 2 cores by default (a Grace node has 48 cores -> 24 workers).
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-48} / 2 ))}"
[ "$WORKERS" -lt 1 ] && WORKERS=1

echo "SIF=$SIF  WORKERS=$WORKERS  sizes: train=$TRAIN_N eval1=$EVAL1_N eval2=$EVAL2_N  frames=$FRAMES"
module load WebProxy 2>/dev/null || true    # outbound internet for the venv build

# Run a bash command string inside the container.
run() { singularity exec --bind /scratch "$SIF" bash -lc "$1"; }

# --- 1) Build the venv once (clone craftium fork + compile Minetest + torch) ---
if [ ! -x .venv/bin/python ]; then
  echo "==> Building venv (first run only; ~15-30 min)"
  run "cd '$PWD' && export UV_CACHE_DIR='$SCRATCH/.uv-cache' UV_PYTHON_INSTALL_DIR='$SCRATCH/.uv-python' HF_HOME='$SCRATCH/.hf-cache' XDG_CACHE_HOME='$SCRATCH/.cache' PIP_CACHE_DIR='$SCRATCH/.pip-cache' && bash hprc/setup_env.sh"
fi

# --- 2) Species split + init the three datasets (single process) ---
echo "==> Preparing datasets (split + init seeds)"
run "cd '$PWD' && source .venv/bin/activate && python tools/agent_splits.py --seen_ratio 0.7 --seed 0 --out_dir datasets/agent_split"
SEEN=$(cat datasets/agent_split/seen.txt)
UNSEEN=$(cat datasets/agent_split/unseen.txt)

init() {  # name  num_levels  seed  entities
  run "cd '$PWD' && source .venv/bin/activate && python dataset_toolkits/generate_raw_data.py \
    --dataset_dir datasets --dataset_name '$1' --env_id '$ENV' --ep_timesteps $FRAMES \
    --seed $3 --init --overwrite_init --num_levels $2 --dynamic_agent_entities \"$4\""
}
init train "$TRAIN_N" 1 "$SEEN"
init eval1 "$EVAL1_N" 2 "$SEEN"
init eval2 "$EVAL2_N" 3 "$UNSEEN"

# --- 3) Generate a dataset with WORKERS parallel workers (data-parallel over
#         seeds via --world_size/--rank; each gets its own Xvfb + port). ---
gen() {  # name  entities
  local NAME="$1" ENTS="$2" r PORT pids=()
  echo "==> Generating $NAME with $WORKERS parallel workers"
  for r in $(seq 0 $((WORKERS - 1))); do
    PORT=$((30000 + r * 4))
    run "cd '$PWD' && source .venv/bin/activate && DISPLAY= python dataset_toolkits/generate_raw_data.py \
      --dataset_dir datasets --dataset_name '$NAME' --env_id '$ENV' --disable_commit_check \
      --ep_timesteps $FRAMES --world_size $WORKERS --rank $r --mt_port $PORT \
      --dynamic_agent_entities \"$ENTS\"" > "logs/${NAME}_rank${r}.out" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
}
gen train "$SEEN"
gen eval1 "$SEEN"
gen eval2 "$UNSEEN"

# --- 4) Record the seeds actually used (captures reseeds) ---
{ echo "dataset,seed"; for N in train eval1 eval2; do
    for d in datasets/$N/raw/$ENV/*/; do [ -d "$d" ] && echo "$N,$(basename "$d")"; done
  done; } > datasets/seeds_used.csv

echo "==> ALL DONE. Levels:"
for N in train eval1 eval2; do
  echo "    $N: $(ls -d datasets/$N/raw/$ENV/*/ 2>/dev/null | wc -l)"
done
echo "    seeds -> datasets/seeds_used.csv"
