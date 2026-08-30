#!/usr/bin/env bash
# Submit the WHOLE pipeline to Slurm as detached, dependency-chained jobs, then
# you can log out / close your laptop - it keeps running on HPRC.
#
#   [image build (FASTER only)] -> setup (venv) -> prepare (split+init) -> 3 gen arrays
#
# Grace (Singularity, image already copied to $SCRATCH/craftium.sif):
#   ACCOUNT=<acct> PARTITION=medium ./hprc/submit_all.sh
#
# FASTER (Charliecloud; image built on-cluster automatically if missing):
#   CONTAINER_KIND=charliecloud CONTAINER_MODULE=charliecloud/0.33 \
#   ACCOUNT=<acct> PARTITION=cpu ./hprc/submit_all.sh
#
# Optional: WS_TRAIN/WS_EVAL1/WS_EVAL2 (array widths = worker counts).
set -e
cd "$SCRATCH/PERSIST"
mkdir -p logs datasets

ACCOUNT="${ACCOUNT:?set ACCOUNT (see: myproject)}"
PARTITION="${PARTITION:?set PARTITION (Grace: medium/long; FASTER: cpu)}"
# Dataset sizes (passed through to prepare.slurm; override for a smoke test).
export TRAIN_N="${TRAIN_N:-100}"
export EVAL1_N="${EVAL1_N:-28}"
export EVAL2_N="${EVAL2_N:-12}"
# Array widths = parallel workers per dataset. Capped to the level count so we
# never launch idle workers (a worker with no seeds does nothing).
WS_TRAIN="${WS_TRAIN:-$(( TRAIN_N < 50 ? TRAIN_N : 50 ))}"
WS_EVAL1="${WS_EVAL1:-$(( EVAL1_N < 28 ? EVAL1_N : 28 ))}"
WS_EVAL2="${WS_EVAL2:-$(( EVAL2_N < 12 ? EVAL2_N : 12 ))}"

SB="sbatch --parsable --account=$ACCOUNT --partition=$PARTITION"

DEP=""
# FASTER/Charliecloud: build the image on-cluster first if it isn't there yet.
if [ "${CONTAINER_KIND:-}" = "charliecloud" ] && [ ! -e "${CH_IMAGE:-$SCRATCH/craftium.sqfs}" ]; then
  B=$($SB hprc/build_image.slurm)
  echo "image    job = $B  (Charliecloud build)"
  DEP="--dependency=afterok:$B"
fi

S=$($SB $DEP hprc/setup_env.slurm)
echo "setup    job = $S"

P=$($SB --dependency=afterok:$S hprc/prepare.slurm)
echo "prepare  job = $P  (after setup)"

G1=$($SB --dependency=afterok:$P --array=0-$((WS_TRAIN-1)) \
      --export=ALL,WORLD_SIZE=$WS_TRAIN,DATASET=train hprc/generate_array.slurm)
G2=$($SB --dependency=afterok:$P --array=0-$((WS_EVAL1-1)) \
      --export=ALL,WORLD_SIZE=$WS_EVAL1,DATASET=eval1 hprc/generate_array.slurm)
G3=$($SB --dependency=afterok:$P --array=0-$((WS_EVAL2-1)) \
      --export=ALL,WORLD_SIZE=$WS_EVAL2,DATASET=eval2 hprc/generate_array.slurm)
echo "train    array = $G1  (after prepare)"
echo "eval1    array = $G2  (after prepare)"
echo "eval2    array = $G3  (after prepare)"

cat <<EOF

Submitted. You can log out now; jobs keep running.
Monitor with:   squeue -u \$USER
Cancel all:     scancel -u \$USER
Logs:           tail -f logs/craftium-*_*.out
EOF
