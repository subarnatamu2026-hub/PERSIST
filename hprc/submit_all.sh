#!/usr/bin/env bash
# Submit the WHOLE pipeline to Slurm as detached, dependency-chained jobs, then
# you can log out / close your laptop - it keeps running on HPRC.
#
#   setup (build venv)  ->  prepare (split + init)  ->  3 generation arrays
#
# Usage (from $SCRATCH/PERSIST):
#   ACCOUNT=<your_acct> PARTITION=medium ./hprc/submit_all.sh
# Optional overrides: WS_TRAIN, WS_EVAL1, WS_EVAL2 (array widths = worker counts).
#   Find your account with `myproject` (HPRC); Grace has time-named partitions: use `medium` (1 day) or `long` (7 days), not `cpu`.
set -e
cd "$SCRATCH/PERSIST"
mkdir -p logs datasets

ACCOUNT="${ACCOUNT:?set ACCOUNT (see: myproject)}"
PARTITION="${PARTITION:?set PARTITION (Grace: medium or long; NOT cpu)}"
WS_TRAIN="${WS_TRAIN:-32}"
WS_EVAL1="${WS_EVAL1:-16}"
WS_EVAL2="${WS_EVAL2:-8}"

SB="sbatch --parsable --account=$ACCOUNT --partition=$PARTITION"

S=$($SB hprc/setup_env.slurm)
echo "setup    job = $S"

P=$($SB --dependency=afterok:$S hprc/prepare.slurm)
echo "prepare  job = $P  (after setup)"

G1=$($SB --dependency=afterok:$P --array=0-$((WS_TRAIN-1)) \
      --export=ALL,WORLD_SIZE=$WS_TRAIN,DATASET=train_1k hprc/generate_array.slurm)
G2=$($SB --dependency=afterok:$P --array=0-$((WS_EVAL1-1)) \
      --export=ALL,WORLD_SIZE=$WS_EVAL1,DATASET=eval1 hprc/generate_array.slurm)
G3=$($SB --dependency=afterok:$P --array=0-$((WS_EVAL2-1)) \
      --export=ALL,WORLD_SIZE=$WS_EVAL2,DATASET=eval2 hprc/generate_array.slurm)
echo "train_1k array = $G1  (after prepare)"
echo "eval1    array = $G2  (after prepare)"
echo "eval2    array = $G3  (after prepare)"

cat <<EOF

Submitted. You can log out now; jobs keep running.
Monitor with:   squeue -u \$USER
Cancel all:     scancel -u \$USER
Logs:           tail -f logs/craftium-*_*.out
EOF
