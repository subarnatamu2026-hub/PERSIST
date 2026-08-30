#!/usr/bin/env bash
# Smoke test: generate ONLY 2 levels for each of train / eval1 / eval2 so you can
# eyeball the RGB video + data_dynamic before committing to the full run. Same
# detached, dependency-chained pipeline as submit_all.sh, just tiny sizes.
#
# Each level still spawns exactly 10 mobs (fixed count, mixed species) and runs
# 600 frames. train/eval1 draw from the SEEN pool, eval2 from the UNSEEN pool.
#
# FASTER:
#   CONTAINER_KIND=charliecloud CONTAINER_MODULE=charliecloud/0.33 \
#   ACCOUNT=142689572675 PARTITION=cpu ./hprc/smoke_test.sh
#
# When happy, run the full suite with ./hprc/submit_all.sh (100 / 28 / 12).
set -e
cd "$SCRATCH/PERSIST"

# 2 levels each; one worker per level (world_size 2).
TRAIN_N=2 EVAL1_N=2 EVAL2_N=2 \
WS_TRAIN=2 WS_EVAL1=2 WS_EVAL2=2 \
exec ./hprc/submit_all.sh
