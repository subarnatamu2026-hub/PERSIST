# Running Craftium dataset generation on TAMU HPRC

Parallel, headless dataset generation on HPRC (Grace/FASTER) using a Slurm job
array. One CPU worker per array task; each worker (`--rank`) processes a disjoint
slice of the dataset's level seeds, so terrains never overlap between workers.

Everything lives in **`$SCRATCH`** (`/scratch/user/subarna_tamu.2026`).

> These are templates. Cluster-specific bits you must fill: the `--account` and
> `--partition` in `generate_array.slurm`. Never run heavy work on the login node.
>
> **On Grace the container tool is `singularity`** (already on PATH at
> `/usr/bin/singularity`; no `module load` needed). Everywhere below that says
> `apptainer`, use `singularity` with the same arguments.

## 0. Connect with MobaXterm
- Session → SSH. Remote host: `grace.hprc.tamu.edu` (or `faster.hprc.tamu.edu`).
  Username: your NetID (`subarna_tamu.2026`). Port 22.
- Authenticate with NetID password + Duo 2FA (approve the push).
- Once in, go to scratch: `cd $SCRATCH` (this is `/scratch/user/subarna_tamu.2026`).

## 1. Get the code
```bash
cd $SCRATCH
git clone -b claude/craftium-dynamic-agents-cauneh \
  https://github.com/subarnatamu2026-hub/PERSIST PERSIST
cd PERSIST
git submodule update --init --recursive
# point the craftium submodule at your fork's branch
cd gym_envs/craftium
git remote add subarna https://github.com/subarnatamu2026-hub/craftium 2>/dev/null || true
git fetch subarna claude/craftium-dynamic-agents-cauneh
git checkout claude/craftium-dynamic-agents-cauneh
git submodule update --init --recursive     # pulls VoxeLibre + mobs
cd $SCRATCH/PERSIST
```

## 2. Build the container (system deps) — once
On Grace (login node has direct internet):
```bash
cd $SCRATCH/PERSIST
singularity build --fakeroot $SCRATCH/craftium.sif hprc/craftium.def
```
If Grace refuses `--fakeroot` (most HPRC sites disable user fakeroot), build the
image on a machine where you have root — e.g. your WSL laptop, which already has a
working setup — and copy it over:
Grace disables user `--fakeroot` (`libsubid: -1`), so build on WSL where you have
root, then copy the finished image:
```bash
# on WSL:
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update && sudo apt install -y apptainer
cd ~/PERSIST && sudo apptainer build craftium.sif hprc/craftium.def   # real root; no --fakeroot
scp craftium.sif subarna_tamu.2026@grace.hprc.tamu.edu:/scratch/user/subarna_tamu.2026/
```
(or drag `craftium.sif` into `/scratch/user/subarna_tamu.2026/` via MobaXterm's SFTP panel).

## RECOMMENDED: run everything detached (survives logout / laptop close)
Interactive `srun --pty` sessions DIE when you disconnect. Instead submit the
whole pipeline as dependency-chained **batch** jobs and walk away. After the
container `.sif` is in `$SCRATCH` (steps 1-2), just run:
```bash
cd $SCRATCH/PERSIST
ACCOUNT=<your_account> PARTITION=cpu ./hprc/submit_all.sh
# find your account with:  myproject
```
This queues: **setup** (build venv) -> **prepare** (split + init) -> **3 generation
arrays** (train_1k / eval1 / eval2), each starting only after the previous
finishes. You can log out immediately. Monitor / manage anytime with:
```bash
squeue -u $USER            # what's running/queued
tail -f logs/craftium-*_*.out
scancel -u $USER           # cancel everything if needed
```
Tune worker counts with `WS_TRAIN`/`WS_EVAL1`/`WS_EVAL2` (they set the array
widths), e.g. `WS_TRAIN=64 ACCOUNT=... PARTITION=cpu ./hprc/submit_all.sh`.

Steps 3-6 below are the manual, step-by-step equivalents if you prefer to run
each phase yourself.

## 3. Build the Python env (torch + craftium) — once
Compiles Minetest into `$SCRATCH/PERSIST/.venv`. Do it in an interactive job.
`WebProxy` gives the compute node outbound internet (Singularity passes it into
the container) so `uv sync` can download torch etc.:
```bash
srun --time=02:00:00 --cpus-per-task=8 --mem=16G --pty bash
module load WebProxy
cd $SCRATCH/PERSIST
singularity exec --bind /scratch $SCRATCH/craftium.sif bash hprc/setup_env.sh
exit    # leave the interactive job
```

## 4. Make the agent split — once
```bash
apptainer exec --bind /scratch $SCRATCH/craftium.sif bash -lc \
  "cd $SCRATCH/PERSIST && source .venv/bin/activate && python tools/agent_splits.py"
# writes datasets/agent_split/{seen.txt,unseen.txt}  (36 seen / 15 unseen)
```

## 5. Initialize each dataset's seeds — once per dataset (light, no Minetest)
```bash
SEEN=$(cat datasets/agent_split/seen.txt)
UNSEEN=$(cat datasets/agent_split/unseen.txt)
run_init () {  # name  nlevels  seed  entities
  apptainer exec --bind /scratch $SCRATCH/craftium.sif bash -lc \
   "cd $SCRATCH/PERSIST && source .venv/bin/activate && \
    uv run python dataset_toolkits/generate_raw_data.py --dataset_dir datasets \
      --dataset_name $1 --env_id OpenWorldCreative-v0 --ep_timesteps 600 \
      --seed $3 --init --num_levels $2 --dynamic_agent_entities '$4'"
}
run_init train_1k 1000 1 "$SEEN"
run_init eval1     140  2 "$SEEN"
run_init eval2     60   3 "$UNSEEN"
```

## 6. Submit the parallel generation (one array per dataset)
Array size == `WORLD_SIZE`. More tasks = faster (bounded by your allocation).
```bash
SEEN=$(cat datasets/agent_split/seen.txt)
UNSEEN=$(cat datasets/agent_split/unseen.txt)

sbatch --array=0-31 --export=ALL,WORLD_SIZE=32,DATASET=train_1k,ENTITIES="$SEEN"  hprc/generate_array.slurm
sbatch --array=0-15 --export=ALL,WORLD_SIZE=16,DATASET=eval1,ENTITIES="$SEEN"     hprc/generate_array.slurm
sbatch --array=0-7  --export=ALL,WORLD_SIZE=8,DATASET=eval2,ENTITIES="$UNSEEN"    hprc/generate_array.slurm
```
Rule: the `--array=0-(N-1)` upper bound must be `WORLD_SIZE-1`.

## 7. Monitor / results
```bash
squeue -u $USER            # running/queued tasks
tail -f logs/craftium-gen_*_*.out
ls datasets/train_1k/raw/OpenWorldCreative-v0/ | wc -l   # levels done so far
```
Re-running an array is safe: existing levels are skipped, only missing seeds are
filled (add `--overwrite_leveldata` only if you want to redo them).

## Sizing tips
- ~2 CPUs + ~6 GB RAM per worker is plenty (one headless Minetest each).
- 600 frames/level is roughly a couple minutes; 1000 levels / 32 workers ≈ a few
  hours. Scale `WORLD_SIZE` (and the array) up to what your allocation allows.
- Keep `datasets/` on `$SCRATCH` (it can get large). Copy finished datasets to
  more permanent storage when done, since scratch is purged periodically.
