#!/usr/bin/env bash
# One-time environment build for HPRC. Run this INSIDE the container, in an
# interactive compute session (not on the login node), e.g.:
#
#   srun --time=02:00:00 --cpus-per-task=8 --mem=16G --pty bash
#   module load Apptainer            # (name may differ: `module spider apptainer`)
#   apptainer exec --bind /scratch $SCRATCH/craftium.sif bash hprc/setup_env.sh
#
# It builds Craftium (compiles Minetest) + installs torch into .venv, all under
# your scratch PERSIST checkout so it persists for the array jobs.
set -e

# Keep all caches on scratch (login-home has small quotas).
export UV_CACHE_DIR="${SCRATCH}/.uv-cache"
export HF_HOME="${SCRATCH}/.hf-cache"
export PLAYWRIGHT_BROWSERS_PATH=0

cd "${SCRATCH}/PERSIST"

echo "==> Building venv (torch + craftium). This compiles Minetest; ~15-30 min."
uv sync --group cu --group env

echo "==> Sanity check:"
source .venv/bin/activate
python -c "import craftium, torch, gymnasium, numpy; print('craftium OK, torch', torch.__version__)"
python -c "import craftium, os; d=os.path.dirname(craftium.__file__); \
print('mod present:', os.path.exists(d+'/craftium-envs/openworld-creative/mods/craftium_env/dynamic_agents.lua'))"
echo "==> Done. venv at ${SCRATCH}/PERSIST/.venv"
