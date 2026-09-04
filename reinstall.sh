#!/usr/bin/env bash
# Reinstall Craftium (engine + mods) with the latest fixes from the fork, then
# (re)build the project venv. Run this after mod changes were pushed to the fork
# branch, or whenever the environment needs to be rebuilt from scratch.
#
#   cd ~/PERSIST && ./reinstall.sh
#
# What it touches:
#   * gym_envs/craftium  -> re-cloned from the fork branch (fresh engine+mods)
#   * .venv              -> synced; craftium is reinstalled so the NEW mods are
#                           copied into site-packages (mods run from there)
# What it does NOT touch:
#   * datasets/          (your generated data.npz / rgb.mp4 are safe)
#   * any of your PERSIST python/toolkit code
set -e
cd ~/PERSIST

FORK=https://github.com/subarnatamu2026-hub/craftium
BRANCH=claude/craftium-dynamic-agents-cauneh

echo "==> Refreshing gym_envs/craftium from $BRANCH"
rm -rf gym_envs/craftium
git clone --recursive -b "$BRANCH" "$FORK" gym_envs/craftium

echo "==> uv sync (cu + env groups) and reinstall craftium so new mods deploy"
uv sync --group cu --group env --reinstall-package craftium

echo "==> Done. Verify the water-only guard is in the installed mod:"
INIT=$(find .venv -path '*openworld-creative*/mods/craftium_env/init.lua' | head -1)
if [ -n "$INIT" ]; then
  if grep -q "add_velocity" "$INIT"; then
    echo "    WARNING: add_velocity still present in $INIT (stale mod?)"
  else
    echo "    OK: no add_velocity in installed player mod ($INIT)"
  fi
else
  echo "    NOTE: could not locate installed init.lua under .venv"
fi
