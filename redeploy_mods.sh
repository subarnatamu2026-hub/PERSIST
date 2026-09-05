#!/usr/bin/env bash
# Fast redeploy of ONLY the Lua mods into the already-built venv - no recompile.
#
# Use this after a mod (.lua) change instead of ./reinstall.sh. Python-only
# changes (generate_raw_data.py, etc.) need NOTHING - they run from ~/PERSIST.
#
#   cd ~/PERSIST && ./redeploy_mods.sh
#
# It (1) updates the craftium fork checkout to the latest branch commit and
# (2) copies the craftium_env .lua files into every matching mod dir under .venv.
# It NEVER uses `find -name init.lua` (that would clobber the engine's
# builtin/init.lua); it is path-guarded to the openworld-* env mods only.
set -e
cd ~/PERSIST

BRANCH=claude/craftium-dynamic-agents-cauneh

# 1) Refresh the fork checkout that holds the mod source.
if [ -d gym_envs/craftium/.git ] || [ -f gym_envs/craftium/.git ]; then
  ( cd gym_envs/craftium
    git fetch origin "$BRANCH" 2>/dev/null || true
    git fetch subarna "$BRANCH" 2>/dev/null || true
    git checkout "$BRANCH" 2>/dev/null || true
    git pull --ff-only 2>/dev/null || true
  )
fi

SRC=gym_envs/craftium/craftium-envs
if [ ! -d "$SRC" ]; then
  echo "ERROR: $SRC not found. Run ./reinstall.sh once first." >&2
  exit 1
fi

# 2) Copy the env-mod .lua files into the venv site-packages (path-guarded).
count=0
while IFS= read -r dst; do
  env=$(echo "$dst" | grep -o 'openworld-[a-z]*')
  base=$(basename "$dst")
  srcf="$SRC/$env/mods/craftium_env/$base"
  if [ -f "$srcf" ]; then
    cp "$srcf" "$dst"
    count=$((count + 1))
  fi
done < <(find .venv -path '*openworld-*/mods/craftium_env/*.lua')

echo "Redeployed $count mod file(s) into .venv."
if [ "$count" -eq 0 ]; then
  echo "  (none found - is the venv built? try ./reinstall.sh once)" >&2
fi
