#!/usr/bin/env python3
"""Delete already-generated levels where the camera shook violently.

New runs already discard water/shaking levels at generation time (they are never
written). Use this only to clean a dataset that was generated BEFORE that logic,
or to double-check an existing one.

It flags a level when the player's horizontal position jumps more than
--shake_max_step blocks between two consecutive frames (far above normal
walking/sprinting ~0.25/frame) - the signature of a violent shake. That signal
is read from the saved data.npz, so it works on any existing level.

Note on water: the per-frame "on_water" flag only exists in the live run log,
which is deleted after generation, so a calm walk INTO water (no shake) in an
OLD dataset can't be detected here. Regenerating with the current code removes
those at the source. This tool catches the violent-shake levels.

Usage:
  # dry run (just report what WOULD be deleted):
  python tools/prune_bad_levels.py "datasets/train/raw/OpenWorldCreative-v0/*"
  # actually delete them:
  python tools/prune_bad_levels.py "datasets/train/raw/OpenWorldCreative-v0/*" --delete
"""
import argparse
import glob
import os
import shutil

import numpy as np


def level_max_step(level_dir):
    """Max horizontal frame-to-frame player displacement, or None if unreadable."""
    data_path = os.path.join(level_dir, "data.npz")
    if not os.path.exists(data_path):
        return None
    try:
        d = np.load(data_path)
        pp = d["player_pos"].astype(np.float64)  # [T, 3] (x, y, z)
    except Exception:
        return None
    if pp.shape[0] < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(pp[:, [0, 2]], axis=0), axis=1)  # horizontal only
    return float(steps.max()) if steps.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_glob", help='e.g. "datasets/train/raw/OpenWorldCreative-v0/*"')
    ap.add_argument("--shake_max_step", type=float, default=2.0,
                    help="flag a level if any horizontal step exceeds this (blocks/frame)")
    ap.add_argument("--delete", action="store_true",
                    help="actually delete flagged levels (default: dry run)")
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(args.dataset_glob) if os.path.isdir(d))
    total = bad = 0
    for lvl in dirs:
        step = level_max_step(lvl)
        if step is None:
            continue
        total += 1
        if step > args.shake_max_step:
            bad += 1
            tag = "DELETED" if args.delete else "would delete"
            print(f"[{tag}] {lvl}  (max horizontal step {step:.2f} > {args.shake_max_step})")
            if args.delete:
                shutil.rmtree(lvl, ignore_errors=True)

    print("-" * 64)
    action = "deleted" if args.delete else "flagged (dry run - rerun with --delete)"
    print(f"levels scanned: {total} | {action}: {bad} | kept: {total - bad}")
    if bad and not args.delete:
        print("Re-run with --delete to remove them.")


if __name__ == "__main__":
    main()
