#!/usr/bin/env python3
"""Check whether every dynamic agent appears in the camera view (RGB) at least once.

Uses ground truth ONLY (no video decoding). A mob counts as "seen" on a frame if
its body point is:
  * in front of the camera,
  * inside the recorded field of view (fov_x horizontally, fov_y vertically), and
  * within --max_dist blocks.
For each level it reports how many of the mobs were seen in >=1 frame, and which
slots (if any) were never seen. A summary line gives the % of levels where ALL
mobs were seen.

Coordinate frame: cam_pos / cam_dir (data.npz) and dyn_pos (data_dynamic.npz) are
in the same world frame with Y = up, so the geometry below is consistent.

Occlusion caveat: this tests the camera *frustum*, not whether terrain hides the
mob, so it can slightly OVER-count (a mob behind a hill still counts). It is a
fast, strong proxy for "was in frame". Lower --max_dist to be stricter.

Usage:
  python tools/check_seen.py "datasets/gtrain/raw/OpenWorldCreative-v0/*"
  python tools/check_seen.py "datasets/train/raw/OpenWorldCreative-v0/*" --max_dist 30
"""
import argparse
import glob
import os

import numpy as np


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-9, 1.0, n)


def _visible(cam_pos, cam_dir, fov_x, fov_y, pts, max_dist):
    """Boolean [T] mask: is world point pts[t] inside the camera frustum at t?"""
    f = _unit(cam_dir)
    right = _unit(np.cross(f, np.array([0.0, 1.0, 0.0])))   # camera right
    up = np.cross(right, f)                                 # camera up
    vec = pts - cam_pos
    z = np.sum(vec * f, axis=-1)                            # depth (forward)
    x = np.sum(vec * right, axis=-1)
    y = np.sum(vec * up, axis=-1)
    dist = np.linalg.norm(vec, axis=-1)
    ha = np.abs(np.arctan2(x, np.maximum(z, 1e-6)))         # horizontal angle
    va = np.abs(np.arctan2(y, np.maximum(z, 1e-6)))         # vertical angle
    return (z > 0.05) & (dist <= max_dist) & (ha <= fov_x * 0.5) & (va <= fov_y * 0.5)


def check_level(lvl, max_dist):
    d = np.load(os.path.join(lvl, "data.npz"))
    dd = np.load(os.path.join(lvl, "data_dynamic.npz"), allow_pickle=True)
    cam_pos = d["cam_pos"].astype(np.float64)
    cam_dir = d["cam_dir"].astype(np.float64)
    fov_x = d["fov_x"].astype(np.float64)
    fov_y = d["fov_y"].astype(np.float64)
    pos = dd["dyn_pos"].astype(np.float64)                  # [T,N,3] bottom-centre
    present = dd["dyn_present"].astype(bool)                # [T,N]
    box = dd["dyn_collisionbox"].astype(np.float64)        # [T,N,6]; y2=index 4=height
    names = [str(x) for x in dd["dyn_names"]]

    T = min(cam_pos.shape[0], pos.shape[0])
    N = pos.shape[1]
    seen_any = np.zeros(N, bool)
    seen_frac = np.zeros(N)
    for m in range(N):
        height = np.clip(box[:T, m, 4], 0.2, 4.0)
        mid = pos[:T, m].copy()
        mid[:, 1] += 0.5 * height                           # also test mid-body point
        vis = np.zeros(T, bool)
        for p in (pos[:T, m], mid):
            vis |= _visible(cam_pos[:T], cam_dir[:T], fov_x[:T], fov_y[:T], p, max_dist)
        vis &= present[:T, m]
        seen_any[m] = bool(vis.any())
        seen_frac[m] = float(vis.mean())
    return names, seen_any, seen_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_glob", help='e.g. "datasets/gtrain/raw/OpenWorldCreative-v0/*"')
    ap.add_argument("--max_dist", type=float, default=40.0)
    ap.add_argument("--verbose", action="store_true", help="print per-mob seen fraction")
    args = ap.parse_args()

    dirs = sorted(glob.glob(args.dataset_glob))
    levels = full = mobs = seen = 0
    for lvl in dirs:
        if not (os.path.exists(os.path.join(lvl, "data.npz"))
                and os.path.exists(os.path.join(lvl, "data_dynamic.npz"))):
            continue
        names, seen_any, frac = check_level(lvl, args.max_dist)
        n, s = len(seen_any), int(seen_any.sum())
        levels += 1
        mobs += n
        seen += s
        full += int(s == n)
        tag = "OK  " if s == n else "MISS"
        missed = [f"{i}:{names[i]}" for i in range(n) if not seen_any[i]]
        line = f"[{tag}] {os.path.basename(lvl)}: {s}/{n} seen"
        if missed:
            line += "  missed: " + ", ".join(missed)
        print(line)
        if args.verbose:
            for i in range(n):
                print(f"        slot {i:2d} {names[i]:<22} seen in {100*frac[i]:5.1f}% of frames")

    print("-" * 64)
    if levels:
        print(f"levels: {levels} | fully-covered (all mobs seen): {full} "
              f"({100*full/levels:.0f}%) | mobs seen: {seen}/{mobs} ({100*seen/mobs:.0f}%)")
    else:
        print("No levels with both data.npz and data_dynamic.npz found for that glob.")


if __name__ == "__main__":
    main()
