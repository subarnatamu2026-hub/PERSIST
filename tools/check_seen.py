#!/usr/bin/env python3
"""Check whether every dynamic agent is ACTUALLY on camera long enough.

Uses ground truth ONLY (no video decoding). A mob counts as "seen" on a frame if
its body point is:
  * in front of the camera,
  * comfortably inside the field of view (within --fov_frac of fov_x/fov_y, so a
    mob clipping the extreme screen edge does NOT count), and
  * within --max_dist blocks (close enough to actually make out).
A mob is reported as SEEN for the level only if that holds for at least
--min_frames frames (default 10) - a one-frame fly-by at the edge no longer
counts. This is what makes the reported count match what you see in rgb.mp4.

Coordinate frame: cam_pos / cam_dir (data.npz) and dyn_pos (data_dynamic.npz)
share the same world frame with Y = up.

Occlusion caveat: this tests the camera *frustum + distance + dwell*, not whether
terrain hides the mob, so it can still slightly OVER-count a mob standing right
behind a hill. Keeping mobs close (they spawn within ~10 blocks, leashed to 12)
makes that rare. Lower --max_dist / raise --min_frames to be stricter.

Usage:
  python tools/check_seen.py "datasets/train/raw/OpenWorldCreative-v0/*"
  python tools/check_seen.py "datasets/train/raw/OpenWorldCreative-v0/*" --min_frames 15 --max_dist 12
"""
import argparse
import glob
import os

import numpy as np


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-9, 1.0, n)


def _visible(cam_pos, cam_dir, fov_x, fov_y, pts, max_dist, fov_frac):
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
    return ((z > 0.05) & (dist <= max_dist)
            & (ha <= fov_x * 0.5 * fov_frac)
            & (va <= fov_y * 0.5 * fov_frac))


def check_level(lvl, max_dist, min_frames, fov_frac):
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
    seen_count = np.zeros(N, int)
    for m in range(N):
        height = np.clip(box[:T, m, 4], 0.2, 4.0)
        mid = pos[:T, m].copy()
        mid[:, 1] += 0.5 * height                           # also test mid-body point
        vis = np.zeros(T, bool)
        for p in (pos[:T, m], mid):
            vis |= _visible(cam_pos[:T], cam_dir[:T], fov_x[:T], fov_y[:T], p, max_dist, fov_frac)
        vis &= present[:T, m]
        c = int(vis.sum())
        seen_count[m] = c
        seen_any[m] = c >= min_frames                       # must be visible >= min_frames
    return names, seen_any, seen_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_glob", help='e.g. "datasets/train/raw/OpenWorldCreative-v0/*"')
    ap.add_argument("--max_dist", type=float, default=15.0,
                    help="max distance (blocks) a mob may be and still count as seen")
    ap.add_argument("--min_frames", type=int, default=10,
                    help="a mob must be visible in at least this many frames to count as seen")
    ap.add_argument("--fov_frac", type=float, default=0.9,
                    help="fraction of the FOV to count as 'in frame' (trims extreme-edge flickers)")
    ap.add_argument("--verbose", action="store_true", help="print per-mob visible-frame counts")
    args = ap.parse_args()

    dirs = sorted(glob.glob(args.dataset_glob))
    levels = full = mobs = seen = 0
    for lvl in dirs:
        if not (os.path.exists(os.path.join(lvl, "data.npz"))
                and os.path.exists(os.path.join(lvl, "data_dynamic.npz"))):
            continue
        names, seen_any, seen_count = check_level(lvl, args.max_dist, args.min_frames, args.fov_frac)
        n, s = len(seen_any), int(seen_any.sum())
        levels += 1
        mobs += n
        seen += s
        full += int(s == n)
        tag = "OK  " if s == n else "MISS"
        missed = [f"{i}:{names[i]}({seen_count[i]}f)" for i in range(n) if not seen_any[i]]
        line = f"[{tag}] {os.path.basename(lvl)}: {s}/{n} seen (>={args.min_frames}f)"
        if missed:
            line += "  missed: " + ", ".join(missed)
        print(line)
        if args.verbose:
            for i in range(n):
                print(f"        slot {i:2d} {names[i]:<22} visible in {seen_count[i]:4d} frames")

    print("-" * 64)
    if levels:
        print(f"levels: {levels} | fully-covered (all mobs seen): {full} "
              f"({100*full/levels:.0f}%) | mobs seen: {seen}/{mobs} ({100*seen/mobs:.0f}%)")
        print(f"criteria: within {args.max_dist:g} blocks, inside {args.fov_frac:g}x FOV, "
              f">= {args.min_frames} frames")
    else:
        print("No levels with both data.npz and data_dynamic.npz found for that glob.")


if __name__ == "__main__":
    main()
