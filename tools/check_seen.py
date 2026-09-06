#!/usr/bin/env python3
"""Check whether every dynamic agent is ACTUALLY visible on camera long enough.

Uses ground truth ONLY (no video decoding). A mob counts as "seen" on a frame if
its body point is:
  * in front of the camera,
  * comfortably inside the field of view (within --fov_frac of fov_x/fov_y), and
  * within --max_dist blocks, and
  * NOT hidden behind terrain (a raycast from the camera to the mob through the
    saved voxel grid finds only air/empty voxels between them).  <-- occlusion test
A mob is reported SEEN for the level only if that holds for >= --min_frames frames.

This matches what you see in rgb.mp4 far better than a plain frustum test: a mob
standing behind a hill or wall, or one that only clips the screen edge for a frame,
no longer counts.

Coordinate frames: cam_pos/cam_dir (data.npz), dyn_pos (data_dynamic.npz) and the
voxel grid (obs_voxel_mt, centered at obs_voxel_center) are all ENU (Y = up), 1
voxel = 1 block, so the geometry below is consistent.

Occlusion notes:
  * "empty" voxels = Minetest air/ignore/unknown (125,126,127) plus the single most
    common voxel value in the level (robust if air was remapped). Everything else
    (dirt, stone, trees, leaves, ...) blocks line of sight - so this is slightly
    CONSERVATIVE (it may under-count a mob seen through leaves), which is the safe
    direction. Disable with --no_occlusion to get the old frustum-only behaviour.
  * If a level has no voxel grid saved, occlusion is skipped automatically.

Usage:
  python tools/check_seen.py "datasets/train/raw/OpenWorldCreative-v0/*"
  python tools/check_seen.py "datasets/train/raw/OpenWorldCreative-v0/*" --verbose
  python tools/check_seen.py "..." --no_occlusion        # frustum only (old behaviour)
"""
import argparse
import glob
import os

import numpy as np

# Minetest fixed special content ids that are NOT solid (see mapnode.h).
_EMPTY_CONTENT = {125, 126, 127}   # CONTENT_UNKNOWN, CONTENT_AIR, CONTENT_IGNORE


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


def _load_voxels(d):
    """Return (vox [T,X,Y,Z] int, center [T,3] float) from a loaded data.npz, or None."""
    if "obs_voxel_mt" not in d or "obs_voxel_center" not in d:
        return None
    vox = d["obs_voxel_mt"]
    if vox.ndim == 5:          # (T,X,Y,Z,C) -> take the content-id channel
        vox = vox[..., 0]
    if vox.ndim != 4:
        return None
    center = d["obs_voxel_center"].astype(np.float64)
    return vox.astype(np.int64), center


def _empty_ids(vox):
    """Set of voxel values treated as empty (air): the fixed specials + the modal value."""
    ids = set(_EMPTY_CONTENT)
    sample = vox[:: max(1, vox.shape[0] // 5)]              # a few frames
    flat = sample[sample >= 0].ravel()
    if flat.size:
        ids.add(int(np.bincount(flat).argmax()))           # most common = air/open space
    return ids


def _ray_clear(cam, target, vox_t, center_t, origin_idx, empty_ids, step):
    """True if the segment cam->target passes only through empty voxels."""
    X, Y, Z = vox_t.shape
    d = target - cam
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return True
    n = max(2, int(dist / step))
    for s in range(1, n):                                   # skip both endpoints
        p = cam + d * (s / n)
        ix = int(round(p[0] - center_t[0])) + origin_idx[0]
        iy = int(round(p[1] - center_t[1])) + origin_idx[1]
        iz = int(round(p[2] - center_t[2])) + origin_idx[2]
        if ix < 0 or iy < 0 or iz < 0 or ix >= X or iy >= Y or iz >= Z:
            continue                                        # outside grid -> can't judge
        if int(vox_t[ix, iy, iz]) not in empty_ids:
            return False                                    # solid voxel blocks view
    return True


def check_level(lvl, max_dist, min_frames, fov_frac, use_occlusion, step):
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

    vox_data = _load_voxels(d) if use_occlusion else None
    if vox_data is not None:
        vox, vcenter = vox_data
        Tv = min(T, vox.shape[0])
        origin_idx = ((vox.shape[1] - 1) // 2, (vox.shape[2] - 1) // 2, (vox.shape[3] - 1) // 2)
        empty_ids = _empty_ids(vox)
    else:
        Tv = 0

    seen_any = np.zeros(N, bool)
    seen_count = np.zeros(N, int)
    for m in range(N):
        height = np.clip(box[:T, m, 4], 0.2, 4.0)
        mid = pos[:T, m].copy()
        mid[:, 1] += 0.5 * height                           # test the mid-body point
        vis = np.zeros(T, bool)
        for p in (pos[:T, m], mid):
            vis |= _visible(cam_pos[:T], cam_dir[:T], fov_x[:T], fov_y[:T], p, max_dist, fov_frac)
        vis &= present[:T, m]

        if vox_data is not None:
            # For frames that pass the frustum test, require a clear line of sight
            # from the camera to the mob's mid-body through the voxel grid.
            for t in np.nonzero(vis[:Tv])[0]:
                if not _ray_clear(cam_pos[t], mid[t], vox[t], vcenter[t], origin_idx, empty_ids, step):
                    vis[t] = False
            vis[Tv:] = False                                # no voxel data past Tv -> don't credit

        c = int(vis.sum())
        seen_count[m] = c
        seen_any[m] = c >= min_frames
    return names, seen_any, seen_count, (vox_data is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_glob", help='e.g. "datasets/train/raw/OpenWorldCreative-v0/*"')
    ap.add_argument("--max_dist", type=float, default=15.0,
                    help="max distance (blocks) a mob may be and still count as seen")
    ap.add_argument("--min_frames", type=int, default=10,
                    help="a mob must be visible in at least this many frames to count as seen")
    ap.add_argument("--fov_frac", type=float, default=0.9,
                    help="fraction of the FOV to count as 'in frame' (trims extreme-edge flickers)")
    ap.add_argument("--no_occlusion", action="store_true",
                    help="disable the terrain-occlusion raycast (frustum-only, old behaviour)")
    ap.add_argument("--vox_step", type=float, default=0.5,
                    help="raycast sampling step in blocks for the occlusion test")
    ap.add_argument("--verbose", action="store_true", help="print per-mob visible-frame counts")
    args = ap.parse_args()

    use_occlusion = not args.no_occlusion
    dirs = sorted(glob.glob(args.dataset_glob))
    levels = full = mobs = seen = 0
    any_vox = False
    for lvl in dirs:
        if not (os.path.exists(os.path.join(lvl, "data.npz"))
                and os.path.exists(os.path.join(lvl, "data_dynamic.npz"))):
            continue
        names, seen_any, seen_count, had_vox = check_level(
            lvl, args.max_dist, args.min_frames, args.fov_frac, use_occlusion, args.vox_step)
        any_vox = any_vox or had_vox
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
        occ = "ON (terrain-aware)" if (use_occlusion and any_vox) else "OFF (frustum only)"
        print(f"levels: {levels} | fully-covered (all mobs seen): {full} "
              f"({100*full/levels:.0f}%) | mobs seen: {seen}/{mobs} ({100*seen/mobs:.0f}%)")
        print(f"criteria: within {args.max_dist:g} blocks, inside {args.fov_frac:g}x FOV, "
              f">= {args.min_frames} frames, occlusion {occ}")
        if use_occlusion and not any_vox:
            print("NOTE: no voxel grid found in these levels; occlusion could not be tested.")
    else:
        print("No levels with both data.npz and data_dynamic.npz found for that glob.")


if __name__ == "__main__":
    main()
