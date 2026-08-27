"""Utilities to read and align the Craftium `data_dynamic` log.

Craftium (when `dynamic_agents_enable` is set) writes one JSON record per
server step to `<world>/data_dynamic.jsonl`. Each record holds the current
player position and the state of the tracked dynamic agents (mobs/animals,
currently sheep). This module reads that log, aligns it to the frames that
were actually collected for a level (by matching the recorded player position
against `level_data["player_pos"]`), converts positions from Minetest's native
axes to the ENU convention used by the rest of the dataset, and packs
everything into fixed-shape numpy arrays saved as `data_dynamic.npz`.

The log is a *superset* of the collected frames (it also covers the init
frames), so we search for the contiguous window whose player-position sequence
best matches the collected one. This makes alignment robust to the exact number
of initialization frames.
"""

import json
import os
from typing import Optional

import numpy as np
from loguru import logger

# Minetest native player axes are (x=east, y=up, z=north). The rest of the
# dataset uses ENU = (east, north, up), obtained by reindexing [0, 2, 1]
# (this mirrors `NueToEnuVoxelObs` in the craftium wrappers).
_MT_TO_ENU = [0, 2, 1]


def _xyz(d):
    return [d["x"], d["y"], d["z"]]


def read_dynamic_log(path: os.PathLike) -> list[dict]:
    """Read the `data_dynamic.jsonl` file, returning the list of records."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A partially-flushed final line can happen if the process was
                # killed mid-write; just drop it.
                logger.warning("Skipping malformed line in dynamic log")
    return records


def _align_offset(log_player_pos: np.ndarray, target_player_pos: np.ndarray) -> tuple[int, float]:
    """Find the offset of the best-matching contiguous window.

    `log_player_pos` is [M, 3] (ENU) and `target_player_pos` is [T, 3] (ENU).
    Returns (offset, rmse) such that log_player_pos[offset:offset+T] best
    matches target_player_pos.
    """
    M = log_player_pos.shape[0]
    T = target_player_pos.shape[0]
    if M < T:
        return 0, float("inf")
    best_off, best_err = 0, float("inf")
    for off in range(0, M - T + 1):
        window = log_player_pos[off:off + T]
        err = float(np.mean(np.sum((window - target_player_pos) ** 2, axis=1)))
        if err < best_err:
            best_err = err
            best_off = off
    return best_off, float(np.sqrt(best_err))


def _to_enu(a: np.ndarray) -> np.ndarray:
    """Reindex the last axis from Minetest (x=east, y=up, z=north) to ENU."""
    return a[..., _MT_TO_ENU]


def _compute_boxes(pos_mt: np.ndarray, cbox: np.ndarray, yaw: np.ndarray):
    """Compute bounding-box ground truth from Minetest-native quantities.

    `pos_mt` [T, N, 3], `cbox` [T, N, 6] = (x1,y1,z1,x2,y2,z2) relative to pos
    (Minetest axes), `yaw` [T, N] radians about the up axis.

    Returns (obb_corners_enu [T, N, 8, 3], aabb_min_enu [T, N, 3],
    aabb_max_enu [T, N, 3]) in ENU world coordinates. The 8 corners are indexed
    by bits: bit0 -> x2 else x1, bit1 -> y2 else y1, bit2 -> z2 else z1 (in
    Minetest axes, before the ENU reindex).
    """
    T, N = pos_mt.shape[0], pos_mt.shape[1]
    x1, y1, z1, x2, y2, z2 = [cbox[..., i] for i in range(6)]
    xs = np.stack([x1, x2], axis=-1)  # [T, N, 2]
    ys = np.stack([y1, y2], axis=-1)
    zs = np.stack([z1, z2], axis=-1)

    local = np.zeros((T, N, 8, 3), dtype=np.float64)
    for c in range(8):
        local[..., c, 0] = xs[..., (c >> 0) & 1]
        local[..., c, 1] = ys[..., (c >> 1) & 1]
        local[..., c, 2] = zs[..., (c >> 2) & 1]

    # Oriented box: rotate the (x, z) plane about the up (y) axis by yaw.
    cos_y = np.cos(yaw)[..., None]  # [T, N, 1]
    sin_y = np.sin(yaw)[..., None]
    lx, ly, lz = local[..., 0], local[..., 1], local[..., 2]
    rx = lx * cos_y - lz * sin_y
    rz = lx * sin_y + lz * cos_y
    rotated = np.stack([rx, ly, rz], axis=-1)  # [T, N, 8, 3], MT axes
    obb_mt = pos_mt[:, :, None, :] + rotated
    obb_enu = _to_enu(obb_mt).astype(np.float32)

    # Axis-aligned box (no rotation): pos + cbox min/max, then to ENU.
    lo_mt = pos_mt + cbox[..., 0:3]
    hi_mt = pos_mt + cbox[..., 3:6]
    lo_enu = _to_enu(lo_mt)
    hi_enu = _to_enu(hi_mt)
    aabb_min = np.minimum(lo_enu, hi_enu).astype(np.float32)
    aabb_max = np.maximum(lo_enu, hi_enu).astype(np.float32)
    return obb_enu, aabb_min, aabb_max


def build_dynamic_arrays(
    records: list[dict],
    target_player_pos: np.ndarray,
    num_agents: Optional[int] = None,
    entity_name: str = "mobs_mc:sheep",
) -> Optional[dict]:
    """Align `records` to `target_player_pos` [T, 3] (ENU) and pack arrays.

    Returns a dict of numpy arrays (or None if the log is empty).
    """
    # Separate the one-off "meta" records (static visual info) from per-frame
    # records. Records written before this feature have no "kind" key -> frame.
    meta_rec = None
    frames = []
    for r in records:
        if r.get("kind") == "meta":
            meta_rec = r  # keep the last meta record seen
        else:
            frames.append(r)

    if len(frames) == 0:
        logger.warning("Dynamic log has no frame records, no data_dynamic will be saved")
        return None

    T = int(target_player_pos.shape[0])
    if num_agents is None:
        num_agents = int(frames[0].get("num_agents", len(frames[0].get("agents", []))))
    N = int(num_agents)

    # Player positions from the log, converted to ENU for matching.
    log_pp = np.array([_xyz(r["player_pos"]) for r in frames], dtype=np.float64)
    log_pp_enu = log_pp[:, _MT_TO_ENU]

    offset, rmse = _align_offset(log_pp_enu, np.asarray(target_player_pos, dtype=np.float64))
    if offset == 0 and len(frames) < T:
        logger.warning(
            f"Dynamic log shorter ({len(frames)}) than collected frames ({T}); "
            f"trailing frames will be marked absent"
        )
    logger.info(f"Aligned dynamic log with offset={offset}, player_pos rmse={rmse:.4f}")

    present = np.zeros((T, N), dtype=np.int8)
    pos = np.zeros((T, N, 3), dtype=np.float32)          # ENU
    pos_mt = np.zeros((T, N, 3), dtype=np.float64)        # Minetest native (for boxes)
    vel = np.zeros((T, N, 3), dtype=np.float32)           # ENU
    yaw = np.zeros((T, N), dtype=np.float32)
    rotation = np.zeros((T, N, 3), dtype=np.float32)      # (pitch, yaw, roll), MT axes
    cbox = np.zeros((T, N, 6), dtype=np.float64)          # collisionbox rel to pos, MT axes
    hp = np.zeros((T, N), dtype=np.float32)
    sheared = np.zeros((T, N), dtype=np.int8)
    baby = np.zeros((T, N), dtype=np.int8)
    color = np.empty((T, N), dtype=object)
    color[:] = ""
    frame_time = np.zeros((T,), dtype=np.float32)
    names = np.array([entity_name] * N, dtype=object)

    for t in range(T):
        idx = offset + t
        if idx >= len(frames):
            break
        rec = frames[idx]
        frame_time[t] = float(rec.get("time", 0.0))
        for agent in rec.get("agents", []):
            slot = int(agent.get("slot", 0)) - 1  # slots are 1-indexed in Lua
            if slot < 0 or slot >= N:
                continue
            present[t, slot] = int(agent.get("present", 0))
            if present[t, slot]:
                p_mt = np.array(_xyz(agent["pos"]), dtype=np.float64)
                pos_mt[t, slot] = p_mt
                pos[t, slot] = p_mt[_MT_TO_ENU].astype(np.float32)
                vel[t, slot] = np.array(_xyz(agent["vel"]), dtype=np.float32)[_MT_TO_ENU]
                yaw[t, slot] = float(agent.get("yaw", 0.0))
                if "rotation" in agent:
                    rotation[t, slot] = _xyz(agent["rotation"])
                cb = agent.get("collisionbox")
                if cb is not None and len(cb) == 6:
                    cbox[t, slot] = cb
                hp[t, slot] = float(agent.get("hp", 0.0))
                sheared[t, slot] = int(agent.get("sheared", 0))
                baby[t, slot] = int(agent.get("baby", 0))
                color[t, slot] = agent.get("color", "") or ""
                nm = agent.get("name")
                if nm:
                    names[slot] = nm

    # Position of each agent relative to the player (ENU), only where present.
    rel_pos = pos - np.asarray(target_player_pos, dtype=np.float32)[:, None, :]
    rel_pos = rel_pos * present[..., None]

    # Bounding-box ground truth (ENU world coordinates).
    obb_corners, aabb_min, aabb_max = _compute_boxes(pos_mt, cbox, yaw.astype(np.float64))
    obb_corners = obb_corners * present[..., None, None]
    aabb_min = aabb_min * present[..., None]
    aabb_max = aabb_max * present[..., None]

    out = {
        "dyn_present": present,          # [T, N] int8
        "dyn_pos": pos,                  # [T, N, 3] float32, ENU world coords (bottom-center)
        "dyn_vel": vel,                  # [T, N, 3] float32, ENU
        "dyn_yaw": yaw,                  # [T, N] float32, radians about up axis
        "dyn_rotation": rotation,        # [T, N, 3] float32, (pitch,yaw,roll) radians, MT axes
        "dyn_hp": hp,                    # [T, N] float32
        "dyn_rel_pos": rel_pos,          # [T, N, 3] float32, agent - player (ENU)
        "dyn_collisionbox": cbox.astype(np.float32),  # [T, N, 6] rel to pos, MT axes
        "dyn_obb_corners": obb_corners,  # [T, N, 8, 3] float32, ENU world OBB corners (yaw-oriented)
        "dyn_aabb_min": aabb_min,        # [T, N, 3] float32, ENU world axis-aligned box min
        "dyn_aabb_max": aabb_max,        # [T, N, 3] float32, ENU world axis-aligned box max
        "dyn_sheared": sheared,          # [T, N] int8 (mob state)
        "dyn_baby": baby,                # [T, N] int8 (mob state)
        "dyn_color": color,              # [T, N] object (wool/dye color where applicable)
        "dyn_names": names,              # [N] object (entity names)
        "dyn_frame_time": frame_time,    # [T] float32, minetest gametime
        "dyn_num_agents": np.array(N, dtype=np.int32),
        "dyn_entity_name": np.array(entity_name, dtype=object),
        "dyn_align_offset": np.array(offset, dtype=np.int32),
        "dyn_align_rmse": np.array(rmse, dtype=np.float32),
    }

    # Static per-slot visual metadata for later mesh drawing.
    mesh = np.array([""] * N, dtype=object)
    textures = np.empty(N, dtype=object)
    visual = np.array([""] * N, dtype=object)
    visual_size = np.ones((N, 3), dtype=np.float32)
    static_cbox = np.zeros((N, 6), dtype=np.float32)
    for n in range(N):
        textures[n] = []
    if meta_rec is not None:
        for agent in meta_rec.get("agents", []):
            slot = int(agent.get("slot", 0)) - 1
            if slot < 0 or slot >= N:
                continue
            mesh[slot] = agent.get("mesh", "") or ""
            textures[slot] = agent.get("textures", []) or []
            visual[slot] = agent.get("visual", "") or ""
            vs = agent.get("visual_size")
            if isinstance(vs, dict):
                visual_size[slot] = [vs.get("x", 1.0), vs.get("y", 1.0), vs.get("z", 1.0)]
            cb = agent.get("collisionbox")
            if cb is not None and len(cb) == 6:
                static_cbox[slot] = cb
    out.update({
        "dyn_mesh": mesh,                 # [N] object, model file name (e.g. mobs_mc_sheep.b3d)
        "dyn_textures": textures,         # [N] object, list of texture file names
        "dyn_visual": visual,             # [N] object, visual type (e.g. "mesh")
        "dyn_visual_size": visual_size,   # [N, 3] float32, model scale
        "dyn_collisionbox_static": static_cbox,  # [N, 6] float32, MT axes
    })
    return out


def collect_dynamic_data(
    run_dir: os.PathLike,
    target_player_pos: np.ndarray,
    world_name: str = "world",
    num_agents: Optional[int] = None,
    entity_name: str = "mobs_mc:sheep",
) -> Optional[dict]:
    """Read + align the dynamic log for a run, returning packed arrays or None.

    `run_dir` is the Minetest run directory (``env.unwrapped.mt.run_dir``).
    """
    log_path = os.path.join(run_dir, "worlds", world_name, "data_dynamic.jsonl")
    if not os.path.exists(log_path):
        logger.warning(f"No dynamic log found at {log_path}")
        return None
    records = read_dynamic_log(log_path)
    return build_dynamic_arrays(
        records, target_player_pos, num_agents=num_agents, entity_name=entity_name
    )
