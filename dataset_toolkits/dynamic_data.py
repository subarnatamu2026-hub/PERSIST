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


def build_dynamic_arrays(
    records: list[dict],
    target_player_pos: np.ndarray,
    num_agents: Optional[int] = None,
    entity_name: str = "mobs_mc:sheep",
) -> Optional[dict]:
    """Align `records` to `target_player_pos` [T, 3] (ENU) and pack arrays.

    Returns a dict of numpy arrays (or None if the log is empty).
    """
    if len(records) == 0:
        logger.warning("Dynamic log is empty, no data_dynamic will be saved")
        return None

    T = int(target_player_pos.shape[0])
    if num_agents is None:
        num_agents = int(records[0].get("num_agents", len(records[0].get("agents", []))))
    N = int(num_agents)

    # Player positions from the log, converted to ENU for matching.
    log_pp = np.array([_xyz(r["player_pos"]) for r in records], dtype=np.float64)
    log_pp_enu = log_pp[:, _MT_TO_ENU]

    offset, rmse = _align_offset(log_pp_enu, np.asarray(target_player_pos, dtype=np.float64))
    if offset == 0 and len(records) < T:
        logger.warning(
            f"Dynamic log shorter ({len(records)}) than collected frames ({T}); "
            f"trailing frames will be marked absent"
        )
    logger.info(f"Aligned dynamic log with offset={offset}, player_pos rmse={rmse:.4f}")

    present = np.zeros((T, N), dtype=np.int8)
    pos = np.zeros((T, N, 3), dtype=np.float32)
    vel = np.zeros((T, N, 3), dtype=np.float32)
    yaw = np.zeros((T, N), dtype=np.float32)
    hp = np.zeros((T, N), dtype=np.float32)
    frame_time = np.zeros((T,), dtype=np.float32)
    names = np.array([entity_name] * N, dtype=object)

    for t in range(T):
        idx = offset + t
        if idx >= len(records):
            break
        rec = records[idx]
        frame_time[t] = float(rec.get("time", 0.0))
        for agent in rec.get("agents", []):
            slot = int(agent.get("slot", 0)) - 1  # slots are 1-indexed in Lua
            if slot < 0 or slot >= N:
                continue
            present[t, slot] = int(agent.get("present", 0))
            if present[t, slot]:
                p = np.array(_xyz(agent["pos"]), dtype=np.float32)[_MT_TO_ENU]
                v = np.array(_xyz(agent["vel"]), dtype=np.float32)[_MT_TO_ENU]
                pos[t, slot] = p
                vel[t, slot] = v
                yaw[t, slot] = float(agent.get("yaw", 0.0))
                hp[t, slot] = float(agent.get("hp", 0.0))
                nm = agent.get("name")
                if nm:
                    names[slot] = nm

    # Position of each agent relative to the player (ENU), only where present.
    rel_pos = pos - np.asarray(target_player_pos, dtype=np.float32)[:, None, :]
    rel_pos = rel_pos * present[..., None]

    return {
        "dyn_present": present,        # [T, N] int8
        "dyn_pos": pos,                # [T, N, 3] float32, ENU world coords
        "dyn_vel": vel,                # [T, N, 3] float32, ENU
        "dyn_yaw": yaw,                # [T, N] float32, radians about up axis
        "dyn_hp": hp,                  # [T, N] float32
        "dyn_rel_pos": rel_pos,        # [T, N, 3] float32, agent - player (ENU)
        "dyn_names": names,            # [N] object (entity names)
        "dyn_frame_time": frame_time,  # [T] float32, minetest gametime
        "dyn_num_agents": np.array(N, dtype=np.int32),
        "dyn_entity_name": np.array(entity_name, dtype=object),
        "dyn_align_offset": np.array(offset, dtype=np.int32),
        "dyn_align_rmse": np.array(rmse, dtype=np.float32),
    }


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
