"""Guided navigation controller for Craftium dataset generation.

Instead of a purely random policy (which stares at the sky, wanders into caves
and rarely faces the mobs), this reads the live per-frame mob + player state that
the `dynamic_agents` mod writes to
    <run_dir>/worlds/<world>/data_dynamic.jsonl
and steers the player so that, over the episode, it turns toward and approaches
each dynamic agent in turn, keeping them in frame. Movement/look are emitted as
ordinary discrete actions (same ones the loop records), so the (obs, action,
obs') transitions stay physically consistent for world-model training.

Everything is computed in Minetest NATIVE coordinates (x=East, y=Up, z=North)
because the log stores player and mobs in that same frame on the same line, so no
frame conversion (and no chance of a frame-mismatch bug) is needed here.

The mouse turn/pitch sign convention is auto-calibrated at runtime by watching
whether a commanded turn actually moved yaw/pitch the intended way, so the
controller is robust regardless of the engine's sign conventions.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque

import numpy as np


def _wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class GuidedNavigator:
    def __init__(
        self,
        run_dir: str,
        actions,                       # env.actions (list of per-group name lists)
        action_shape,                  # env.action_space.shape  -> (num_groups,)
        world_name: str = "world",
        fov_deg: float = 90.0,         # horizontal fov used for the "observed" cone
        pitch_limit_deg: float = 25.0,
        observe_dist: float = 18.0,    # a mob within this + in the fov cone counts as observed
        close_dist: float = 5.0,       # stop approaching once this close (avoid walking into it)
        stuck_window: int = 20,
        stuck_eps: float = 0.6,
    ):
        self.log_path = os.path.join(run_dir, "worlds", world_name, "data_dynamic.jsonl")
        self.action_shape = tuple(int(x) for x in action_shape)
        self.num_groups = self.action_shape[0] if self.action_shape else 4

        # Resolve action indices (loop convention: action[g]=0 is noop, and
        # env.actions[g][action[g]-1] is the name, so name_index + 1).
        def idx0(name, default):
            try:
                return actions[0].index(name) + 1
            except (ValueError, IndexError):
                return default
        self.a_forward   = idx0("forward", 1)
        self.a_jump      = idx0("jump", 9)
        self.a_fwd_jump  = idx0("forward+jump", 10)
        self.a_fwd_left  = idx0("forward+left", 5)
        self.a_fwd_right = idx0("forward+right", 6)
        # groups 2 (mouse x) and 3 (mouse y): option 1 and 2 are the two directions.
        self.MX_A, self.MX_B = 1, 2
        self.MY_A, self.MY_B = 1, 2

        self.fov = math.radians(fov_deg)
        self.pitch_limit = math.radians(pitch_limit_deg)
        self.observe_dist = observe_dist
        self.close_dist = close_dist

        self.observed = set()
        # Yaw-aim convention: +1 assumes look dir = (-sin yaw, cos yaw) (Minetest
        # default). If aiming is mirrored, the stall watchdog flips this so the
        # controller self-heals instead of spinning away from the mobs.
        self.conv = 1
        self.stall = 0
        self.pos_hist = deque(maxlen=stuck_window)
        self.stuck_eps = stuck_eps
        self.escape = 0
        self.escape_dir = 1

        # sign auto-calibration state
        self.mx_plus_incr = None      # does pressing MX_B increase yaw? (None=unknown)
        self.my_plus_incr = None
        self._prev_yaw = None
        self._prev_pitch = None
        self._last_mx = 0             # -1 pressed A, +1 pressed B, 0 none
        self._last_my = 0

    # --- reading the live log ------------------------------------------------
    def _read_latest_frame(self):
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 65536))
                chunk = f.read().decode("utf-8", "ignore")
        except (FileNotFoundError, OSError):
            return None
        for line in reversed(chunk.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind", "frame") == "frame" and "player_pos" in r:
                return r
        return None

    # --- helpers -------------------------------------------------------------
    def _desired_yaw(self, dx: float, dz: float) -> float:
        # Minetest look dir horizontal = (-sin(yaw), cos(yaw)) in (x=East, z=North).
        # `conv` flips the dx sign if the aim turns out mirrored (self-healed).
        return math.atan2(-self.conv * dx, dz)

    def _turn_action(self, want_increase_yaw: bool):
        """Return the group-2 option that moves yaw toward the target, given the
        calibrated sign; also record what we pressed for next-frame calibration."""
        if self.mx_plus_incr is None:
            press_B = True                       # unknown: just try B, calibrate next frame
        else:
            press_B = (want_increase_yaw == self.mx_plus_incr)
        self._last_mx = +1 if press_B else -1
        return self.MX_B if press_B else self.MX_A

    def _pitch_action(self, want_increase_pitch: bool):
        if self.my_plus_incr is None:
            press_B = True
        else:
            press_B = (want_increase_pitch == self.my_plus_incr)
        self._last_my = +1 if press_B else -1
        return self.MY_B if press_B else self.MY_A

    def _calibrate(self, yaw, pitch):
        if self._prev_yaw is not None and self._last_mx != 0:
            dyaw = _wrap(yaw - self._prev_yaw)
            if abs(dyaw) > 1e-3:
                incr = dyaw > 0
                # did pressing B (last_mx=+1) increase yaw?
                self.mx_plus_incr = incr if self._last_mx > 0 else (not incr)
        if self._prev_pitch is not None and self._last_my != 0:
            dp = pitch - self._prev_pitch
            if abs(dp) > 1e-4:
                incr = dp > 0
                self.my_plus_incr = incr if self._last_my > 0 else (not incr)

    # --- main step -----------------------------------------------------------
    def act(self):
        action = np.zeros(self.action_shape, dtype=np.int32)
        self._last_mx = 0
        self._last_my = 0

        rec = self._read_latest_frame()
        if rec is None:
            action[2] = self.MX_B            # no data yet -> slow scan
            self._last_mx = +1
            return action

        p = rec["player_pos"]
        px, py, pz = float(p["x"]), float(p.get("y", 0.0)), float(p["z"])
        yaw = float(rec.get("player", {}).get("yaw", 0.0))
        pitch = float(rec.get("player", {}).get("rotation", {}).get("x", 0.0)) \
            if isinstance(rec.get("player", {}).get("rotation"), dict) else 0.0

        self._calibrate(yaw, pitch)

        # collect present mobs, update "observed" set
        cand = []
        for a in rec.get("agents", []):
            if a.get("present", 0) != 1:
                continue
            ap = a.get("pos", {})
            ax, ay, az = float(ap.get("x", 0)), float(ap.get("y", 0)), float(ap.get("z", 0))
            dx, dz = ax - px, az - pz
            dist = math.hypot(dx, dz)
            yaw_err = _wrap(self._desired_yaw(dx, dz) - yaw)
            slot = a.get("slot")
            cand.append((slot, dist, yaw_err, ay - py))
            if dist <= self.observe_dist and abs(yaw_err) <= self.fov * 0.45:
                self.observed.add(slot)

        # Watchdog: if there are still-unobserved mobs but we make no progress for
        # a long stretch, the yaw aim is probably mirrored -> flip the convention
        # so the controller self-heals instead of steering away forever.
        n_present = len(cand)
        n_unobs = sum(1 for c in cand if c[0] not in self.observed)
        if getattr(self, "_last_obs_count", 0) < len(self.observed):
            self.stall = 0
        elif n_unobs > 0 and n_present > 0:
            self.stall += 1
        self._last_obs_count = len(self.observed)
        if self.stall > 250:
            self.conv = -self.conv
            self.stall = 0

        # stuck detection
        self.pos_hist.append((px, pz))
        stuck = False
        if len(self.pos_hist) == self.pos_hist.maxlen:
            xs = [q[0] for q in self.pos_hist]
            zs = [q[1] for q in self.pos_hist]
            if (max(xs) - min(xs)) < self.stuck_eps and (max(zs) - min(zs)) < self.stuck_eps:
                stuck = True

        # escape maneuver (turn + strafe-forward for a few frames to get unstuck)
        if self.escape > 0:
            self.escape -= 1
            action[0] = self.a_fwd_left if self.escape_dir > 0 else self.a_fwd_right
            action[2] = self._turn_action(self.escape_dir > 0)
            self._prev_yaw, self._prev_pitch = yaw, pitch
            return action
        if stuck:
            self.escape = 14
            self.escape_dir = -self.escape_dir
            self.pos_hist.clear()

        # choose target: nearest not-yet-observed mob, else nearest overall
        unobs = [c for c in cand if c[0] not in self.observed]
        pool = unobs if unobs else cand
        if not pool:
            action[2] = self._turn_action(True)     # nothing seen yet -> scan
            self._prev_yaw, self._prev_pitch = yaw, pitch
            return action
        slot, dist, yaw_err, dy = min(pool, key=lambda c: c[1])

        # --- yaw: turn toward the target ---
        if abs(yaw_err) > 0.12:
            action[2] = self._turn_action(yaw_err > 0)

        # --- pitch: aim a touch toward the target's height, clamped near horizon ---
        desired_pitch = math.atan2(dy, max(dist, 1e-3))
        desired_pitch = max(-self.pitch_limit, min(self.pitch_limit, desired_pitch))
        # engine pitch sign unknown; drive current pitch (rec player pitch) toward desired
        if abs(desired_pitch - pitch) > math.radians(4):
            action[3] = self._pitch_action(desired_pitch > pitch)

        # --- movement: walk toward the target once roughly facing it ---
        if abs(yaw_err) < 0.6 and dist > self.close_dist:
            action[0] = self.a_forward
        # if close, don't advance (keep it framed); movement stays 0

        self._prev_yaw, self._prev_pitch = yaw, pitch
        return action
