# fmt:off
import copy
import json
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import git
import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
import tyro
import utils3d
from loguru import logger
from xvfbwrapper import Xvfb

# Make both the repo root (for gym_envs/utils) and this script's directory
# (for dynamic_data) importable regardless of how the script is launched.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gym_envs.craftium.craftium.wrappers import NueToEnuVoxelObs, enu_to_nue
from utils import get_file_hash, seed_everything
from utils.action_util import MultiDiscreteActionWrapper
from dynamic_data import collect_dynamic_data
from guided_nav import GuidedNavigator

# Reuse the QA visibility logic (frustum + distance + FOV + terrain occlusion) so
# the in-generation "min visibility" acceptance gate matches tools/check_seen.py
# exactly. Guarded so generation still runs if the tools package can't be imported.
try:
    from tools.check_seen import (
        _visible as _cs_visible,
        _ray_clear as _cs_ray_clear,
        _empty_ids as _cs_empty_ids,
        _load_voxels as _cs_load_voxels,
    )
    _HAVE_CHECK_SEEN = True
except Exception:  # noqa: BLE001
    _HAVE_CHECK_SEEN = False

EMPTY_SPACE_NODE_IDS = [126, 127]


@dataclass
class Args:
    init: bool = False
    """If True, only the dataset_params.json and level_seeds.txt files will be generated at the root directory. 
    Script must be run again with this set to False to start generating the data."""

    dataset_name: str = "testdataset"

    dataset_dir: str = "datasets"

    env_id: str = "OpenWorldCreative-v0"

    mt_port: int = 49152
    """TCP port used by Minetest server and client communication. Multiple envs will use successive ports."""

    mt_run_dir: str = "outputs/mt_runs"
    """Directory where the Minetest working directories will be created (defaults to the current one)"""

    seed: int = 0
    """Random seed that will be used to generate new downstream seeds for each environment"""

    num_levels: int = 100_000
    """Number of levels to generate"""

    ep_timesteps: int = 400
    "number of timesteps for each episode (1 episode per level)"

    voxel_obs_rx: int = 24
    "x radius of the voxel observation"

    voxel_obs_ry: int = 24
    "y radius of the voxel observation"

    voxel_obs_rz: int = 24
    "z radius of the voxel observation"

    resolution_w: int = 640
    "resolution of the generated RGB observations"

    resolution_h: int = 360
    "resolution of the generated RGB observations"

    fov: int = 72  # we get an horizontal fov of ~105 degrees with 16:9 aspect ratio
    "vertical field of view of the generated RGB observations in degrees"

    starting_inventory: str = "mcl_core:wood,mcl_core:sand,mcl_core:glass,mcl_doors:wooden_door,mcl_core:sandstone,mcl_core:brick_block,mcl_torches:torch,mcl_core:cobble,mcl_core:stone"
    "Starting inventory for the agent, as a comma-separated list of items."
    # 'mcl_core:stone,mcl_torches:torch,mcl_core:cobble,mcl_core:stripped_oak,mcl_core:dirt,mcl_core:wood,mcl_core:sandstone,mcl_core:brick_block,mcl_core:sand,mcl_core:glass,mcl_stairs:stair_wood,mcl_doors:wooden_door'

    randomize_world_start_time: bool = False
    "Randomize the time of day the episode starts at"

    randomize_inventory: bool = False
    "If True, the number of items and their order will be randomized (but sampled from the starting_inventory list)."

    init_frames: int = 200
    "number of frames to wait before starting the episode"

    fps_max: int = 24
    "target fps for the environment"

    pmul: int = 1
    """Physics speed multiplier. Defaults to the default value of CraftiumEnv."""

    rank: int = 0
    """Process rank"""

    world_size: int = 1
    """Total number of processes"""

    overwrite_init: bool = False
    """If True, dataset_params.json and level_seeds.txt will be overwritten when --init flag is provided"""

    overwrite_leveldata: bool = False
    """If True, existing data for a given level will be overwritten"""

    disable_commit_check: bool = False
    """If True, a mismatch between the current codebase commits and the commits recorded in dataset_params.json will only trigger a warning."""

    regen_sha256_only: bool = False
    """If True, only the sha256.txt file will be regenerated for existing levels"""

    compute_extrinsics_only: bool = False
    """If True, only the extrinsics will be re-computed and saved to the dataset"""

    compute_extrinsics_while_collecting: bool = True
    """If True, extrinsics will be computed while collecting data and saved to the dataset."""

    gen_sha256_while_collecting: bool = True
    """If True, sha256 will be generated while collecting data and saved to the dataset."""

    no_cuda: bool = False
    """If True, will not use CUDA even if available. Otherwise GPU will be used to process data if CUDA is available."""

    debug: bool = False
    """Enable debug mode. Will run the code in a single process."""

    navigation_only: bool = True
    """If True, the agent only performs navigation actions (walk, strafe, jump, sneak,
    sprint and looking around). Interaction actions (dig/place, and therefore hitting
    mobs) are disabled. Set to False to restore the original mixed action distribution."""

    guided_navigation: bool = False
    """If True, AFTER the spin the player is driven by a goal-directed controller that
    turns toward / approaches each mob in turn. DISABLED by default: it makes the player
    rotate continuously (turning to face each mob), which is not wanted. With it off the
    player observes its surroundings with ONE 360 spin (player_spin_once) and otherwise
    just does its normal navigation task - no continuous rotation."""

    player_observe_hold_frames: int = 30
    """At the very START of the episode the player stands still and looks level at the
    forward cluster of freshly-spawned mobs for this many frames, so every mob (spawned
    in-view with a clear line of sight) is captured for >= min_frames before the player
    wanders off. Set 0 to disable. This is the main lever that makes ALL mobs visible."""

    player_spin_once: bool = True
    """If True, once per episode (at a random time - see player_spin_min/max_seconds)
    the player stands still and turns a single full 360 degrees to observe its
    surroundings, then hands over to the guided tour (or random nav). Pure horizontal
    turn - it never looks up/down - and spread over player_spin_seconds so it is slow."""

    player_spin_seconds: float = 3.0
    """How long the single 360 spin takes, in seconds (default ~3s = a natural,
    human-speed look-around). The turn is spread evenly across this many frames (turning
    only on the frames needed to track a linear ramp to 360 deg, standing still
    otherwise). Lower = faster spin; higher = slower."""

    player_spin_min_seconds: float = 2.0
    """Earliest time (seconds into the episode) the one-off 360 spin may start."""

    player_spin_max_seconds: float = 5.0
    """Latest time (seconds into the episode) the one-off 360 spin may start. The exact
    start time is drawn uniformly in [min, max] seconds, independently per level, so it
    varies across the dataset. Converted to frames via fps_max."""

    player_spin_max_frames: int = 240
    """Hard safety cap on how many frames the 360 spin may take (must exceed
    player_spin_seconds * fps_max; the spin normally ends on its own at ~360 deg)."""

    camera_pitch_limit_deg: float = 35.0
    """Keep the camera pitch within +-this many degrees of the horizon, so the player
    looks around the environment (where the mobs are) instead of drifting up and
    staring at the sky (or down at the ground) for long stretches. Larger = more
    freedom to look up/down; smaller = more strictly horizontal."""

    clean_rgb: bool = True
    """If True, hide the HUD (hotbar, health/breath bars, etc.) and the first-person
    wielded hand/item from the RGB observation. Purely visual; does not change actions,
    observations or the ground-truth data. Set to False to keep the default HUD."""

    collect_dynamic_data: bool = True
    """If True, spawn dynamic agents (mobs/animals) in Craftium and save their per-frame
    state to an extra `data_dynamic.npz` file alongside `data.npz` for each level."""

    disable_world_mob_spawning: bool = True
    """If True, disable VoxeLibre's own natural mob-spawning system (mobs_spawn=false)
    so the ONLY mobs in the world are the fixed set our craftium_env mod places. Without
    this the game keeps spawning ambient animals/monsters all over the terrain, flooding
    the scene with far more than the intended count. Our mod uses direct add_entity, which
    is unaffected by this setting."""

    dynamic_agents_spawn_once: bool = True
    """If True (default), the full set of mobs is spawned ONCE at the start and no mob
    is ever created or relocated again mid-episode - so the environment never changes
    after the start (no new mob appears behind/around the player). Overrides the
    respawn top-up and the leash relocation failsafe."""

    dynamic_agents_free_roam: bool = False
    """If True, mobs wander freely with their own AI: NO leash (they are never pulled
    back to the player) and NO respawn top-up. If False (default now), a SOFT leash
    keeps them near the player - a mob that strays beyond the leash radius gently walks
    back (no teleport under spawn_once) - so all mobs stay close and observable for the
    whole clip. Set True for natural/uncontrolled trajectories that may leave the frame."""

    num_dynamic_agents: int = 8
    """Number of dynamic agents to spawn (used when min==max). Superseded by the
    min/max range below when they differ."""

    num_dynamic_agents_min: int = 7
    """Minimum number of dynamic agents per level (randomized per level)."""

    num_dynamic_agents_max: int = 10
    """Maximum number of dynamic agents per level (randomized per level). Each level
    picks a random count in [min, max] (default 7-10), seeded by the level seed, so the
    number of mobs varies per level; the *mix* of species varies too because each slot
    draws a random entity from the seen/unseen list. Set min==max to fix the count."""

    dynamic_agents_leash_radius: float = 12.0
    """Soft leash: mobs wander freely within this horizontal distance of the player;
    a mob that strays beyond it gently walks back (velocity nudge, no teleport under
    spawn_once), so mobs stay close and observable for the whole clip. 0 disables."""

    dynamic_agents_min_radius: float = 3.0
    """Inner radius of the ring mobs are spawned into around the player."""

    dynamic_agents_max_radius: float = 9.0
    """Outer radius of the spawn ring (kept small so mobs spawn CLOSE to the player and
    within its view; with spawn_in_view they land inside the forward cone with a clear
    line of sight to the player)."""

    dynamic_agents_min_separation: float = 2.0
    """Minimum horizontal spacing between mobs at spawn, so the herd stays sparse but
    still fits inside the forward view cone (small enough that up to 10 mobs can be
    placed in-view with a clear line of sight)."""

    dynamic_agents_max_speed: float = 0.0
    """Optional cap on a mob's horizontal speed (blocks/second). Default 0 = NO cap,
    so every mob moves at its NATIVE VoxeLibre speed (normal game behaviour). Set > 0
    only to slow unusually fast mobs: it then lowers each mob's configured walk/run
    speed AND hard-clamps its velocity every frame. Vertical (fall/jump) is untouched."""

    dynamic_agents_view_half_angle: float = 65.0
    """Half-angle (degrees) of the player's view cone used to decide whether a spawn/
    relocation spot is on-screen. Mobs are only ever spawned or relocated OUTSIDE this
    cone (prefer behind the player), so the player never sees a mob appear or teleport
    -- it only discovers mobs by turning toward them. Wider = more conservative."""

    dynamic_agents_spawn_in_view: bool = True
    """If True, the whole INITIAL population is spawned inside the player's forward view
    cone so every mob starts on-screen (instead of scattered all around). Mid-episode
    respawns, if any, are still placed off-screen. Falls back to anywhere-around if the
    cone can't fit all the mobs."""

    dynamic_agents_spawn_view_half_angle: float = 40.0
    """Half-angle (degrees) of the forward cone the initial mobs are spawned into. Kept
    a little narrower than the real horizontal FOV (~52 deg at fov=72, 16:9) so the mobs
    render comfortably inside the frame rather than clipping at the edges."""

    dynamic_agent_entity: str = "mobs_mc:sheep"
    """Single entity id, used only if `dynamic_agent_entities` is empty."""

    dynamic_agent_entities: str = (
        # Passive / neutral land animals
        "mobs_mc:sheep,mobs_mc:cow,mobs_mc:pig,mobs_mc:chicken,mobs_mc:rabbit,"
        "mobs_mc:mooshroom,mobs_mc:horse,mobs_mc:donkey,mobs_mc:mule,mobs_mc:llama,"
        "mobs_mc:wolf,mobs_mc:dog,mobs_mc:cat,mobs_mc:ocelot,mobs_mc:polar_bear,"
        "mobs_mc:killer_bunny,mobs_mc:skeleton_horse,mobs_mc:zombie_horse,"
        "mobs_mc:iron_golem,mobs_mc:snowman,mobs_mc:villager,"
        # Hostile *bodies*, neutralized to wander like animals (no attacking, no HP loss)
        "mobs_mc:zombie,mobs_mc:baby_zombie,mobs_mc:husk,mobs_mc:baby_husk,"
        "mobs_mc:skeleton,mobs_mc:stray,mobs_mc:witherskeleton,mobs_mc:silverfish,"
        "mobs_mc:endermite,mobs_mc:spider,mobs_mc:cave_spider,mobs_mc:villager_zombie,"
        "mobs_mc:zombified_piglin,mobs_mc:baby_zombified_piglin,mobs_mc:pigman,"
        "mobs_mc:baby_pigman,mobs_mc:piglin,mobs_mc:piglin_brute,mobs_mc:sword_piglin,"
        "mobs_mc:hoglin,mobs_mc:baby_hoglin,mobs_mc:zoglin,mobs_mc:vindicator,"
        "mobs_mc:pillager,mobs_mc:slime_big,mobs_mc:slime_small,mobs_mc:slime_tiny,"
        "mobs_mc:magma_cube_big,mobs_mc:magma_cube_small,mobs_mc:magma_cube_tiny"
    )
    """Comma-separated VoxeLibre entity ids to mix. Each agent slot is randomly
    assigned one of these per level (seeded). This default is a LAND-only set curated
    from a VoxeLibre build's registered mobs: passive animals plus hostile bodies that
    are neutralized (see --neutralize_agents) to wander with no attacking and no HP
    loss. Excluded on purpose: water mobs, flyers (bat/parrot/blaze/ghast/vex), bosses
    (enderdragon/wither), lava striders, wall-mounted shulkers, summoners/casters
    (evoker/illusioner/witch) and creeper-type stalkers. Any id your build lacks is
    filtered out automatically. Adjust freely to your build's ids."""

    neutralize_agents: bool = True
    """If True, every spawned mob (including hostile species) is reconfigured to
    behave like a passive land animal: no attacking/chasing, no self-destruct, and
    no death from sunlight/fire/water - it just wanders. Uses the hostile mesh/body
    but animal behavior. Set False to keep each mob's native AI."""

    spawn_on_land: bool = True
    """If True, relocate the player onto dry, solid ground at episode start so it
    does not spawn in a water body (and jump in place). Terrain-only navigation."""

    keep_on_land: bool = True
    """If True, the mod records per frame whether the player is on/over water (used by
    skip_on_water below). It no longer teleports the player at the water's edge - that
    snap-back was what made the camera shake - so a level where the player DOES reach
    water is instead discarded and reseeded (see skip_on_water)."""

    skip_on_water: bool = True
    """If True, after generating a level, discard it (save nothing) and retry with a
    fresh random seed when the player entered the water at any frame OR the camera shook
    violently (a large frame-to-frame position jump). Guarantees the kept dataset has no
    underwater / violently-shaking clips. Retries up to max_reseed_attempts times."""

    max_reseed_attempts: int = 12
    """Max fresh seeds to try for one level slot before giving up (avoids an infinite
    loop if a rank keeps landing near water or failing the visibility gate)."""

    shake_max_step: float = 2.0
    """A level is 'violently shaking' (and is discarded) if the player's horizontal
    position jumps more than this many blocks between two consecutive frames - far above
    normal walking/sprinting (~0.25/frame), so only physics glitches/teleports trip it."""

    require_min_visibility: bool = True
    """If True, a level is only ACCEPTED when enough of its mobs are actually seen on
    camera (frustum + distance + FOV + terrain-occlusion, same test as check_seen); a
    level with too many unseen mobs is discarded and the slot is retried with a fresh
    seed. Guarantees the kept dataset has good mob coverage."""

    max_unseen_allowed: int = 2
    """Accept a level only if at most this many mobs are NOT seen. With the default 2
    that means >= N-2 mobs seen: 8/10, 7/9, 6/8, 5/7, ... Otherwise the level is
    discarded and reseeded. Set 0 to require EVERY mob be seen, 1 for >= N-1."""

    visibility_min_frames: int = 10
    """A mob counts as 'seen' for the acceptance gate only if visible in >= this many
    frames (matches check_seen --min_frames)."""

    visibility_max_dist: float = 15.0
    """Max distance (blocks) a mob may be and still count toward the acceptance gate
    (matches check_seen --max_dist)."""

    visibility_fov_frac: float = 0.9
    """Fraction of the FOV a mob must be inside to count toward the acceptance gate
    (matches check_seen --fov_frac)."""

    water_avoid_radius: float = 12.0
    """Spawn the player at least this many blocks from any water, so episodes don't
    start on a shoreline."""

    water_lookahead: float = 12.0
    """During the episode, detect water within this many blocks ahead and steer the
    player away before it reaches the shore (smooth velocity nudge, no teleport)."""

    water_push_strength: float = 2.5
    """How strongly the player is pushed away from nearby water (blocks/second at the
    edge of the look-ahead; stronger the closer the water). Higher = turns away harder."""


def make_env(craftium_kwargs, mt_port_offset):
    (
        craftium_kwargs["voxel_obs_rx"],
        craftium_kwargs["voxel_obs_ry"],
        craftium_kwargs["voxel_obs_rz"],
    ) = enu_to_nue(
        craftium_kwargs["voxel_obs_rx"],
        craftium_kwargs["voxel_obs_ry"],
        craftium_kwargs["voxel_obs_rz"],
    )
    craftium_kwargs["mt_port"] += mt_port_offset
    craftium_kwargs["enable_voxel_obs"] = True

    env = gym.make(**craftium_kwargs)
    env = MultiDiscreteActionWrapper(env.env)
    env = NueToEnuVoxelObs(env)
    return env


def get_codebase_commits():
    voxel_wm_rel_path = "."
    craftium_rel_path = "gym_envs/craftium"
    minetest_rel_path = "gym_envs/craftium/craftium-envs/minetest_game"
    voxel_libre2_rel_path = "gym_envs/craftium/craftium-envs/common_games/mineclone2"

    # Function to get repository information
    def get_repo_info(path):
        try:
            repo = git.Repo(path)
            url = next(repo.remote().urls)
            commit = repo.head.commit.hexsha
            return {"local_path": path, "url": url, "commit": commit}
        except (git.InvalidGitRepositoryError, git.NoSuchPathError) as err:
            return {"local_path": path, "url": "Repository not found", "commit": "N/A"}

    # Get information for each repository
    voxel_wm_info = get_repo_info(voxel_wm_rel_path)
    craftium_info = get_repo_info(craftium_rel_path)
    minetest_info = get_repo_info(minetest_rel_path)
    voxel_libre2_info = get_repo_info(voxel_libre2_rel_path)

    return {
        "voxel_wm": voxel_wm_info,
        "craftium": craftium_info,
        "minetest": minetest_info,
        "voxel-libre2": voxel_libre2_info,
    }


def generate_dataset_meta(args):
    dataset_root = os.path.join(args.dataset_dir, args.dataset_name)
    if os.path.exists(os.path.join(dataset_root, "dataset_params.json")):
        if args.overwrite_init:
            logger.info(
                f"Existing {os.path.join(dataset_root, 'dataset_params.json')} found, will be overwritten"
            )
        else:
            raise ValueError(
                f"Existing {os.path.join(dataset_root, 'dataset_params.json')} found, "
                f"set --overwrite_init to overwrite."
            )
    if os.path.exists(os.path.join(dataset_root, "level_seeds.txt")):
        if args.overwrite_init:
            logger.info(
                f"Existing {os.path.join(dataset_root, 'level_seeds.txt')} found, will be overwritten"
            )
        else:
            raise ValueError(
                f"Existing {os.path.join(dataset_root, 'level_seeds.txt')} found, "
                f"set --overwrite_init to overwrite."
            )

    seed_everything(args.seed)

    # dataset metadata
    dataset_params = {
        "info": {
            "name": args.dataset_name,
            "description": "Minetest dataset",
            "script_name": os.path.basename(__file__),
            "script_args": vars(args),
            "version": "0.0.1",
            "minetest_rundir_root": args.mt_run_dir,
            "codebase_commit": get_codebase_commits(),
            "agent_info": {"action_space": "multi-discrete"},
            "model_pipeline_info": {},  # (added at a later stage of the dataset building pipeline, TODO)
        },
        "minetest_voxel_info": {
            "dims": (
                2 * args.voxel_obs_rx + 1,
                2 * args.voxel_obs_ry + 1,
                2 * args.voxel_obs_rz + 1,
                3,
            ),
            "dtype": str(np.int16),
            "xyz": "ENU",
            "origin": "agent",
            "origin_idx": (args.voxel_obs_rx, args.voxel_obs_ry, args.voxel_obs_rz),
        },
        "rgb_info": {
            "dims": (args.resolution_h, args.resolution_w, 3),
            "dtype": str(np.uint8),
            "hud": False,  # can be set to a dict in the future if enabled
        },
        "voxel_info": {
            "dims": (
                2 * args.voxel_obs_rx,
                2 * args.voxel_obs_ry,
                2 * args.voxel_obs_rz,
            ),
            "dtype": str(np.int16),
            "xyz": "ENU",
            "origin": "agent",
            "origin_idx": None,
            "active_voxel_preprocessing": None,
            "extrinsics_key": "extrinsics_local",
        },
    }
    set_voxel_info(dataset_params, args)
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)
    with open(os.path.join(dataset_root, "dataset_params.json"), "w") as f:
        json.dump(dataset_params, f, indent=4)

    seeds = np.random.randint(0, 2**31, args.num_levels).tolist()
    with open(os.path.join(dataset_root, "level_seeds.txt"), "w") as f:
        for s in seeds:
            f.write(f"{s}\n")


def check_codebase_match(dataset_params, args):
    codebase_commits = get_codebase_commits()
    if codebase_commits != dataset_params["info"]["codebase_commit"]:
        err_msg = (
            f"Codebase state does not match dataset_params.json. "
            f"\n DATASET_PARAMS: \n {dataset_params['info']['codebase_commit']}, "
            f"\n CODEBASE: \n {codebase_commits}"
        )
        if args.disable_commit_check:
            logger.warning(err_msg)
        else:
            raise ValueError(err_msg)


def avoid_load_error(data_npz):
    data_clean = {}
    for k in data_npz:
        try:
            data_clean[k] = data_npz[k]
        except ValueError as e:
            pass
    return data_clean


def set_voxel_info(dataset_params, args):
    mt_vox_shape = np.array(dataset_params["minetest_voxel_info"]["dims"][:3])
    output_vox_grid = np.array(dataset_params["voxel_info"]["dims"][:3])
    crop_left = (mt_vox_shape - output_vox_grid) // 2
    crop_right = crop_left + output_vox_grid
    dataset_params["voxel_info"]["active_voxel_preprocessing"] = {
        "empty_space_node_ids": EMPTY_SPACE_NODE_IDS,
        "crop": {"left": crop_left.tolist(), "right": crop_right.tolist()},
        "pad": False,
        "scale": False,
    }
    dataset_params["voxel_info"]["origin_idx"] = (
        np.array(dataset_params["minetest_voxel_info"]["origin_idx"]) - crop_left
    ).tolist()


def compute_cam_pose_local(cam_pos, voxel_center_pos, dataset_params):
    vox_preprocessing = dataset_params["voxel_info"]["active_voxel_preprocessing"]
    assert (
        vox_preprocessing["pad"] == False
        and vox_preprocessing["scale"] == False
        and vox_preprocessing["crop"] is not None
    ), "Only cropping is supported when computing intrinsics and extrinsics"
    assert all(vox_preprocessing["crop"]["left"]) == 0, (
        "Only right crop is supported when computing intrinsics and extrinsics"
    )

    new_voxel_center_pos = (
        voxel_center_pos - 0.5
    )  # shift from coordinate being in the center of the voxel to the corner
    cam_pos_local = cam_pos - new_voxel_center_pos
    voxel_grid_size = torch.tensor(
        dataset_params["voxel_info"]["dims"][:3], dtype=torch.float32
    ).to(cam_pos.device)
    cam_pos_local = cam_pos_local / voxel_grid_size.reshape(1, 1, 3)

    return cam_pos_local


def compute_extrinsics(cam_pos, cam_dir, eps=1e-8):
    f = cam_dir / (cam_dir.norm(dim=-1, keepdim=True).clamp_min(eps))  # unit forward
    look_at_pos = cam_pos + f
    ups = torch.tile(torch.tensor([0, 0, 1], dtype=cam_pos.dtype), cam_pos.shape[:-1] + (1,)).to(
        cam_pos.device
    )
    return utils3d.torch.extrinsics_look_at(cam_pos, look_at_pos, ups)


def compute_intrisincs_extrinsics(level_data, dataset_params, device=torch.device("cpu")):
    data = list(level_data.values())
    # upscaling to float64 to avoid numerical issues when computing extrinsics
    cam_pos = torch.tensor(np.array([(d["cam_pos"]) for d in data]), dtype=torch.float64).to(device)
    cam_dir = torch.tensor(np.array([(d["cam_dir"]) for d in data]), dtype=torch.float64).to(device)
    voxel_center_pos = torch.tensor(
        np.array([(d["obs_voxel_center"]) for d in data]), dtype=torch.float64
    ).to(device)
    fov_x = torch.tensor(np.array([(d["fov_x"]) for d in data]), dtype=torch.float64).to(device)
    fov_y = torch.tensor(np.array([(d["fov_y"]) for d in data]), dtype=torch.float64).to(device)
    cam_pos_local = compute_cam_pose_local(cam_pos, voxel_center_pos, dataset_params)
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov_x, fov_y).to(torch.float32)
    extrinsics_local = compute_extrinsics(cam_pos_local, cam_dir).to(torch.float32)
    extrinsics_global = compute_extrinsics(cam_pos, cam_dir).to(torch.float32)

    for i, s in enumerate(level_data):
        level_data[s]["intrinsics"] = intrinsics[i].cpu().numpy()
        level_data[s]["extrinsics_local"] = extrinsics_local[i].cpu().numpy()
        level_data[s]["extrinsics_global"] = extrinsics_global[i].cpu().numpy()


def _count_seen(level_data, dyn, args):
    """Return (n_mobs, n_seen) using the SAME visibility test as tools/check_seen.py:
    a mob is 'seen' if it is in the camera frustum, within max_dist, inside fov_frac of
    the FOV, and NOT occluded by terrain (voxel raycast), for >= min_frames frames.
    Returns (0, 0) if the visibility helpers or mob data are unavailable."""
    if not _HAVE_CHECK_SEEN or dyn is None:
        return 0, 0
    cam_pos = np.asarray(level_data["cam_pos"], dtype=np.float64)
    cam_dir = np.asarray(level_data["cam_dir"], dtype=np.float64)
    fov_x = np.asarray(level_data["fov_x"], dtype=np.float64)
    fov_y = np.asarray(level_data["fov_y"], dtype=np.float64)
    pos = np.asarray(dyn["dyn_pos"], dtype=np.float64)          # [T,N,3] bottom-centre
    present = np.asarray(dyn["dyn_present"], dtype=bool)        # [T,N]
    box = np.asarray(dyn["dyn_collisionbox"], dtype=np.float64)  # [T,N,6]; height=idx 4
    T = min(cam_pos.shape[0], pos.shape[0])
    N = pos.shape[1]
    if N == 0 or T == 0:
        return N, 0

    vox_data = _cs_load_voxels(level_data)
    if vox_data is not None:
        vox, vcenter = vox_data
        Tv = min(T, vox.shape[0])
        origin = ((vox.shape[1] - 1) // 2, (vox.shape[2] - 1) // 2, (vox.shape[3] - 1) // 2)
        empty_ids = _cs_empty_ids(vox)
    else:
        Tv = 0

    md, mf, ff, step = (args.visibility_max_dist, args.visibility_min_frames,
                        args.visibility_fov_frac, 0.5)
    n_seen = 0
    for m in range(N):
        height = np.clip(box[:T, m, 4], 0.2, 4.0)
        mid = pos[:T, m].copy()
        mid[:, 1] += 0.5 * height
        vis = np.zeros(T, bool)
        for p in (pos[:T, m], mid):
            vis |= _cs_visible(cam_pos[:T], cam_dir[:T], fov_x[:T], fov_y[:T], p, md, ff)
        vis &= present[:T, m]
        if vox_data is not None:
            for t in np.nonzero(vis[:Tv])[0]:
                if not _cs_ray_clear(cam_pos[t], mid[t], vox[t], vcenter[t], origin, empty_ids, step):
                    vis[t] = False
            vis[Tv:] = False
        if int(vis.sum()) >= mf:
            n_seen += 1
    return N, n_seen


def _level_is_bad(level_data, run_dir, args):
    """Return (bad, reason) for a just-collected level.

    A level is 'bad' (to be discarded + reseeded) if the player entered the water
    at any frame (authoritative per-frame `on_water` flag written by the mod) or if
    the camera shook violently (a large frame-to-frame horizontal position jump,
    e.g. a physics glitch). Purely a read of already-collected data + the mob log.
    """
    # 1) Violent shake: any horizontal player-position jump far above normal motion.
    pp = np.asarray(level_data["player_pos"], dtype=np.float64)  # [T, 3] ENU-ish (x,y,z)
    if pp.shape[0] >= 2:
        steps = np.linalg.norm(np.diff(pp[:, [0, 2]], axis=0), axis=1)  # horizontal only
        if steps.size and float(steps.max()) > args.shake_max_step:
            return True, f"violent shake (max horizontal step {float(steps.max()):.2f} > {args.shake_max_step})"

    # 2) Player entered water: scan the mob log for any player record with on_water=1.
    if args.skip_on_water:
        log_path = os.path.join(run_dir, "worlds", "world", "data_dynamic.jsonl")
        try:
            n_water = 0
            with open(log_path, "r") as f:
                for line in f:
                    if '"on_water"' not in line:      # cheap pre-filter
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pl = rec.get("player") or {}
                    if pl.get("on_water"):
                        n_water += 1
            if n_water > 0:
                return True, f"player entered water ({n_water} frames)"
        except FileNotFoundError:
            pass  # no log (dynamic data off) -> can't check water; rely on shake test
    return False, ""


def sample_repeat(action):
    if action == "dig":
        return np.random.randint(7, 25)
    elif action.startswith("place"):
        return np.random.randint(5, 7)
    elif action.startswith("mouse x"):
        return np.random.randint(1, 10)
    elif action.startswith("mouse y"):
        return np.random.randint(1, 15)
    elif action.startswith("slot_"):
        return 1  # no repeat for slot actions
    elif action != "noop":
        return np.random.randint(5, 40)
    else:
        return 0  # default (noop action)


def generate_level_chunk(seeds, args, dataset_params, device=torch.device("cpu")):
    level_meta_template = {
        "seed": None,
        "fov_x": np.rad2deg(
            2
            * np.arctan(args.resolution_w / args.resolution_h * np.tan(0.5 * np.deg2rad(args.fov)))
        ),
        "fov_y": args.fov,
        "spawn_pos": {"player_pos": None, "player_pitch": None, "player_yaw": None},
        "minetest_conf": None,
    }

    level_data_template = {
        "timestep_craftium": [],  # [Nsteps]
        "dt_minetest": [],  # [Nsteps]
        "player_pos": [],  # [Nsteps, 3]
        "player_vel": [],  # [Nsteps, 3]
        "player_pitch": [],  # [Nsteps]
        "player_yaw": [],  # [Nsteps]
        "cam_pos": [],  # [Nsteps, 3]
        "cam_dir": [],  # [Nsteps, 3]
        "fov_x": [],  # [Nsteps]
        "fov_y": [],  # [Nsteps]
        "obs_rgb": [],  # [Nsteps, C, H, W]
        "obs_voxel_mt": [],  # [Nsteps, C, X, Y, Z]
        "obs_voxel_center": [],  # [Nsteps, 3]
        "action": [],  # [Nsteps]
        "termination_flag": [],  # [Nsteps]
        "truncation_flag": [],  # [Nsteps]
        "intrinsics": [],  # [Nsteps, 3, 3]
        "extrinsics_local": [],  # [Nsteps, 4, 4]
        "extrinsics_global": [], # [Nsteps, 4, 4]
    }

    # Set random action probs and compute cumulative probs for sampling
    action_probs = [
        [1 / 72 for _ in range(72)],  # "noop", all movements
        [0.01, 0.36] + [0.07 for _ in range(9)],  # "noop", "dig", "place+slot_1", ... , "place+slot_9"
        # [0.05] + [0.1 for _ in range(9)] + [0.05],  # "noop", "slot 1-9", "drop"
        [0.6] + [0.2 for _ in range(2)],  # "noop", "mouse x-0.9", ... , "mouse x+0.9"
        [0.6] + [0.2 for _ in range(2)],  # "noop", "mouse y-0.9", ... , "mouse y+0.9"
    ]
    assert np.allclose([np.sum(prob) for prob in action_probs], 1)

    if args.navigation_only:
        # Keep only navigation: walking/jumping/sneaking/sprinting (group 0) and
        # looking around (groups 2 and 3). Disable the interaction group (dig and
        # place); note that "dig" (left click) is also how the player would hit
        # mobs, so this also prevents attacking the sheep.
        action_probs[1] = [1.0] + [0.0 for _ in range(len(action_probs[1]) - 1)]

    action_cmf = [np.cumsum(prob) for prob in action_probs]

    if args.gen_sha256_while_collecting:
        assert args.compute_extrinsics_while_collecting, (
            "When generate sha256 on the fly, we need to compute extrinsics on the fly as well. Otherwise the generated sha256 is useless."
        )

    seeds_to_gen = []
    logger.info("Checking if levels already exist")
    if not args.overwrite_leveldata:
        for seed in seeds:
            seed = int(seed)
            dataset_folder = Path(args.dataset_dir) / args.dataset_name / "raw" / args.env_id
            level_folder = dataset_folder / str(seed)
            data_path = level_folder / "data.npz"
            if data_path.exists():
                if args.regen_sha256_only:
                    logger.info(f"Level {seed} already exists, will regenerate sha256.txt")
                    with open(level_folder / "sha256.txt", "w") as f:
                        f.write(get_file_hash(data_path))
                if args.compute_extrinsics_only:
                    logger.info(f"Level {seed} already exists, recomputing extrinsics")
                    data = np.load(data_path)
                    data = avoid_load_error(data)
                    level_meta = json.load(open(level_folder / "level_metadata.json"))
                    seed = level_meta["seed"]
                    data = {seed: data}
                    compute_intrisincs_extrinsics(data, dataset_params, device=device)
                    np.savez_compressed(data_path, **data[seed])
                    logger.info("Regenerating sha256.txt since extrinsics were recomputed")
                    with open(os.path.join(level_folder, "sha256.txt"), "w") as f:
                        f.write(get_file_hash(data_path))
                if not args.regen_sha256_only and not args.compute_extrinsics_only:
                    logger.info(f"Level {seed} already exists, skipping")
            else:
                seeds_to_gen.append(seed)
    else:
        seeds_to_gen = [int(seed) for seed in seeds]

    if len(seeds_to_gen) == 0 or args.regen_sha256_only or args.compute_extrinsics_only:
        return

    # collect and save data according to the following structure:
    #     - raw
    #         - dataset_name (allows for mixing datasets later)
    #             - seed_folder (one per level)
    #                 - level_metadata.json
    #                 - data.npz
    dataset_root = Path(args.dataset_dir) / args.dataset_name
    raw_data_root = dataset_root / "raw" / args.env_id
    if not raw_data_root.exists():
        logger.info("Creating raw data folders")
        raw_data_root.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("Raw data folders already exist")
    seeds_to_gen = deque(seeds_to_gen)

    def env_process_executor(seed):
        logger.info(f"Start collecting level {seed}")
        t_start = time.time()
        # env setup
        craftium_kwargs = dict(
            id=f"Craftium/{args.env_id}",
            mt_port=args.mt_port,
            run_dir_prefix=args.mt_run_dir,
            rgb_observations=True,
            enable_voxel_obs=True,
            obs_width=args.resolution_w,
            obs_height=args.resolution_h,
            voxel_obs_rx=args.voxel_obs_rx,
            voxel_obs_ry=args.voxel_obs_ry,
            voxel_obs_rz=args.voxel_obs_rz,
            init_frames=args.init_frames,
            fps_max=args.fps_max,
            pmul=args.pmul,
            minetest_conf={
                "fov": args.fov,
                "world_start_time": 12000,
                "creative_mode": True,
                # note: not sure if voxel_obs_rx or voxel_obs_rx + 1
                "viewing_range": int(args.voxel_obs_rx * 1.25),
                "shadow_map_max_distance": int(args.voxel_obs_rx * 1.25),
                "active_object_send_range_blocks": (int(args.voxel_obs_rx * 1.25) // 16) + 1,
                "max_block_send_distance": (int(args.voxel_obs_rx * 1.25) // 16) + 1,
                "enable_fog": True,
                "directional_colored_fog": True,
                "fog_start": 0.8,
                "performance_tradeoffs": False,
                "font_size": 5,
                "mono_font_size": 5,
                "font_shadow": False,
                "repeat_place_time": 0.16,
                # Disable VoxeLibre's ambient mob spawner so only our controlled
                # set of mobs exists (our mod uses add_entity, unaffected by this).
                "mobs_spawn": "false" if args.disable_world_mob_spawning else "true",
                # Belt-and-suspenders: our mod also removes any mob it didn't spawn.
                "dynamic_agents_cull_wild": "true" if args.disable_world_mob_spawning else "false",
                # Dynamic-agent (mobs/animals) spawning + logging in Craftium.
                # Written to minetest.conf and read by the craftium_env mod.
                "dynamic_agents_enable": "true" if args.collect_dynamic_data else "false",
                "dynamic_agents_count": args.num_dynamic_agents,
                "dynamic_agents_count_min": args.num_dynamic_agents_min,
                "dynamic_agents_count_max": args.num_dynamic_agents_max,
                # Free roam: leash off (0) and no respawn top-up, so mobs wander
                # naturally and may travel far from the player.
                "dynamic_agents_leash_radius":
                    0.0 if args.dynamic_agents_free_roam else args.dynamic_agents_leash_radius,
                "dynamic_agents_maintain":
                    "false" if args.dynamic_agents_free_roam else "true",
                # Spawn once at the start; never create/relocate a mob mid-episode.
                "dynamic_agents_spawn_once":
                    "true" if args.dynamic_agents_spawn_once else "false",
                "dynamic_agents_min_radius": args.dynamic_agents_min_radius,
                "dynamic_agents_max_radius": args.dynamic_agents_max_radius,
                "dynamic_agents_min_separation": args.dynamic_agents_min_separation,
                "dynamic_agents_max_speed": args.dynamic_agents_max_speed,
                "dynamic_agents_view_half_angle": args.dynamic_agents_view_half_angle,
                # Spawn the initial population inside the player's forward view cone.
                "dynamic_agents_spawn_in_view":
                    "true" if args.dynamic_agents_spawn_in_view else "false",
                "dynamic_agents_spawn_view_half_angle": args.dynamic_agents_spawn_view_half_angle,
                "dynamic_agents_entity": args.dynamic_agent_entity,
                "dynamic_agents_entities": args.dynamic_agent_entities,
                # Make hostile mobs wander like animals (no attacking).
                "dynamic_agents_neutral": "true" if args.neutralize_agents else "false",
                # Hide HUD + first-person wielded hand/item from the RGB (visual only).
                "clean_rgb": "true" if args.clean_rgb else "false",
                # Relocate the player onto dry land at spawn (terrain-only).
                "spawn_on_land": "true" if args.spawn_on_land else "false",
                # Keep the player on dry land for the whole episode.
                "keep_player_on_land": "true" if args.keep_on_land else "false",
                # Water avoidance: spawn this far from water, and steer away from
                # water within the look-ahead (smoothly, no snap).
                "water_avoid_radius": args.water_avoid_radius,
                "water_lookahead": args.water_lookahead,
                "water_push_strength": args.water_push_strength,
            },
        )
        # mt_port should remain in the range [args.mt_port [default:49152], 65535]
        try:
            env = make_env(craftium_kwargs, mt_port_offset=0)
        except Exception as e:
            logger.error(f"Failed to create environment with seed {seed}: {e}")
            return "error"

        try:
            t_start = time.time()
            repeat = np.zeros(env.action_space.shape, dtype=np.int32)
            ts = 0
            if args.randomize_world_start_time:
                # truncated normal distribution [0, 23999]: ~80% day [6000 - 18000], ~20% night (otherwise)
                world_start_time = int(np.fmod(np.random.randn(), 2.5) * 4800 + 12000)
            else:
                world_start_time = craftium_kwargs["minetest_conf"]["world_start_time"] #defaults to 12000 (noon)
            if args.randomize_inventory:
                # squashed normal distribution to favor middle values
                num_items = int(np.clip(np.random.randn() * 2 + 5, 1, 9))
                # num_items = np.random.randint(1, 10) # 9 inventory slots in hotbar
                possible_items = args.starting_inventory.split(",")
                inventory = np.random.choice(possible_items, num_items, replace=False).tolist()
                inventory = ",".join(inventory)
            else:
                inventory = args.starting_inventory
            obs, info = env.reset(
                seed=seed,
                options={"minetest_conf": {"world_start_time": world_start_time, "starting_inventory_creative": inventory}},
            )
            level_meta = copy.deepcopy(level_meta_template)
            level_data = copy.deepcopy(level_data_template)
            level_meta["seed"] = seed
            level_meta["spawn_pos"] = {
                "player_pos": info["player_pos"].tolist(),
                "pitch": info["player_pitch"],
                "yaw": info["player_yaw"],
            }
            level_meta["minetest_conf"] = env.unwrapped.get_mt_config()

            # Optional goal-directed controller: reads the live mob positions and
            # steers the player to turn toward / approach each agent in turn. Falls
            # back to the random policy below if it can't be built.
            navigator = None
            if args.guided_navigation and args.collect_dynamic_data:
                try:
                    navigator = GuidedNavigator(
                        run_dir=env.unwrapped.mt.run_dir,
                        actions=env.actions,
                        action_shape=env.action_space.shape,
                        fov_deg=args.fov,
                        pitch_limit_deg=args.camera_pitch_limit_deg,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Guided navigator unavailable ({e}); using random policy")
                    navigator = None

            L = float(args.camera_pitch_limit_deg)  # keep the view within +-L of horizon

            # Schedule a one-off 360-degree spin-in-place at a random time in the
            # [min, max] SECONDS window (converted to frames via fps_max). When the
            # spin is disabled, mark it already-done so the guided tour (if any) can
            # run from the first frame.
            spin_active, spin_accum, spin_prev_yaw, spin_frames = False, 0.0, 0.0, 0
            spin_done = not args.player_spin_once
            spin_turn_idx = 2  # group-2 (mouse x) turn option
            # Frames over which the 360 is spread (slow, smooth spin).
            spin_total_frames = max(1, int(round(args.player_spin_seconds * args.fps_max)))
            if args.player_spin_once:
                lo = max(1, int(round(args.player_spin_min_seconds * args.fps_max)))
                hi = max(lo + 1, int(round(args.player_spin_max_seconds * args.fps_max)))
                # leave room for the spin itself to finish before the episode ends
                cap = args.ep_timesteps - args.player_spin_max_frames - 10
                if cap > lo:
                    hi = min(hi, cap)
                spin_start = int(np.random.randint(lo, hi))
            else:
                spin_start = -1

            while ts < args.ep_timesteps:
                if navigator is not None and spin_done:
                    # Post-spin goal-directed tour: turn toward / approach the next
                    # unobserved mob so each one is brought into frame in turn.
                    action = navigator.act()
                    repeat = np.zeros_like(repeat)
                    # Force the camera pitch back toward the horizon here (NOT inside the
                    # controller, whose pitch-sign calibration could drift and make the
                    # player stare at the sky while spinning). mouse-y idx 1 = look down,
                    # 2 = look up; correct whichever way reduces |pitch|.
                    pitch = info["player_pitch"]
                    action[3] = (1 if pitch > 0 else 2) if abs(pitch) > 3.0 else 0
                else:
                    # Camera-pitch controller (recomputed every frame so it also acts
                    # mid action-repeat): bias the look-up/look-down probabilities to
                    # keep the pitch within a mostly-horizontal band [-L, +L], so the
                    # player looks around the environment instead of locking onto the
                    # sky or the ground. Distances are measured to +-L (not +-90), so
                    # the avoidance kicks in early and hard.
                    pitch = info["player_pitch"]
                    dist_up = np.clip((L - pitch) / L, 0.0, 2.0)    # small near the top
                    dist_down = np.clip((L + pitch) / L, 0.0, 2.0)  # small near the bottom
                    sharpness = 4
                    w_up = dist_up**sharpness
                    w_down = dist_down**sharpness
                    denom = w_up + w_down
                    if denom <= 0:
                        p_up_total, p_down_total = 0.5, 0.5
                    else:
                        p_up_total = w_up / denom
                        p_down_total = w_down / denom
                    action_probs[3][1] = p_down_total * 0.38 + 0.01  # hardcode the camera action idx
                    action_probs[3][2] = p_up_total * 0.38 + 0.01  # hardcode the camera action idx
                    action_cmf[3] = np.cumsum(action_probs[3])  # hardcode the camera action idx
                    # The pitch action (index 1 or 2) with the higher probability is
                    # the one that pulls the view back toward the horizon.
                    corr_idx = 1 if action_probs[3][1] >= action_probs[3][2] else 2

                    if not repeat.any():
                        action = np.zeros_like(repeat)
                        for j in range(len(action)):
                            action[j] = np.argmax(np.random.rand() < action_cmf[j])
                            repeat[j] = sample_repeat(env.actions[j][action[j] - 1]) if action[j] > 0 else 0
                    else:
                        repeat = (repeat - 1).clip(min=0)
                        action[repeat == 0] = 0

                    # Hard guard: if the pitch has left the allowed band, override the
                    # camera axis to correct back toward the horizon this frame instead
                    # of lingering (e.g. staring at the sky) through a long repeat.
                    if abs(pitch) > L:
                        action[3] = corr_idx
                        repeat[3] = 1

                # One-off 360-degree spin-in-place: overrides whatever policy chose,
                # standing still and turning until a full circle is accumulated.
                if args.player_spin_once and not spin_done:
                    if ts == spin_start:
                        spin_active, spin_accum, spin_prev_yaw, spin_frames = True, 0.0, info["player_yaw"], 0
                    if spin_active:
                        dyaw = ((info["player_yaw"] - spin_prev_yaw + 180.0) % 360.0) - 180.0
                        spin_accum += abs(dyaw)
                        spin_prev_yaw = info["player_yaw"]
                        spin_frames += 1
                        if spin_accum >= 350.0 or spin_frames > args.player_spin_max_frames:
                            spin_active, spin_done = False, True
                        else:
                            # Slow, smooth spin: spread the 360 evenly over
                            # spin_total_frames. Only turn on the frames where we are
                            # BEHIND a linear ramp to 360 deg; stand still otherwise.
                            # Self-calibrates to the engine's per-step turn magnitude.
                            action = np.zeros_like(action)
                            repeat = np.zeros_like(repeat)
                            target = 360.0 * min(1.0, spin_frames / float(spin_total_frames))
                            if spin_accum < target:
                                action[2] = spin_turn_idx   # pure horizontal turn (stand still)

                # Initial observation hold: for the first N frames stand still and look
                # LEVEL at the freshly-spawned forward cluster of mobs, so every mob is
                # captured on camera for >= min_frames before the player wanders off.
                # (Overrides the policy/spin; the spin starts >= 2 s in, so no overlap.)
                if ts < args.player_observe_hold_frames:
                    action = np.zeros_like(action)
                    repeat = np.zeros_like(repeat)
                    pitch = info["player_pitch"]
                    if abs(pitch) > 3.0:
                        action[3] = 1 if pitch > 0 else 2   # level the view toward the horizon

                # we store interaction tuples in the format
                # (obs_t, info_t, action_t, reward_t+1, termination_t+1, truncation_t+1)
                level_data["obs_rgb"].append(obs)
                level_data["obs_voxel_mt"].append(info["voxel_obs"])
                level_data["obs_voxel_center"].append(info["voxel_obs_center"].tolist())
                level_data["timestep_craftium"].append(ts)
                level_data["dt_minetest"].append(info["mt_dtime"])
                multihot_action = env.multihot(action)  # convert to multihot before saving
                level_data["action"].append(multihot_action)
                level_data["player_pitch"].append(info["player_pitch"])
                level_data["player_yaw"].append(info["player_yaw"])
                level_data["player_pos"].append(info["player_pos"].tolist())
                level_data["player_vel"].append(info["player_vel"].tolist())
                level_data["cam_pos"].append(info["cam_pos"].tolist())
                level_data["cam_dir"].append(info["cam_dir"].tolist())
                level_data["fov_x"].append(info["cam_fov_x"])
                level_data["fov_y"].append(info["cam_fov_y"])

                obs, reward, term, trunc, info = env.step(action)

                level_data["termination_flag"].append(term)
                level_data["truncation_flag"].append(trunc)
                ts += 1

            logger.info(f"Finish collecting level {seed}, length:{ts}, time:{time.time() - t_start}")
            t_start = time.time()
            level_data = {k: np.asarray(v) for k, v in level_data.items()}
            if np.any(level_data["obs_voxel_mt"] > 8192):
                logger.info(f"Corrupt voxel data detected in level {seed}, skipping level")
                env.unwrapped.close(not args.debug)
                return "skipped"

            # Discard (and let the caller reseed) levels where the player went into
            # the water or the camera shook violently - keeps those out of the dataset.
            # This runs BEFORE any file is written, so a bad level is never added to the
            # dataset. As a belt-and-suspenders guard we also delete the level folder if
            # one somehow already exists (e.g. a partial write from an earlier crashed
            # run), so a bad level is never left on disk.
            bad, reason = _level_is_bad(level_data, env.unwrapped.mt.run_dir, args)
            if bad:
                logger.info(f"Discarding level {seed}: {reason}. Deleting any partial "
                            f"output and reseeding.")
                stale = raw_data_root / str(seed)
                if stale.exists():
                    shutil.rmtree(stale, ignore_errors=True)
                env.unwrapped.close(not args.debug)
                return "skipped"

            # Collect the mob ground truth NOW (before saving) so we can both enforce the
            # minimum-visibility gate and reuse it when writing data_dynamic.npz. Must run
            # before env.close() clears the run directory.
            dyn = None
            if args.collect_dynamic_data:
                try:
                    dyn = collect_dynamic_data(
                        env.unwrapped.mt.run_dir,
                        np.asarray(level_data["player_pos"]),
                        num_agents=None,   # auto-detect (count is randomized per level)
                        entity_name=args.dynamic_agent_entity,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to collect dynamic data for level {seed}: {e}")
                    dyn = None

            # Minimum-visibility gate: accept only if at most max_unseen_allowed mobs are
            # NOT actually seen on camera (>= N-1 by default -> 6/7, 5/6, 4/5, 3/4). Uses
            # the same frustum+distance+FOV+occlusion test as tools/check_seen.py.
            if args.require_min_visibility and dyn is not None and _HAVE_CHECK_SEEN:
                n_mobs, n_seen = _count_seen(level_data, dyn, args)
                if n_mobs > 0 and (n_mobs - n_seen) > args.max_unseen_allowed:
                    logger.info(
                        f"Discarding level {seed}: only {n_seen}/{n_mobs} mobs visible "
                        f"(need >= {n_mobs - args.max_unseen_allowed}). Reseeding.")
                    stale = raw_data_root / str(seed)
                    if stale.exists():
                        shutil.rmtree(stale, ignore_errors=True)
                    env.unwrapped.close(not args.debug)
                    return "skipped"

            level_data["obs_voxel_mt"] = level_data["obs_voxel_mt"].astype(np.int16)
            level_folder = raw_data_root / str(seed)
            level_folder.mkdir(exist_ok=True, parents=True)
            with open(level_folder / "level_metadata.json", "w") as f:
                json.dump(level_meta, f, indent=4)

            if args.compute_extrinsics_while_collecting:
                compute_intrisincs_extrinsics({seed: level_data}, dataset_params, device=device)

            iio.imwrite(
                level_folder / "rgb.mp4",
                level_data["obs_rgb"],
                fps=args.fps_max,
                codec="h264",  # or "libx264" depending on your ffmpeg build
                quality=10,  # 0 (worst) .. 10 (best), tradeoff size vs quality
                macro_block_size=1
            )
            level_data.pop("obs_rgb")
            np.savez_compressed(level_folder / "data.npz", **level_data)
            if args.gen_sha256_while_collecting:
                with open(level_folder / "sha256.txt", "w") as f:
                    f.write(get_file_hash(level_folder / "data.npz"))

            # Save the mob ground truth collected above (reused, not recomputed).
            if args.collect_dynamic_data and dyn is not None:
                try:
                    np.savez_compressed(level_folder / "data_dynamic.npz", **dyn)
                    logger.info(f"Saved data_dynamic.npz for level {seed}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to save dynamic data for level {seed}: {e}")

            logger.info(f"Finish saving level {seed}, length:{ts}, time:{time.time() - t_start}")

            env.unwrapped.close(not args.debug) # in debug mode, close(False) will not delete the mt_run_dir
            return "saved"
        except Exception as e:
            logger.error(
                f"Error happened during processing. Skipping level {seed}. Traceback: {e}"
            )
            if env.unwrapped.mt_chann.is_open():
                env.unwrapped.mt.close_pipes()
                env.unwrapped.mt.wait_close()
            if not args.debug:
                env.unwrapped.mt.clear()
            return "error"

    # Track every seed we've used (initial + reseeds) so replacements never collide
    # or duplicate terrain.
    used_seeds = set(int(s) for s in total_level_seeds)
    reseed_rng = np.random.default_rng(1_000_003 * (args.rank + 1) + int(args.seed))

    def _fresh_seed():
        s = int(reseed_rng.integers(1, 2**31 - 1))
        while s in used_seeds:
            s = int(reseed_rng.integers(1, 2**31 - 1))
        used_seeds.add(s)
        return s

    for seed_to_gen in seeds_to_gen:
        status = env_process_executor(int(seed_to_gen))
        # If the level was discarded (player in water / violent shake / corrupt),
        # retry the SLOT with fresh random seeds until one is clean or we hit the cap.
        attempts = 0
        while status == "skipped" and args.skip_on_water and attempts < args.max_reseed_attempts:
            new_seed = _fresh_seed()
            attempts += 1
            logger.info(
                f"Reseed attempt {attempts}/{args.max_reseed_attempts} for a discarded "
                f"level: trying seed {new_seed}"
            )
            status = env_process_executor(new_seed)
        if status == "skipped":
            logger.warning(
                f"Gave up on a level after {args.max_reseed_attempts} reseed attempts "
                f"(kept landing in water / shaking)."
            )


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    if args.init:
        generate_dataset_meta(args)
        sys.exit(0)

    dataset_params = json.load(
        open(os.path.join(args.dataset_dir, args.dataset_name, "dataset_params.json"))
    )
    check_codebase_match(dataset_params, args)
    total_level_seeds = np.loadtxt(
        Path(args.dataset_dir) / args.dataset_name / "level_seeds.txt", dtype=np.int32
    )
    start = len(total_level_seeds) * args.rank // args.world_size
    end = len(total_level_seeds) * (args.rank + 1) // args.world_size
    level_seeds_to_process = total_level_seeds[start:end]

    # If a display is already available (e.g. when launched via `xvfb-run`),
    # use it directly. Otherwise spawn our own virtual display. We bound the
    # display number to a small value because some Xvfb builds fail on the very
    # large random display numbers xvfbwrapper picks by default.
    vdisplay = None
    if not os.environ.get("DISPLAY"):
        try:
            vdisplay = Xvfb(display=99 + args.rank)
        except TypeError:
            # Older xvfbwrapper without the `display` kwarg.
            vdisplay = Xvfb()
        vdisplay.start()
    try:
        generate_level_chunk(level_seeds_to_process, args, dataset_params, device)
    finally:
        if vdisplay is not None:
            vdisplay.stop()
