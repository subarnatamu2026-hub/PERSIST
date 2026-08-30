# fmt:off
import copy
import json
import os
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

    clean_rgb: bool = True
    """If True, hide the HUD (hotbar, health/breath bars, etc.) and the first-person
    wielded hand/item from the RGB observation. Purely visual; does not change actions,
    observations or the ground-truth data. Set to False to keep the default HUD."""

    collect_dynamic_data: bool = True
    """If True, spawn dynamic agents (mobs/animals) in Craftium and save their per-frame
    state to an extra `data_dynamic.npz` file alongside `data.npz` for each level."""

    num_dynamic_agents: int = 10
    """Number of dynamic agents to spawn (used when min==max). Superseded by the
    min/max range below when they differ."""

    num_dynamic_agents_min: int = 10
    """Minimum number of dynamic agents per level (randomized per level)."""

    num_dynamic_agents_max: int = 10
    """Maximum number of dynamic agents per level (randomized per level). Each level
    picks a random count in [min, max], seeded by the level seed. Set equal to the
    min (default 10) to spawn exactly that many agents in every level; the *mix* of
    species still varies per level because each slot draws a random entity from the
    seen/unseen list."""

    dynamic_agents_leash_radius: float = 12.0
    """Soft leash: mobs wander freely within this horizontal distance of the player.
    A mob that strays beyond it is quietly relocated back ONLY while it is off-screen
    (out of the player's view cone), so mobs stay observable within the 600-frame
    episode without ever being seen to move. Set to 0 to let mobs wander freely."""

    dynamic_agents_min_radius: float = 5.0
    """Inner radius of the ring mobs are (re)spawned into around the player."""

    dynamic_agents_max_radius: float = 10.0
    """Outer radius of the (re)spawn ring (clamped inside the leash by the mod)."""

    dynamic_agents_view_half_angle: float = 65.0
    """Half-angle (degrees) of the player's view cone used to decide whether a spawn/
    relocation spot is on-screen. Mobs are only ever spawned or relocated OUTSIDE this
    cone (prefer behind the player), so the player never sees a mob appear or teleport
    -- it only discovers mobs by turning toward them. Wider = more conservative."""

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
    """If True, keep the player on dry land for the WHOLE episode: any step onto or
    into water is undone by snapping it back to the last solid-ground spot (an
    invisible wall at the water's edge). Normal walking/jumping on land is untouched."""


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
                # Dynamic-agent (mobs/animals) spawning + logging in Craftium.
                # Written to minetest.conf and read by the craftium_env mod.
                "dynamic_agents_enable": "true" if args.collect_dynamic_data else "false",
                "dynamic_agents_count": args.num_dynamic_agents,
                "dynamic_agents_count_min": args.num_dynamic_agents_min,
                "dynamic_agents_count_max": args.num_dynamic_agents_max,
                "dynamic_agents_leash_radius": args.dynamic_agents_leash_radius,
                "dynamic_agents_min_radius": args.dynamic_agents_min_radius,
                "dynamic_agents_max_radius": args.dynamic_agents_max_radius,
                "dynamic_agents_view_half_angle": args.dynamic_agents_view_half_angle,
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
            },
        )
        # mt_port should remain in the range [args.mt_port [default:49152], 65535]
        try:
            env = make_env(craftium_kwargs, mt_port_offset=0)
        except Exception as e:
            logger.error(f"Failed to create environment with seed {seed}: {e}")
            return

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
            while ts < args.ep_timesteps:
                if not repeat.any():
                    # distance from top/bottom boundary
                    dist_up = (90 - info["player_pitch"]) / 90  # 2 at -90, 0 at 90
                    dist_down = (90 + info["player_pitch"]) / 90  # 2 at 90, 0 at -90
                    # squash with nonlinearity so effect is only near boundary
                    sharpness = 4
                    w_up = dist_up**sharpness
                    w_down = dist_down**sharpness
                    p_up_total = w_up / (w_up + w_down)
                    p_down_total = w_down / (w_up + w_down)
                    action_probs[3][1] = p_down_total * 0.38 + 0.01  # hardcode the camera action idx
                    action_probs[3][2] = p_up_total * 0.38 + 0.01  # hardcode the camera action idx
                    action_cmf[3] = np.cumsum(action_probs[3])  # hardcode the camera action idx
                    action = np.zeros_like(repeat)
                    for j in range(len(action)):
                        action[j] = np.argmax(np.random.rand() < action_cmf[j])
                        repeat[j] = sample_repeat(env.actions[j][action[j] - 1]) if action[j] > 0 else 0
                else:
                    repeat = (repeat - 1).clip(min=0)
                    action[repeat == 0] = 0

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
                return

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

            # Collect dynamic-agent (mobs/animals) data logged by Craftium and
            # save it as an extra data_dynamic.npz aligned to the collected
            # frames. Must run before env.close() clears the run directory.
            if args.collect_dynamic_data:
                try:
                    dyn = collect_dynamic_data(
                        env.unwrapped.mt.run_dir,
                        level_data["player_pos"],
                        num_agents=None,  # auto-detect (count is randomized per level)
                        entity_name=args.dynamic_agent_entity,
                    )
                    if dyn is not None:
                        np.savez_compressed(level_folder / "data_dynamic.npz", **dyn)
                        logger.info(f"Saved data_dynamic.npz for level {seed}")
                except Exception as e:
                    logger.warning(f"Failed to collect dynamic data for level {seed}: {e}")

            logger.info(f"Finish saving level {seed}, length:{ts}, time:{time.time() - t_start}")

            env.unwrapped.close(not args.debug) # in debug mode, close(False) will not delete the mt_run_dir
        except Exception as e:
            logger.error(
                f"Error happened during processing. Skipping level {seed}. Traceback: {e}"
            )
            if env.unwrapped.mt_chann.is_open():
                env.unwrapped.mt.close_pipes()
                env.unwrapped.mt.wait_close()
            if not args.debug:
                env.unwrapped.mt.clear()
            return

    for seed_to_gen in seeds_to_gen:
        env_process_executor(seed_to_gen)


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
