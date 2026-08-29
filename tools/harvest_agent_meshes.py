#!/usr/bin/env python3
"""Collect the 3D model (mesh) + textures for each dynamic-agent (mob) type.

For every entity id in the agent pool this copies the model file (e.g.
``mobs_mc_sheep.b3d``) and its texture PNGs out of the VoxeLibre game assets into
``<out>/<mob>/`` and writes ``<out>/manifest.json`` mapping each mob to its files.
You can then load / convert those in Cursor to visualize each agent's body.

The id -> (mesh, textures) mapping is resolved from, in order:
  1. the live values already stored in generated ``data_dynamic.npz`` files
     (``dyn_mesh`` / ``dyn_textures``) - authoritative, from the running engine;
  2. best-effort parsing of the ``mcl_mobs.register_mob`` definitions in the
     VoxeLibre source (covers mobs not seen in any dataset yet).

Model files (.b3d/.x) are Blitz3D/DirectX meshes; trimesh can't open them
directly - convert to glTF/OBJ (e.g. via Blender) for viewing. The manifest tells
you which model + textures belong together.

Usage:
  python tools/harvest_agent_meshes.py --out agent_meshes
  python tools/harvest_agent_meshes.py --from_dataset datasets/mob_test3 --out agent_meshes
  python tools/harvest_agent_meshes.py --entities mobs_mc:sheep,mobs_mc:cow
"""
import argparse
import json
import os
import re
import shutil

# The default land-only pool (matches generate_raw_data.py's default).
DEFAULT_POOL = [
    "mobs_mc:sheep", "mobs_mc:cow", "mobs_mc:pig", "mobs_mc:chicken", "mobs_mc:rabbit",
    "mobs_mc:mooshroom", "mobs_mc:horse", "mobs_mc:donkey", "mobs_mc:mule", "mobs_mc:llama",
    "mobs_mc:wolf", "mobs_mc:dog", "mobs_mc:cat", "mobs_mc:ocelot", "mobs_mc:polar_bear",
    "mobs_mc:killer_bunny", "mobs_mc:skeleton_horse", "mobs_mc:zombie_horse",
    "mobs_mc:iron_golem", "mobs_mc:snowman", "mobs_mc:villager",
    "mobs_mc:zombie", "mobs_mc:baby_zombie", "mobs_mc:husk", "mobs_mc:baby_husk",
    "mobs_mc:skeleton", "mobs_mc:stray", "mobs_mc:witherskeleton", "mobs_mc:silverfish",
    "mobs_mc:endermite", "mobs_mc:spider", "mobs_mc:cave_spider", "mobs_mc:villager_zombie",
    "mobs_mc:zombified_piglin", "mobs_mc:baby_zombified_piglin", "mobs_mc:pigman",
    "mobs_mc:baby_pigman", "mobs_mc:piglin", "mobs_mc:piglin_brute", "mobs_mc:sword_piglin",
    "mobs_mc:hoglin", "mobs_mc:baby_hoglin", "mobs_mc:zoglin", "mobs_mc:vindicator",
    "mobs_mc:pillager", "mobs_mc:slime_big", "mobs_mc:slime_small", "mobs_mc:slime_tiny",
    "mobs_mc:magma_cube_big", "mobs_mc:magma_cube_small", "mobs_mc:magma_cube_tiny",
]

MODEL_EXTS = (".b3d", ".x", ".obj", ".gltf", ".glb")


def index_files(game_dir):
    """Index model and texture files by basename (first match wins)."""
    models, textures = {}, {}
    for root, _, files in os.walk(game_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith(MODEL_EXTS):
                models.setdefault(fn, os.path.join(root, fn))
            elif low.endswith(".png"):
                textures.setdefault(fn, os.path.join(root, fn))
    return models, textures


def mapping_from_datasets(dataset_dirs):
    """id -> {mesh, textures} from generated data_dynamic.npz (authoritative)."""
    import glob
    import numpy as np
    mapping = {}
    for dd in dataset_dirs:
        for f in glob.glob(os.path.join(dd, "raw", "*", "*", "data_dynamic.npz")):
            try:
                d = np.load(f, allow_pickle=True)
            except Exception:
                continue
            names = d["dyn_names"].tolist() if "dyn_names" in d else []
            mesh = d["dyn_mesh"].tolist() if "dyn_mesh" in d else []
            tex = d["dyn_textures"].tolist() if "dyn_textures" in d else []
            for i, nm in enumerate(names):
                m = str(mesh[i]) if i < len(mesh) else ""
                ts = [str(t) for t in tex[i]] if i < len(tex) and tex[i] is not None else []
                if nm and m and nm not in mapping:
                    mapping[nm] = {"mesh": m, "textures": ts, "source": "dataset"}
    return mapping


def _balanced_table(s, start):
    """Return the substring of the {...} table beginning at/after index start."""
    i = s.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i:j + 1]
    return None


def parse_lua_def(game_dir, entity_id):
    """Best-effort parse of a mob's register_mob({...}) literal for mesh/textures."""
    pat = re.compile(r'register_mob\(\s*["\']' + re.escape(entity_id) + r'["\']\s*,')
    for root, _, files in os.walk(game_dir):
        for fn in files:
            if not fn.endswith(".lua"):
                continue
            path = os.path.join(root, fn)
            try:
                s = open(path, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            m = pat.search(s)
            if not m:
                continue
            # Only handle literal-table definitions: `register_mob("id", { ... })`.
            rest = s[m.end():]
            if rest.lstrip()[:1] != "{":
                return None  # def passed as a variable; skip (dataset covers it)
            block = _balanced_table(s, m.end())
            if not block:
                return None
            mesh = None
            mm = re.search(r'mesh\s*=\s*["\']([^"\']+)["\']', block)
            if mm:
                mesh = mm.group(1)
            textures = []
            tm = re.search(r'textures\s*=\s*{', block)
            if tm:
                tex_block = _balanced_table(block, tm.end() - 1)
                if tex_block:
                    textures = re.findall(r'["\']([^"\']+\.png)["\']', tex_block)
            if mesh:
                return {"mesh": mesh, "textures": textures, "source": "lua"}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game_dir",
                    default="gym_envs/craftium/craftium-envs/common_games/VoxeLibre",
                    help="Path to the VoxeLibre game (with mods/ENTITIES/...).")
    ap.add_argument("--entities", default="",
                    help="Comma-separated entity ids. Defaults to the built-in pool.")
    ap.add_argument("--from_dataset", default="",
                    help="Dataset dir (e.g. datasets/mob_test3): read its pool from "
                         "dataset_params.json and use its data_dynamic.npz mappings.")
    ap.add_argument("--datasets_glob", default="datasets/*",
                    help="Where to look for data_dynamic.npz mappings (glob of dataset dirs).")
    ap.add_argument("--out", default="agent_meshes", help="Output folder.")
    ap.add_argument("--copy_all", action="store_true",
                    help="Also dump every mob model/texture into _all_models/_all_textures.")
    args = ap.parse_args()

    if not os.path.isdir(args.game_dir):
        raise SystemExit(f"game_dir not found: {args.game_dir}")

    # Resolve the pool.
    if args.entities.strip():
        pool = [e.strip() for e in args.entities.split(",") if e.strip()]
    elif args.from_dataset:
        params = json.load(open(os.path.join(args.from_dataset, "dataset_params.json")))
        s = params["info"]["script_args"].get("dynamic_agent_entities", "")
        pool = [e.strip() for e in s.split(",") if e.strip()] or DEFAULT_POOL
    else:
        pool = DEFAULT_POOL

    # Gather id -> mesh/textures mappings from datasets first, then Lua for the rest.
    import glob
    dataset_dirs = []
    if args.from_dataset:
        dataset_dirs.append(args.from_dataset)
    dataset_dirs += [d for d in glob.glob(args.datasets_glob) if os.path.isdir(d)]
    mapping = mapping_from_datasets(dataset_dirs) if dataset_dirs else {}

    models_idx, textures_idx = index_files(args.game_dir)
    os.makedirs(args.out, exist_ok=True)

    manifest = {}
    for eid in pool:
        info = mapping.get(eid) or parse_lua_def(args.game_dir, eid)
        short = eid.split(":")[-1]
        dst = os.path.join(args.out, short)
        os.makedirs(dst, exist_ok=True)
        rec = {"entity": eid, "source": None, "mesh": None, "mesh_file": None,
               "textures": [], "texture_files": [], "missing": []}
        if not info:
            rec["missing"].append("no mapping found (variable-based def; try generating a "
                                   "dataset that includes this mob, or inspect the source)")
            manifest[eid] = rec
            continue
        rec["source"] = info.get("source")
        # Copy the model.
        mesh = info.get("mesh")
        rec["mesh"] = mesh
        if mesh and mesh in models_idx:
            shutil.copy2(models_idx[mesh], os.path.join(dst, mesh))
            rec["mesh_file"] = mesh
        elif mesh:
            rec["missing"].append(f"model file not found: {mesh}")
        # Copy the textures (dedup).
        for t in dict.fromkeys(info.get("textures", [])):
            rec["textures"].append(t)
            if t in textures_idx:
                shutil.copy2(textures_idx[t], os.path.join(dst, t))
                rec["texture_files"].append(t)
            else:
                rec["missing"].append(f"texture file not found: {t}")
        manifest[eid] = rec

    if args.copy_all:
        for name, path in models_idx.items():
            if os.path.basename(path).startswith("mobs_mc"):
                d = os.path.join(args.out, "_all_models"); os.makedirs(d, exist_ok=True)
                shutil.copy2(path, os.path.join(d, name))
        for name, path in textures_idx.items():
            if name.startswith("mobs_mc"):
                d = os.path.join(args.out, "_all_textures"); os.makedirs(d, exist_ok=True)
                shutil.copy2(path, os.path.join(d, name))

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    ok = sum(1 for r in manifest.values() if r["mesh_file"])
    print(f"Pool: {len(pool)} mobs -> {args.out}/")
    print(f"  models copied: {ok}/{len(pool)}")
    miss = [e for e, r in manifest.items() if r["missing"]]
    if miss:
        print(f"  with issues ({len(miss)}): " + ", ".join(miss))
        print("  (see manifest.json 'missing' fields; --copy_all dumps every mob asset as a fallback)")
    print(f"  manifest: {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
