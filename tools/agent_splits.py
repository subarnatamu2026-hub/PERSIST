#!/usr/bin/env python3
"""Deterministically split the dynamic-agent pool into a 'seen' and 'unseen' set.

'seen' (default 70%) is used for the training and eval-1 datasets; 'unseen' (30%)
is used only for the eval-2 (zero-shot) dataset. The split is reproducible from a
fixed seed and written to text files so the whole pipeline uses the same partition.

Pool source (in order): --entities, --pool_file, else parsed from the default
`dynamic_agent_entities` in dataset_toolkits/generate_raw_data.py.
"""
import argparse
import os
import random
import re


def default_pool():
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "..", "dataset_toolkits", "generate_raw_data.py")
    src = open(src_path, encoding="utf-8").read()
    i = src.index("dynamic_agent_entities: str = (")
    j = src.index('"""', i)          # docstring starts right after the closing ')'
    block = src[i:j]
    ids = re.findall(r"mobs_mc:[a-z_0-9]+", block)
    # dedupe preserving order
    seen = set()
    out = []
    for e in ids:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", default="", help="Comma-separated pool (overrides default).")
    ap.add_argument("--pool_file", default="", help="File with one entity id per line.")
    ap.add_argument("--seen_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="datasets/agent_split")
    args = ap.parse_args()

    if args.entities.strip():
        pool = [e.strip() for e in args.entities.split(",") if e.strip()]
    elif args.pool_file:
        pool = [l.strip() for l in open(args.pool_file) if l.strip()]
    else:
        pool = default_pool()

    pool = sorted(set(pool))                      # canonical order for reproducibility
    rng = random.Random(args.seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    n_seen = round(args.seen_ratio * len(pool))
    seen = sorted(shuffled[:n_seen])
    unseen = sorted(shuffled[n_seen:])

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "seen.txt"), "w") as f:
        f.write(",".join(seen))
    with open(os.path.join(args.out_dir, "unseen.txt"), "w") as f:
        f.write(",".join(unseen))

    print(f"pool: {len(pool)}   seen(train/eval1): {len(seen)}   unseen(eval2): {len(unseen)}")
    print(f"seed={args.seed} ratio={args.seen_ratio}  -> {args.out_dir}/seen.txt, unseen.txt")
    print("\nSEEN (train + eval1):")
    for e in seen:
        print("  ", e)
    print("\nUNSEEN (eval2, zero-shot):")
    for e in unseen:
        print("  ", e)


if __name__ == "__main__":
    main()
