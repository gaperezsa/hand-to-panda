#!/usr/bin/env python3
"""Concatenate LeRobot datasets episode by episode into a new one (re-encodes the videos).

    python tools/merge_datasets.py --out outputs/datasets/T_Hself outputs/datasets/T_H outputs/datasets/self_T_H_*
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from humanoid_challenge.sim_dataset import create_dataset  # noqa: E402
from make_training_sets import copy_episodes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("roots", nargs="+", type=Path)
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    ds = create_dataset(f"local/{a.out.name}", a.out, fps=10)
    total = 0
    for root in a.roots:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        n = LeRobotDataset(f"local/{root.name}", root=str(root)).meta.total_episodes
        total += copy_episodes(str(root), f"local/{root.name}", list(range(n)), ds)
        print(f"  {root.name}: {n} episodes")
    print(f"{a.out}: {total} episodes")


if __name__ == "__main__":
    main()
