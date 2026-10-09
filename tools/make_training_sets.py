#!/usr/bin/env python3
"""Cut the three training sets out of the replay datasets.

    T_H   human-derived episodes: LIBERO-predicate successes of the production replay
    T_R   reference-derived episodes through the same replay, subsampled to the same
          number of episodes per task as T_H (the robot-sourced control at matched N)
    T_HR  both
    T_Hm  the human set subsampled to T_R's per-task counts (the comparison at matched N;
          with fewer reference than human successes, T_R is every reference success)

The replay writes an episode for every rollout that passed the predicate OR the physical
criterion, in log order, so the k-th written episode is the k-th log row that passed
either. Training uses predicate successes only unless there are fewer than 10, in which
case the physical ones are included and the README says so.

    python tools/make_training_sets.py --human R3 --reference Rref
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from humanoid_challenge.sim_dataset import append_episode, create_dataset  # noqa: E402
from plot_results import parse_log  # noqa: E402

MIN_PREDICATE = 10


def written_rows(run):
    """Log rows that became dataset episodes, in dataset order, with their index."""
    rows = [r for r in parse_log(f"outputs/build_{run}.log") if r["ok"] or r["phys"]]
    for k, r in enumerate(rows):
        r["episode"] = k
        r["task"] = r["chunk"].split("_t")[-1]     # 'A_t6' and 'ref_t6' both -> '6'
    return rows


def copy_episodes(src_root, src_repo, episodes, ds):
    """Append the chosen episodes of one replay dataset to `ds`, frame by frame."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    src = LeRobotDataset(src_repo, root=src_root, video_backend="torchcodec")
    ep_col = np.asarray(src.hf_dataset["episode_index"])
    for ep in episodes:
        idx = np.flatnonzero(ep_col == ep)
        frames, task = [], None
        for i in idx:
            item = src[int(i)]
            f = {}
            for k in ("observation.images.image", "observation.images.image2"):
                img = item[k]
                assert img.dtype == torch.float32 and img.shape[0] == 3, (k, img.dtype, img.shape)
                f[k] = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            f["observation.state"] = item["observation.state"].numpy().astype(np.float32)
            f["action"] = item["action"].numpy().astype(np.float32)
            frames.append(f)
            task = item["task"]
        append_episode(ds, frames, task)
    return len(episodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="R3")
    ap.add_argument("--reference", default="Rref")
    ap.add_argument("--reference-root", type=Path, default=Path("outputs/datasets/ref_clean"),
                    help="the reference replay's dataset root (its log is outputs/build_<reference>.log)")
    ap.add_argument("--out", type=Path, default=Path("outputs/datasets"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma separated subset of T_H,T_R,T_HR,T_Hm")
    a = ap.parse_args()

    h = written_rows(a.human)
    h_pred = [r for r in h if r["ok"]]
    use_physical = len(h_pred) < MIN_PREDICATE
    h_sel = h if use_physical else h_pred
    need = Counter(r["task"] for r in h_sel)
    r_ok = [r for r in written_rows(a.reference) if r["ok"]]
    rng = random.Random(a.seed)
    r_sel = []
    for task, n in sorted(need.items()):
        pool = [r for r in r_ok if r["task"] == task]
        if len(pool) < n:
            print(f"  task {task}: only {len(pool)} reference successes for {n} human ones")
        r_sel += rng.sample(pool, min(n, len(pool)))
    print(f"T_H : {len(h_sel)} episodes ({'predicate+physical' if use_physical else 'predicate only'}), per task {dict(need)}")
    print(f"T_R : {len(r_sel)} episodes, per task {dict(Counter(r['task'] for r in r_sel))}")
    have = Counter(r["task"] for r in r_sel)
    hm_sel = []
    for task, n in sorted(have.items()):
        hm_sel += rng.sample([r for r in h_sel if r["task"] == task], n)
    print(f"T_Hm: {len(hm_sel)} episodes, per task {dict(Counter(r['task'] for r in hm_sel))}")

    roots = {a.human: a.out / a.human, a.reference: a.reference_root}
    sets = {"T_H": [(a.human, h_sel)], "T_R": [(a.reference, r_sel)],
            "T_HR": [(a.human, h_sel), (a.reference, r_sel)], "T_Hm": [(a.human, hm_sel)]}
    for name, parts in sets.items():
        if a.only and name not in a.only.split(","):
            continue
        root = a.out / name
        if root.exists():
            import shutil; shutil.rmtree(root)
        ds = create_dataset(f"local/{name}", root, fps=10)
        n = sum(copy_episodes(str(roots[run]), f"local/{roots[run].name}", [r["episode"] for r in rows], ds)
                for run, rows in parts)
        print(f"{name}: {n} episodes -> {root}")
    (a.out / "training_sets.txt").write_text(
        f"human={a.human} reference={a.reference} seed={a.seed} use_physical={use_physical}\n"
        + "".join(f"T_H  {r['chunk']}_{r['ep']}\n" for r in h_sel)
        + "".join(f"T_R  {r['chunk']} ep{r['ep']}\n" for r in r_sel)
        + "".join(f"T_Hm {r['chunk']}_{r['ep']}\n" for r in hm_sel))


if __name__ == "__main__":
    main()
