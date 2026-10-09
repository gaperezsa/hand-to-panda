#!/usr/bin/env python3
"""Eval success per arm with Wilson intervals, and the training loss curves.

    python tools/plot_eval.py T_H T_R T_Hm T_HR     # -> docs/figures/eval_success.png, train_loss.png
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_eval import TASKS, wilson  # noqa: E402

LABEL = {"T_H": "human-derived\n43 ep", "T_R": "robot-derived\n28 ep",
         "T_Hm": "human-derived\n28 ep (matched)", "T_HR": "both\n71 ep"}
FIG = Path("docs/figures")


def main(arms):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, arm in enumerate(arms):
        f = Path(f"outputs/eval/{arm}/eval_info.json")
        if not f.exists():
            continue
        per = {int(t["task_id"]): t["metrics"].get("successes") or [] for t in json.load(open(f))["per_task"]}
        k = sum(sum(map(bool, s)) for s in per.values()); n = sum(len(s) for s in per.values())
        lo, hi = wilson(k, n); p = 100 * k / max(n, 1)
        ax.bar(i, p, 0.6, color="0.75", edgecolor="k")
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="k", capsize=4)
        for j, t in enumerate(TASKS):
            if t in per:
                ax.plot(i - 0.2 + 0.1 * j, 100 * np.mean(list(map(bool, per[t]))), "o", ms=3, color="tab:blue")
        ax.text(i, hi + 2, f"{k}/{n}", ha="center", fontsize=8)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([LABEL.get(a, a) for a in arms], fontsize=8)
    ax.set_ylabel("LIBERO success [%]"); ax.set_ylim(0, 100)
    ax.set_title("SmolVLA on libero_spatial {2,0,8,1,6}, 20 episodes per task; dots = per task", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "eval_success.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for arm in arms:
        log = Path(f"outputs/train/{arm}.log")
        if not log.exists():
            continue
        s = log.read_text(errors="ignore").replace("\r", "\n")
        pts = [(float(m[1]) * (1000 if m[2] == "K" else 1), float(m[3]))
               for m in re.finditer(r"step:([0-9.]+)(K?) .*?loss:([0-9.]+)", s)]
        if pts:
            ax.plot(*zip(*pts), label=LABEL.get(arm, arm).replace("\n", ", "), lw=1)
    ax.set_xlabel("training step"); ax.set_ylabel("loss"); ax.set_yscale("log"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "train_loss.png", dpi=150); plt.close(fig)
    print("figures ->", FIG)


if __name__ == "__main__":
    main(sys.argv[1:] or ["T_H", "T_R", "T_Hm", "T_HR"])
