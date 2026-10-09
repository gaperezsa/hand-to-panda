#!/usr/bin/env python3
"""Per-task and pooled success of the evaluated arms, with Wilson 95 % intervals.

    python tools/summarize_eval.py T_H T_R T_Hm T_HR      # reads outputs/eval/<arm>/eval_info.json
"""
import json
import sys
from math import sqrt
from pathlib import Path

TASKS = [2, 0, 8, 1, 6]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def main(arms):
    print("| arm | " + " | ".join(f"task {t}" for t in TASKS) + " | pooled | 95 % CI |")
    print("|---|" + "---|" * (len(TASKS) + 2))
    for arm in arms:
        f = Path(f"outputs/eval/{arm}/eval_info.json")
        if not f.exists():
            print(f"| {arm} | (no eval_info.json) |")
            continue
        info = json.load(open(f))
        by_task = {}
        for t in info["per_task"]:
            s = t["metrics"].get("successes") or []
            by_task[int(t["task_id"])] = (int(sum(bool(x) for x in s)), len(s))
        cells = [f"{by_task[t][0]}/{by_task[t][1]}" if t in by_task else "-" for t in TASKS]
        k = sum(v[0] for v in by_task.values()); n = sum(v[1] for v in by_task.values())
        lo, hi = wilson(k, n)
        print(f"| {arm} | " + " | ".join(cells) + f" | **{k}/{n}** ({100*k/max(n,1):.0f} %) | {lo:.0f}–{hi:.0f} |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["T_H", "T_R", "T_Hm", "T_HR"])
