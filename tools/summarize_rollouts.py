#!/usr/bin/env python3
"""Tables and a figure from policy_rollout.py logs.

    python tools/summarize_rollouts.py outputs/rollouts/E1_*.jsonl      # generalization table + figure
    python tools/summarize_rollouts.py outputs/rollouts/E2_*.jsonl      # self-rollout yield
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_eval import wilson  # noqa: E402


MAX_BOWL_ERR_MM = 20


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    keys = [(r["policy"], r["task"], r["init_id"], r["dx"], r["dy"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicated episodes across workers"
    return rows


def radius_cm(r):
    return round(100 * (r["dx"] ** 2 + r["dy"] ** 2) ** 0.5)


def main(paths):
    rows = load(paths)
    # A shifted layout is dropped for every policy if the bowl did not stay where it was put
    # (it landed on a neighbour). Stock layouts are the benchmark's own and always count.
    bad_keys = {(r["task"], r["init_id"], r["dx"], r["dy"]) for r in rows
                if (r["dx"] or r["dy"]) and r.get("bowl_err_mm", 0) > MAX_BOWL_ERR_MM}
    rows = [r for r in rows if (r["task"], r["init_id"], r["dx"], r["dy"]) not in bad_keys]
    print(f"{len(bad_keys)} shifted layouts dropped (bowl displaced > {MAX_BOWL_ERR_MM} mm at reset)")
    known = ["T_H", "T_R", "T_Hm", "T_HR", "T_Haug15", "T_Haug40", "T_Hself", "T_H_s1", "T_R_s1"]
    pols = sorted({r["policy"] for r in rows}, key=lambda p: known.index(p) if p in known else 99)
    radii = sorted({radius_cm(r) for r in rows})
    print(f"{len(rows)} episodes\n")
    print("| policy | " + " | ".join(f"bowl shifted {c} cm" for c in radii) + " |")
    print("|---|" + "---|" * len(radii))
    curves = {}
    for p in pols:
        cells, curve = [], []
        for c in radii:
            s = [r["success"] for r in rows if r["policy"] == p and radius_cm(r) == c]
            k, n = sum(s), len(s)
            lo, hi = wilson(k, n)
            cells.append(f"{k}/{n} ({100*k/max(n,1):.0f} %, {lo:.0f}–{hi:.0f})")
            curve.append((c, 100 * k / max(n, 1), lo, hi))
        curves[p] = curve
        base = curve[0][1] if radii[0] == 0 and curve[0][1] > 0 else None
        keep = "  retained: " + ", ".join(f"{c} cm {100*y/base:.0f} %" for c, y, _, _ in curve[1:]) if base and len(curve) > 1 else ""
        print(f"| {p} | " + " | ".join(cells) + " |" + keep)
    print("\nper task (all shifts pooled):")
    for p in pols:
        by = defaultdict(list)
        for r in rows:
            if r["policy"] == p:
                by[r["task"]].append(r["success"])
        print(f"  {p}: " + "  ".join(f"t{t} {sum(v)}/{len(v)}" for t, v in sorted(by.items())))
    if len(radii) > 1:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for k, (p, curve) in enumerate(curves.items()):
            c, y, lo, hi = zip(*curve)
            c = [x + 0.12 * (k - (len(curves) - 1) / 2) for x in c]     # dodge so the bars do not overlap
            ax.errorbar(c, y, yerr=[[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]],
                        marker="o", capsize=3, label=p)
        ax.set_xlabel("bowl shifted from the benchmark's own position [cm]")
        ax.set_ylabel("success [%]"); ax.set_ylim(0, 100); ax.set_xticks(radii); ax.legend(fontsize=8)
        ax.set_title("Success when the bowl is moved off the benchmark layout (95 % intervals)", fontsize=9)
        fig.tight_layout(); Path("docs/figures").mkdir(exist_ok=True)
        fig.savefig("docs/figures/generalization.png", dpi=150); print("\nfigure -> docs/figures/generalization.png")


if __name__ == "__main__":
    main(sys.argv[1:])
