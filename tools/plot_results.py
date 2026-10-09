#!/usr/bin/env python3
"""Figures for the README, read from the replay logs and the interim files.

Nothing here touches the pipeline; it only reads what the runs already wrote.

    python tools/plot_results.py            # writes docs/figures/*.png and prints the tables
"""
import glob
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHUNKS = ["A_t2", "A_t0", "A_t8", "A_t1", "A_t6"]
# (key, label, log files under outputs/, drawn in the ladder figure?)
RUNS = [("R0", "hand path, literal", "build_R0.log", True),
        ("R1np", "+ wrist held at its home angle", "build_R1np_*.log", True),
        ("R5", "+ grasp height from the bowl's shape", "build_R5_*.log", True),
        ("R3", "+ grasp and release pinned to\nknown object positions", "build_R3.log", True),
        ("Rref", "robot's own demonstration paths", "build_Rref.log", True),
        ("R1", "wrist lock, release pinned", "build_R1.log", False),
        ("R2", "grasp pinned, no wrist lock", "build_R2.log", False)]
USES_POSITIONS = {"R3"}
RIM_R_MM, RIM_TOP_MM = 56, 52      # akita black bowl, from the compiled MuJoCo mesh
FIG = Path("docs/figures")
HDR = re.compile(r"\s*\[\s*\d+/\d+\]\s+(\S+)\s+t(\d+)\s+(SUCCESS|failed)\s+\d+ steps.*?(data/interim/(\w+)_(\d+)\.npz|lerobot/libero ep (\d+))")
PHY = re.compile(r"lifted\s+(-?\d+) mm\s+moved\s+(\d+) mm\s+bowl->plate\s+(\d+) mm\s+physical=(\w+)")


def parse_log(path):
    """One row per replayed episode: chunk, episode id, predicate, physical, bowl motion."""
    rows, cur = [], None
    for line in Path(path).read_text().splitlines():
        m = HDR.match(line)
        if m:
            cur = dict(chunk=m[5] or f"ref_t{m[2]}", ep=m[6] or m[7], ok=m[3] == "SUCCESS")
            rows.append(cur)
            continue
        p = PHY.search(line)
        if p and cur is not None and "lifted" not in cur:
            cur.update(lifted=int(p[1]), moved=int(p[2]), to_target=int(p[3]), phys=p[4] == "True")
    return [r for r in rows if "lifted" in r]


def failure_mode(r):
    if r["ok"]:
        return "predicate"
    if r["phys"]:
        return "physical only"
    if r["lifted"] < 30 and r["moved"] >= 50:
        return "pushed, not lifted"
    return "no contact"


def grasp_geometry(cfg):
    """Per capture: distance and height of the pinch midpoint relative to the sim bowl,
    and the angle between the jaw axis (EEF y) and the radial line, at the grasp."""
    start = {f"A_t{p['task_ids']}": np.array(p["start_m"], float)
             for p in cfg["layouts"]["A"]["positions"]}
    out = {}
    for f in sorted(glob.glob("data/interim/A_*.npz")):
        ep = Path(f).stem
        d = np.load(f)
        P, R, g = d["eef_pos"], d["eef_rotmat"], d["gripper"]
        tr = np.flatnonzero(np.diff(g) != 0)
        if len(tr) < 2:
            continue
        gi = tr[0] + 1
        rad = P[gi, :2] - start["_".join(ep.split("_")[:2])]
        rhat = rad / np.linalg.norm(rad)
        jaw = R[gi][:2, 1] / np.linalg.norm(R[gi][:2, 1])
        out[ep] = dict(r_mm=np.linalg.norm(rad) * 1000, z_mm=P[gi, 2] * 1000,
                       jaw_deg=np.degrees(np.arccos(min(1.0, abs(jaw @ rhat)))))
    return out


def reference_geometry():
    """Same three numbers for the LIBERO demonstrations of the five tasks."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    bowl = {38: (-0.075, 0.0), 34: (-0.05, 0.20), 36: (0.01, 0.31), 37: (-0.18, 0.32), 30: (0.13, -0.07)}
    ds = LeRobotDataset("lerobot/libero")
    df = ds.hf_dataset.select_columns(
        ["episode_index", "frame_index", "task_index", "action", "observation.state"]).to_pandas()
    ept = df.groupby("episode_index")["task_index"].first()
    out = []
    for gti, b in bowl.items():
        for ep in ept.index[ept == gti].tolist()[:8]:
            e = df[df.episode_index == ep].sort_values("frame_index")
            st = np.stack(e["observation.state"].to_numpy())
            ac = np.stack(e["action"].to_numpy())
            gi = np.flatnonzero(np.diff(ac[:, 6]) != 0)[0] + 1
            rad = st[gi, :2] - np.array(b)
            rhat = rad / np.linalg.norm(rad)
            R = cv2.Rodrigues(st[gi, 3:6])[0]
            jaw = R[:2, 1] / np.linalg.norm(R[:2, 1])
            out.append(dict(r_mm=np.linalg.norm(rad) * 1000, z_mm=(st[gi, 2] - 0.90) * 1000,
                            jaw_deg=np.degrees(np.arccos(min(1.0, abs(jaw @ rhat))))))
    return out


def fig_grasp_geometry(geo, ref, outcomes):
    colours = {"predicate": "tab:green", "physical only": "tab:olive",
               "pushed, not lifted": "tab:orange", "no contact": "tab:red"}
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvline(RIM_R_MM, color="k", lw=0.8, ls="--")
    ax.axhline(RIM_TOP_MM, color="k", lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.text(RIM_R_MM + 2, -140, "rim radius", fontsize=8)
    ax.text(5, RIM_TOP_MM + 3, "rim top", fontsize=8)
    ax.text(5, 3, "table surface", fontsize=8)
    ax.scatter([r["r_mm"] for r in ref], [r["z_mm"] for r in ref], s=18, c="0.6", marker="s",
               label=f"LIBERO demonstrations (n={len(ref)})")
    for mode, c in colours.items():
        pts = [(geo[e]["r_mm"], geo[e]["z_mm"]) for e, o in outcomes.items()
               if e in geo and failure_mode(o) == mode]
        if pts:
            ax.scatter(*zip(*pts), s=22, c=c, label=f"human capture, {mode} (n={len(pts)})")
    ax.set_xlim(0, 130); ax.set_ylim(-150, 200)
    ax.set_xlabel("pinch midpoint distance from bowl centre at grasp [mm]")
    ax.set_ylabel("pinch midpoint height above table at grasp [mm]")
    ax.set_title("Grasp pose as retargeted (R0) vs the robot's own demonstrations")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "grasp_geometry.png", dpi=150); plt.close(fig)


def fig_jaw_axis(geo, ref):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.linspace(0, 90, 19)
    ax.hist([g["jaw_deg"] for g in geo.values()], bins, alpha=0.7, label="human captures (forearm yaw)")
    ax.hist([r["jaw_deg"] for r in ref], bins, alpha=0.7, label="LIBERO demonstrations")
    ax.set_xlabel("angle between jaw axis and the bowl's radial line at grasp [deg]")
    ax.set_ylabel("episodes"); ax.legend(fontsize=8)
    ax.set_title("0 deg = jaws straddle the rim; 90 deg = jaws close along it")
    fig.tight_layout(); fig.savefig(FIG / "jaw_axis.png", dpi=150); plt.close(fig)


def yield_table(logs):
    print("\n| run | " + " | ".join(CHUNKS) + " | all pick (predicate) | predicate or physical |")
    print("|---|" + "---|" * (len(CHUNKS) + 2))
    table = {}
    for run, label, _, _ in RUNS:
        if run not in logs:
            continue
        rows = logs[run]
        cells = []
        for ch in CHUNKS:
            rr = [r for r in rows if r["chunk"] == ch]
            cells.append(f"{sum(r['ok'] for r in rr)}/{len(rr)}" if rr else "-")
        n_ok, n_ph = sum(r["ok"] for r in rows), sum(r["ok"] or r["phys"] for r in rows)
        table[run] = (n_ok, n_ph, len(rows))
        label = label.replace("\n", " ")
        print(f"| {run} {label} | " + " | ".join(cells)
              + f" | **{n_ok}/{len(rows)}** ({100*n_ok/len(rows):.0f} %) | {n_ph}/{len(rows)} |")
    print("\nFailure modes per run:")
    for run, _, _, _ in RUNS:
        if run in logs:
            print(f"  {run}: " + ", ".join(f"{k} {v}" for k, v in Counter(map(failure_mode, logs[run])).most_common()))
    return table


def fig_yield(table):
    """The replay ladder: what each added piece of knowledge buys, on the 59 pick captures."""
    runs = [(r, l) for r, l, _, show in RUNS if show and r in table][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for y, (r, label) in enumerate(runs):
        k, _, n = table[r]
        colour = "0.8" if r == "Rref" else "tab:orange" if r in USES_POSITIONS else "tab:blue"
        ax.barh(y, 100 * k / n, color=colour)
        ax.text(100 * k / n + 1.5, y, f"{k}/{n}", va="center", fontsize=9)
    ax.set_yticks(range(len(runs))); ax.set_yticklabels([l for _, l in runs], fontsize=8)
    ax.set_xlim(0, 100); ax.set_xlabel("replays that complete the task in LIBERO [%]")
    ax.legend(handles=[Patch(color="tab:blue", label="human capture, no object positions used"),
                       Patch(color="tab:orange", label="human capture, object positions known"),
                       Patch(color="0.8", label="control")], fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "replay_yield.png", dpi=150); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    logs = {}
    for run, _, pattern, _ in RUNS:
        rows = [r for f in sorted(glob.glob(f"outputs/{pattern}")) for r in parse_log(f)]
        if rows:
            logs[run] = rows
    geo = grasp_geometry(cfg)
    ref = reference_geometry()
    r0 = {f"{r['chunk']}_{r['ep']}": r for r in logs.get("R0", [])}
    fig_grasp_geometry(geo, ref, r0)
    fig_jaw_axis(geo, ref)
    fig_yield(yield_table(logs))
    hz = [g["z_mm"] for e, g in geo.items() if e.startswith(("A_t0", "A_t8", "A_t6"))]
    print(f"\nreference grasp: r={np.mean([r['r_mm'] for r in ref]):.0f}+/-{np.std([r['r_mm'] for r in ref]):.0f} mm, "
          f"z={np.mean([r['z_mm'] for r in ref]):.0f}+/-{np.std([r['z_mm'] for r in ref]):.0f} mm, "
          f"jaw {np.mean([r['jaw_deg'] for r in ref]):.0f}+/-{np.std([r['jaw_deg'] for r in ref]):.0f} deg")
    print(f"human grasp (well-registered chunks A_t0/A_t8/A_t6): z={np.mean(hz):.0f}+/-{np.std(hz):.0f} mm, "
          f"jaw {np.mean([g['jaw_deg'] for g in geo.values()]):.0f}+/-{np.std([g['jaw_deg'] for g in geo.values()]):.0f} deg")
    print(f"figures -> {FIG}/")


if __name__ == "__main__":
    main()
