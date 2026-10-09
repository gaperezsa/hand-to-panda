#!/usr/bin/env python3
"""Quality-check one recorded chunk: consistency, and which episodes to keep.

The bowl and plate do not move within a chunk, so the grasp and release points should
repeat. Their spread is the single best measure of whether the whole chain -- markers,
frame fusion, hand tracking, depth lifting -- is working on this footage. An episode is
rejected when it is physically wrong (no clean grasp, no real lift, grasp nowhere near
the bowl), not merely noisy.

    python tools/check_chunk.py A_t2
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from humanoid_challenge.interim import read  # noqa: E402

MIN_LIFT_MM = 100.0
# An episode is an outlier if its grasp sits far from where the OTHER episodes in the
# same chunk grasped -- not from the nominal mark. The simulated scene is set from the
# capture, so a bowl placed a few cm off its mark is harmless as long as it stayed there;
# what ruins an episode is the hand tracker losing the bowl in that one take.
# Measured: chunk A_t1 grasps clustered to +-13 mm but 76 mm from the nominal mark, and
# a nominal-distance test threw away 4 perfectly good episodes.
MAX_GRASP_SCATTER_MM = 45.0


def chunk_targets(cfg, chunk):
    a = cfg["layouts"]["A"]
    tgt = np.array(a["target_position_m"], dtype=float)
    for p in a["positions"]:
        if f"A_t{p['task_ids']}" == chunk:
            return np.array(p["start_m"], dtype=float), tgt, p["instruction"]
    raise SystemExit(f"unknown chunk {chunk}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk")
    ap.add_argument("--interim", type=Path, default=Path("data/interim"))
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    a = ap.parse_args()
    bowl, plate, instr = chunk_targets(yaml.safe_load(a.config.read_text()), a.chunk)
    print(f"{a.chunk}: \"{instr}\"")
    print(f"  bowl at ({bowl[0]:+.3f}, {bowl[1]:+.3f})   plate at ({plate[0]:+.3f}, {plate[1]:+.3f})\n")

    good, bad, cand = [], [], []
    for f in sorted(glob.glob(str(a.interim / f"{a.chunk}_*.npz"))):
        d, m = read(f)
        X, g = d["eef_pos"], d["gripper"]
        name = Path(f).stem
        tr = np.flatnonzero(np.diff(g) != 0)
        if len(tr) < 2:
            bad.append((name, None, None, 0, 0, f"only {len(tr)} gripper transition(s)")); continue
        gi, ri = tr[0], tr[-1]
        lift = (X[gi:ri + 1, 2].max() - X[gi, 2]) * 1000
        why = []
        if len(tr) != 2:
            why.append(f"{len(tr)} gripper transitions")
        if lift < MIN_LIFT_MM:
            why.append(f"lift {lift:.0f} mm")
        cand.append([name, X[gi, :2], X[ri, :2], lift, m["arkit_drift_deg"], why])

    # second pass: outliers measured against the chunk's own median grasp
    if cand:
        med = np.median(np.array([c[1] for c in cand]), axis=0)
        for c in cand:
            sc = float(np.linalg.norm(c[1] - med)) * 1000
            if sc > MAX_GRASP_SCATTER_MM:
                c[5].append(f"grasp {sc:.0f} mm off the chunk median")
    for c in cand:
        (bad if c[5] else good).append(tuple(c[:5]) + ("; ".join(c[5]),))

    for b in bad:
        print(f"  REJECT  {b[0]}  -- {b[-1]}")
    print(f"\n  accepted {len(good)}/{len(good) + len(bad)}")
    if not good:
        return 1
    G = np.array([x[1] for x in good]); R = np.array([x[2] for x in good])
    L = np.array([x[3] for x in good]); D = np.array([x[4] for x in good])
    print(f"  grasp    spread +-({G[:,0].std()*1000:.0f}, {G[:,1].std()*1000:.0f}) mm"
          f"   {np.linalg.norm(G.mean(0)-bowl)*1000:.0f} mm from the nominal mark")
    print(f"  release  spread +-({R[:,0].std()*1000:.0f}, {R[:,1].std()*1000:.0f}) mm"
          f"   {np.linalg.norm(R.mean(0)-plate)*1000:.0f} mm from plate centre")
    print(f"  lift     {L.mean():.0f} +- {L.std():.0f} mm")
    print(f"  drift    {D.mean():.2f} +- {D.std():.2f} deg")
    print(f"  hand displacement {np.round((R-G).mean(0),3)} vs needed {np.round(plate-bowl,3)}"
          f"  -> mismatch {np.linalg.norm((R-G).mean(0)-(plate-bowl))*1000:.0f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
