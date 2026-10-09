#!/usr/bin/env python3
"""Measure where the markers actually ended up, from the calibration sweep clip.

Writes config/board_calibration.yaml. Run once after taping, and again if anything is
bumped. Everything downstream reads the result instead of the nominal layout.

    python tools/calibrate_board_from_sweep.py data/raw/sweep.r3d
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from humanoid_challenge.board_calibration import calibrate_board, layout_to_yaml  # noqa: E402
from humanoid_challenge.markers import make_detector  # noqa: E402
from humanoid_challenge.record3d import load_r3d  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--out", type=Path, default=Path("config/board_calibration.yaml"))
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    mk = cfg["markers"]
    L = float(mk["marker_length_m"])
    if not mk.get("marker_length_measured", False):
        print("WARNING: config.yaml marker_length_measured is still false. The calibrated "
              "layout will carry the same scale error as the nominal marker size.\n")

    cap = load_r3d(a.sweep)
    for n in cap.notes:
        print(f"NOTE  {n}")
    print(f"{cap.n_frames} frames, using every {a.stride}")

    frames = (cv2.cvtColor(cap.rgb(i), cv2.COLOR_RGB2GRAY)
              for i in range(0, cap.n_frames, a.stride))
    res = calibrate_board(frames, L, cap.K, detector=make_detector(mk["dictionary"]))

    print(f"\nframes with >=2 markers : {res['n_frames_with_pair']}")
    print(f"times each marker seen  : {dict(sorted(res['seen_counts'].items()))}")
    if res["unreached"]:
        print(f"!! never linked to ID 0 : {res['unreached']}  -- reshoot the sweep so these "
              f"appear in the same frame as a marker that was")

    print("\nmeasured layout (metres, relative to ID 0, in ID 0's frame):")
    for i, (x, y, yaw) in sorted(res["layout"].items()):
        print(f"  ID {i}:  x {x:+.4f}  y {y:+.4f}  yaw {np.degrees(yaw):+7.2f} deg"
              f"   | out-of-plane {res['out_of_plane_mm'][i]:.1f} mm")

    # Compare to the intended layout only AFTER aligning the frames. ID 0's frame is set
    # by how ID 0 happens to be taped, so raw coordinates differ from the nominal ones by
    # whatever rotation that is -- comparing them directly reports a huge, meaningless
    # error. The rigid fit is also what the retargeting uses to reach LIBERO's frame.
    from humanoid_challenge.geometry import rigid_fit
    nominal = {int(k): v for k, v in mk["nominal_layout_m"].items()}
    ids = sorted(set(res["layout"]) & set(nominal))
    src = np.array([[res["layout"][i][0], res["layout"][i][1], 0.0] for i in ids])
    dst = np.array([[nominal[i][0], nominal[i][1], 0.0] for i in ids])
    T, mx, mean = rigid_fit(src, dst)
    aligned = src @ T[:3, :3].T + T[:3, 3]
    yaw_to_libero = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    print(f"\nafter aligning ID 0's frame to LIBERO's (rotation {yaw_to_libero:+.1f} deg):")
    for k, i in enumerate(ids):
        e = np.hypot(*(aligned[k][:2] - dst[k][:2])) * 1000
        print(f"  ID {i}:  measured ({aligned[k][0]:+.4f}, {aligned[k][1]:+.4f})   "
              f"intended ({dst[k][0]:+.4f}, {dst[k][1]:+.4f})   off by {e:5.1f} mm")
    print(f"  taping error: {mean*1000:.1f} mm mean, {mx*1000:.1f} mm worst")

    # side lengths are the cleanest check on both taping and marker_length
    import itertools
    print("\n  side lengths, measured vs intended (centre to centre):")
    for a_, b_ in itertools.combinations(ids, 2):
        m = np.hypot(res["layout"][a_][0]-res["layout"][b_][0],
                     res["layout"][a_][1]-res["layout"][b_][1])*1000
        n_ = np.hypot(nominal[a_][0]-nominal[b_][0], nominal[a_][1]-nominal[b_][1])*1000
        print(f"    {a_}-{b_}: {m:7.1f} mm   intended {n_:7.1f} mm   diff {m-n_:+6.1f} mm")

    worst = max(res["out_of_plane_mm"].values())
    print(f"\nworst out-of-plane residual: {worst:.1f} mm "
          f"({'flat enough' if worst < 5 else 'CHECK: a marker is lifted or curled'})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(layout_to_yaml(res["layout"], extra={
        "marker_length_m": L,
        "source_sweep": str(a.sweep),
        "out_of_plane_mm": {int(k): round(v, 2) for k, v in res["out_of_plane_mm"].items()},
        "seen_counts": {int(k): int(v) for k, v in res["seen_counts"].items()},
    }))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
