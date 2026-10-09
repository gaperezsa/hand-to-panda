#!/usr/bin/env python3
"""Check a Record3D capture before trusting it. Run this on the 10 s test clip FIRST.

Every check prints PASS/FAIL with the number that decided it, so a failure says what to
change about the recording rather than just that something is wrong.

    python tools/validate_capture.py data/raw/test.r3d
    python tools/validate_capture.py data/raw/test.r3d --sweep     # relax marker checks
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from humanoid_challenge.markers import make_board, make_detector, solve_board_pose  # noqa: E402
from humanoid_challenge.record3d import load_r3d  # noqa: E402

OK, BAD = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, passed, detail):
    results.append((OK if passed else BAD, name, detail))
    print(f"  [{OK if passed else BAD}] {name:34s} {detail}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--sweep", action="store_true",
                    help="calibration sweep: expect all four markers, not an anchor-then-hands episode")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    mk = cfg["markers"]
    marker_length = float(mk["marker_length_m"])
    if not mk.get("marker_length_measured", False):
        print("  NOTE  config.yaml still has marker_length_measured: false -- the value "
              f"below ({marker_length:.4f} m) is nominal, not your ruler reading.\n")

    layout_file = Path(mk.get("board_calibration", ""))
    if layout_file.exists():
        layout = {int(k): tuple(v) for k, v in yaml.safe_load(layout_file.read_text())["markers"].items()}
        src = str(layout_file)
    else:
        layout = {int(k): (v[0], v[1], 0.0) for k, v in mk["nominal_layout_m"].items()}
        src = "config.yaml nominal_layout_m (board not calibrated yet)"

    print(f"capture : {a.capture}")
    print(f"layout  : {src}\n")

    cap = load_r3d(a.capture)
    for n in cap.notes:
        print(f"  NOTE  {n}")
    if cap.notes:
        print()

    print("structure")
    check("frames", cap.n_frames >= 30, f"{cap.n_frames} frames, {cap.timestamps[-1]:.1f} s "
          f"at {cap.fps:g} fps")
    check("rgb resolution", min(cap.rgb_size) >= 480, f"{cap.rgb_size[0]}x{cap.rgb_size[1]}")
    check("depth resolution", min(cap.depth_size) >= 120, f"{cap.depth_size[0]}x{cap.depth_size[1]}")
    f = (cap.K[0, 0] + cap.K[1, 1]) / 2
    check("intrinsics plausible", 0.5 * cap.rgb_size[0] < f < 2.0 * cap.rgb_size[0],
          f"f={f:.0f}px  c=({cap.K[0,2]:.0f},{cap.K[1,2]:.0f})  for width {cap.rgb_size[0]}")
    dt = np.diff(cap.timestamps)
    check("timestamps monotonic", bool(np.all(dt > 0)),
          f"median dt {np.median(dt)*1000:.1f} ms, max gap {dt.max()*1000:.1f} ms")

    print("\ndepth")
    idx = list(range(0, cap.n_frames, max(1, cap.n_frames // 8)))[:8]
    meds, valids = [], []
    for i in idx:
        try:
            d = cap.depth(i)
        except Exception as e:
            check("depth decode", False, str(e))
            break
        valids.append(float(np.isfinite(d).mean()))
        meds.append(float(np.nanmedian(d)))
    else:
        check("depth decodes", True, f"{len(idx)} frames sampled")
        check("depth valid fraction", np.mean(valids) > 0.5,
              f"{np.mean(valids)*100:.0f}% of pixels finite")
        check("depth units look metric", 0.15 < np.median(meds) < 3.0,
              f"median {np.median(meds):.3f} m (expect ~0.4-0.8 m for an episode, ~1.2-1.5 m for a sweep)")

    print("\nARKit poses")
    t = cap.T_world_cam[:, :3, 3]
    check("poses move", float(np.linalg.norm(t - t[0], axis=1).max()) > 0.01,
          f"max displacement {np.linalg.norm(t - t[0], axis=1).max():.3f} m")
    R0 = cap.T_world_cam[:, :3, :3]
    orth = max(float(np.abs(R.T @ R - np.eye(3)).max()) for R in R0[::max(1, len(R0)//20)])
    check("rotations orthonormal", orth < 1e-4, f"max |RtR - I| = {orth:.2e}")

    print("\nmarkers")
    det = make_detector(mk["dictionary"])
    board = make_board(layout, marker_length, mk["dictionary"])
    import cv2
    seen, poses = [], []
    for i in range(0, cap.n_frames, a.stride):
        g = cv2.cvtColor(cap.rgb(i), cv2.COLOR_RGB2GRAY)
        corners, ids, _ = det.detectMarkers(g)
        got = sorted(int(v) for v in np.ravel(ids)) if ids is not None else []
        seen.append(got)
        p = solve_board_pose(board, corners, ids, cap.K)
        poses.append(p if p.ok else None)
    any_seen = sum(1 for s in seen if s)
    allfour = sum(1 for s in seen if len(s) >= 4)
    ids_ever = sorted({i for s in seen for i in s})
    check("markers detected at all", any_seen > 0,
          f"{any_seen}/{len(seen)} sampled frames; IDs seen {ids_ever}")
    check("all four IDs appear", set(ids_ever) >= set(mk["ids"]),
          f"expected {mk['ids']}, saw {ids_ever}")

    if a.sweep:
        check("sweep sees all four together", allfour >= 5,
              f"{allfour} frames with >=4 markers (need several, from different angles)")
    else:
        anchor_n = max(1, int(cfg['markers']['min_anchor_seconds'] * cap.fps / a.stride))
        anchored = sum(1 for s in seen[:anchor_n] if len(s) >= cfg['markers']['min_markers_at_start'])
        check("episode is anchored at start",
              anchored >= anchor_n * 0.5,
              f"{anchored}/{anchor_n} of the first {cfg['markers']['min_anchor_seconds']:g}s "
              f"have >={cfg['markers']['min_markers_at_start']} marker(s)")

    good = [(i, p) for i, p in enumerate(poses) if p is not None]
    if len(good) >= 2:
        rms = np.median([p.reproj_px for _, p in good])
        check("marker reprojection", rms < 2.0, f"median {rms:.2f} px over {len(good)} frames")
        # ARKit drift: marker-derived camera pose vs ARKit's, first solve against last
        i0, p0 = good[0]
        i1, p1 = good[-1]
        rel_marker = np.linalg.inv(p0.T_table_cam) @ p1.T_table_cam
        rel_arkit = np.linalg.inv(cap.T_world_cam[i0 * a.stride]) @ cap.T_world_cam[i1 * a.stride]
        d = np.linalg.inv(rel_marker) @ rel_arkit
        drift_m = float(np.linalg.norm(d[:3, 3]))
        drift_deg = float(np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1))))
        check("ARKit vs marker drift", drift_m < 0.05 and drift_deg < 5.0,
              f"{drift_m*1000:.0f} mm, {drift_deg:.2f} deg over "
              f"{cap.timestamps[i1*a.stride]-cap.timestamps[i0*a.stride]:.1f} s")

    if not a.sweep:
        print("\nhands")
        try:
            from humanoid_challenge.hands import HandTracker
            with HandTracker() as hands:
                got, pinches, scores = 0, [], []
                n = 0
                for i in range(0, cap.n_frames, a.stride):
                    hf = hands(cap.rgb(i), int(cap.timestamps[i] * 1000))
                    n += 1
                    if hf.ok:
                        got += 1
                        pinches.append(hf.pinch_m)
                        scores.append(hf.score)
            check("hand detected", got > 0.5 * n, f"{got}/{n} sampled frames"
                  + (f", mean confidence {np.mean(scores):.2f}" if scores else ""))
            if pinches:
                pin = np.array(pinches) * 1000
                check("pinch aperture varies", pin.max() - pin.min() > 15,
                      f"thumb-index {pin.min():.0f}-{pin.max():.0f} mm "
                      f"(needs a clear open/closed swing to threshold on)")
        except FileNotFoundError as e:
            print(f"  [skip] hand model missing: {e}")
        except Exception as e:
            print(f"  [skip] hand tracking unavailable: {type(e).__name__}: {e}")

    nbad = sum(1 for r, _, _ in results if r == BAD)
    print(f"\n{'=' * 64}\n{len(results) - nbad}/{len(results)} checks passed"
          f"{'' if nbad == 0 else f'  --  {nbad} FAILED'}")
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
