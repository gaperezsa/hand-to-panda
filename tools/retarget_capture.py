#!/usr/bin/env python3
"""One capture -> one intermediate-format episode. The whole vision half, end to end.

    python tools/retarget_capture.py data/raw/A_t2_001.r3d
    python tools/retarget_capture.py 'data/raw/A_*.r3d' --glob

Everything it decides is printed, including the frames it had to discard and why. The
gripper thresholds are the one thing that must be set from your own hand -- run
`--report-pinch` on a few episodes first and read the histogram.
"""
import argparse
import glob as globmod
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from humanoid_challenge import interim  # noqa: E402
from humanoid_challenge.geometry import rigid_fit  # noqa: E402
from humanoid_challenge.hands import HandTracker  # noqa: E402
from humanoid_challenge.markers import make_board, make_detector, solve_board_pose  # noqa: E402
from humanoid_challenge.record3d import load_r3d  # noqa: E402
from humanoid_challenge.retarget import retarget  # noqa: E402
from humanoid_challenge.trajectory import fuse  # noqa: E402


def load_layout(cfg):
    """Measured board if calibrated, nominal otherwise, plus the transform into
    LIBERO's table frame (the board is anchored at ID 0; LIBERO's frame is defined by
    where the markers were meant to go)."""
    mk = cfg["markers"]
    nominal = {int(k): (float(v[0]), float(v[1]), 0.0) for k, v in mk["nominal_layout_m"].items()}
    cal = Path(mk.get("board_calibration", ""))
    if not cal.exists():
        return nominal, np.eye(4), 0.0, "nominal (board not calibrated)"
    measured = {int(k): tuple(v) for k, v in yaml.safe_load(cal.read_text())["markers"].items()}
    ids = sorted(set(measured) & set(nominal))
    src = np.array([[measured[i][0], measured[i][1], 0.0] for i in ids])
    dst = np.array([[nominal[i][0], nominal[i][1], 0.0] for i in ids])
    T, mx, mean = rigid_fit(src, dst)
    return measured, T, mx, f"measured ({cal}), fit residual max {mx*1000:.1f} mm"


def process(path, cfg, args):
    mk = cfg["markers"]
    L = float(mk["marker_length_m"])
    layout, T_lib_board, fit_mx, layout_src = load_layout(cfg)

    cap = load_r3d(path)
    for n in cap.notes:
        print(f"    note: {n}")

    det = make_detector(mk["dictionary"])
    board = make_board(layout, L, mk["dictionary"])
    poses = {}
    for i in range(0, cap.n_frames, args.stride):
        g = cv2.cvtColor(cap.rgb(i), cv2.COLOR_RGB2GRAY)
        c, ids, _ = det.detectMarkers(g)
        p = solve_board_pose(board, c, ids, cap.K)
        if p.ok and p.reproj_px < 3.0:
            poses[i] = T_lib_board @ p.T_table_cam
    traj = fuse(cap.T_world_cam, poses)

    with HandTracker(num_hands=1) as hands:
        hfs = [hands(cap.rgb(i), int(cap.timestamps[i] * 1000))
               for i in range(0, cap.n_frames, args.stride)]
    n_hand = sum(h.ok for h in hfs)

    print(f"    layout      : {layout_src}")
    print(f"    marker      : {traj.n_sightings} sightings, drift "
          f"{traj.drift_m*1000:.0f} mm / {traj.drift_deg:.2f} deg, fit resid "
          f"{traj.fit_resid_m*1000:.1f} mm")
    print(f"    hands       : {n_hand}/{len(hfs)} frames")

    if args.report_pinch:
        # Report the DEPTH-DERIVED pinch, which is what retarget() thresholds on.
        # MediaPipe's own pinch_m comes from a generic metric hand model, not from this
        # capture, so thresholds taken from it would not match what is actually used.
        from humanoid_challenge.retarget import lift_point
        Krgb, Kd = cap.K, cap.K_depth()
        from humanoid_challenge.hands import INDEX_TIP, THUMB_TIP
        pin = []
        for k, h in enumerate(hfs):
            if not h.ok:
                continue
            i = k * args.stride
            dep = cap.depth(i)
            a = lift_point(h.landmarks_px[THUMB_TIP], dep, Krgb, Kd, traj.T_table_cam[i])
            b = lift_point(h.landmarks_px[INDEX_TIP], dep, Krgb, Kd, traj.T_table_cam[i])
            if a is not None and b is not None:
                pin.append(float(np.linalg.norm(a - b)) * 1000)
        if pin:
            pin = np.asarray(pin, dtype=float)
            qs = np.percentile(pin, [2, 10, 25, 50, 75, 90, 98])
            print(f"    pinch (mm)  : p2 {qs[0]:.0f}  p10 {qs[1]:.0f}  p25 {qs[2]:.0f}  "
                  f"med {qs[3]:.0f}  p75 {qs[4]:.0f}  p90 {qs[5]:.0f}  p98 {qs[6]:.0f}")
            # The pinch signal is bimodal: a tight cluster while gripping and a broad
            # one while the hand is open. Percentiles land inside a mode; what we want
            # is the valley between them, so split with Otsu and set the hysteresis band
            # symmetrically around it.
            h, edges = np.histogram(pin, bins=48, range=(0, np.percentile(pin, 99)))
            cen = (edges[:-1] + edges[1:]) / 2
            w0 = np.cumsum(h); w1 = w0[-1] - w0
            m0 = np.cumsum(h * cen) / np.maximum(w0, 1)
            m1 = (np.sum(h * cen) - np.cumsum(h * cen)) / np.maximum(w1, 1)
            k = int(np.argmax(w0 * w1 * (m0 - m1) ** 2))
            split = cen[k]
            lo = float(np.median(pin[pin <= split])) if (pin <= split).any() else split
            hi = float(np.median(pin[pin > split])) if (pin > split).any() else split
            closed, opened = split * 0.75 + lo * 0.25, split * 0.75 + hi * 0.25
            print(f"    modes       : gripping ~{lo:.0f} mm, open ~{hi:.0f} mm, "
                  f"valley at {split:.0f} mm")
            print(f"    suggestion  : --pinch-closed {closed:.0f} --pinch-open {opened:.0f}")
        return None

    res = retarget(cap, traj, hfs,
                   pinch_closed_mm=args.pinch_closed, pinch_open_mm=args.pinch_open,
                   stride=args.stride, min_state_frames=args.min_state_frames)
    for n in res.notes:
        print(f"    note: {n}")

    flips = int((np.diff(res.gripper) != 0).sum())
    print(f"    retarget    : {res.valid.sum()}/{len(res.valid)} usable, "
          f"path {np.linalg.norm(np.diff(res.eef_pos, axis=0), axis=1).sum():.3f} m, "
          f"z {res.eef_pos[:,2].min():.3f}..{res.eef_pos[:,2].max():.3f} m, "
          f"{flips} gripper transition(s)")
    if flips == 0:
        print("    WARNING: the gripper never changes state -- thresholds are probably "
              "wrong for your hand. Run with --report-pinch.")

    ep = Path(path).stem
    arrays = dict(t=res.t, eef_pos=res.eef_pos, eef_rotmat=res.eef_rotmat,
                  gripper=res.gripper, pinch_mm=res.pinch_mm, hand_conf=res.hand_conf,
                  valid=res.valid, cam_T_table=traj.T_table_cam[::args.stride],
                  anchored=traj.anchored[::args.stride])
    meta = dict(source_clip=str(path), n_frames=int(len(res.t)),
                duration_s=float(res.t[-1]), capture_fps=float(cap.fps),
                marker_length_m=L, arkit_drift_m=traj.drift_m,
                arkit_drift_deg=traj.drift_deg, board_fit_resid_m=fit_mx,
                n_marker_sightings=int(traj.n_sightings),
                frames_without_hand=int(len(hfs) - n_hand),
                layout=ep.split("_")[0], task_ids=int(ep.split("_")[1].lstrip("t"))
                if "_t" in ep else None)
    out = interim.write(args.out, ep, arrays, meta)
    print(f"    wrote       : {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--glob", action="store_true")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/interim"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pinch-closed", type=float, default=35.0)
    ap.add_argument("--pinch-open", type=float, default=55.0)
    ap.add_argument("--min-state-frames", type=int, default=8)
    ap.add_argument("--report-pinch", action="store_true",
                    help="print the thumb-index distribution instead of writing output")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    if not cfg["markers"].get("marker_length_measured", False):
        print("WARNING: config.yaml marker_length_measured is false -- every distance "
              "below inherits the printer's scale error.\n")

    paths = sorted(globmod.glob(a.capture)) if a.glob else [a.capture]
    if not paths:
        raise SystemExit(f"no captures matched {a.capture!r}")
    nok = 0
    for p in paths:
        print(f"\n{p}")
        try:
            if process(p, cfg, a) is not None:
                nok += 1
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
    if not a.report_pinch:
        print(f"\n{nok}/{len(paths)} captures retargeted -> {a.out}")


if __name__ == "__main__":
    main()
