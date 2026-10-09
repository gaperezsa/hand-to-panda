"""Validate the marker geometry against synthetic views with known ground truth.

These run without any recorded data, which is the point: if the board geometry is
wrong, it should fail here and not after fifty episodes have been shot.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from humanoid_challenge.markers import (  # noqa: E402
    BoardPose, make_board, make_detector, marker_object_points, solve_board_pose,
)
from humanoid_challenge.synthetic import camera_matrix, look_at, render_board  # noqa: E402

MARKER_LENGTH = 0.100
# The nominal taped layout from config.yaml: markers left and right of the object area.
LAYOUT = {
    0: (0.107, 0.457, 0.0),
    1: (0.107, -0.207, 0.0),
    2: (-0.157, -0.207, 0.0),
    3: (-0.157, 0.457, 0.0),
}
W, H, FOCAL = 1280, 960, 1100.0
BOARD_CENTRE = np.array([-0.025, 0.125, 0.0])


def _solve(layout, eye, target=BOARD_CENTRE, marker_length=MARKER_LENGTH):
    K = camera_matrix(W, H, FOCAL)
    rvec_t, tvec_t = look_at(np.asarray(eye, float), np.asarray(target, float))
    img = render_board(layout, marker_length, K, rvec_t, tvec_t, W, H)
    corners, ids, _ = make_detector().detectMarkers(img)
    board = make_board(layout, marker_length)
    return solve_board_pose(board, corners, ids, K), (rvec_t, tvec_t), ids, img


def _pose_error(pose: BoardPose, rvec_t, tvec_t):
    pos = np.linalg.norm(pose.tvec.ravel() - tvec_t.ravel())
    R_est, _ = cv2.Rodrigues(pose.rvec)
    R_true, _ = cv2.Rodrigues(rvec_t)
    cos = (np.trace(R_est.T @ R_true) - 1) / 2
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return pos, ang


def test_marker_object_points_geometry():
    pts = marker_object_points(0.0, 0.0, 0.0, 0.1)
    assert pts.shape == (4, 3)
    assert np.allclose(pts[:, 2], 0.0), "markers must be flat in z=0"
    side = np.linalg.norm(pts[0] - pts[1])
    assert side == pytest.approx(0.1, abs=1e-9)
    # cv2.aruco order: top-left, top-right, bottom-right, bottom-left seen from +z
    assert pts[0][0] < pts[1][0] and pts[0][1] > pts[3][1]


def test_yaw_rotates_in_plane():
    pts = marker_object_points(0.0, 0.0, np.pi / 2, 0.1)
    assert np.allclose(pts[:, 2], 0.0)
    assert np.linalg.norm(pts[0] - pts[1]) == pytest.approx(0.1, abs=1e-9)
    assert pts[0][0] == pytest.approx(-0.05, abs=1e-9)   # TL rotates to (-h,-h)
    assert pts[0][1] == pytest.approx(-0.05, abs=1e-9)


def test_all_four_markers_recover_pose():
    eye = BOARD_CENTRE + np.array([0.10, 0.05, 0.95])
    pose, truth, ids, _ = _solve(LAYOUT, eye)
    assert pose.ok and pose.n_markers == 4, f"saw {pose.n_markers} markers"
    pos_err, ang_err = _pose_error(pose, *truth)
    assert pos_err < 2e-3, f"position error {pos_err*1000:.2f} mm"
    assert ang_err < 0.3, f"rotation error {ang_err:.3f} deg"
    assert pose.reproj_px < 1.0


@pytest.mark.parametrize("eye_offset", [
    (0.10, 0.05, 0.95),      # overhead, slightly off centre
    (0.35, -0.10, 0.70),     # oblique from the operator's side
    (-0.30, 0.30, 0.80),     # oblique from the far corner
    (0.05, 0.00, 1.40),      # high, as in the calibration sweep
])
def test_pose_from_several_viewpoints(eye_offset):
    pose, truth, _, _ = _solve(LAYOUT, BOARD_CENTRE + np.array(eye_offset))
    assert pose.ok, "no pose recovered"
    pos_err, ang_err = _pose_error(pose, *truth)
    assert pos_err < 3e-3, f"position error {pos_err*1000:.2f} mm"
    assert ang_err < 0.5, f"rotation error {ang_err:.3f} deg"


@pytest.mark.parametrize("visible", [[0], [0, 1], [1, 2], [0, 1, 2]])
def test_pose_from_marker_subsets(visible):
    """Head-mounted, most frames see one or two markers, not four."""
    sub = {k: v for k, v in LAYOUT.items() if k in visible}
    centre = np.array([np.mean([v[0] for v in sub.values()]),
                       np.mean([v[1] for v in sub.values()]), 0.0])
    # Stand back far enough that the subset actually fits in frame. Markers 0 and 1 are
    # 66 cm apart, which at 60 cm height simply does not fit -- the same reason the real
    # calibration sweep has to be shot from higher than the episodes.
    xs = [v[0] for v in sub.values()] + [v[0] for v in sub.values()]
    ys = [v[1] for v in sub.values()]
    extent = max(max(xs) - min(xs), max(ys) - min(ys)) + 0.15
    height = max(0.60, 1.6 * extent * FOCAL / min(W, H))
    pose, truth, _, _ = _solve(sub, centre + np.array([0.05, 0.05, height]), target=centre)
    assert pose.ok, f"no pose from markers {visible}"
    pos_err, ang_err = _pose_error(pose, *truth)
    # Tolerances measured on the printed board, with ~2x headroom on the
    # measured medians. The loose rotation bound is NOT slack: pairs that share an x
    # (0+1, 2+3) are a 5.4:1 elongated configuration seen from 1.4 m, and genuinely
    # resolve rotation ~2.5x worse than a single marker seen from 0.55 m.
    elongated = len(visible) == 2 and len({LAYOUT[i][0] for i in visible}) == 1
    tol_pos, tol_ang = (4e-3, 1.2) if elongated else (4e-3, 0.6)
    assert pos_err < tol_pos, f"{visible}: position error {pos_err*1000:.2f} mm"
    assert ang_err < tol_ang, f"{visible}: rotation error {ang_err:.3f} deg"


def test_marker_length_error_scales_the_translation():
    """A mis-measured printed marker biases every distance by the same factor.

    This is why config.yaml demands the ruler value: it is a pure scale error on the
    whole reconstruction, and nothing downstream can detect or correct it.
    """
    eye = BOARD_CENTRE + np.array([0.0, 0.0, 1.30])   # straight down, degenerate "up"
    K = camera_matrix(W, H, FOCAL)
    rvec_t, tvec_t = look_at(eye, BOARD_CENTRE)
    img = render_board(LAYOUT, MARKER_LENGTH, K, rvec_t, tvec_t, W, H)
    corners, ids, _ = make_detector().detectMarkers(img)

    wrong = 0.98 * MARKER_LENGTH                       # believe the marker is 2% smaller
    layout_wrong = {k: (x * 0.98, y * 0.98, yaw) for k, (x, y, yaw) in LAYOUT.items()}
    pose = solve_board_pose(make_board(layout_wrong, wrong), corners, ids, K)
    assert pose.ok
    ratio = np.linalg.norm(pose.tvec) / np.linalg.norm(tvec_t)
    assert ratio == pytest.approx(0.98, rel=0.02), f"scale ratio {ratio:.4f}"


def test_no_markers_returns_not_ok():
    board = make_board(LAYOUT, MARKER_LENGTH)
    assert not solve_board_pose(board, [], None, camera_matrix(W, H, FOCAL)).ok


# --------------------------------------------------------------------------------------
# Board calibration: can a sweep recover where the markers actually are?
# --------------------------------------------------------------------------------------

def _sweep_images(layout, marker_length=MARKER_LENGTH, n=28, seed=0):
    """Synthetic calibration sweep: a slow pan with all markers in view."""
    rng = np.random.default_rng(seed)
    K = camera_matrix(W, H, FOCAL)
    centre = np.array([np.mean([v[0] for v in layout.values()]),
                       np.mean([v[1] for v in layout.values()]), 0.0])
    imgs = []
    for i in range(n):
        a = 2 * np.pi * i / n
        eye = centre + np.array([0.22 * np.cos(a), 0.22 * np.sin(a),
                                 1.45 + 0.1 * rng.uniform(-1, 1)])
        rvec, tvec = look_at(eye, centre + rng.normal(0, 0.01, 3))
        imgs.append(render_board(layout, marker_length, K, rvec, tvec, W, H))
    return imgs, K


def test_board_calibration_recovers_a_taped_layout():
    """Markers taped by hand: offset from nominal, each at its own yaw."""
    from humanoid_challenge.board_calibration import calibrate_board

    rng = np.random.default_rng(3)
    true = {}
    for mid, (x, y, _) in LAYOUT.items():
        true[mid] = (x + rng.uniform(-.02, .02), y + rng.uniform(-.02, .02),
                     rng.uniform(-np.pi / 6, np.pi / 6))
    imgs, K = _sweep_images(true)
    res = calibrate_board(imgs, MARKER_LENGTH, K)

    assert not res["unreached"], f"markers never linked to ID 0: {res['unreached']}"
    assert set(res["layout"]) == set(true)

    # ID 0 defines the frame -- and it has a yaw of its own, so the recovered layout
    # lives in ID 0's rotated frame, not in table axes. Rotate the ground truth into it.
    yaw0 = true[0][2]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    for mid in sorted(true):
        dx = true[mid][0] - true[0][0]
        dy = true[mid][1] - true[0][1]
        ex, ey = c * dx - s * dy, s * dx + c * dy
        gx, gy, _ = res["layout"][mid]
        err = np.hypot(gx - ex, gy - ey)
        assert err < 5e-3, f"marker {mid} position off by {err*1000:.1f} mm"
        dyaw = np.degrees(abs(np.arctan2(np.sin(res['layout'][mid][2] - (true[mid][2] - true[0][2])),
                                         np.cos(res['layout'][mid][2] - (true[mid][2] - true[0][2])))))
        assert dyaw < 2.0, f"marker {mid} yaw off by {dyaw:.2f} deg"


def test_board_calibration_reports_out_of_plane_residual():
    """The residual is the honest check that the markers really are flat."""
    from humanoid_challenge.board_calibration import calibrate_board
    imgs, K = _sweep_images({k: (v[0], v[1], 0.0) for k, v in LAYOUT.items()})
    res = calibrate_board(imgs, MARKER_LENGTH, K)
    for mid, mm in res["out_of_plane_mm"].items():
        assert mm < 5.0, f"marker {mid} fitted {mm:.1f} mm off-plane for a flat board"
