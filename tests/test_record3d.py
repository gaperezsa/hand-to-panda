"""Round-trip the Record3D parser against a synthetic bundle with known ground truth."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from humanoid_challenge.markers import make_board, make_detector, solve_board_pose  # noqa: E402
from humanoid_challenge.record3d import load_r3d  # noqa: E402
from make_fake_r3d import build  # noqa: E402

LAYOUT = {0: (0.110, 0.500, 0.0), 1: (0.110, -0.250, 0.0),
          2: (-0.155, -0.250, 0.0), 3: (-0.155, 0.500, 0.0)}
L = 0.100


@pytest.fixture(scope="module")
def cap(tmp_path_factory):
    p = build(tmp_path_factory.mktemp("r3d") / "fake.r3d", LAYOUT, L, n=20)
    return load_r3d(p)


@pytest.fixture(scope="module")
def cap_hires(tmp_path_factory):
    """The convention test needs markers that are more than ~40 px wide, or pose noise
    swamps the thing being tested. Measured: absolute camera-pose error is 4.4 mm median
    at 640x480 (43 px markers) and 1.7 mm at 1280x960 (86 px) -- it halves as resolution
    doubles, which is noise, not bias. The phone records 1440x1920, so hi-res is also the
    more representative case."""
    p = build(tmp_path_factory.mktemp("r3dhi") / "fake_hi.r3d", LAYOUT, L, n=12,
              w=1280, h=960, dw=320, dh=240)
    return load_r3d(p)


def test_structure(cap):
    assert cap.n_frames == 20
    assert cap.rgb_size == (640, 480)
    assert cap.depth_size == (160, 120)
    assert cap.timestamps[0] == 0
    assert np.all(np.diff(cap.timestamps) > 0)


def test_intrinsics_column_major_recovered(cap):
    # K was written column-major; the parser must transpose it back.
    assert cap.K[2, 2] == pytest.approx(1.0)
    assert cap.K[0, 2] == pytest.approx(320.0, abs=1)
    assert cap.K[1, 2] == pytest.approx(240.0, abs=1)
    assert cap.K[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_depth_is_metric(cap):
    d = cap.depth(0)
    assert d.shape == (120, 160)
    med = float(np.nanmedian(d))
    assert 0.8 < med < 2.5, f"median depth {med} m is not a table-sized distance"


def test_depth_agrees_with_marker_pose(cap):
    """Depth and pose come from independent paths; they must agree on camera height."""
    det, board = make_detector(), make_board(LAYOUT, L)
    import cv2
    g = cv2.cvtColor(cap.rgb(0), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = det.detectMarkers(g)
    pose = solve_board_pose(board, corners, ids, cap.K)
    assert pose.ok
    cam_height = pose.T_table_cam[2, 3]
    centre_depth = float(np.nanmedian(cap.depth(0)))
    assert abs(cam_height - centre_depth) < 0.25, \
        f"marker says camera is {cam_height:.3f} m up, depth says {centre_depth:.3f} m"


def test_arkit_pose_convention_matches_markers(cap_hires):
    cap = cap_hires
    """The ARKit->OpenCV axis flip must be right, or relative motion disagrees."""
    det, board = make_detector(), make_board(LAYOUT, L)
    import cv2
    got = []
    for i in (0, cap.n_frames // 2, cap.n_frames - 1):
        g = cv2.cvtColor(cap.rgb(i), cv2.COLOR_RGB2GRAY)
        corners, ids, _ = det.detectMarkers(g)
        p = solve_board_pose(board, corners, ids, cap.K)
        assert p.ok
        got.append((i, p.T_table_cam))
    for (i0, A0), (i1, A1) in zip(got, got[1:]):
        rel_marker = np.linalg.inv(A0) @ A1
        rel_arkit = np.linalg.inv(cap.T_world_cam[i0]) @ cap.T_world_cam[i1]
        dt = np.linalg.norm(rel_marker[:3, 3] - rel_arkit[:3, 3])
        dR = np.degrees(np.arccos(np.clip(
            (np.trace(rel_marker[:3, :3].T @ rel_arkit[:3, :3]) - 1) / 2, -1, 1)))
        # tolerance from measurement: 1.7 mm median absolute pose error at this
        # resolution, and a relative comparison differences two independent solves.
        assert dt < 0.008, f"relative translation disagrees by {dt*1000:.1f} mm"
        assert dR < 0.5, f"relative rotation disagrees by {dR:.2f} deg"


def test_k_depth_scales(cap):
    Kd = cap.K_depth()
    assert Kd[0, 0] == pytest.approx(cap.K[0, 0] * 160 / 640, rel=1e-6)
    assert Kd[1, 2] == pytest.approx(cap.K[1, 2] * 120 / 480, rel=1e-6)


def test_absolute_pose_matches_markers(cap_hires):
    """In the fixture the ARKit world frame IS the table frame, so marker-derived and
    ARKit camera poses must agree outright -- a sharper check than relative motion,
    and the one that would catch a wrong axis flip rather than merely a noisy solve."""
    import cv2
    det, board = make_detector(), make_board(LAYOUT, L)
    errs = []
    for i in range(cap_hires.n_frames):
        g = cv2.cvtColor(cap_hires.rgb(i), cv2.COLOR_RGB2GRAY)
        corners, ids, _ = det.detectMarkers(g)
        p = solve_board_pose(board, corners, ids, cap_hires.K)
        if not p.ok:
            continue
        errs.append(np.linalg.norm(p.T_table_cam[:3, 3] - cap_hires.T_world_cam[i][:3, 3]))
    assert len(errs) >= 8
    assert np.median(errs) < 4e-3, f"median absolute pose error {np.median(errs)*1000:.1f} mm"
