"""Exact tests for the retargeting maths, using synthetic ground truth."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from humanoid_challenge.geometry import apply, rigid_fit, sample_depth, unproject  # noqa: E402
from humanoid_challenge.markers import make_board, make_detector, solve_board_pose  # noqa: E402
from humanoid_challenge.record3d import load_r3d  # noqa: E402
from humanoid_challenge.synthetic import camera_matrix  # noqa: E402
from humanoid_challenge.trajectory import fuse  # noqa: E402
from make_fake_r3d import build  # noqa: E402

LAYOUT = {0: (0.110, 0.500, 0.0), 1: (0.110, -0.250, 0.0),
          2: (-0.155, -0.250, 0.0), 3: (-0.155, 0.500, 0.0)}
L = 0.100


def test_unproject_roundtrip():
    K = camera_matrix(1280, 960, 1100.0)
    for p in [np.array([0.1, -0.05, 0.7]), np.array([-0.3, 0.2, 1.4])]:
        uv = (K @ p)[:2] / p[2]
        assert np.allclose(unproject(uv, p[2], K), p, atol=1e-9)


def test_rigid_fit_recovers_a_known_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(6, 3))
    ang = 0.7
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    t = np.array([0.3, -0.2, 0.05])
    dst = src @ R.T + t
    T, mx, mean = rigid_fit(src, dst)
    assert mx < 1e-9
    assert np.allclose(T[:3, :3], R, atol=1e-9)
    assert np.allclose(T[:3, 3], t, atol=1e-9)


def test_rigid_fit_reports_residual_when_points_disagree():
    """Taping error must surface as a residual, not be absorbed silently."""
    src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    dst = src.copy()
    dst[2] += np.array([0.02, 0, 0])           # one marker 20 mm out of place
    _, mx, mean = rigid_fit(src, dst)
    assert 0.005 < mx < 0.02, f"residual {mx}"
    assert mean > 0


def test_sample_depth_prefers_the_near_surface():
    """A patch straddling a finger edge contains the table behind it; take the finger."""
    K = camera_matrix(640, 480, 500.0)
    depth = np.full((120, 160), 1.20, np.float32)
    depth[58:62, 78:82] = 0.60                  # a near object
    Kd = K.copy(); Kd[0] *= 160 / 640; Kd[1] *= 120 / 480
    d = sample_depth(depth, np.array([320.0, 240.0]), K, Kd, radius=3)
    assert 0.55 < d < 0.75, f"got {d}, expected the near surface"


def test_sample_depth_returns_nan_on_a_hole():
    K = camera_matrix(640, 480, 500.0)
    depth = np.full((120, 160), np.nan, np.float32)
    Kd = K.copy(); Kd[0] *= 160 / 640; Kd[1] *= 120 / 480
    assert np.isnan(sample_depth(depth, np.array([320.0, 240.0]), K, Kd))


def test_sample_depth_out_of_frame_is_nan():
    K = camera_matrix(640, 480, 500.0)
    depth = np.ones((120, 160), np.float32)
    Kd = K.copy(); Kd[0] *= 160 / 640; Kd[1] *= 120 / 480
    assert np.isnan(sample_depth(depth, np.array([-50.0, 240.0]), K, Kd))


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    """An episode shaped like the real protocol: markers for the first 2 s, then gone."""
    p = build(tmp_path_factory.mktemp("traj") / "ep.r3d", LAYOUT, L, n=40,
              w=1280, h=960, dw=320, dh=240, anchor_frames=24)
    cap = load_r3d(p)
    det, board = make_detector(), make_board(LAYOUT, L)
    poses = {}
    for i in range(cap.n_frames):
        g = cv2.cvtColor(cap.rgb(i), cv2.COLOR_RGB2GRAY)
        c, ids, _ = det.detectMarkers(g)
        pr = solve_board_pose(board, c, ids, cap.K)
        if pr.ok:
            poses[i] = pr.T_table_cam
    return cap, poses


def test_fusion_places_unanchored_frames_in_the_table_frame(episode):
    """The point of fusing: frames with no marker still land in the table frame."""
    cap, poses = episode
    traj = fuse(cap.T_world_cam, poses)
    assert traj.n_sightings >= 10
    assert not traj.anchored[-1], "fixture should have no markers at the end"
    # the fixture's ARKit world IS the table frame, so fused poses must match it
    err = np.linalg.norm(traj.T_table_cam[:, :3, 3] - cap.T_world_cam[:, :3, 3], axis=1)
    assert np.median(err) < 5e-3, f"median {np.median(err)*1000:.1f} mm"
    assert err.max() < 2e-2, f"max {err.max()*1000:.1f} mm"


def test_fusion_reports_drift(episode):
    cap, poses = episode
    traj = fuse(cap.T_world_cam, poses)
    assert traj.drift_m < 0.02 and traj.drift_deg < 2.0
    assert traj.fit_resid_m < 0.02


def test_fusion_refuses_an_unanchored_episode():
    """An episode with no marker sighting has no common frame and must fail loudly."""
    with pytest.raises(ValueError, match="no marker sightings"):
        fuse(np.tile(np.eye(4), (10, 1, 1)), {})
