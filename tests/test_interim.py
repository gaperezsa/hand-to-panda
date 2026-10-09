"""The intermediate-format writer must reject malformed episodes at write time."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from humanoid_challenge.interim import read, validate, workspace_check, write  # noqa: E402


def good(T=20):
    return dict(t=np.arange(T) / 30.0,
                eef_pos=np.random.default_rng(0).normal(0, .05, (T, 3)),
                eef_rotmat=np.tile(np.eye(3), (T, 1, 1)),
                gripper=np.where(np.arange(T) < T // 2, -1, 1).astype(np.int8),
                pinch_mm=np.linspace(60, 10, T))


def test_roundtrip(tmp_path):
    a = good()
    p = write(tmp_path, "A_t2_001", a, {"task_ids": 2, "instruction": "x"})
    d, meta = read(p)
    assert meta["episode_id"] == "A_t2_001"
    assert np.allclose(d["eef_pos"], a["eef_pos"])
    assert d["gripper"].dtype == np.int8


def test_rejects_non_monotonic_time(tmp_path):
    a = good(); a["t"][5] = a["t"][4]
    with pytest.raises(ValueError, match="strictly increasing"):
        write(tmp_path, "bad", a, {})


def test_rejects_non_orthonormal_rotation(tmp_path):
    a = good(); a["eef_rotmat"][3] *= 1.05
    with pytest.raises(ValueError, match="orthonormal"):
        write(tmp_path, "bad", a, {})


def test_rejects_reflection(tmp_path):
    """A determinant of -1 is orthonormal but mirrored -- it would flip the gripper."""
    a = good(); a["eef_rotmat"][7] = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="determinant"):
        write(tmp_path, "bad", a, {})


def test_rejects_non_binary_gripper(tmp_path):
    a = good(); a["gripper"] = a["gripper"].astype(float); a["gripper"][2] = 0.4
    with pytest.raises(ValueError, match="gripper contains"):
        write(tmp_path, "bad", a, {})


def test_rejects_ragged_arrays(tmp_path):
    a = good(); a["pinch_mm"] = a["pinch_mm"][:-3]
    with pytest.raises(ValueError, match="rows but 't' has"):
        write(tmp_path, "bad", a, {})


def test_rejects_nan_positions(tmp_path):
    a = good(); a["eef_pos"][4, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        write(tmp_path, "bad", a, {})


def test_workspace_check_flags_escapes():
    pos = np.zeros((10, 3)); pos[3] = [5.0, 0, 0]
    n, frac = workspace_check(pos, (-0.232, 0.182, -0.122, 0.372))
    assert n == 1 and 0 < frac < 1
