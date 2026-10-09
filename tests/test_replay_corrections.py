"""The two path corrections must move only what they claim to move."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from humanoid_challenge.tracking import anchor_grasp, place_correction  # noqa: E402

GRASP, RELEASE = 12, 30


def path(n=40):
    p = np.zeros((n, 7))
    p[:, 0] = np.linspace(-0.20, 0.10, n)
    p[:, 1] = 0.05
    p[:, 2] = np.linspace(1.10, 0.95, n)
    p[:, 3:6] = [0.1, 0.2, 0.3]
    p[:, 6] = -1
    p[GRASP:RELEASE, 6] = 1
    return p


def test_anchor_lands_the_grasp_and_keeps_the_approach_above_it():
    p = path()
    p[:GRASP, 2] = 0.90                                             # approach dragging on the table
    anchor = (0.0, 0.047, 0.938)
    q, delta = anchor_grasp(p, GRASP, anchor)
    assert np.allclose(q[GRASP, :3], anchor)
    assert np.allclose(q[GRASP:, :3] - p[GRASP:, :3], delta)       # rigid after the grasp
    assert np.allclose(q[0, :2], p[0, :2])                          # approach start xy untouched
    assert np.all(q[:GRASP - 5, 2] >= anchor[2] + 0.03 - 1e-9)      # clearance until the descent
    assert np.all(np.diff(q[GRASP - 5:GRASP + 1, 2]) <= 1e-9)       # then monotone descent
    assert np.array_equal(q[:, 3:], p[:, 3:])                       # orientation, gripper


def test_anchor_then_place_lands_the_object_on_the_target():
    obj, tgt = (-0.05, 0.20), (0.06, 0.20)
    q, _ = anchor_grasp(path(), GRASP, (obj[0], obj[1] + 0.047, 0.938))
    q, _ = place_correction(q, GRASP, RELEASE, obj, tgt)
    held = q[GRASP, :2] - np.asarray(obj)
    assert np.allclose(q[RELEASE, :2] - held, tgt)
