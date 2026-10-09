"""
As described in docs/intermediate_format.md, this module reads and writes the intermediate
format used to store episodes.

The writer enforces the spec's validity rules at write time.

A wrong or corrupted episode file should fail here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REQUIRED = ("t", "eef_pos", "eef_rotmat", "gripper")


def validate(d: dict, episode_id: str = "?") -> None:
    def bad(msg):
        raise ValueError(f"episode {episode_id}: {msg}")

    for k in REQUIRED:
        if k not in d:
            bad(f"missing array '{k}'")
    T = len(d["t"])
    if T < 2:
        bad(f"only {T} frames")
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and k not in ("object_pos", "target_pos"):
            if len(v) != T:
                bad(f"'{k}' has {len(v)} rows but 't' has {T}")
    if not np.all(np.diff(d["t"]) > 0):
        bad("timestamps are not strictly increasing")
    R = np.asarray(d["eef_rotmat"], float)
    if R.shape[1:] != (3, 3):
        bad(f"eef_rotmat has shape {R.shape}, expected (T, 3, 3)")
    err = np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max()
    if err > 1e-6:
        bad(f"eef_rotmat is not orthonormal (max |RRt - I| = {err:.2e})")
    dets = np.linalg.det(R)
    if np.abs(dets - 1).max() > 1e-6:
        bad(f"eef_rotmat determinant ranges {dets.min():.6f}..{dets.max():.6f}, expected +1")
    g = np.asarray(d["gripper"])
    if not set(np.unique(g)).issubset({-1, 1}):
        bad(f"gripper contains {sorted(set(np.unique(g).tolist()))}, expected only -1 and +1")
    if not np.all(np.isfinite(d["eef_pos"])):
        bad("eef_pos contains non-finite values")


def write(out_dir: Path, episode_id: str, arrays: dict, meta: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validate(arrays, episode_id)
    npz = out_dir / f"{episode_id}.npz"
    np.savez_compressed(npz, **{k: v for k, v in arrays.items() if isinstance(v, np.ndarray)})
    (out_dir / f"{episode_id}.json").write_text(
        json.dumps({"episode_id": episode_id, **meta}, indent=1, default=float))
    return npz


def read(npz_path: Path):
    npz_path = Path(npz_path)
    d = dict(np.load(npz_path))
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    validate(d, meta.get("episode_id", npz_path.stem))
    return d, meta


def workspace_check(eef_pos: np.ndarray, layout_bounds, margin: float = 0.15):
    """Spec rule 6: a path outside the taped area is a tracking failure, not a reach."""
    (x0, x1, y0, y1) = layout_bounds
    out = ((eef_pos[:, 0] < x0 - margin) | (eef_pos[:, 0] > x1 + margin) |
           (eef_pos[:, 1] < y0 - margin) | (eef_pos[:, 1] > y1 + margin))
    return int(out.sum()), float(out.mean())
