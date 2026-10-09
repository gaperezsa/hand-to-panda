
"""
Recover where the markers actually ended up, from a calibration sweep.

One slow camera pan with all four in view is enough to measure them, provided we use the
fact that they are coplanar.

There are 4 markers, with IDs 0, 1, 2 and 3, placed as docs/recording/setup_sheet.pdf states.

Method, in order:

1. Per frame, solve each visible marker's pose independently (IPPE, coplanar 4-point).
2. For every frame where markers i and j are both visible, record the relative transform
   i -> j. Averaging these over many viewpoints cancels most of the per-frame noise.
3. Chain outward from ID 0 by breadth-first search, so a marker never seen beside ID 0
   is still placed via one that was.
4. Project the result onto a common plane: fit the plane through the marker origins, drop
   each origin onto it, and keep only each marker's in-plane yaw.

This is where the coplanarity assumption pays off, since it removes the tilt error entirely
rather than averaging it down.

Output is a layout dict {id: (x, y, yaw)} in the table frame, with ID 0 at the origin.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .markers import make_board, marker_object_points


def _single_marker_poses(corners, ids, marker_length, K, dist):
    """T_cam_marker for each detected marker, solved independently."""
    out = {}
    if ids is None:
        return out
    obj = marker_object_points(0.0, 0.0, 0.0, marker_length)
    for c, i in zip(corners, np.ravel(ids)):
        ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float32), K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj, c.reshape(4, 2).astype(np.float32), K, dist,
                                          rvec, tvec)
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        out[int(i)] = T
    return out


def _average_transforms(Ts: list[np.ndarray]) -> np.ndarray:
    """Median translation, chordal-mean rotation. Median because a sweep inevitably
    contains a few frames with a marker caught at a grazing angle."""
    t = np.median(np.stack([T[:3, 3] for T in Ts]), axis=0)
    R = np.mean(np.stack([T[:3, :3] for T in Ts]), axis=0)
    U, _, Vt = np.linalg.svd(R)                       # nearest orthonormal matrix
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def calibrate_board(frames, marker_length, K, dist=None, detector=None, ref_id=0,
                    min_pair_observations=3):
    """frames: iterable of grayscale images from the calibration sweep."""
    from .markers import make_detector
    detector = detector or make_detector()
    dist = np.zeros(5) if dist is None else dist

    pairs: dict[tuple[int, int], list[np.ndarray]] = {}
    seen_counts: dict[int, int] = {}
    n_frames_used = 0
    for img in frames:
        corners, ids, _ = detector.detectMarkers(img)
        poses = _single_marker_poses(corners, ids, marker_length, K, dist)
        for i in poses:
            seen_counts[i] = seen_counts.get(i, 0) + 1
        if len(poses) >= 2:
            n_frames_used += 1
        for i, Ti in poses.items():
            for j, Tj in poses.items():
                if i != j:
                    pairs.setdefault((i, j), []).append(np.linalg.inv(Ti) @ Tj)

    rel = {k: _average_transforms(v) for k, v in pairs.items()
           if len(v) >= min_pair_observations}

    # BFS outward from the reference marker
    T_ref = {ref_id: np.eye(4)}
    q = deque([ref_id])
    while q:
        i = q.popleft()
        for (a, b), T in rel.items():
            if a == i and b not in T_ref:
                T_ref[b] = T_ref[i] @ T
                q.append(b)
    unreached = sorted(set(seen_counts) - set(T_ref))

    # Coplanar projection: fit a plane through the origins, then keep only yaw.
    origins = np.stack([T_ref[i][:3, 3] for i in sorted(T_ref)])
    centroid = origins.mean(axis=0)
    if len(origins) >= 3:
        _, _, Vt = np.linalg.svd(origins - centroid)
        normal = Vt[-1]
    else:
        normal = T_ref[ref_id][:3, 2]
    if normal @ T_ref[ref_id][:3, 2] < 0:
        normal = -normal

    # In-plane basis anchored on the reference marker's own x axis
    ex = T_ref[ref_id][:3, 0] - (T_ref[ref_id][:3, 0] @ normal) * normal
    ex /= np.linalg.norm(ex)
    ey = np.cross(normal, ex)
    origin = T_ref[ref_id][:3, 3]

    layout, residual_mm = {}, {}
    for i in sorted(T_ref):
        p = T_ref[i][:3, 3] - origin
        x, y, z = p @ ex, p @ ey, p @ normal
        xa = T_ref[i][:3, 0] - (T_ref[i][:3, 0] @ normal) * normal
        yaw = float(np.arctan2(xa @ ey, xa @ ex))
        layout[i] = (float(x), float(y), yaw)
        residual_mm[i] = float(abs(z) * 1000)         # how far off-plane it really was

    return dict(layout=layout, out_of_plane_mm=residual_mm, seen_counts=seen_counts,
                n_frames_with_pair=n_frames_used, unreached=unreached,
                pair_observations={f"{a}->{b}": len(v) for (a, b), v in pairs.items()})


def layout_to_yaml(layout: dict[int, tuple[float, float, float]], extra: dict | None = None) -> str:
    import yaml
    d = {"markers": {int(k): [round(v[0], 6), round(v[1], 6), round(v[2], 6)]
                     for k, v in sorted(layout.items())}}
    if extra:
        d.update(extra)
    return yaml.safe_dump(d, sort_keys=True)
