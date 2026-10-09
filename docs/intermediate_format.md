# Intermediate trajectory format

The seam between the two halves of this project. `humanoid-vision` (OpenCV, ArUco,
hand pose) writes these files; `humanoid-sim` (LeRobot, LIBERO, MuJoCo) reads them. The
two environments never import each other -- they cannot, since they need incompatible
`cv2` builds -- so this file is the whole contract.

One file per recorded episode: `data/interim/<episode_id>.npz`, plus a sidecar
`<episode_id>.json` holding everything a human would want to read.

## Why this shape

The retargeting produces an **absolute end-effector path in the table frame**. It does
not produce actions. That is deliberate, and it follows from three measured facts:

- LIBERO's BDDL table frame and the robot end-effector frame share the same x,y axes
  (measured offset 0.002 +- 0.020 m over 30 reference demos). The table surface is at
  `z = 0.90 m`. So a pose in the ArUco table frame becomes a robot-frame pose by adding
  0.90 to z -- no learned mapping, no fitting.
- The gripper command is **binary** (-1 open, +1 closed) in every reference demo, never
  intermediate. So the file stores a boolean, not a continuous aperture.
- The simulator's controller cannot be driven open-loop from a path. Converting a path
  to actions requires the live measured pose at each step, so it must happen on the sim
  side, at replay time. See `src/humanoid_challenge/tracking.py`.

Storing a path rather than actions also means the same recording can be replayed under
different controllers, control rates and object layouts without re-running any vision.

## Arrays (`.npz`)

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `t` | `(T,)` | float64 | seconds from episode start, from the capture's own timestamps |
| `eef_pos` | `(T, 3)` | float64 | pinch-midpoint position, **table frame**, metres |
| `eef_rotmat` | `(T, 3, 3)` | float64 | pinch orientation, table frame, rotation matrix |
| `gripper` | `(T,)` | int8 | -1 open, +1 closed. Thresholded thumb-index distance |
| `pinch_mm` | `(T,)` | float64 | raw thumb-index tip distance in mm, before thresholding |
| `hand_conf` | `(T,)` | float64 | hand-pose confidence, 0-1; low values mark untrusted frames |
| `cam_T_table` | `(T, 4, 4)` | float64 | camera pose in the table frame, drift-corrected |
| `cam_T_table_arkit` | `(T, 4, 4)` | float64 | the same, from raw ARKit only -- kept for the drift figure |
| `markers_seen` | `(T, 4)` | bool | which of IDs 0-3 were detected in each frame |
| `object_pos` | `(3,)` | float64 | estimated start position of the manipulated object, table frame |
| `target_pos` | `(3,)` | float64 | estimated position of the target (plate / goal zone) |

`eef_pos` is the **thumb-tip/index-tip midpoint**, not the wrist. The Panda's end-effector
site sits between its fingers; a human wrist is roughly 10 cm behind that and rotated away
from it, so retargeting the wrist puts the gripper in the wrong place.

Rotation matrices rather than quaternions or axis-angle: no sign ambiguity, no convention
argument, and the consumer converts once at the point of use.

## Sidecar (`.json`)

```json
{
  "episode_id": "A_t2_003",
  "layout": "A",
  "task_suite": "libero_spatial",
  "task_ids": 2,
  "global_task_index": 38,
  "instruction": "pick up the black bowl from table center and place it on the plate",
  "source_clip": "data/raw/2026-10-06_1412.r3d",
  "n_frames": 412,
  "duration_s": 13.7,
  "capture_fps": 30.0,
  "marker_length_m": 0.1002,
  "board_calibration": "config/board_calibration.yaml",
  "table_scale": 1.0,
  "anchor_frames": 37,
  "arkit_drift_m": 0.021,
  "arkit_drift_deg": 1.4,
  "frames_without_hand": 6,
  "notes": ""
}
```

`arkit_drift_m` / `arkit_drift_deg` are the disagreement between the marker-derived pose
and the ARKit-propagated pose at the last frame where a marker was visible. They are the
honest error bar on everything derived from this episode, and they go in the README.

## Conventions, stated once

- **Table frame**: origin at ArUco ID 0's centre, x toward the operator, y to the
  operator's left, z up out of the table. Markers lie in z = 0.
- **Units**: metres, seconds, radians throughout. No millimetres except `pinch_mm`,
  which is named for it.
- **Time**: `t[0] = 0`. The capture rate is whatever the phone recorded; the sim side
  resamples. Do not assume 30 Hz.
- The sim side runs LIBERO at **`control_freq=20`**, not the 10 fps the
  `lerobot/libero` dataset declares. Replaying recorded actions at 10 Hz succeeds 0% of
  the time; at 20 Hz, 80%. This is a property of the simulator, not of this file --
  recorded here so nobody re-derives it.

## Validity rules

A reader may assume, and a writer must guarantee:

1. `T >= 2`, and all arrays share the same leading `T`.
2. `t` strictly increasing.
3. `eef_rotmat[i]` orthonormal with `det = +1` to 1e-6.
4. `gripper` contains only -1 and +1.
5. At least `anchor_frames >= 1` frames have any marker visible, and they are at the
   start -- the frame cannot be anchored retroactively.
6. `eef_pos` lies within the taped working area expanded by 0.15 m. Anything outside is
   a tracking failure, not a reach.

Writer-side violations should fail loudly at write time rather than produce a file that
breaks a training run six hours later.
