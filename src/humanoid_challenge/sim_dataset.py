"""
Here we replay end-effector paths in LIBERO and record them as a LeRobot dataset.

The capture supplies the actions, the simulator the observations.

Using these, a path is tracked and at every control step the two cameras, the proprioceptive
state and the action sent are recorded as frames, in the exact schema of lerobot/libero.

Two conventions copied from it: frames are rotated 180 degrees, and the dataset declares
fps 10 although control runs at 20 Hz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robosuite.utils.transform_utils as T

from .tracking import axisangle_to_matrix, track_step

DUMMY = [0, 0, 0, 0, 0, 0, -1]

# Matches lerobot/libero exactly, so policy configs and normalisation stats line up.
LIBERO_FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (256, 256, 3),
                                 "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "video", "shape": (256, 256, 3),
                                  "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def obs_to_frame(obs) -> dict:
    """Pull the four dataset fields out of a raw LIBERO observation."""
    return {
        "observation.images.image": obs["agentview_image"][::-1, ::-1].copy(),
        "observation.images.image2": obs["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
        "observation.state": np.concatenate([
            obs["robot0_eef_pos"],
            T.quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]).astype(np.float32),
    }


def resample(path, n_waypoints=100):
    """Thin a path to fit the 280-step episode budget at 2 control steps per waypoint."""
    idx = np.unique(np.linspace(0, len(path) - 1, n_waypoints).round().astype(int))
    return path[idx]


def rollout_path(env, obs, path, *, kp=4.0, kr=3.0, dwell=2, gripper_hold=10,
                 approach_steps=20, settle_steps=30, max_steps=280, record=True,
                 on_step=None):
    """Track an absolute EEF path, recording (observation, action) at every control step.

    Returns (success, frames, steps). `frames` holds the observation BEFORE each action
    together with that action, which is the convention LeRobot datasets use.

    `on_step(env)` is called after every control step.
    """
    frames, steps = [], 0
    prev_grip = path[0][6]
    # keep steps for the settle phase; judged mid-drop, a placed bowl still fails
    path_budget = max_steps - settle_steps
    for w, tgt in enumerate(path):
        # the arm starts at its home pose, so the first waypoint gets extra steps
        hold = approach_steps if w == 0 else (gripper_hold if tgt[6] != prev_grip else dwell)
        tgt_R = axisangle_to_matrix(tgt[3:6])

        for _ in range(hold):
            if steps >= path_budget:
                break
            action = track_step(obs["robot0_eef_pos"], T.quat2mat(obs["robot0_eef_quat"]),
                                tgt[:3], tgt_R, tgt[6], kp=kp, kr=kr)
            if record:
                f = obs_to_frame(obs)
                f["action"] = action.astype(np.float32)
                frames.append(f)
            obs, _, _, _ = env.step(action)
            steps += 1
            if on_step:
                on_step(env)
            if env.check_success():
                return True, frames, steps
        prev_grip = tgt[6]
        if steps >= path_budget:
            break

    # retreat upward, gripper open, and let the bowl come to rest before judging
    for _ in range(settle_steps):
        if steps >= max_steps:
            break
        a = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        if record:
            f = obs_to_frame(obs)
            f["action"] = a
            frames.append(f)
        obs, _, _, _ = env.step(a)
        steps += 1
        if on_step:
            on_step(env)
        if env.check_success():
            return True, frames, steps
    return False, frames, steps


def create_dataset(repo_id: str, root: str | Path, fps: int = 10,
                   robot_type: str = "panda"):
    """fps must be an int: the video encoder builds a rational frame rate from it and
    a float raises a message saying: 'float' object has no attribute 'numerator'
    deep inside the encoder thread, long after the frames were accepted."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"{root} already exists; remove it or pick another root")
    return LeRobotDataset.create(repo_id=repo_id, fps=int(fps), features=LIBERO_FEATURES,
                                 root=root, robot_type=robot_type, use_videos=True)


def append_episode(ds, frames: list[dict], task: str) -> None:
    """Write one successful rollout. Only successes belong in a training set."""
    for f in frames:
        ds.add_frame({**f, "task": task})
    ds.save_episode()
