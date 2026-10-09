#!/usr/bin/env python3
"""Work out which LIBERO init state each recorded episode actually started from.

The obvious guess -- the n-th episode of a task used the n-th init state -- is wrong.
Each task ships 50 init states, and the 92-dim vector sets the robot's joints as well as
the object poses, so the home end-effector position varies by up to 41 mm between them.
Matching episode to init state by that home position recovers the pairing to 2-7 mm,
against 9-25 mm for the naive guess.

This matters: a replay started from the wrong init state is chasing a path that is
offset from the first frame, which silently destroys any absolute-path tracking.
"""
import numpy as np
import torch
from pathlib import Path
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DUMMY = [0, 0, 0, 0, 0, 0, -1]


def home_positions(task, settle=10, control_freq=20):
    """Home EEF position after reset+settle, for every init state of a task."""
    init = np.asarray(torch.load(
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
        weights_only=False))
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64,
                             control_freq=control_freq)
    try:
        out = []
        for i in range(len(init)):
            env.seed(0)
            env.reset()
            obs = env.set_init_state(init[i])
            for _ in range(settle):
                obs, _, _, _ = env.step(DUMMY)
            out.append(obs["robot0_eef_pos"].copy())
    finally:
        env.close()
    return init, np.asarray(out)


def match(home, state0):
    """Index of the init state whose home pose best explains a recorded first frame."""
    d = np.linalg.norm(home - np.asarray(state0)[:3], axis=1)
    return int(d.argmin()), float(d.min())
