#!/usr/bin/env python3
"""Pick the LIBERO init state whose object layout best matches a human capture.

LIBERO ships 50 init states per task and they jitter the objects by a couple of
centimetres. The human recorded against ONE physical layout, so replaying against an
arbitrary init state introduces an offset that nothing downstream can correct -- the
hand releases where the real plate was, not where the simulated one is.

Measured on the first real capture: an arbitrary init state left the bowl 63 mm from the
plate centre, close enough to grasp and carry correctly but not close enough to satisfy
the success predicate.

This is the cheap half of "make the scene follow the data". The thorough half is to
write the object poses into the init state directly; this just picks the best of 50.
"""
import numpy as np
import torch
from pathlib import Path
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DUMMY = [0, 0, 0, 0, 0, 0, -1]
TABLE_Z = 0.90


def object_positions(task, bodies, settle=10):
    """Where each named body sits, for every init state of a task."""
    init = np.asarray(torch.load(
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
        weights_only=False))
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=64, camera_widths=64, control_freq=20)
    try:
        out = []
        for i in range(len(init)):
            env.seed(0); env.reset(); env.set_init_state(init[i])
            for _ in range(settle):
                env.step(DUMMY)
            # re-fetch sim each iteration: LIBERO rebuilds it on a hard reset, so a
            # handle grabbed before the loop goes stale
            sim = env.env.sim
            out.append({b: sim.data.body_xpos[sim.model.body_name2id(b)].copy()
                        for b in bodies})
    finally:
        env.close()
    return init, out


def best_init(positions, grasp_xy, release_xy, obj_body, tgt_body):
    """Init state minimising how far the human's grasp and release land from the
    simulated object and target."""
    rows = []
    for i, p in enumerate(positions):
        d_o = float(np.linalg.norm(p[obj_body][:2] - grasp_xy))
        d_t = float(np.linalg.norm(p[tgt_body][:2] - release_xy))
        rows.append((d_o + d_t, d_o, d_t, i))
    rows.sort()
    return rows


# The init-state vector is 92 long while qpos is 48, offset by exactly one element:
# init[j + 1] == qpos[j]. Verified by reading an object's free-joint address out of the
# model and comparing the stored vector against the live simulation.
INIT_QPOS_OFFSET = 1


def set_object_xy(init_vec, sim_model, body_joint, x, y):
    """Write an object's x,y into a copy of an init-state vector.

    This is what makes the simulated scene follow the capture instead of the other way
    round: the operator recorded against one physical layout, and LIBERO's 50 stock init
    states jitter the objects by a couple of centimetres around a different one. Picking
    the closest stock state only moves things ~18 mm; writing the measured position in
    removes the mismatch entirely.
    """
    import numpy as np
    jid = sim_model.joint_name2id(body_joint)
    adr = sim_model.jnt_qposadr[jid] + INIT_QPOS_OFFSET
    out = np.asarray(init_vec, dtype=float).copy()
    out[adr] = x
    out[adr + 1] = y
    return out
