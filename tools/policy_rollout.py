#!/usr/bin/env python3
"""Roll a trained SmolVLA checkpoint out in LIBERO, with the bowl optionally moved off the
benchmark's own layout, and optionally record the successful episodes as a LeRobot dataset.

This is what lerobot-eval cannot do: choose the init states, move an object, and keep the
successes as training data. One JSON line per episode goes to --out.

    # generalization: bowl shifted off the stock position
    python tools/policy_rollout.py --ckpt outputs/train/T_H/checkpoints/015000/pretrained_model \
        --policy T_H --init-ids 0-1 --shifts "0.04,0;-0.04,0;0,0.04;0,-0.04" --out outputs/rollouts/E1_T_H.jsonl
    # self-training rollouts on init states the eval never uses, recording the successes
    python tools/policy_rollout.py --ckpt ... --policy T_H --init-ids 20-49 --record outputs/datasets/self_T_H
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TASKS = [2, 0, 8, 1, 6]
BOWL_JOINT = "akita_black_bowl_1_joint0"
BOWL_BODY = "akita_black_bowl_1_main"


def parse_ids(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def parse_shifts(s):
    return [tuple(float(x) for x in p.split(",")) for p in s.split(";")] if s else [(0.0, 0.0)]


def add_batch(d):
    return {k: add_batch(v) if isinstance(v, dict) else (v[None] if isinstance(v, np.ndarray) else v)
            for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--policy", required=True, help="label written to the jsonl")
    ap.add_argument("--tasks", default="2,0,8,1,6")
    ap.add_argument("--init-ids", default="0-19")
    ap.add_argument("--shifts", default="", help='"dx,dy;dx,dy;..." in metres; empty = stock layout')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--record", type=Path, default=None, help="LeRobot dataset root for successes")
    ap.add_argument("--worker", default="0/1", help="i/n: this process takes every n-th job")
    ap.add_argument("--n-action-steps", type=int, default=10,
                    help="actions executed per predicted chunk; 10 is what tools/eval_arm.sh uses")
    ap.add_argument("--resume", action="store_true", help="skip episodes already in --out (no --record)")
    a = ap.parse_args()
    assert not (a.resume and a.record), "recording cannot resume: the dataset is recreated"

    from lerobot.envs.libero import LiberoEnv, _get_suite
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    import robosuite.utils.transform_utils as T
    from match_scene import set_object_xy
    from humanoid_challenge.sim_dataset import append_episode, create_dataset

    wi, wn = (int(x) for x in a.worker.split("/"))
    jobs = [(t, k, s) for t in (int(x) for x in a.tasks.split(","))
            for k in parse_ids(a.init_ids) for s in parse_shifts(a.shifts)]
    jobs = jobs[wi::wn]
    if a.resume and a.out.exists():
        done = {(r["task"], r["init_id"], r["dx"], r["dy"]) for r in map(json.loads, a.out.read_text().splitlines()) if r}
        jobs = [j for j in jobs if (j[0], j[1], j[2][0], j[2][1]) not in done]
    print(f"{len(jobs)} episodes for worker {a.worker}", flush=True)

    torch.set_num_threads(4)            # many workers share the machine
    policy = SmolVLAPolicy.from_pretrained(a.ckpt)
    policy.config.n_action_steps = a.n_action_steps   # the checkpoint itself says 50 (open loop)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=a.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}})
    env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    suite = _get_suite("libero_spatial")
    ds = None
    if a.record:
        if a.record.exists():
            import shutil; shutil.rmtree(a.record)
        ds = create_dataset(f"local/{a.record.name}", a.record, fps=10)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    env, env_task = None, None
    with open(a.out, "a") as log:
        for task_id, init_id, (dx, dy) in jobs:
            if env_task != task_id:
                if env is not None:
                    env.close()
                env = LiberoEnv(suite, task_id, task_suite_name="libero_spatial", obs_type="pixels_agent_pos",
                                observation_height=256, observation_width=256, control_freq=20,
                                control_mode="relative")
                env._ensure_env()
                stock = env._init_states.copy()
                env_task = task_id
            # The scene comes from the benchmark's own init state; only the bowl moves, by the
            # requested shift from wherever that state put it.
            v = stock[init_id].copy()
            m = env._env.sim.model
            adr = m.jnt_qposadr[m.joint_name2id(BOWL_JOINT)] + 1      # state vector = [time, qpos...]
            want_xy = np.array([v[adr] + dx, v[adr + 1] + dy])
            if dx or dy:
                v = set_object_xy(v, m, BOWL_JOINT, *want_xy)
            env._init_states = np.stack([v]); env.init_state_id = 0
            obs, _ = env.reset(seed=init_id)
            # A shifted bowl can land on a neighbour and get thrown; such layouts are not a
            # test of the policy. Log how far the bowl ended up from where it was put.
            sim = env._env.sim
            bowl = sim.data.body_xpos[sim.model.body_name2id(BOWL_BODY)]
            bowl_err_mm = round(float(np.linalg.norm(bowl[:2] - want_xy)) * 1000, 1)
            policy.reset(); pre.reset(); post.reset()
            frames, success, t0 = [], False, time.time()
            for step in range(env._max_episode_steps):
                o = preprocess_observation(add_batch(obs))
                o["task"] = [env.task_description]
                o = pre(env_pre(o))
                with torch.inference_mode():
                    act = post(policy.select_action(o))
                act = act.cpu().numpy()[0].astype(np.float32)
                if ds is not None:
                    rs = obs["robot_state"]
                    frames.append({
                        "observation.images.image": obs["pixels"]["image"][::-1, ::-1].copy(),
                        "observation.images.image2": obs["pixels"]["image2"][::-1, ::-1].copy(),
                        "observation.state": np.concatenate([rs["eef"]["pos"], T.quat2axisangle(rs["eef"]["quat"]),
                                                             rs["gripper"]["qpos"]]).astype(np.float32),
                        "action": act,
                    })
                obs, _, term, trunc, info = env.step(act)
                if info.get("is_success"):
                    success = True
                    break
                if term or trunc:
                    break
            if ds is not None and success:
                append_episode(ds, frames, env.task_description)
            rec = dict(policy=a.policy, task=task_id, init_id=init_id, dx=dx, dy=dy, bowl_err_mm=bowl_err_mm,
                       success=bool(success), steps=step + 1, seconds=round(time.time() - t0, 1))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
