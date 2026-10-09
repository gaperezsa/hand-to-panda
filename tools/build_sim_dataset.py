#!/usr/bin/env python3
"""Replay end-effector paths in LIBERO and write the successful rollouts as a LeRobot dataset.

Runs on the retargeted phone captures (data/interim) or on lerobot/libero's own
demonstration paths, which is the control arm.

    # test / control arm: ground-truth paths
    python tools/build_sim_dataset.py --source reference --episodes-per-task 3 \
        --root outputs/datasets/ref_replay --repo-id local/ref_replay

    # the real thing, once data/interim is populated
    python tools/build_sim_dataset.py --source interim --root outputs/datasets/human
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import robosuite.utils.transform_utils as T
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from humanoid_challenge.sim_dataset import (append_episode, create_dataset,  # noqa: E402
                                            resample, rollout_path)
from humanoid_challenge.tracking import anchor_grasp, place_correction  # noqa: E402
from match_scene import set_object_xy  # noqa: E402
from match_init_states import home_positions, match  # noqa: E402

DUMMY = [0, 0, 0, 0, 0, 0, -1]
TABLE_Z = 0.90                  # LIBERO table surface in world coordinates
# Per suite: the manipulated object's body and joint, and where it must end up. Layout A
# moves the bowl onto the plate; layout B pushes the plate into the stove-front region
# (BDDL x in [-0.09, -0.01], y in [0.17, 0.25]; its centre is the target).
SCENE = {"libero_spatial": dict(body="akita_black_bowl_1_main", joint="akita_black_bowl_1_joint0",
                                target=(0.060, 0.200), fixed=("plate_1_joint0", (0.060, 0.200))),
         "libero_goal": dict(body="plate_1_main", joint="plate_1_joint0",
                             target=(-0.050, 0.210), fixed=None)}
# Where the Panda itself grasps this bowl, measured over 40 lerobot/libero demonstrations
# of the five tasks: 47 +/- 14 mm from the bowl centre along the jaw axis, 38 +/- 19 mm
# above the table (the rim top is at 52 mm).
GRASP_R = 0.047
GRASP_Z = 0.038
TASKS = [("libero_spatial", 2, 38), ("libero_spatial", 0, 34), ("libero_spatial", 8, 36),
         ("libero_spatial", 1, 37), ("libero_spatial", 6, 30)]


def reference_paths(n_per_task):
    """Ground-truth EEF paths, as a stand-in for retargeted human paths."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("lerobot/libero")
    df = ds.hf_dataset.select_columns(
        ["episode_index", "frame_index", "task_index", "action", "observation.state"]).to_pandas()
    ep_task = df.groupby("episode_index")["task_index"].first()
    out = []
    for suite_name, tid, gti in TASKS:
        for ep in ep_task.index[ep_task == gti].tolist()[:n_per_task]:
            e = df[df.episode_index == ep].sort_values("frame_index")
            st = np.stack(e["observation.state"].to_numpy())
            ac = np.stack(e["action"].to_numpy())
            out.append(dict(suite=suite_name, task_ids=tid,
                            path=np.concatenate([st[:, :6], ac[:, 6:7]], axis=1),
                            state0=st[0], source=f"lerobot/libero ep {ep}"))
    return out


def interim_paths(interim_dir: Path, cfg, chunks=None, trim=True, pre_s=1.0, post_s=0.5):
    """Paths from the phone captures (docs/intermediate_format.md), trimmed to 1 s before
    the grasp .. 0.5 s after the release. Task and instruction come from config.yaml,
    keyed on the chunk name."""
    import cv2
    A = cfg["layouts"]["A"]
    by_chunk = {f"A_t{p['task_ids']}": ("libero_spatial", int(p["task_ids"]),
                                        p["instruction"], np.array(p["start_m"], float))
                for p in A["positions"]}
    B = cfg["layouts"]["B"]
    by_chunk["B_push"] = (B["task_suite"], int(B["task_ids"]), B["instruction"],
                          np.array(B["start_m"], float))
    out = []
    for npz in sorted(interim_dir.glob("*.npz")):
        chunk = "_".join(npz.stem.split("_")[:2])
        if chunk not in by_chunk or (chunks and chunk not in chunks):
            continue
        suite, tid, instr, start = by_chunk[chunk]
        d = np.load(npz)
        t, R, g = d["t"], d["eef_rotmat"], d["gripper"]
        tr = np.flatnonzero(np.diff(g) != 0)
        if len(tr) < 2 and chunk != "B_push":
            continue
        keep = np.ones(len(t), bool)
        # the push never closes the gripper; replay it whole
        if trim and len(tr) >= 2:
            keep = (t >= t[tr[0]] - pre_s) & (t <= t[tr[-1]] + post_s)
        aa = np.stack([cv2.Rodrigues(R[i])[0].ravel() for i in np.flatnonzero(keep)])
        pos = d["eef_pos"][keep].copy()
        # interim z is relative to the table surface, which sits at 0.90 m in the world
        pos[:, 2] += TABLE_Z
        path = np.concatenate([pos, aa, g[keep][:, None]], axis=1)
        out.append(dict(suite=suite, task_ids=tid, instruction=instr, path=path,
                        state0=None, source=str(npz), chunk=chunk,
                        object_start=start, nominal_start=start,
                        grasp_xy=d["eef_pos"][tr[0], :2] if len(tr) else None,
                        release_xy=d["eef_pos"][tr[-1], :2] if len(tr) else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["reference", "interim"], default="reference")
    ap.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    ap.add_argument("--episodes-per-task", type=int, default=3)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--repo-id", default="local/humanoid_replay")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--chunks", default=None, help="comma separated, e.g. A_t2,A_t0")
    ap.add_argument("--episodes", default=None, help="comma separated, e.g. A_t0_005,A_t8_003")
    ap.add_argument("--strips", type=Path, default=None,
                    help="also save an 8-frame agentview strip per rollout here (for docs)")
    ap.add_argument("--save-states", type=Path, default=None,
                    help="also save every rollout's simulator states here (to re-render it later)")
    ap.add_argument("--video-dir", type=Path, default=None,
                    help="also save every rollout's agentview frames as an mp4 here")
    ap.add_argument("--waypoints", type=int, default=100)
    ap.add_argument("--keep-physical", action="store_true", default=True)
    ap.add_argument("--only-success", dest="keep_physical", action="store_false")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="replay each capture with the bowl moved uniformly within +/- this many "
                         "metres of where it was recorded")
    ap.add_argument("--draws", type=int, default=1, help="jittered replays per capture")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--yaw-lock", action="store_true", default=True,
                    help="hold the robot's home wrist orientation for the whole path")
    ap.add_argument("--no-yaw-lock", dest="yaw_lock", action="store_false")
    ap.add_argument("--anchor-grasp", action="store_true", default=True,
                    help="snap the grasp waypoint to the robot's own grasp pose on the rim")
    ap.add_argument("--no-anchor-grasp", dest="anchor_grasp", action="store_false")
    ap.add_argument("--anchor-height-only", action="store_true",
                    help="take only the grasp HEIGHT from the object (its shape); keep the "
                         "horizontal grasp position the hand measured")
    ap.add_argument("--place-correction", action="store_true", default=True)
    ap.add_argument("--no-place-correction", dest="place_correction", action="store_false")
    ap.add_argument("--scene-from-data", action="store_true", default=True,
                    help="place the simulated objects where the capture says they were")
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    items = (reference_paths(a.episodes_per_task) if a.source == "reference"
             else interim_paths(a.interim_dir, cfg,
                                chunks=set(a.chunks.split(",")) if a.chunks else None))
    if a.episodes:
        want = set(a.episodes.split(","))
        items = [it for it in items if Path(it["source"]).stem in want]
    # One capture, many layouts: the grasp anchor and the place correction are defined
    # relative to the bowl, so the same hand path can be re-targeted to a bowl that sits
    # somewhere else (the MimicGen idea, with a phone capture as the source demo).
    if a.jitter > 0:
        rng = np.random.default_rng(a.seed)
        items = [dict(it, object_start=it["nominal_start"] + rng.uniform(-a.jitter, a.jitter, 2),
                      source=f"{it['source']} draw {k}")
                 for it in items if it.get("nominal_start") is not None for k in range(a.draws)]
    if not items:
        raise SystemExit("no input paths found")
    print(f"{len(items)} paths to replay\n")

    if a.overwrite and a.root.exists():
        shutil.rmtree(a.root)
    ds = create_dataset(a.repo_id, a.root, fps=a.fps)

    suites, homes, inits = {}, {}, {}
    n_ok = n_written = 0
    for i, it in enumerate(items):
        key = (it["suite"], it["task_ids"])
        if key not in suites:
            suite = benchmark.get_benchmark_dict()[it["suite"]]()
            task = suite.get_task(it["task_ids"])
            suites[key] = task
            inits[key] = np.asarray(torch.load(
                Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                weights_only=False))
            # home poses only matter for matching a reference episode to its init state
            homes[key] = home_positions(task)[1] if a.source == "reference" else None
        task = suites[key]
        init_idx = match(homes[key], it["state0"])[0] if it["state0"] is not None else i % len(inits[key])

        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=256, camera_widths=256, control_freq=20)
        try:
            # put the objects where the operator had them, not where this init state does
            env.seed(0); env.reset(); env.set_init_state(inits[key][init_idx])
            env.step(DUMMY)
            v = inits[key][init_idx].copy()
            scene = SCENE[it["suite"]]
            TARGET_XY = scene["target"]
            if a.scene_from_data and it.get("object_start") is not None:
                m = env.env.sim.model
                v = set_object_xy(v, m, scene["joint"], *it["object_start"])
                if scene["fixed"]:
                    v = set_object_xy(v, m, *scene["fixed"][0:1], *scene["fixed"][1])
            env.seed(0); env.reset()
            obs = env.set_init_state(v)
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY)
            wp = resample(it["path"], a.waypoints)
            tr = np.flatnonzero(np.diff(wp[:, 6]) != 0)
            # the grasp happens at the first waypoint with the new gripper state
            gi = int(tr[0]) + 1 if len(tr) else None
            # Hold the wrist at its home orientation, as every LIBERO demonstration does
            # (yaw change -3 +/- 7 deg). Following the forearm put the jaws ~76 deg off
            # the rim's radial line.
            if a.yaw_lock:
                wp[:, 3:6] = cv2.Rodrigues(T.quat2mat(obs["robot0_eef_quat"]))[0].ravel()
            if a.anchor_grasp and gi is not None and it.get("object_start") is not None:
                side = np.sign(it["grasp_xy"][1] - it["nominal_start"][1]) or 1.0
                anchor = (it["object_start"][0], it["object_start"][1] + side * GRASP_R,
                          TABLE_Z + GRASP_Z)
                if a.anchor_height_only:
                    anchor = (wp[gi, 0], wp[gi, 1], TABLE_Z + GRASP_Z)
                wp, _ = anchor_grasp(wp, gi, anchor)
            if (a.place_correction and len(tr) >= 2
                    and it.get("object_start") is not None):
                wp, dmm = place_correction(wp, gi, int(tr[-1]),
                                           it["object_start"], TARGET_XY)
            sim = env.env.sim
            bid = sim.model.body_name2id(scene["body"])
            b0 = sim.data.body_xpos[bid].copy()
            states = [env.get_sim_state().copy()]
            ok, frames, steps = rollout_path(env, obs, wp,
                                             on_step=lambda e: states.append(e.get_sim_state().copy()))
            if a.save_states:
                a.save_states.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(a.save_states / f"{Path(it['source']).stem}.npz", states=np.stack(states),
                                    waypoints=wp, bddl=str(Path(task.problem_folder) / task.bddl_file), success=ok)
            bf = sim.data.body_xpos[bid].copy()
            lifted = float(bf[2] - b0[2])
            moved = float(np.linalg.norm(bf[:2] - b0[:2]))
            to_target = float(np.linalg.norm(bf[:2] - np.asarray(TARGET_XY)))
            # "physical" = grasped, carried and set down near the plate, whether or not
            # the predicate fired; the predicate rate is the headline number regardless
            physical = ((lifted > 0.03 or moved > 0.10) and to_target < 0.12
                        if it["suite"] == "libero_spatial" else
                        moved > 0.05 and to_target < 0.06)
        finally:
            env.close()

        if not a.quiet:
            print(f"  [{i+1:3d}/{len(items)}] {it['suite']} t{it['task_ids']:<2} "
                  f"{'SUCCESS' if ok else 'failed '} {steps:3d} steps  {len(frames):3d} frames  "
                  f"{it['source']}")
        if ok:
            n_ok += 1
        if ok or (a.keep_physical and physical):
            append_episode(ds, frames, task.language)
            n_written += 1
        if a.strips and frames:
            a.strips.mkdir(parents=True, exist_ok=True)
            pick = np.linspace(0, len(frames) - 1, 8).round().astype(int)
            strip = np.concatenate([frames[k]["observation.images.image"] for k in pick], axis=1)
            cv2.imwrite(str(a.strips / f"{Path(it['source']).stem}_{'ok' if ok else 'fail'}.png"),
                        cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        if a.video_dir and frames:
            a.video_dir.mkdir(parents=True, exist_ok=True)
            vw = cv2.VideoWriter(str(a.video_dir / f"{Path(it['source']).stem}_{'ok' if ok else 'fail'}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 20, (256, 256))
            for f in frames:
                vw.write(cv2.cvtColor(f["observation.images.image"], cv2.COLOR_RGB2BGR))
            vw.release()
        if not a.quiet:
            print(f"        lifted {lifted*1000:4.0f} mm  moved {moved*1000:4.0f} mm  "
                  f"bowl->plate {to_target*1000:4.0f} mm  physical={physical}")

    print(f"\nLIBERO predicate success : {n_ok}/{len(items)} ({n_ok/len(items)*100:.0f}%)")
    print(f"episodes written          : {n_written}/{len(items)} "
          f"({n_written/len(items)*100:.0f}%)  -> {a.root}")
    print("Written = predicate successes" + (" + physical successes" if a.keep_physical else "") + ".")


if __name__ == "__main__":
    main()
