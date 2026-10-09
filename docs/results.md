# Results — numbers only

Everything here is read from the run logs under `outputs/`, most of it by
`tools/plot_results.py`; nothing is rounded beyond what the tools print. Figures in
`docs/figures/`.

## Data recorded

71 Record3D captures (iPhone 16 Pro Max, egocentric, 30 Hz RGB + LiDAR depth + ARKit pose),
one table, four ArUco markers, one bowl, one plate.

| chunk | task (`libero_spatial` id) | bowl start (x, y) m | episodes |
|---|---|---|---|
| A_t2 | bowl at table centre (2) | (−0.075, 0.000) | 11 |
| A_t0 | between plate and ramekin (0) | (−0.050, 0.200) | 12 |
| A_t8 | next to plate (8) | (0.010, 0.310) | 14 |
| A_t1 | next to ramekin (1) | (−0.180, 0.320) | 12 |
| A_t6 | next to cookie box (6) | (0.130, −0.070) | 10 |
| B_push | push the plate (`libero_goal` 5) | plate at (0.050, −0.020) | 11 |

## Instrument: the marker frame

| chunk | marker drift, median (mm / deg) | fusion residual, median (mm) | hand frames usable | camera height above table (mm) | episodes with hand below −50 mm |
|---|---|---|---|---|---|
| A_t2 | 7 / 7.5 | 48 | 0.74 | 762 | 7 |
| A_t0 | 16 / 0.9 | 74 | 0.79 | 718 | 1 |
| A_t8 | 12 / 0.8 | 49 | 0.81 | 715 | 0 |
| A_t1 | 8 / 0.6 | 57 | 0.81 | 725 | 12 |
| A_t6 | 17 / 1.0 | 36 | 0.82 | 678 | 2 |
| B_push | 11 / 0.6 | 24 | 0.83 | 675 | 0 |

Board re-fitted from one clip each of A_t0, A_t6, A_t1: every marker within 5 mm of the
calibration. Grasp repeatability within a chunk, well-registered chunks: pinch midpoint
42 ± 5 mm from the bowl centre (A_t0).

## Grasp geometry: human pinch vs the robot's own demonstrations (40 demos, 8 per task)

| | distance from bowl centre (mm) | height above table (mm) | jaw axis vs radial line (deg) | wrist yaw change over the episode (deg) |
|---|---|---|---|---|
| LIBERO demonstrations | 47 ± 14 | 38 ± 19 | 21 ± 13 | −3 ± 7 |
| human captures, as retargeted (A_t0/A_t8/A_t6) | 42 ± 5 (A_t0) | 60 ± 12 | 76 ± 13 | follows the forearm, ±45–150 |

Rim: radius 56 mm, top at 52 mm. Pinch threshold fires 6–36 mm (per-chunk mean) after
the hand's lowest point.

## Replay of the 59 pick captures in LIBERO

Success = LIBERO's own predicate (bowl resting on the plate). Figure:
`docs/figures/replay_yield.png`.

| replay | object positions used | A_t2 | A_t0 | A_t8 | A_t1 | A_t6 | all |
|---|---|---|---|---|---|---|---|
| hand path, literal | release only | 0/11 | 0/12 | 0/14 | 0/12 | 1/10 | 1/59 |
| + gripper held at its home orientation | none | 2/11 | 3/12 | 4/14 | 0/12 | 8/10 | 17/59 (29 %) |
| + grasp height 38 mm (14 mm below the rim); horizontal grasp and release as the hand measured them | none | 4/11 | 11/12 | 11/14 | 3/12 | 9/10 | **38/59 (64 %)** |
| + grasp anchored to the bowl, release to the plate | bowl, plate | 8/11 | 11/12 | 10/14 | 7/12 | 7/10 | 43/59 (73 %) |
| control: LIBERO's own demonstration paths, same tracker | – | – | – | – | – | – | 28/40 (70 %) |

The two constants (home orientation; 38 mm) are read off 40 LIBERO demonstrations: wrist yaw
changes by −3 ± 7° over an episode, grasp height 38 ± 19 mm. In the last human row the bowl
position is the session's tape mark and the plate position is the layout's; that is
privileged information in building the demonstrations (not in the trained policies, which
see only cameras, robot state and text).

Other combinations, for completeness: wrist lock with the release pinned 17/59; grasp
anchored without the wrist lock 4/59. Failure modes (no contact / pushed, not lifted):
literal 29/24, wrist lock 23/18, height 11/8, anchored 6/8, control 6/1. Push task
(`libero_goal` 5, 11 captures): 0/11.

Attempt to get the bowl position from each capture's own LiDAR instead (points more than
25 mm above the table within 12 cm of the tape mark, circle fit to the rim): the bowl is in
view in 95-100 % of pre-grasp frames, but the depth map shows it in only 16 of 59 captures,
and there the estimate is 39-47 mm (median, by session) from the tape mark with 1-4 cm of
frame-to-frame scatter. Not usable for a grasp, which needs about 1 cm.

## Training sets

| set | episodes | frames | source |
|---|---|---|---|
| T_H | 43 | 8389 | successes of the anchored replay (human-derived) |
| T_R | 28 | 5636 | successes of the control replay (robot-derived) |
| T_Hm | 28 | 5468 | T_H subsampled to T_R's per-task counts (seed 0) |
| T_HR | 71 | 14025 | T_H ∪ T_R |
| T_Hz | 38 | 6979 | successes of the height-only replay (no object positions) |
| T_Haug15 / T_Haug40 | 118 / 102 | 22759 / 19771 | each capture replayed onto 3 jittered bowl positions (±15 / ±40 mm) |
| T_Hself | 120 | 25289 | T_H + 77 successful rollouts of the T_H policy |

Recipe, identical for every arm (`tools/train_arm.sh`): SmolVLA, pretrained SmolVLM
backbone, action expert from scratch, batch 64, 15 000 steps, seed 0, one A100 each.

## Policy evaluation

`tools/eval_arm.sh`: `libero_spatial` tasks {2, 0, 8, 1, 6}, 20 episodes per task, stock
init states, `n_action_steps=10`, seed 0, final (15 000-step) checkpoint. One training seed
per arm. Wilson 95 % intervals on the pooled 100 episodes. Figure: `docs/figures/eval_success.png`.

| arm | task 2 | task 0 | task 8 | task 1 | task 6 | pooled | 95 % CI |
|---|---|---|---|---|---|---|---|
| T_H  human-derived, 43 ep | 6/20 | 12/20 | 12/20 | 14/20 | 12/20 | **56/100** (56 %) | 46–65 |
| T_R  robot-derived, 28 ep | 8/20 | 11/20 | 13/20 | 16/20 | 12/20 | **60/100** (60 %) | 50–69 |
| T_Hm human-derived, 28 ep matched to T_R per task | 4/20 | 8/20 | 0/20 | 9/20 | 14/20 | **35/100** (35 %) | 26–45 |
| T_HR both, 71 ep | 4/20 | 6/20 | 13/20 | 11/20 | 14/20 | **48/100** (48 %) | 38–58 |

Per-task episode counts: T_H t0 11 / t1 7 / t2 8 / t6 7 / t8 10; T_R and T_Hm t0 4 / t1 6 /
t2 6 / t6 7 / t8 5. Final training loss: T_H 0.075, T_R 0.099, T_Hm 0.059, T_HR 0.130
(`docs/figures/train_loss.png`).

Context: the published `pi05_libero_finetuned` reports 97.0 on libero_spatial; I did not run
it here (its tokenizer is gated). `smolvla_base` cannot be evaluated zero-shot here either:
it expects three cameras and LIBERO provides two.

This table was rendered at 360 px (the `lerobot-eval` default) on Mesa's software renderer,
because of a launch-environment mistake of mine; the renderer does not
change the physics. The training data is 256 px; the harness tables below render at 256 px.

## Second evaluation, through `tools/policy_rollout.py` (256 px render, the training resolution)

Same checkpoints, same five tasks, same init states 0–19, `n_action_steps=10`.

| arm | harness, 256 px | `lerobot-eval`, 360 px (table above) |
|---|---|---|
| T_H  human-derived, 43 ep | **58/100** (48–67) | 56/100 |
| T_R  robot-derived, 28 ep | **54/100** (44–63) | 60/100 |
| T_Hm human-derived, 28 ep matched | **50/100** (40–60) | 35/100 |
| T_HR both, 71 ep | **47/100** (38–57) | 48/100 |

T_Hm differs between the two evaluations (not resolved: render size or sampling noise).

## Layout sensitivity: bowl moved off the benchmark's own position

Bowl shifted in ±x, ±y; init states 0–4; 28 of 200 shifted layouts dropped for every policy
(the bowl landed on a neighbour). Instruction text unchanged. Figure:
`docs/figures/generalization.png`.

| arm | stock | shifted 4 cm | shifted 8 cm |
|---|---|---|---|
| T_H | 58/100 | 29/87 (33 %, 24–44) | 11/85 (13 %, 7–22) |
| T_R | 54/100 | 22/87 (25 %, 17–35) | 4/85 (5 %, 2–11) |
| T_Hm | 50/100 | 30/87 (34 %, 25–45) | 6/85 (7 %, 3–15) |
| T_HR | 47/100 | 30/87 (34 %, 25–45) | 11/85 (13 %, 7–22) |
| T_Haug15 (jittered ±15 mm, 118 ep) | **65/100** (55–74) | 36/87 (41 %, 32–52) | 10/85 (12 %, 7–20) |
| T_Haug40 (jittered ±40 mm, 102 ep) | 56/100 (46–65) | 30/87 (34 %, 25–45) | 10/85 (12 %, 7–20) |
| T_Hself (T_H + 77 self-rollouts, 120 ep) | 52/100 (42–62) | 27/87 (31 %, 22–41) | 6/85 (7 %, 3–15) |
| T_Hz (height-only replay, no object positions, 38 ep) | 46/100 (37–56) | 24/87 (28 %, 19–38) | 7/85 (8 %, 4–16) |

## One capture, many layouts (jittered replay)

LIBERO's own init states move the bowl ±15 mm; the replays above used one position per task.
`build_sim_dataset.py --jitter J --draws 3` replays each capture onto a moved bowl.

| set | bowl jitter | replays | predicate successes |
|---|---|---|---|
| T_H | none | 59 | 43 (73 %) |
| T_Haug15 | ±15 mm | 177 | 118 (67 %) |
| T_Haug40 | ±40 mm | 177 | 102 (58 %) |

## Self-training

T_H rolled out on init states 20–49 (disjoint from the evaluation's 0–19): 77/150 succeed.
T_Hself = T_H's 43 episodes + those 77 (120 episodes, 25 289 frames).

## Second training seed (harness, stock layouts)

| arm | seed 0 | seed 1 |
|---|---|---|
| T_H | 58/100 | 63/100 |
| T_R | 54/100 | 45/100 |

T_Hz per task on stock layouts (training episodes in brackets): task 2 9/20 (4), task 0
14/20 (11), task 8 15/20 (11), task 1 0/20 (3), task 6 8/20 (9).
