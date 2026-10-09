# Recording protocol

Follow in order. Steps 0-3 happen once; step 4 is the loop.

Print `docs/recording/setup_sheet.pdf` and keep it on the table.

---

## 0. Measure the printed marker  (2 minutes, do not skip)

Put a ruler across the **black square** on any printed page — not the white border.

```bash
source env.sh
# edit config.yaml:
#   marker_length_m: 0.1002        <- your measured value, in METRES
#   marker_length_measured: true
```

Why this first: a printer that scaled by 2% puts a 2% error on every distance the
pipeline ever computes — hand position, object position, trajectory length. Nothing
downstream can detect or correct it, because the scale error is self-consistent. It is
the cheapest possible thing to get right and the most expensive to get wrong.

If the square measures 100 mm exactly, still set `marker_length_measured: true`.

## 1. Tape the markers  (15 minutes)

Work from `setup_sheet.pdf`. Sit at a long edge of the table — that edge is "near".

1. Measure and mark the four marker centres: two distances each, from the near edge and
   from the left edge.
2. **Check the rectangle before taping**: 265 mm short side, 750 mm long side,
   **795 mm diagonal**. If the diagonal is off, the rectangle is not square — fix it now.
   (Exact placement is not critical; the sweep measures where they really are. A square
   rectangle just makes the numbers easier to sanity-check.)
3. Tape all four corners of each marker flat. A curled corner tilts the marker and biases
   its pose.
4. Keep the full white border. Do not trim close to the black edge and do not let tape
   cover any white.

Then mark the object positions with a pencil cross or a small dot of masking tape. The
nearest object sits 57 mm from the nearest marker tile — keep that gap clear.

**Once taped, do not move the markers.** If one is bumped, redo step 2.

## 2. Calibration sweep  (one clip, ~30 s)

One slow pan with **all four markers visible together**, from several angles.

- Hold the phone higher than you will during episodes — around 1.2-1.5 m. The markers
  span 750 mm and will not all fit in frame from episode height.
- Move slowly. Motion blur destroys corner accuracy, which is the one thing this clip
  exists to measure.
- Circle the table so the board is seen obliquely from several directions, not just
  straight down. Out-of-plane tilt is the weakest direction of a single view and
  varied viewpoints are what pin it down.
- No hands in frame.

Save as `data/raw/sweep.r3d`.

```bash
conda activate $VISION_ENV
python tools/dump_record3d.py data/raw/sweep.r3d        # what is actually in the file
python tools/validate_capture.py data/raw/sweep.r3d --sweep
python tools/calibrate_board_from_sweep.py data/raw/sweep.r3d
```

The last command writes `config/board_calibration.yaml`. Check its output:
- every marker linked to ID 0 (nothing listed as "never linked")
- out-of-plane residual under 5 mm per marker — larger means a marker is lifted or curled
- positions within ~20 mm of nominal — larger is fine but means your taping drifted

## 3. One test episode, validated end to end  (10 s clip)

Before recording fifty of anything, record one and check it.

```bash
python tools/validate_capture.py data/raw/test.r3d
```

Every check prints the number that decided it. Fix anything that fails before continuing.
The check most likely to fail first is **"episode is anchored at start"** — see below.

## 4. The episodes

### Layout A — 50 episodes, 10 at each of 5 bowl positions

Only one bowl is on the table at a time. Plate stays put.

Each episode:

1. **Anchor.** Start recording with the table in view and at least one marker clearly
   visible. Hold still for **two full seconds**. This is not optional: ARKit sets its
   world origin wherever the phone happened to be, so without a marker sighting the
   episode has no common frame and is unusable. Two seconds of looking at a marker is
   cheap; a re-shoot is not.
2. **Act.** Look down at your hands and do the task: reach, pinch the bowl rim from
   directly above, lift a few centimetres, move it over the plate, lower, release.
   Roughly 8-14 seconds, which is what the reference demonstrations take.
3. **Stop.** No need to re-show a marker at the end, though it costs nothing and gives a
   second drift estimate.

Grasp from **above**, pinching the rim between thumb and index. That is what the Panda
does on every reference demo, and the retargeting maps your thumb-index midpoint onto the
point between its fingers.

Vary naturally between the ten repeats of a position — slightly different approach,
speed, grip point. Identical repeats teach a policy nothing.

File naming, which the tools expect:

```
data/raw/A_t2_001.r3d  ...  A_t2_010.r3d     bowl at "from table center"
data/raw/A_t0_001.r3d  ...  A_t0_010.r3d     "between the plate and the ramekin"
data/raw/A_t8_001.r3d  ...  A_t8_010.r3d     "next to the plate"
data/raw/A_t1_001.r3d  ...  A_t1_010.r3d     "next to the ramekin"
data/raw/A_t6_001.r3d  ...  A_t6_010.r3d     "next to the cookie box"
```

### Layout B — 10 push episodes

Take the bowl away. Move the plate to its push start position, put a small box at the
obstacle position, and mark the goal zone.

Push the plate from its start into the goal zone with a **flat open hand**, keeping your
hand low and in contact — the reference demo keeps the gripper within ~2 cm of the table
for the whole stroke and never closes it. Steer around the box rather than through it.

```
data/raw/B_push_001.r3d  ...  B_push_010.r3d
```

Expect this one to transfer worse than the picks. The plate never leaves the table, so
success depends on contact geometry with no grasp to capture the object. That is the
point of recording it — it isolates the 6-DoF path from the gripper mapping.

---

## Things that will cost you a re-shoot

| symptom | cause | fix |
| --- | --- | --- |
| "episode is anchored at start" fails | started recording already looking at your hands | hold the table in view for 2 s first |
| markers detected in very few frames | glare on the paper, or motion blur | matte paper, diffuse light, move slower |
| a marker never detected | covered by a hand, an object, or tape | check the 57 mm clearance; nothing on the white border |
| depth units check fails | wrong export format | export `.r3d`, not EXR / USDZ / plain video |
| ARKit drift over 50 mm | long episode, fast motion, or a featureless scene | keep episodes under ~15 s |
| calibration says a marker was never linked to ID 0 | the sweep never saw it beside another marker | reshoot the sweep from further back |

## One-time: the hand-pose model

Not in git (7.8 MB of third-party weights). Fetch once:

```bash
mkdir -p data/models
curl -L -o data/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

On a headless box `mediapipe` also needs the GLVND GLES dispatch library, which Ubuntu
does not ship: `conda install -c conda-forge libgles`. Without it `HandLandmarker` dies
with `OSError: libGLESv2.so.2: cannot open shared object file`.

## Capture settings

- Record3D, **rear camera** (the LiDAR is on the back), head-strapped, egocentric.
- Export **`.r3d`** — that is the bundle the parser reads. If the app gives you something
  else, run `tools/dump_record3d.py` on it and we will adapt the parser.
- Keep episodes short. Drift grows with time, and 10-15 s is all the task needs.
