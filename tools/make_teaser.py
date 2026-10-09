#!/usr/bin/env python3
"""The 26-second animation in the README: one capture, from the phone's view, out into the
3-D scene the pipeline reconstructs, and into the simulator where the Panda follows the path
-- first literally (it fails), then with the two constants (it succeeds).

Three stages, because the capture reader and the simulator live in different environments:

    $VISION_ENV/bin/python tools/make_teaser.py real      # point-cloud frames + camera path
    $SIM_ENV/bin/python    tools/make_teaser.py sim       # the saved replay, same camera
    $VISION_ENV/bin/python tools/make_teaser.py compose   # overlays, captions, mp4 + gif

The replay states come from tools/build_sim_dataset.py --source interim --episodes A_t0_003
--root /tmp/x, once with  --no-yaw-lock --no-anchor-grasp --save-states outputs/teaser/states_literal
and once with  --anchor-height-only --no-place-correction --save-states outputs/teaser/states
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EP, CHUNK = "A_t0_003", "A_t0"
OUT = Path("outputs/teaser")
FPS, W, H = 30, 1280, 720
# shot boundaries [s]: phone video | fade to cloud | orbit | fade to sim | literal replay | fixed replay | end
T_RGB, T_FADE, T_REAL, T_CUT, T_LIT, T_SIM, T_END = 1.8, 3.0, 8.5, 9.5, 15.5, 24.5, 26.0
LITERAL_STEPS = 150                               # the literal replay never grasps; its start is enough
N = int(T_END * FPS)
TABLE_Z = 0.90
TARGET = np.array([-0.08, 0.14, 0.20])            # what the orbiting camera looks at (table frame)
F_ORBIT = 760.0


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def capture_time(tau):
    """Animation time -> capture time: the approach slightly slowed, the carry slightly sped up."""
    return np.interp(tau, [0.0, T_FADE, T_REAL, T_END], [0.4, 2.0, 10.3, 10.3])


def look_at(eye, target, up):
    z = target - eye; z = z / np.linalg.norm(z)
    x = np.cross(z, up); x = x / np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])          # camera axes (x right, y down, z forward) as rows
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = -R @ eye
    return T


def orbit_eye(tau):
    # a steady sweep over the reconstruction, then half as fast while the robot moves
    az = np.radians(np.interp(tau, [T_FADE, T_REAL, T_END], [-28.0, -7.4, 30.0]))
    return TARGET + np.array([1.25 * np.cos(az), 1.25 * np.sin(az), 0.50])


def camera(tau, T_table_from_phone, f_phone):
    """The virtual camera: the phone itself until the fade ends, then easing out to an orbit."""
    e = ease((tau - T_FADE) / 2.2)
    eye_p = T_table_from_phone[:3, 3]; fwd_p = T_table_from_phone[:3, 2]; up_p = -T_table_from_phone[:3, 1]
    eye = (1 - e) * eye_p + e * orbit_eye(tau)
    target = (1 - e) * (eye_p + 0.7 * fwd_p) + e * TARGET
    up = (1 - e) * up_p + e * np.array([0.0, 0.0, 1.0])
    return look_at(eye, target, up), (1 - e) * f_phone + e * F_ORBIT


def project(P, T, f):
    Pc = P @ T[:3, :3].T + T[:3, 3]
    z = np.where(Pc[:, 2] > 0.05, Pc[:, 2], np.nan)
    return np.stack([f * Pc[:, 0] / z + W / 2, f * Pc[:, 1] / z + H / 2], 1), Pc[:, 2]


def splat(P, C, T, f, img):
    uv, z = project(P, T, f)
    ok = np.isfinite(uv).all(1)
    uv, z, C = uv[ok].astype(np.int32), z[ok], C[ok]
    order = np.argsort(-z); uv, C = uv[order], C[order]          # far first, near overwrites
    for du in range(2):
        for dv in range(2):
            u, v = uv[:, 0] + du, uv[:, 1] + dv
            m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            img[v[m], u[m]] = C[m]


def stage_real():
    from humanoid_challenge.record3d import load_r3d
    d = np.load(f"data/interim/{EP}.npz"); cap = load_r3d(f"data/raw/{CHUNK}/{EP}.r3d")
    t, T_tab = d["t"], d["cam_T_table"]
    stride = round(cap.n_frames / len(t))
    cw, ch = 640, 480
    K = cap.K_depth().copy(); K[0] *= cw / cap.depth_size[0]; K[1] *= ch / cap.depth_size[1]
    u, v = np.meshgrid(np.arange(cw), np.arange(ch))

    def cloud(i):
        rgb = cv2.resize(cap.rgb(i * stride), (cw, ch), interpolation=cv2.INTER_AREA)
        raw = cap.depth(i * stride)
        step = np.maximum(np.abs(np.gradient(raw, axis=0)), np.abs(np.gradient(raw, axis=1)))
        raw = np.where(step > 0.02, np.nan, raw)          # a depth edge: these points belong to neither surface
        dep = cv2.resize(raw, (cw, ch), interpolation=cv2.INTER_NEAREST)
        P = np.stack([(u - K[0, 2]) / K[0, 0] * dep, (v - K[1, 2]) / K[1, 1] * dep, dep], -1).reshape(-1, 3)
        ok = np.isfinite(P).all(1) & (P[:, 2] > 0.15) & (P[:, 2] < 2.0)
        return P[ok] @ T_tab[i][:3, :3].T + T_tab[i][:3, 3], rgb.reshape(-1, 3)[ok]

    # What the phone saw at other moments, dimmed: fused from frames without the hand in view.
    parts = [cloud(i) for i in range(len(t)) if not d["valid"][i]][::3]
    bgP = np.concatenate([p for p, _ in parts]); bgC = np.concatenate([c for _, c in parts])
    _, keep = np.unique(np.floor(bgP / 0.004).astype(np.int64), axis=0, return_index=True)
    bgP, bgC = bgP[keep], (bgC[keep] * 0.45).astype(np.uint8)

    from humanoid_challenge.hands import HandTracker, INDEX_TIP, THUMB_TIP
    from humanoid_challenge.markers import make_detector
    hands, detector = HandTracker(num_hands=1), make_detector("DICT_4X4_50")
    f_phone = cap.K[0, 0] * W / cap.rgb_size[0]
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "real").mkdir(exist_ok=True)
    cams, fs, idx = [], [], []
    for k in range(N):
        tau = k / FPS
        i = int(np.argmin(np.abs(t - capture_time(tau))))
        T, f = camera(tau, T_tab[i], f_phone)
        cams.append(T); fs.append(f); idx.append(i)
        if tau >= T_CUT:
            continue
        img = np.full((H, W, 3), 14, np.uint8)
        if tau >= T_RGB:
            # hide remembered points that the phone is looking at right now: the live frame owns them
            pc = (bgP - T_tab[i][:3, 3]) @ T_tab[i][:3, :3]
            uu, vv = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
            seen = (pc[:, 2] > 0) & (uu > 0) & (uu < cw) & (vv > 0) & (vv < ch)
            splat(bgP[~seen], bgC[~seen], T, f, img)
            splat(*cloud(i), T, f, img)
        if tau < T_FADE:
            rgb = cap.rgb(i * stride).copy()
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            hf = hands(rgb, int(tau * 1000))
            if ids is not None:
                cv2.polylines(rgb, [c.astype(np.int32) for c in corners], True, (60, 255, 60), 7, cv2.LINE_AA)
            if hf.ok:
                px = hf.landmarks_px.astype(int)
                for q in px:
                    cv2.circle(rgb, tuple(q), 7, (40, 170, 255), -1, cv2.LINE_AA)
                cv2.line(rgb, tuple(px[THUMB_TIP]), tuple(px[INDEX_TIP]), (235, 60, 60), 5, cv2.LINE_AA)
            h = rgb.shape[1] * H // W; y0 = (rgb.shape[0] - h) // 2
            rgb = cv2.resize(rgb[y0:y0 + h], (W, H), interpolation=cv2.INTER_AREA)
            a = ease((tau - T_RGB) / (T_FADE - T_RGB))
            img = cv2.addWeighted(rgb, 1 - a, img, a, 0)
        cv2.imwrite(str(OUT / "real" / f"{k:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    hands.close()
    np.savez(OUT / "cams.npz", T=np.stack(cams), f=np.array(fs), idx=np.array(idx), t=t, hand=d["eef_pos"],
             gripper=d["gripper"], phone=T_tab[:, :3, 3], phone_R=T_tab[:, :3, :3])
    print("real frames ->", OUT / "real")


def stage_sim():
    import robosuite.utils.transform_utils as RT
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    c = np.load(OUT / "cams.npz"); z = np.load(OUT / "states" / f"{EP}.npz")
    literal = np.load(OUT / "states_literal" / f"{EP}.npz")["states"][:LITERAL_STEPS]
    fixed = z["states"]
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) / str(z["bddl"])),
                             camera_heights=H, camera_widths=W)
    env.seed(0); env.reset()
    sim = env.sim; cid = sim.model.camera_name2id("agentview")
    (OUT / "sim").mkdir(exist_ok=True)
    for k in range(int(T_REAL * FPS), N):
        tau = k / FPS
        if tau < T_LIT:
            state = literal[int(round(np.interp(tau, [T_CUT, T_LIT - 0.2], [0, len(literal) - 1])))]
        else:
            state = fixed[int(round(np.interp(tau, [T_LIT + 0.2, T_SIM], [0, len(fixed) - 1])))]
        T = c["T"][k]; R = T[:3, :3]
        # MuJoCo cameras look along -z with y up; ours look along +z with y down
        sim.model.cam_pos[cid] = -R.T @ T[:3, 3] + [0.0, 0.0, TABLE_Z]
        q = RT.mat2quat(R.T @ np.diag([1.0, -1.0, -1.0]))             # xyzw
        sim.model.cam_quat[cid] = [q[3], q[0], q[1], q[2]]
        sim.model.cam_fovy[cid] = np.degrees(2 * np.arctan(H / 2 / c["f"][k]))
        env.set_state(state); sim.forward()
        img = sim.render(camera_name="agentview", height=H, width=W)[::-1]
        cv2.imwrite(str(OUT / "sim" / f"{k:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    env.close()
    print("sim frames ->", OUT / "sim")


def polyline(img, P, T, f, colour, width=3):
    uv, z = project(P, T, f)
    for a, b, za, zb in zip(uv[:-1], uv[1:], z[:-1], z[1:]):
        if np.isfinite(a).all() and np.isfinite(b).all():
            cv2.line(img, tuple(int(x) for x in a), tuple(int(x) for x in b), colour, width, cv2.LINE_AA)


def caption(img, text, sub=None):
    from PIL import Image, ImageDraw, ImageFont
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    im = Image.fromarray(img); dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, H - 86, W, H], fill=(0, 0, 0, 150))
    dr.text((36, H - 74), text, font=ImageFont.truetype(font, 28), fill=(255, 255, 255))
    if sub:
        dr.text((36, H - 36), sub, font=ImageFont.truetype(font, 19), fill=(200, 200, 200))
    return np.asarray(im)


def stage_compose(end_line):
    c = np.load(OUT / "cams.npz")
    t, hand, grip = c["t"], c["hand"], c["gripper"]
    OPEN, CLOSED, PHONE = (255, 170, 40), (235, 60, 60), (120, 200, 255)
    (OUT / "final").mkdir(exist_ok=True)
    for k in range(N):
        tau = k / FPS; T, f, i = c["T"][k], c["f"][k], c["idx"][k]
        real = cv2.cvtColor(cv2.imread(str(OUT / "real" / f"{k:04d}.jpg")), cv2.COLOR_BGR2RGB) if tau < T_CUT else None
        simg = cv2.cvtColor(cv2.imread(str(OUT / "sim" / f"{k:04d}.jpg")), cv2.COLOR_BGR2RGB) if tau >= T_REAL else None
        if real is not None and simg is not None:
            a = ease((tau - T_REAL) / (T_CUT - T_REAL)); img = cv2.addWeighted(real, 1 - a, simg, a, 0)
        else:
            img = real if real is not None else simg
        img = np.ascontiguousarray(img)
        if tau >= T_FADE:                                   # the hand's path so far, by gripper state
            j0 = int(np.argmin(np.abs(t - capture_time(0.0))))
            for a_, b_ in zip(range(j0, i), range(j0 + 1, i + 1)):
                polyline(img, hand[[a_, b_]], T, f, CLOSED if grip[b_] > 0 else OPEN)
        if T_FADE + 0.8 <= tau < T_REAL:                    # where the phone is, and where it has been
            polyline(img, c["phone"][:i + 1:3], T, f, PHONE, 1)
            R = c["phone_R"][i]; o = c["phone"][i]
            corners = [o + 0.10 * (R @ np.array([sx * 0.66, sy * 0.5, 1.0])) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            for q in corners:
                polyline(img, np.array([o, q]), T, f, PHONE, 1)
            polyline(img, np.array(corners + corners[:1]), T, f, PHONE, 1)
        if tau < T_FADE:
            img = caption(img, "A phone on my head records colour, depth and its own motion")
        elif tau < T_REAL:
            img = caption(img, "The hand is tracked and lifted into 3-D, in a frame fixed to the table",
                          "orange: pinch open   red: pinch closed   blue: the phone's path")
        elif tau < T_LIT:
            img = caption(img, "Followed literally, the gripper closes sideways and too high",
                          "1 of 59 captures replay successfully")
        elif tau < T_SIM:
            img = caption(img, "With two constants from the robot's own demonstrations, it works",
                          "wrist held at its home angle; grasp 1.4 cm below the rim; no object positions used")
        else:
            img = caption(cv2.addWeighted(img, 0.55, np.zeros_like(img), 0.45, 0), "38 of 59 phone captures replay successfully", end_line)
        cv2.imwrite(str(OUT / "final" / f"{k:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    mp4, gif = "docs/examples/teaser.mp4", "docs/figures/teaser.gif"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(OUT / "final" / "%04d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "25", "-movflags", "+faststart", mp4], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp4, "-vf",
                    "fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    gif], check=True)
    print("wrote", mp4, gif)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["real", "sim", "compose"])
    ap.add_argument("--end-line", default="no knowledge of where the objects are")
    a = ap.parse_args()
    {"real": stage_real, "sim": stage_sim, "compose": lambda: stage_compose(a.end_line)}[a.stage]()
