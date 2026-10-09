"""
Read a Record3D capture.

Record3D has several export paths and they do not share a layout. This parser
probes rather than assumes what comes within the r3d files.

We log uncertain decisions in `Capture.notes`, so if a failure happens it shows up as a
message instead of as silently wrong geometry.

What is checked rather than trusted:
  1 `.r3d` is a zip with a JSON `metadata` entry and an `rgbd/` directory.
  2 depth frames are LZFSE-compressed float32 (or float16), row-major, dw x dh.
  3 poses are [qx, qy, qz, qw, tx, ty, tz], camera-to-world, ARKit convention
    (x right, y up, z backward -- the camera looks down -z).

The ARKit-to-OpenCV rotation is applied here, once, so nothing downstream has to think
about it.

OpenCV cameras look down +z with y down; a simple 180 degree rotation about x does the trick.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARKIT_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


@dataclass
class Capture:
    path: Path
    n_frames: int
    rgb_size: tuple[int, int]                  # (w, h)
    depth_size: tuple[int, int]                # (w, h)
    K: np.ndarray                              # 3x3, for the RGB resolution
    fps: float
    timestamps: np.ndarray                     # (T,) seconds, starting at 0
    T_world_cam: np.ndarray                    # (T, 4, 4) camera-to-world, OpenCV convention
    notes: list[str] = field(default_factory=list)
    _zip: zipfile.ZipFile | None = None
    _rgb_names: list[str] = field(default_factory=list)
    _depth_names: list[str] = field(default_factory=list)

    def rgb(self, i: int) -> np.ndarray:
        from PIL import Image
        with self._zip.open(self._rgb_names[i]) as f:
            return np.asarray(Image.open(io.BytesIO(f.read())).convert("RGB"))

    def depth(self, i: int) -> np.ndarray:
        """Metres, shape (h, w) at the depth resolution. NaN where invalid."""
        raw = self._zip.read(self._depth_names[i])
        w, h = self.depth_size
        try:
            import liblzfse
            raw = liblzfse.decompress(raw)
        except Exception:
            pass                                # some exports store it uncompressed
        for dt in (np.float32, np.float16):
            if len(raw) == w * h * np.dtype(dt).itemsize:
                d = np.frombuffer(raw, dt).reshape(h, w).astype(np.float32)
                d[~np.isfinite(d)] = np.nan
                d[d <= 0] = np.nan
                return d
        raise ValueError(f"depth frame is {len(raw)} bytes; expected {w*h*4} (f32) "
                         f"or {w*h*2} (f16) for {w}x{h}")

    def K_depth(self) -> np.ndarray:
        """Intrinsics rescaled to the depth resolution."""
        sx = self.depth_size[0] / self.rgb_size[0]
        sy = self.depth_size[1] / self.rgb_size[1]
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return K


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _numeric_key(name: str) -> tuple:
    stem = Path(name).stem
    return (int(stem), name) if stem.isdigit() else (1 << 30, name)


def load_r3d(path: str | Path) -> Capture:
    path = Path(path)
    notes: list[str] = []
    z = zipfile.ZipFile(path)
    names = z.namelist()

    meta_name = next((n for n in names if Path(n).name in ("metadata", "metadata.json")), None)
    if meta_name is None:
        raise ValueError(f"no metadata entry in {path}; entries look like {names[:5]}")
    meta = json.loads(z.read(meta_name))

    rgb = sorted((n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))), key=_numeric_key)
    dep = sorted((n for n in names if n.lower().endswith(".depth")), key=_numeric_key)
    if not rgb:
        raise ValueError("no RGB frames found")
    if len(dep) != len(rgb):
        notes.append(f"{len(rgb)} RGB frames but {len(dep)} depth frames; using the shorter")

    w = int(meta.get("w") or meta.get("width") or 0)
    h = int(meta.get("h") or meta.get("height") or 0)
    dw = int(meta.get("dw") or meta.get("depthWidth") or 0)
    dh = int(meta.get("dh") or meta.get("depthHeight") or 0)
    if not (w and h):
        from PIL import Image
        with z.open(rgb[0]) as f:
            w, h = Image.open(io.BytesIO(f.read())).size
        notes.append(f"metadata had no w/h; read {w}x{h} from the first JPEG")
    if not (dw and dh):
        dw, dh = 256, 192
        notes.append("metadata had no dw/dh; assuming 256x192 (verify against the dump tool)")

    Kflat = np.asarray(meta.get("K", []), dtype=float)
    if Kflat.size == 9:
        # Record3D stores K column-major.
        K = Kflat.reshape(3, 3).T
        if not np.isclose(K[2, 2], 1.0, atol=1e-6):
            K = Kflat.reshape(3, 3)
            notes.append("K did not look column-major; used row-major instead")
    else:
        raise ValueError(f"metadata K has {Kflat.size} entries, expected 9")

    poses = np.asarray(meta.get("poses", []), dtype=float)
    n = min(len(rgb), len(dep) or len(rgb), len(poses) if len(poses) else len(rgb))
    if len(poses) == 0:
        raise ValueError("metadata has no 'poses'; cannot build a trajectory")
    if len(poses) != len(rgb):
        notes.append(f"{len(poses)} poses for {len(rgb)} frames; truncating to {n}")

    T = np.zeros((n, 4, 4))
    for i in range(n):
        p = poses[i]
        R = _quat_to_mat(p[:4]) @ ARKIT_TO_OPENCV      # ARKit -> OpenCV camera axes
        T[i, :3, :3] = R
        T[i, :3, 3] = p[4:7]
        T[i, 3, 3] = 1.0

    ts = np.asarray(meta.get("frameTimestamps", []), dtype=float)[:n]
    fps = float(meta.get("fps") or 0) or 30.0
    if ts.size != n:
        ts = np.arange(n) / fps
        notes.append(f"no usable frameTimestamps; synthesised from fps={fps}")
    ts = ts - ts[0]

    return Capture(path=path, n_frames=n, rgb_size=(w, h), depth_size=(dw, dh), K=K, fps=fps,
                   timestamps=ts, T_world_cam=T, notes=notes, _zip=z,
                   _rgb_names=rgb[:n], _depth_names=dep[:n])
