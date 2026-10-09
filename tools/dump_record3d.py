#!/usr/bin/env python3
"""Print everything about a Record3D export. Assume nothing; parse nothing.

Record3D has several export paths (.r3d bundle, "export to folder", EXR sequences,
USDZ, plain mp4) and they do not share a layout. Rather than guess, this walks whatever
it is given and reports structure, sizes, dtypes and the actual shape of the metadata,
so the real parser can be written against the real file.

Where it does interpret something -- JSON keys, image shapes, the byte-size of a depth
frame against a candidate resolution -- it says so and shows its working.

    python tools/dump_record3d.py /path/to/capture.r3d
    python tools/dump_record3d.py /path/to/exported_folder/
"""
import argparse
import json
import struct
import zipfile
from collections import Counter
from pathlib import Path

MAX_LIST = 3


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f}{u}" if u != "B" else f"{n}B"
        n /= 1024


def describe_json(obj, indent="    ", depth=0):
    pad = indent + "  " * depth
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                n = len(v)
                kind = "dict" if isinstance(v, dict) else "list"
                print(f"{pad}{k}: {kind}[{n}]")
                if kind == "list" and n:
                    e = v[0]
                    print(f"{pad}  first element: {type(e).__name__} "
                          f"{'len=' + str(len(e)) if isinstance(e, (list, dict)) else repr(e)[:90]}")
                    if isinstance(e, list) and n > 1:
                        print(f"{pad}  [0] = {e}")
                        print(f"{pad}  [-1] = {v[-1]}")
                elif kind == "dict" and depth < 2:
                    describe_json(v, indent, depth + 1)
            else:
                print(f"{pad}{k}: {type(v).__name__} = {repr(v)[:110]}")
    else:
        print(f"{pad}{type(obj).__name__}: {repr(obj)[:200]}")


def probe_depth_bytes(nbytes, candidates=((256, 192), (320, 240), (640, 480), (192, 256))):
    """A raw depth frame's byte count constrains (w, h, dtype). Show the options."""
    out = []
    for (w, h) in candidates:
        for name, size in (("float16", 2), ("float32", 4)):
            if nbytes == w * h * size:
                out.append(f"{w}x{h} {name}")
    return out or ["no exact match -- probably compressed (Record3D uses LZFSE for .depth)"]


def dump_file(path: Path, data: bytes | None = None, prefix="  "):
    ext = path.suffix.lower()
    n = len(data) if data is not None else path.stat().st_size
    print(f"{prefix}{path.name}  {human(n)}")
    raw = data if data is not None else (path.read_bytes() if n < 64 * 1024 * 1024 else None)
    if raw is None:
        return
    if ext in (".json", "") and raw[:1] in (b"{", b"["):
        try:
            describe_json(json.loads(raw), indent=prefix + "    ")
            return
        except Exception as e:
            print(f"{prefix}    (not JSON: {e})")
    if ext in (".jpg", ".jpeg", ".png"):
        try:
            import io

            from PIL import Image
            im = Image.open(io.BytesIO(raw))
            print(f"{prefix}    image {im.size[0]}x{im.size[1]} mode={im.mode} format={im.format}")
        except Exception as e:
            print(f"{prefix}    (image read failed: {e})")
    elif ext == ".npy":
        try:
            import io

            import numpy as np
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            print(f"{prefix}    ndarray shape={arr.shape} dtype={arr.dtype} "
                  f"min={arr.min():.4f} max={arr.max():.4f}")
        except Exception as e:
            print(f"{prefix}    (npy read failed: {e})")
    elif ext in (".depth", ".conf", ".bin"):
        print(f"{prefix}    magic={raw[:4]!r}  candidates: {', '.join(probe_depth_bytes(n))}")
    else:
        print(f"{prefix}    magic={raw[:8]!r}")


def dump_zip(path: Path):
    print(f"ZIP bundle: {path}  ({human(path.stat().st_size)})")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        print(f"  {len(names)} entries")
        by_dir = Counter(str(Path(n).parent) for n in names)
        by_ext = Counter(Path(n).suffix.lower() or "(none)" for n in names)
        print("  by directory:", dict(by_dir.most_common(10)))
        print("  by extension:", dict(by_ext.most_common(10)))
        print("\n  --- metadata-ish entries, in full ---")
        for n in names:
            if Path(n).suffix.lower() in ("", ".json", ".plist", ".txt") and Path(n).name:
                info = z.getinfo(n)
                if info.file_size < 8 * 1024 * 1024:
                    dump_file(Path(n), z.read(n), prefix="  ")
        print("\n  --- first few of each other extension ---")
        for ext in by_ext:
            if ext in ("", ".json", ".plist", ".txt"):
                continue
            sample = [n for n in names if Path(n).suffix.lower() == ext][:MAX_LIST]
            for n in sample:
                dump_file(Path(n), z.read(n), prefix="  ")


def dump_dir(path: Path):
    print(f"DIRECTORY: {path}")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    print(f"  {len(files)} files, {human(sum(p.stat().st_size for p in files))}")
    by_ext = Counter(p.suffix.lower() or "(none)" for p in files)
    print("  by extension:", dict(by_ext.most_common(12)))
    print("  directories:", sorted({str(p.parent.relative_to(path)) for p in files})[:12])
    print()
    for ext in by_ext:
        print(f"  --- {ext} ({by_ext[ext]} files) ---")
        for p in [q for q in files if (q.suffix.lower() or "(none)") == ext][:MAX_LIST]:
            dump_file(p, prefix="  ")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    a = ap.parse_args()
    p = a.path
    if not p.exists():
        raise SystemExit(f"no such path: {p}")
    if p.is_dir():
        dump_dir(p)
    elif zipfile.is_zipfile(p):
        dump_zip(p)
    else:
        dump_file(p, prefix="")


if __name__ == "__main__":
    main()
