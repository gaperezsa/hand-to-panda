#!/usr/bin/env python3
"""Rename Record3D exports into the episode naming scheme, by capture order.

Record3D names files by timestamp. Renaming 61 clips by hand on a phone is a reliable
way to mislabel one, and a mislabelled episode trains the policy on the wrong language
instruction. So: drop clips into the folder for their position, keep the app's names,
and this assigns A_t2_001 ... in the order they were captured.

Dry run by default; pass --apply to actually rename.

    python tools/ingest_raw.py
    python tools/ingest_raw.py --apply
"""
import argparse
import hashlib
from pathlib import Path

FOLDERS = {"sweep": "sweep", "test": "test", "A_t2": "A_t2", "A_t0": "A_t0",
           "A_t8": "A_t8", "A_t1": "A_t1", "A_t6": "A_t6", "B_push": "B_push"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    total = 0
    for folder, prefix in FOLDERS.items():
        d = a.raw / folder
        if not d.is_dir():
            continue
        # capture order = mtime; Record3D preserves it and it is what the phone recorded
        clips = sorted((p for p in d.glob("*.r3d")), key=lambda p: p.stat().st_mtime)

        # Shellfish sometimes retries an upload and leaves "name 2.r3d" beside "name.r3d".
        # A duplicate silently becomes a phantom extra episode, so quarantine byte-identical
        # copies instead of letting them through. Hash only files that share a size.
        by_size = {}
        for c in clips:
            by_size.setdefault(c.stat().st_size, []).append(c)
        dup_dir = a.raw / ".duplicates"
        for size, group in by_size.items():
            if len(group) < 2:
                continue
            seen = {}
            for c in sorted(group, key=lambda p: (len(p.name), p.name)):
                h = hashlib.sha256(c.read_bytes()).hexdigest()
                if h in seen:
                    print(f"   DUPLICATE of {seen[h].name}: {c.name} -> .duplicates/")
                    if a.apply:
                        dup_dir.mkdir(exist_ok=True)
                        c.rename(dup_dir / c.name)
                    clips = [x for x in clips if x != c]
                else:
                    seen[h] = c
        already = [p for p in clips if p.stem.startswith(prefix + "_")]
        todo = [p for p in clips if p not in already]
        if not clips:
            continue
        print(f"\n{folder}/  {len(clips)} clip(s), {len(already)} already named")
        n = len(already)
        for p in todo:
            n += 1
            dst = p.with_name(f"{prefix}_{n:03d}.r3d")
            while dst.exists():
                n += 1
                dst = p.with_name(f"{prefix}_{n:03d}.r3d")
            size = p.stat().st_size / 1e6
            print(f"   {p.name:<36} -> {dst.name}   {size:6.1f} MB")
            if a.apply:
                p.rename(dst)
            total += 1
    if total == 0:
        print("nothing to rename")
    elif not a.apply:
        print(f"\n{total} file(s) would be renamed. Re-run with --apply.")
    else:
        print(f"\nrenamed {total} file(s)")


if __name__ == "__main__":
    main()
