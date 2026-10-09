#!/usr/bin/env python3
"""What has been recorded, what has been processed, and what is still missing.

Exists because a real gap went unnoticed: three clips finished uploading after their
chunk had already been retargeted, were renamed during a later chunk's ingest, and so
sat on disk with valid names and no intermediate file. Nothing compared the raw count
against the processed count.

    python tools/status.py
"""
import sys
from pathlib import Path

CHUNKS = ["sweep", "test", "A_t2", "A_t0", "A_t8", "A_t1", "A_t6", "B_push"]


def main():
    raw = Path("data/raw")
    interim = Path("data/interim")
    print(f"{'chunk':<9}{'raw':>5}{'processed':>11}{'uploading':>11}{'unnamed':>9}  missing")
    total_raw = total_done = 0
    for c in CHUNKS:
        d = raw / c
        if not d.is_dir():
            continue
        named = sorted(d.glob(f"{c}_*.r3d"))
        other = [p for p in d.glob("*.r3d") if p not in named]
        up = len(list(d.glob("_shellfish_uploading*")))
        done = {p.stem for p in interim.glob(f"{c}_*.npz")}
        missing = [p.stem for p in named if p.stem not in done]
        total_raw += len(named) + len(other)
        total_done += len(done)
        flag = ", ".join(missing) if missing else ""
        if other:
            flag = (flag + "; " if flag else "") + f"{len(other)} not yet renamed"
        print(f"{c:<9}{len(named) + len(other):5d}{len(done):11d}{up:11d}{len(other):9d}  {flag}")
    print(f"\n{'TOTAL':<9}{total_raw:5d}{total_done:11d}")
    if total_raw != total_done:
        print("\nTo process everything outstanding:")
        print("  python tools/ingest_raw.py --apply")
        print("  python tools/retarget_capture.py 'data/raw/<chunk>/<chunk>_*.r3d' --glob "
              "--stride 2 --pinch-closed 40 --pinch-open 95")
    return 0


if __name__ == "__main__":
    sys.exit(main())
