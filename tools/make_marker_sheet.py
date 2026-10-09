#!/usr/bin/env python3
"""Generate printable ArUco marker sheets for the table frame.

One marker per page so nothing competes for the quiet zone. Each page carries a
100 mm verification ruler OUTSIDE the quiet zone -- measure it after printing and
put the measured value into config.yaml as marker_length_m. Printer scaling is the
single most common way to silently corrupt a marker-based pose pipeline.

Usage:  python tools/make_marker_sheet.py [--out-dir docs/recording]
"""
import argparse
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

DICT_NAME = "DICT_4X4_50"
IDS = [0, 1, 2, 3]
MARKER_MM = 100.0          # side of the full black square, border bits included
QUIET_MM = 25.0            # white margin on each side; part of the printed tile
RENDER_PX = 2400           # 6 cells x 400 px -> ~600 dpi at 100 mm


def marker_png(marker_id: int, path: Path) -> Path:
    d = aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))
    # borderBits=1 is the ArUco default: the black border is INSIDE MARKER_MM.
    img = aruco.generateImageMarker(d, marker_id, RENDER_PX, borderBits=1)
    Image.fromarray(img).save(path)
    return path


def draw_page(c: canvas.Canvas, page_w: float, page_h: float, marker_id: int, png: Path) -> None:
    tile = (MARKER_MM + 2 * QUIET_MM) * mm
    tx, ty = (page_w - tile) / 2, (page_h - tile) / 2

    # Cut guides at the tile edge, drawn as short ticks so no line crosses the quiet zone.
    c.setStrokeColorRGB(.65, .65, .65)
    c.setLineWidth(.4)
    for x in (tx, tx + tile):
        for y0, y1 in ((ty - 9 * mm, ty - 2 * mm), (ty + tile + 2 * mm, ty + tile + 9 * mm)):
            c.line(x, y0, x, y1)
    for y in (ty, ty + tile):
        for x0, x1 in ((tx - 9 * mm, tx - 2 * mm), (tx + tile + 2 * mm, tx + tile + 9 * mm)):
            c.line(x0, y, x1, y)

    c.drawImage(str(png), tx + QUIET_MM * mm, ty + QUIET_MM * mm,
                width=MARKER_MM * mm, height=MARKER_MM * mm)

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(tx, ty + tile + 14 * mm, f"ArUco  {DICT_NAME}   ID {marker_id}")
    c.setFont("Helvetica", 9.5)
    c.drawString(tx, ty + tile + 9 * mm,
                 f"Black square {MARKER_MM:.0f} mm  |  quiet zone {QUIET_MM:.0f} mm each side"
                 f"  |  tile {MARKER_MM + 2 * QUIET_MM:.0f} mm")

    # 100 mm verification ruler, below the tile, clear of the quiet zone.
    ry = ty - 20 * mm
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(1.1)
    c.line(tx, ry, tx + 100 * mm, ry)
    for i in range(11):
        x = tx + i * 10 * mm
        h = 4.5 * mm if i % 5 == 0 else 2.6 * mm
        c.line(x, ry, x, ry + h)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(tx + 102 * mm, ry - 1.2 * mm, "100 mm")
    c.setFont("Helvetica", 8.6)
    c.setFillColorRGB(.32, .32, .32)
    c.drawString(tx, ry - 7 * mm,
                 "PRINT AT 100% / ACTUAL SIZE -- turn OFF 'fit to page' and 'shrink oversized pages'.")
    c.drawString(tx, ry - 12 * mm,
                 "Then measure this line AND the black square with a ruler. If they are not 100 mm,")
    c.drawString(tx, ry - 17 * mm,
                 "do not reprint -- just put the measured black-square size into config.yaml.")
    c.showPage()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/recording", type=Path)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = a.out_dir / "_markers_png"
    tmp.mkdir(exist_ok=True)
    pngs = {i: marker_png(i, tmp / f"aruco_{DICT_NAME}_{i}.png") for i in IDS}

    for label, size in (("letter", letter), ("a4", A4)):
        out = a.out_dir / f"aruco_markers_{label}.pdf"
        c = canvas.Canvas(str(out), pagesize=size)
        c.setTitle(f"ArUco {DICT_NAME} IDs {IDS} - {MARKER_MM:.0f} mm")
        for i in IDS:
            draw_page(c, size[0], size[1], i, pngs[i])
        c.save()
        print(f"wrote {out}  ({len(IDS)} pages, {label})")


if __name__ == "__main__":
    main()
