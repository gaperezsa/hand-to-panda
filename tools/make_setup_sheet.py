#!/usr/bin/env python3
"""Printable table-setup sheet: where every marker and object goes, in tape-measure terms.

The LIBERO table frame has no physical anchor, so the sheet converts it into the only two
measurements that exist on a real table: distance from the near edge and from the left
edge. Those assume the layout is centred on a 70 x 120 cm table; residual error is
measured and corrected by the calibration sweep, so being a few millimetres out is fine.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

TX, TY = 0.70, 1.20
TILE, MK_SIDE = 0.150, 0.100
MK = {0: (0.110, 0.500), 1: (0.110, -0.250), 2: (-0.155, -0.250), 3: (-0.155, 0.500)}
BOWLS = [("1", (-0.075, 0.000), "#d6453f"), ("2", (-0.050, 0.200), "#de8c28"),
         ("3", (0.010, 0.310), "#3e8c5c"), ("4", (-0.180, 0.320), "#306abc"),
         ("5", (0.130, -0.070), "#7e48aa")]
BOWL_TAG = {"1": "A_t2", "2": "A_t0", "3": "A_t8", "4": "A_t1", "5": "A_t6"}
PLATE_A, PLATE_B, GOAL_B, BOX_B = (0.060, 0.200), (0.050, -0.020), (-0.09, -0.01, 0.17, 0.25), (-0.050, 0.130)

xmax = max(v[0] for v in MK.values()) + TILE / 2
xmin = min(v[0] for v in MK.values()) - TILE / 2
ymax = max(v[1] for v in MK.values()) + TILE / 2
ymin = min(v[1] for v in MK.values()) - TILE / 2
X_NEAR = xmax + (TX - (xmax - xmin)) / 2
Y_RIGHT = ymax + (TY - (ymax - ymin)) / 2
near = lambda x: round((X_NEAR - x) * 1000)
right = lambda y: round((Y_RIGHT - y) * 1000)
left = lambda y: round((y - (Y_RIGHT - TY)) * 1000)

ROWS = [("MARKER ID 0      near-RIGHT", *MK[0]), ("MARKER ID 1      near-LEFT", *MK[1]),
        ("MARKER ID 2      far-LEFT", *MK[2]), ("MARKER ID 3      far-RIGHT", *MK[3]),
        ("PLATE   target, NEVER MOVES", *PLATE_A),
        ("chunk 1   A_t2   bowl", -0.075, 0.000),
        ("chunk 2   A_t0   bowl", -0.050, 0.200),
        ("chunk 3   A_t8   bowl", 0.010, 0.310),
        ("chunk 4   A_t1   bowl", -0.180, 0.320),
        ("chunk 5   A_t6   bowl", 0.130, -0.070),
        ("push: plate start", *PLATE_B),
        ("push: goal zone centre", -0.050, 0.210),
        ("push: obstacle box", *BOX_B)]


def draw(ax, layout, title):
    """One layout at a time. Drawing A and B together reads as a collision that is not
    there -- only one bowl is ever on the table, and the push obstacle sits where bowl t0
    goes in layout A."""
    ax.add_patch(Rectangle((Y_RIGHT - TY, X_NEAR - TX), TY, TX, fc="#ecdfc8", ec="#6b5c44", lw=2.0))
    for mid, (x, y) in MK.items():
        ax.add_patch(Rectangle((y - TILE / 2, x - TILE / 2), TILE, TILE, fc="white", ec="#777", lw=1.1, zorder=3))
        ax.add_patch(Rectangle((y - MK_SIDE / 2, x - MK_SIDE / 2), MK_SIDE, MK_SIDE, fc="#1a1a1a", zorder=4))
        ax.text(y, x, str(mid), color="white", ha="center", va="center", fontsize=12,
                fontweight="bold", zorder=5)
    ax.plot([MK[0][1], MK[1][1], MK[2][1], MK[3][1], MK[0][1]],
            [MK[0][0], MK[1][0], MK[2][0], MK[3][0], MK[0][0]],
            color="#555", lw=.9, ls=(0, (4, 3)), zorder=2)

    if layout == "A":
        px, py = PLATE_A
        ax.add_patch(Circle((py, px), 0.048, fc="#fbf6f6", ec="#bd6a6c", lw=2.2, zorder=4))
        ax.text(py, px, "P", color="#aa5c5e", ha="center", va="center", fontsize=11,
                fontweight="bold", zorder=5)
        for tid, (x, y), col in BOWLS:
            ax.add_patch(Circle((y, x), 0.0475, fc=col, ec="#222", lw=1.3, zorder=4))
            ax.text(y, x, tid, color="white", ha="center", va="center", fontsize=12,
                    fontweight="bold", zorder=5)
            ax.text(y, x - 0.062, BOWL_TAG[tid], color=col, ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", zorder=5)
            ax.plot([y, py], [x, px], color=col, lw=1.0, alpha=.5, zorder=2)
        ax.text(Y_RIGHT - TY + .02, X_NEAR - TX + .03, "one bowl at a time\n10 episodes each",
                fontsize=8.5, color="#2e6b3e", va="top",
                bbox=dict(fc="#eaf5ec", ec="#9cc4a6", lw=.7, pad=3))
    else:
        gx0, gx1, gy0, gy1 = GOAL_B
        ax.add_patch(Rectangle((gy0, gx0), gy1 - gy0, gx1 - gx0, fc="#8fd18f", ec="#2e7d32",
                               lw=1.4, ls=(0, (4, 3)), alpha=.55, zorder=3))
        ax.text((gy0 + gy1) / 2, (gx0 + gx1) / 2, "goal", color="#1b5e20", ha="center",
                va="center", fontsize=8.5, fontweight="bold", zorder=5)
        ax.add_patch(Circle((BOX_B[1], BOX_B[0]), 0.030, fc="#4a74c8", ec="#222", lw=1.3, zorder=4))
        ax.text(BOX_B[1], BOX_B[0] - .055, "small box", color="#24408a", ha="center",
                fontsize=8.5, zorder=5)
        px, py = PLATE_B
        ax.annotate("", xy=(-(gy0 + gy1) / 2, (gx0 + gx1) / 2), xytext=(-py, px),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#666", ls="--",
                                    shrinkA=14, shrinkB=12), zorder=3)
        ax.add_patch(Circle((py, px), 0.048, fc="#fbf6f6", ec="#bd6a6c", lw=2.2, zorder=4))
        ax.text(py, px, "P", color="#aa5c5e", ha="center", va="center", fontsize=11,
                fontweight="bold", zorder=5)
        ax.text(Y_RIGHT - TY + .02, X_NEAR - TX + .03, "no bowl\npush the plate, open gripper",
                fontsize=8.5, color="#8a4a12", va="top",
                bbox=dict(fc="#fdf0e2", ec="#d6a877", lw=.7, pad=3))

    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=16)
    ax.text(Y_RIGHT - TY / 2, X_NEAR + .055, "YOU SIT ON THIS EDGE", ha="center",
            fontsize=9, fontweight="bold", color="#444")
    ax.text(Y_RIGHT - TY + .01, X_NEAR + .055, "your LEFT", ha="left", fontsize=8.5, color="#666")
    ax.text(Y_RIGHT - .01, X_NEAR + .055, "your RIGHT", ha="right", fontsize=8.5, color="#666")
    ax.text(Y_RIGHT - TY / 2, X_NEAR - TX - .045, "robot / agentview camera edge",
            ha="center", va="bottom", fontsize=8, color="#888")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(Y_RIGHT - TY - .04, Y_RIGHT + .04)
    ax.set_ylim(X_NEAR - TX - .10, X_NEAR + .09)
    ax.invert_yaxis()


def main():
    fig = plt.figure(figsize=(11.7, 13.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[.72, .72, 1.0], hspace=.10)
    fig.suptitle("Table setup sheet  -  70 x 120 cm table, 1:1 scale",
                 fontsize=17, fontweight="bold", x=.04, ha="left", y=.985)
    fig.text(.04, .962, "Distances from the NEAR edge (where you sit) and the LEFT edge. Smaller \"from LEFT\" = further left. "
             "Markers first, then the objects.", fontsize=10.5, color="#555")
    draw(fig.add_subplot(gs[0]), "A", "LAYOUT A   -   50 episodes, 10 at each of 5 bowl positions")
    draw(fig.add_subplot(gs[1]), "B", "LAYOUT B   -   10 push episodes (same markers, objects moved)")

    ax = fig.add_subplot(gs[2]); ax.axis("off")
    rows = [[n, f"{near(x)} mm", f"{left(y)} mm", f"{right(y)} mm"] for n, x, y in ROWS]
    t = ax.table(cellText=rows,
                 colLabels=["item", "from NEAR edge", "from LEFT edge", "(from RIGHT)"],
                 cellLoc="left", colWidths=[.46, .18, .18, .18], loc="upper center")
    t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor("#ececec")
        elif r <= 4:
            cell.set_facecolor("#eef3fa")
        elif r == 5:
            cell.set_facecolor("#fdeaea")
        elif r <= 10:
            cell.set_facecolor("#fbfbf8")
        else:
            cell.set_facecolor("#fdf3e8")
        if c:
            cell.set_text_props(ha="right", family="monospace")
    ax.text(.0, -.06, "SMALLER \"from LEFT\" = FURTHER LEFT.  The two lateral columns always add to 1200 mm -- use that as a check.\n\n"
            "Blue rows: the four ArUco markers. Tape flat, all four corners down.\n"
            "Pink row: the PLATE. It never moves, in any of the 50 episodes.\n"
            "Cream rows: the 5 chunks. Between chunks you move ONLY the bowl, to the position on that row.\n"
            "Orange rows: layout B, the 10 push episodes. Move the plate, add the small box, remove the bowl.\n\n"
            "Markers are INSET, not at the corners: 14.2 cm from near/far, 15.0 cm from left/right.\n"
            "Check the marker rectangle before taping anything else:  265 mm short side, 750 mm long side, "
            "795 mm diagonal.\nIf the diagonal is wrong the rectangle is not square.\n\n"
            "Mark object positions with a small pencil cross or a dot of masking tape -- NOT anything that "
            "covers a marker.\nMinimum clearance between any object and any marker tile is 57 mm; keep it.",
            transform=ax.transAxes, fontsize=10, va="top", color="#333")

    out = Path("docs/recording")
    fig.savefig(out / "setup_sheet.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "setup_sheet.png", dpi=110, bbox_inches="tight", facecolor="white")
    print("wrote docs/recording/setup_sheet.pdf and .png")


if __name__ == "__main__":
    main()
