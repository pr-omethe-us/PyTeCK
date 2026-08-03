"""Model-against-data, square cut.

The snake is the model prediction climbing through experimental points that
don't quite sit on it. Proportions follow PyKED, because the face is PyKED's
own pixels: a head roughly a third of the mark's width, on a chunky body.
"""

import math
import pathlib
import random

from lib import (
    GOLD,
    GOLD_DK,
    GREEN,
    GREEN_DK,
    GREEN_HI,
    GREEN_LO,
    GREEN_MD,
    INK,
    PYKED_FACE,
    Pix,
    chaikin,
    flame_outline,
    stamp,
    write_variants,
)

OUT = pathlib.Path(__file__).resolve().parent.parent

rnd = random.Random(17)
p = Pix()

# ---- axes
AX_C, AX_R = 4, 51
for r in range(2, AX_R + 1):
    p.set(AX_C, r, INK)
for c in range(AX_C, 47):
    p.set(c, AX_R, INK)

# ---- the model curve
CEN = chaikin(
    [(12.0, 47.0), (18.0, 44.0), (24.0, 37.0), (29.0, 28.0), (33.0, 19.0), (36.5, 13.0)],
    2,
    closed=False,
)


def body_color(c, r, u, t, side):
    h = (c * 7919 + r * 104729) % 100
    if side > 0 and u > 0.48:
        return GOLD_DK if u > 0.86 and rnd.random() < 0.4 else GOLD
    if side < 0 and u > 0.62:
        return GREEN_DK if h < 55 else GREEN_LO
    if h < 12:
        return GREEN_HI
    if h < 22:
        return GREEN_MD
    return GREEN


p.stroke(CEN, 4.0, 7.6, body_color, bbox_pad=8)

# ---- PyKED's face, mirrored so its neck meets the top of the curve
stamp(p, PYKED_FACE, 32, 2, mirror=True)

p.outline(INK, only={GREEN, GREEN_HI, GREEN_MD, GREEN_LO, GREEN_DK, GOLD, GOLD_DK})


# ---- experimental data points scattered either side of the curve
def at(u):
    i = max(1, min(len(CEN) - 2, int(u * (len(CEN) - 1))))
    (x0, y0), (x1, y1) = CEN[i - 1], CEN[i + 1]
    L = math.hypot(x1 - x0, y1 - y0) or 1.0
    return CEN[i], (-(y1 - y0) / L, (x1 - x0) / L)


for u, off in [(0.30, 6.4), (0.40, -6.8), (0.52, 7.0), (0.64, -6.8), (0.78, 6.6), (0.86, -6.4)]:
    (x, y), (nx, ny) = at(u)
    dc, dr = round(x + nx * off), round(y + ny * off)
    for c in range(dc - 2, dc + 3):  # ink ring
        for r in range(dr - 2, dr + 3):
            if abs(c - dc) == 2 or abs(r - dr) == 2:
                p.set(c, r, INK)
    for c in range(dc - 1, dc + 2):
        for r in range(dr - 1, dr + 2):
            p.set(c, r, "#F3521A")
    p.set(dc, dr, "#FBC707")

# ---- flame at the tail, bottom left
fw, fh = 12.0, 18.0
flame = [(3.5 + x * fw, 49.5 - y * fh) for x, y in flame_outline()]
p.erosion_bands(p.poly_cells(flame), ["#ED1C22", "#F3521A", "#F78F10", "#FBC707", "#FDE702"])

write_variants(p, OUT, "pyteck-model-data-square", title="PyTeCK model vs data (square)")
print("ok")
