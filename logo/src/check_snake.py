"""Concept 1 - pixel 'check' snake with flame tip."""

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
    Pix,
    chaikin,
    flame_outline,
    write_variants,
)

OUT = pathlib.Path(__file__).resolve().parent.parent

rnd = random.Random(11)
p = Pix()

CEN = [(9.0, 11.5), (14.5, 22.0), (30.0, 3.0)]


def body_color(c, r, u, t, side):
    if side > 0 and u > 0.44:
        return GOLD_DK if u > 0.84 and rnd.random() < 0.45 else GOLD
    h = (c * 7919 + r * 104729) % 100
    if side < 0 and u > 0.58:
        return GREEN_DK if h < 55 else GREEN_LO
    if h < 12:
        return GREEN_HI
    if h < 22:
        return GREEN_MD
    return GREEN


p.stroke(CEN, 6.8, 3.6, body_color)


def rot(pts, cx, cy, deg):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return [
        (cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca) for x, y in pts
    ]


# ---- head: broad slab facing left, sitting on top of the short arm
hx, hy = 5.0, 7.0
raw = [
    (hx - 6.6, hy - 3.6),
    (hx - 5.4, hy - 4.8),
    (hx + 3.6, hy - 4.8),
    (hx + 5.2, hy - 3.2),
    (hx + 5.2, hy + 3.4),
    (hx + 3.6, hy + 4.8),
    (hx - 5.6, hy + 4.8),
    (hx - 7.4, hy + 3.0),
    (hx - 7.4, hy - 1.6),
]
head = chaikin(rot(raw, hx, hy, 11.0), 2)
for c, r in p.poly_cells(head):
    h = (c * 7919 + r * 104729) % 100
    d = (r - hy) + 0.2 * (c - hx)
    col = GREEN
    if d > 2.4:
        col = GREEN_LO if h < 60 else GREEN_DK
    elif h < 14:
        col = GREEN_HI
    elif h < 26:
        col = GREEN_MD
    p.set(c, r, col)

# brow, eye, mouth, tongue
for c in range(-6, -2):  # brow
    p.set(hx + c, hy - 3, INK)
for c in (-6, -5, -4, -3):  # eye ring
    for r in (-2, -1, 0):
        p.set(hx + c, hy + r, INK)
for c in (-5, -4):
    for r in (-2, -1):
        p.set(hx + c, hy + r, "#FFFFFF")
p.set(hx - 5, hy - 2, "#680A0C")
p.set(hx - 5, hy - 1, "#680A0C")
p.set(hx - 1, hy - 3, INK)
p.set(hx, hy - 3, INK)  # nostril dots
for c in range(-7, 4):  # mouth line
    p.set(hx + c, hy + 2 + (1 if c > 1 else 0), INK)
for c, r in [(-8, 3), (-9, 3), (-10, 3), (-11, 2), (-11, 4), (-12, 1), (-12, 5)]:
    p.set(hx + c, hy + r, "#ED1C22")

p.outline(INK)

# ---- flame rising from the tip of the long arm
fw, fh = 13.5, 21.0
fx, fy = 23.6, -19.0
flame = [(fx + x * fw, fy + (1 - y) * fh) for x, y in chaikin(flame_outline(), 1)]
p.erosion_bands(
    p.poly_cells(flame), ["#ED1C22", "#F3521A", "#F78F10", "#FBC707", "#FDE702", "#FFFAA0"]
)

write_variants(p, OUT, "pyteck-check-snake", title="PyTeCK check-snake")
print("ok")
