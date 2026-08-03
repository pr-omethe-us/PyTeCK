"""A 16x16 sprite for the docs favicon.

The full marks cannot be resampled this small - pixel art has no detail below
its own grid - so the icon is drawn on a 16-cell grid of its own: check-snake
reduced to the two things that still read at this size, a tick and a flame.
Rendered at 16, 32, 48 and 64 px, every cell lands on a whole number of device
pixels, so all four are crisp.
"""

import pathlib

from lib import GOLD, GREEN, GREEN_HI, INK, Pix, write_svg

OUT = pathlib.Path(__file__).resolve().parent.parent
N = 16

p = Pix()

# ---- the tick, in PyKED green with a gold underside
CEN = [(3.0, 8.0), (5.8, 11.6), (12.2, 3.4)]


def body_color(c, r, u, t, side):
    if side > 0 and u > 0.5:
        return GOLD
    return GREEN_HI if u < 0.35 else GREEN


p.stroke(CEN, 2.8, 2.8, body_color, bbox_pad=4)
p.outline(INK)

# ---- flame on the tip
for (c, r), col in {
    (12, 0): "#ED1C22",
    (11, 1): "#ED1C22",
    (12, 1): "#F78F10",
    (13, 1): "#ED1C22",
    (10, 2): "#ED1C22",
    (11, 2): "#FDE702",
    (12, 2): "#FDE702",
    (13, 2): "#F3521A",
    (10, 3): "#F3521A",
    (11, 3): "#FDE702",
    (12, 3): "#F78F10",
    (13, 3): "#ED1C22",
    (11, 4): "#ED1C22",
    (12, 4): "#F3521A",
    (13, 4): "#ED1C22",
}.items():
    p.set(c, r, col)

# trim anything that wandered outside the 16-cell frame
for k in [k for k in p.g if not (0 <= k[0] < N and 0 <= k[1] < N)]:
    del p.g[k]

svg = p.svg(pad=0, px=1, title="PyTeCK favicon")
# the sprite is exactly 16x16 whatever the ink reached, so pin the viewBox
svg = svg.replace(
    svg.split(">", 1)[0] + ">",
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
    'width="%d" height="%d" shape-rendering="crispEdges">' % (N, N, N, N),
    1,
)
write_svg(OUT / "pyteck-favicon.svg", svg)
print("ok", len(p.g), "cells")
