"""Tiny engines for PyTeCK logo concepts: pixel-art (PyKED style) and low-poly (Prometheus style)."""

import math

# ---------------------------------------------------------------- palettes
# sampled from pyked-logo.png
INK = "#221E1E"
GREEN_HI = "#CDE532"
GREEN = "#A5CD1C"
GREEN_MD = "#81BD28"
GREEN_LO = "#5EA52A"
GREEN_DK = "#428F2A"
GOLD = "#FBC707"
GOLD_DK = "#D9910A"
CREAM = "#F4EFE4"  # ink stand-in for dark grounds


# ---------------------------------------------------------------- geometry
def chaikin(pts, n=2, closed=True):
    for _ in range(n):
        out = []
        m = len(pts)
        rng = range(m) if closed else range(m - 1)
        for i in rng:
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % m]
            out.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            out.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        if not closed:
            out = [pts[0]] + out + [pts[-1]]
        pts = out
    return pts


def inside(poly, p):
    x, y = p
    c = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            if x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
                c = not c
    return c


def seg_dist(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy), t


def polyline_dist(p, pts):
    """distance to polyline + normalized arclength position + signed side"""
    best = (1e9, 0.0, 0.0)
    segs = []
    total = 0.0
    for i in range(len(pts) - 1):
        L = math.dist(pts[i], pts[i + 1])
        segs.append((pts[i], pts[i + 1], total, L))
        total += L
    for a, b, s0, L in segs:
        d, t = seg_dist(p, a, b)
        if d < best[0]:
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            best = (d, (s0 + t * L) / total, math.copysign(1.0, cross or 1.0))
    return best


def flame_outline(licks=True):
    """normalized flame silhouette, x,y in 0..1, y up, tip at top"""
    p = [(0.50, 0.00), (0.72, 0.03), (0.86, 0.13), (0.92, 0.28), (0.89, 0.40)]
    if licks:
        p += [(1.00, 0.47), (0.96, 0.62), (0.86, 0.56)]
    p += [
        (0.80, 0.64),
        (0.72, 0.76),
        (0.63, 0.87),
        (0.545, 1.00),
        (0.47, 0.85),
        (0.415, 0.73),
        (0.35, 0.67),
    ]
    if licks:
        p += [(0.31, 0.78), (0.20, 0.72), (0.185, 0.57), (0.25, 0.49)]
    p += [(0.145, 0.41), (0.085, 0.28), (0.15, 0.13), (0.29, 0.035)]
    return p


# ---------------------------------------------------------------- pixel art
class Pix:
    def __init__(self):
        self.g = {}

    def set(self, c, r, color):
        self.g[(int(c), int(r))] = color

    def get(self, c, r):
        return self.g.get((int(c), int(r)))

    def stroke(self, pts, w0, w1, colorfn, bbox_pad=6):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        for r in range(int(min(ys)) - bbox_pad, int(max(ys)) + bbox_pad):
            for c in range(int(min(xs)) - bbox_pad, int(max(xs)) + bbox_pad):
                d, t, side = polyline_dist((c + 0.5, r + 0.5), pts)
                half = (w0 + (w1 - w0) * t) / 2.0
                if d <= half:
                    col = colorfn(c, r, d / max(half, 1e-6), t, side)
                    if col:
                        self.set(c, r, col)

    def outline(self, color, only=None, diag=True):
        add = {}
        offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diag:
            offs += [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        body = {k: v for k, v in self.g.items() if only is None or v in only}
        for c, r in body:
            for dc, dr in offs:
                k = (c + dc, r + dr)
                if k not in self.g:
                    add[k] = color
        self.g.update(add)

    def erosion_bands(self, cells, colors):
        """cells: set of (c,r). colors[0] = outermost band."""
        remaining = set(cells)
        depth = {}
        d = 0
        while remaining:
            edge = {
                k
                for k in remaining
                if any(
                    (k[0] + dc, k[1] + dr) not in remaining
                    for dc, dr in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
            }
            if not edge:
                edge = set(remaining)
            for k in edge:
                depth[k] = d
            remaining -= edge
            d += 1
        maxd = max(depth.values()) if depth else 0
        for k, dd in depth.items():
            idx = min(len(colors) - 1, int(round(dd / max(maxd, 1) * (len(colors) - 1))))
            self.set(k[0], k[1], colors[idx])
        return depth

    def poly_cells(self, poly):
        cells = set()
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        for r in range(int(min(ys)) - 1, int(max(ys)) + 2):
            for c in range(int(min(xs)) - 1, int(max(xs)) + 2):
                if inside(poly, (c + 0.5, r + 0.5)):
                    cells.add((c, r))
        return cells

    def copy(self):
        q = Pix()
        q.g = dict(self.g)
        return q

    def boundary_cells(self, color):
        """cells of `color` that touch the transparent background"""
        offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        return {
            k
            for k, v in self.g.items()
            if v == color and any((k[0] + dc, k[1] + dr) not in self.g for dc, dr in offs)
        }

    def svg(self, pad=1, px=1, title="", adaptive=None):
        """adaptive: {grid_color: (light_hex, dark_hex)} - follows prefers-color-scheme"""
        ks = list(self.g)
        c0 = min(k[0] for k in ks) - pad
        c1 = max(k[0] for k in ks) + pad + 1
        r0 = min(k[1] for k in ks) - pad
        r1 = max(k[1] for k in ks) + pad + 1
        w, h = (c1 - c0) * px, (r1 - r0) * px
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" shape-rendering="crispEdges">'
        ]
        if title:
            parts.append(f"<title>{title}</title>")
        cls = {}
        if adaptive:
            rules = []
            dark = []
            for i, (key, (light_hex, dark_hex)) in enumerate(adaptive.items()):
                cls[key] = f"k{i}"
                rules.append(f".k{i}{{fill:{light_hex}}}")
                dark.append(f".k{i}{{fill:{dark_hex}}}")
            parts.append(
                "<style>"
                + "".join(rules)
                + "@media(prefers-color-scheme:dark){"
                + "".join(dark)
                + "}</style>"
            )
        # merge runs per row for smaller output
        for r in range(r0, r1):
            c = c0
            while c < c1:
                col = self.get(c, r)
                if col is None:
                    c += 1
                    continue
                n = 1
                while self.get(c + n, r) == col:
                    n += 1
                paint = f'class="{cls[col]}"' if col in cls else f'fill="{col}"'
                parts.append(
                    f'<rect x="{(c - c0) * px}" y="{(r - r0) * px}" width="{n * px}" '
                    f'height="{px}" {paint}/>'
                )
                c += n
        parts.append("</svg>")
        return "\n".join(parts)


def write_svg(path, text):
    """write text ending in exactly one newline, matching end-of-file-fixer"""
    path.write_text(text.rstrip("\n") + "\n")


def write_variants(pix, out_dir, stem, px=10, title="", pad=1):
    """Write three cuts of a mark: the light-ground original, one for dark
    grounds, and one that follows prefers-color-scheme.

    Only the ink that touches the background is recoloured. Interior detail --
    the eye, the mouth line, the nostrils -- stays dark, because it sits on the
    snake's own green and needs the contrast either way.
    """
    KEY = "#221E1F"  # sentinel, never rendered
    edge = pix.boundary_cells(INK)

    write_svg(out_dir / (stem + ".svg"), pix.svg(pad=pad, px=px, title=title))

    dark = pix.copy()
    for k in edge:
        dark.g[k] = CREAM
    write_svg(
        out_dir / (stem + "-dark.svg"), dark.svg(pad=pad, px=px, title=title + " (dark grounds)")
    )

    auto = pix.copy()
    for k in edge:
        auto.g[k] = KEY
    write_svg(
        out_dir / (stem + "-auto.svg"),
        auto.svg(pad=pad, px=px, title=title + " (auto)", adaptive={KEY: (INK, CREAM)}),
    )


# ---------------------------------------------------------------- snake head
# The face is lifted cell-for-cell off pyked-logo.png (26 px pixel grid,
# origin 18,22) rather than redrawn: eye slits, nostrils, the open mouth with
# its two white fangs, and the forked tongue are PyKED's exact pixels. The
# right-hand columns are the start of PyKED's neck, kept so the body has
# something to join onto.
PYKED_FACE = [
    "     KKKKKKKKKKKK      ",
    "    KhhhhhhhhhhhhK     ",
    "   KhggKggggKgggghK    ",
    "  KhgggKggggKggggghK   ",
    " KhggghhKgKghhggggghK  ",
    " KggKgggggggggggKggghK ",
    " KgglKKKKKKKKKKKlggggK ",
    " KgggKWxKRRKxWxKggggghK",
    " KlmglKKorKKKKKlgmllggK",
    "  KlmmKorKDllllmmlKlglK",
    "   KlKoKrKDmmmmmlKlgggK",
    "    KoKDKrKDllllKYKgggK",
    "     KKKKKKKKKKKYyKglgK",
]

FACE_KEYS = {
    "K": INK,
    "h": GREEN_HI,
    "g": GREEN,
    "m": GREEN_MD,
    "l": GREEN_LO,
    "D": "#3A7A26",
    "y": GOLD,
    "Y": GOLD_DK,
    "r": "#ED1C22",
    "R": "#AF1418",
    "o": "#F1461C",
    "x": "#680A0C",
    "W": "#FFFFFF",
}


def stamp(pix, pattern, c0, r0, mirror=False, keys=None):
    """paint a character-grid sprite at (c0, r0); spaces stay transparent"""
    keys = keys or FACE_KEYS
    for dr, line in enumerate(pattern):
        row = line[::-1] if mirror else line
        for dc, k in enumerate(row):
            if k != " ":
                pix.set(c0 + dc, r0 + dr, keys[k])
