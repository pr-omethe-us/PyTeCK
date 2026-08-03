"""The 1280x640 image GitHub wants for Settings -> Social preview.

Writes an SVG that embeds the wide cut alongside the wordmark; render it to PNG
with rsvg-convert (GitHub will not take an SVG).
"""

import pathlib
import re

from lib import write_svg

OUT = pathlib.Path(__file__).resolve().parent.parent
W, H = 1280, 640

PAPER, INK_T, MUTED, FLAME = "#FBFAF6", "#16120F", "#5A5147", "#D94A07"
SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"
SERIF = "Iowan Old Style, Charter, Palatino, Georgia, serif"
MONO = "SF Mono, Menlo, Consolas, monospace"

mark = OUT.joinpath("pyteck-model-data.svg").read_text()
vb = re.search(r'viewBox="0 0 (\d+) (\d+)"', mark)
mw, mh = int(vb.group(1)), int(vb.group(2))
body = re.sub(r"<svg[^>]*>", "", mark, count=1).replace("</svg>", "")
body = re.sub(r"<title>.*?</title>", "", body, flags=re.S)

MH = 384  # mark height on the card
MW = MH * mw / mh
mx, my = 74, (H - MH) / 2
tx = 668

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<title>PyTeCK</title>
<rect width="{W}" height="{H}" fill="{PAPER}"/>
<g transform="translate({mx} {my:.0f}) scale({MW / mw:.5f})" shape-rendering="crispEdges">{body}</g>
<text x="{tx}" y="312" font-family="{SANS}" font-size="96" font-weight="700"
      letter-spacing="-3" fill="{INK_T}">PyTeCK</text>
<rect x="{tx}" y="342" width="96" height="6" fill="{FLAME}"/>
<text x="{tx}" y="404" font-family="{SERIF}" font-size="26" fill="{MUTED}">Automated testing of chemical kinetic models</text>
<text x="{tx}" y="450" font-family="{MONO}" font-size="19" fill="{MUTED}">github.com/pr-omethe-us/PyTeCK</text>
</svg>'''

write_svg(OUT / "pyteck-social-card.svg", svg)
print("ok - render with:  rsvg-convert -w 1280 pyteck-social-card.svg -o pyteck-social-card.png")
