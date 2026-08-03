# Logo

<img src="pyteck-model-data-square.png" align="right" width="160" />

This directory contains the logo files for PyTeCK. Two marks, plus an icon and a
social card, are provided:

- **Check-snake** (`pyteck-check-snake.*`) — the snake bent into a checkmark with a
  flame at the tail tip. The secondary mark, and the one to reach for where the frame is
  small or square: it holds down to about 48 px, well past where the plot mark gives up.
- **Model against data** (`pyteck-model-data.*`, `pyteck-model-data-square.*`) — the
  model prediction as a snake climbing through scattered experimental points, which is
  what PyTeCK actually computes. The square version (1 : 1) is the primary cut, for docs
  headers and anywhere square; the wide version (1.5 : 1) suits banners and social cards.
- **Favicon** (`pyteck-favicon.*`) — a 16-cell sprite, drawn separately. The marks above
  cannot be resampled below their own pixel grid: check-snake is fine at 48 px, marginal at
  32 and unreadable at 16. So the icon is check-snake reduced to the two things that still
  read that small, a tick and a flame. `pyteck-favicon.ico` carries 16/32/48 px.
- **Social card** (`pyteck-social-card.png`) — 1280 × 640, the size GitHub wants under
  Settings → General → Social preview. It has to be uploaded by hand; GitHub will not take
  an SVG.

Both marks continue the family started by the
[Prometheus/ChemKED](https://niemeyer-research-group.github.io/images/software/prometheus-logo.svg)
flame and the [PyKED](https://github.com/pr-omethe-us/PyKED) snake: the same 8-bit grid,
ink outline, and color palette, sampled directly from the PyKED logo file.

The snake's face in the model-against-data marks is not a redraw — it is lifted from
`pyked-logo.png` cell for cell (26 px pixel grid, origin 18,22) and stamped mirrored, so
its neck meets the top of the curve. All 244 cells match the source exactly, including
the two shades of red inside the mouth. That fidelity sets the proportions: PyKED's head
is about a third of its logo's width, so these marks carry the same big-headed,
thick-bodied build. Everything else — curve, flame, data markers, axes — is drawn from
scratch as pixel geometry, and no stock imagery is used.

## Light and dark grounds

The near-black ink outline that defines these marks disappears against a dark
background, so each mark ships in three cuts:

| Suffix | Outline | Use on |
| --- | --- | --- |
| *(none)* | ink `#221E1E` | light backgrounds |
| `-dark` | cream `#F4EFE4` | dark backgrounds |
| `-auto` | follows `prefers-color-scheme` | pages that theme themselves |

Only the outline that touches the background is recolored. Interior detail — the eye
slits, the nostrils, the open mouth — stays dark, because it sits on the snake's own green
and needs the contrast on either ground.

**`-auto` follows the reader's system setting, not the background it is placed on.** On a
page that is light in light mode and dark in dark mode, it is the file you want. On a
fixed background — a dark hero on an otherwise light site — use the explicit `-dark`
file, or a `<picture>` element, which is what GitHub honours in a README:

```html
<picture>
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/pr-omethe-us/PyTeCK/main/logo/pyteck-model-data-square-dark.png">
  <img src="https://raw.githubusercontent.com/pr-omethe-us/PyTeCK/main/logo/pyteck-model-data-square.png"
       width="220" alt="PyTeCK">
</picture>
```

That is the block in the project README. It uses absolute `raw.githubusercontent.com`
URLs rather than relative paths because `README.md` is also the PyPI long description,
and PyPI does not resolve relative image paths. PyPI drops the `<picture>` and `<source>`
wrapper it does not recognise and keeps the inner `<img>`, so it falls back to the light
cut — which is what you want there.

## Regenerating

The marks are generated rather than hand-drawn, so they can be re-cut at a different
size, weight, or color without redrawing. The scripts need only the standard library:

    > python src/check_snake.py
    > python src/model_data.py
    > python src/model_data_square.py
    > python src/favicon.py
    > python src/social_card.py

Each mark script writes all three cuts into this directory, whatever the working
directory. `src/lib.py` holds the pixel-grid engine (stroking, outlining, flame color
banding), the `PYKED_FACE` sprite with `stamp` to place it, the palette constants sampled
from PyKED, and `write_variants`, which derives the `-dark` and `-auto` cuts from the same
grid.

Then render the rasters — 20 pixels per grid cell for the marks:

    > rsvg-convert -w 940 pyteck-check-snake.svg -o pyteck-check-snake.png
    > rsvg-convert -w 940 pyteck-check-snake-dark.svg -o pyteck-check-snake-dark.png
    > rsvg-convert -w 1200 pyteck-model-data.svg -o pyteck-model-data.png
    > rsvg-convert -w 1200 pyteck-model-data-dark.svg -o pyteck-model-data-dark.png
    > rsvg-convert -w 1040 pyteck-model-data-square.svg -o pyteck-model-data-square.png
    > rsvg-convert -w 1040 pyteck-model-data-square-dark.svg -o pyteck-model-data-square-dark.png

    > rsvg-convert -w 1280 pyteck-social-card.svg -o pyteck-social-card.png

    > for s in 16 32 48; do rsvg-convert -w $s -h $s pyteck-favicon.svg -o pyteck-favicon-$s.png; done
    > python src/pack_ico.py

The `-auto` cut is SVG only; a PNG cannot carry the media query.

## Where these are used

| Place | File |
| --- | --- |
| `README.md` header | `pyteck-model-data-square{,-dark}.png`, by absolute raw.githubusercontent URL |
| Docs sidebar, every page | `docs/_static/pyteck-model-data-square.svg` (`html_logo`) |
| Docs browser tab | `docs/_static/pyteck-favicon.ico` (`html_favicon`) |
| GitHub social preview | `pyteck-social-card.png`, uploaded by hand |

The two files under `docs/_static/` are **copies**. Refresh them when the marks change:

    > cp pyteck-model-data-square.svg pyteck-favicon.ico ../docs/_static/

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">PyTeCK Logo</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Kyle Niemeyer</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>, matching the PyKED logo it descends from. The match is not
incidental: the face is PyKED's own pixels, so these marks are a direct derivative and
carry the same terms.
