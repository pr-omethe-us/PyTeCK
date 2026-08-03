"""Pack the rendered favicon PNGs into a single multi-size .ico.

Run after rendering pyteck-favicon-{16,32,48}.png from pyteck-favicon.svg.
ICO entries hold whole PNG files, which every browser in use understands.
"""

import pathlib
import struct

OUT = pathlib.Path(__file__).resolve().parent.parent
SIZES = (16, 32, 48)

imgs = []
for s in SIZES:
    f = OUT / f"pyteck-favicon-{s}.png"
    if not f.exists():
        raise SystemExit(f"missing {f.name} - render it from pyteck-favicon.svg first")
    imgs.append((s, f.read_bytes()))

head = struct.pack("<HHH", 0, 1, len(imgs))
offset = len(head) + 16 * len(imgs)
entries, blobs = b"", b""
for s, data in imgs:
    entries += struct.pack("<BBBBHHII", s, s, 0, 0, 1, 32, len(data), offset)
    blobs += data
    offset += len(data)

ico = OUT / "pyteck-favicon.ico"
ico.write_bytes(head + entries + blobs)
print(f"ok - {ico.name}, {len(imgs)} sizes, {ico.stat().st_size} bytes")
