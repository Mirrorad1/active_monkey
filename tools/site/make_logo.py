#!/usr/bin/env python3
"""Turn the white-background monkey art into a transparent PNG + favicons.

Removes ONLY the outer white background (flood-fill from the borders), so the
white eye-highlights and teeth inside the figure are preserved. Then emits a
square, padded logo plus the favicon / apple-touch sizes the page expects.

Usage:
    python3 tools/site/make_logo.py [path/to/raw_monkey.png]

Default input: site/assets/monkey-raw.png  ->  writes site/assets/monkey.png (+ sizes).
"""
import sys
from collections import deque
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else "site/assets/monkey-raw.png"
WHITE = 236      # pixels with R,G,B all >= this are "background white"
PAD = 0.06       # transparent padding around the trimmed figure (fraction)

img = Image.open(SRC).convert("RGBA")
w, h = img.size
px = img.load()

def is_white(p):
    return p[0] >= WHITE and p[1] >= WHITE and p[2] >= WHITE

# flood-fill the connected outer-white region from every border pixel
seen = [[False] * w for _ in range(h)]
q = deque()
for x in range(w):
    for y in (0, h - 1):
        if is_white(px[x, y]):
            q.append((x, y)); seen[y][x] = True
for y in range(h):
    for x in (0, w - 1):
        if is_white(px[x, y]) and not seen[y][x]:
            q.append((x, y)); seen[y][x] = True

while q:
    x, y = q.popleft()
    r, g, b, _ = px[x, y]
    px[x, y] = (r, g, b, 0)                      # make it transparent
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and is_white(px[nx, ny]):
            seen[ny][nx] = True; q.append((nx, ny))

# trim to the visible figure, then pad to a centered square
bbox = img.getbbox()
if bbox:
    img = img.crop(bbox)
side = int(max(img.size) * (1 + 2 * PAD))
canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
canvas.paste(img, ((side - img.size[0]) // 2, (side - img.size[1]) // 2), img)
img = canvas

img.save("site/assets/monkey.png")
for size in (32, 180, 512):
    img.resize((size, size), Image.LANCZOS).save(f"site/assets/monkey-{size}.png")
img.resize((48, 48), Image.LANCZOS).save("favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
print(f"ok — wrote site/assets/monkey.png ({img.size[0]}px) + favicons from {SRC}")
