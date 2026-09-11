#!/usr/bin/env python3
"""Build sample.jpg: distinct objects on plain background for SAM layer-split demo."""
from PIL import Image, ImageDraw
import math
import os

W, H = 960, 600
img = Image.new("RGB", (W, H), (243, 240, 233))  # warm cream background
d = ImageDraw.Draw(img)

# soft floor band
d.rectangle([0, H - 110, W, H], fill=(228, 224, 214))

# 1) emerald circle (sun)
d.ellipse([670, 70, 870, 270], fill=(16, 163, 110), outline=(11, 120, 82), width=6)

# 2) amber star
cx, cy, R, r = 230, 190, 130, 52
pts = []
for k in range(10):
    ang = -math.pi / 2 + k * math.pi / 5
    rad = R if k % 2 == 0 else r
    pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
d.polygon(pts, fill=(240, 168, 44), outline=(178, 120, 20), width=5)

# 3) terracotta triangle (mountain)
d.polygon([(390, 520), (560, 250), (730, 520)], fill=(198, 106, 78), outline=(148, 74, 52), width=5)

# 4) dark teal rounded rect (door/panel)
d.rounded_rectangle([90, 360, 330, 520], radius=22, fill=(28, 101, 116), outline=(16, 66, 78), width=5)

# 5) small sky-blue bird (two ellipses + beak)
d.ellipse([430, 120, 470, 150], fill=(78, 152, 214))
d.ellipse([455, 115, 500, 142], fill=(78, 152, 214))
d.polygon([(448, 128), (462, 118), (466, 132)], fill=(240, 168, 44))

out = os.path.join(os.path.dirname(__file__), "..", "sample.jpg")
img.save(out, quality=90)
print("saved", os.path.abspath(out), img.size)
