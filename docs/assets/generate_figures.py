"""Regenerate README diagrams as SVG and PNG: python docs/assets/generate_figures.py."""

import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
INK, MUTED, LINE = "#172B4D", "#61718A", "#DCE4EE"
BLUE, ORANGE, TEAL = "#3974D9", "#D8872E", "#199786"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class Canvas:
    def __init__(self, height):
        self.height = height
        self.image = Image.new("RGB", (2400, height * 2), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
            '<rect width="1200" height="100%" fill="white"/>',
        ]

    def text(self, x, y, text, size=18, color=INK, bold=False, center=False):
        font = ImageFont.truetype(BOLD if bold else FONT, size * 2)
        self.draw.text((x * 2, y * 2), text, font=font, fill=color, anchor="mt" if center else "lt")
        self.svg.append(
            f'<text x="{x}" y="{y + size * 0.82}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{"middle" if center else "start"}" fill="{color}">{html.escape(text)}</text>'
        )

    def box(self, x, y, w, h, fill="white", stroke=LINE, radius=16):
        self.draw.rounded_rectangle(
            (x * 2, y * 2, (x + w) * 2, (y + h) * 2), radius=radius * 2, fill=fill, outline=stroke, width=2
        )
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'
        )

    def circle(self, x, y, r, fill="white", stroke=LINE, width=1):
        self.draw.ellipse(
            ((x - r) * 2, (y - r) * 2, (x + r) * 2, (y + r) * 2), fill=fill, outline=stroke, width=width * 2
        )
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def line(self, points, color=LINE, width=2, arrow=False):
        self.draw.line([(x * 2, y * 2) for x, y in points], fill=color, width=width * 2)
        self.svg.append(
            f'<polyline points="{" ".join(f"{x},{y}" for x, y in points)}" fill="none" stroke="{color}" stroke-width="{width}"/>'
        )
        if arrow:
            x, y = points[-1]
            px, py = points[-2]
            a = math.atan2(y - py, x - px)
            ends = [
                (x - 9 * math.cos(a - 0.5), y - 9 * math.sin(a - 0.5)),
                (x, y),
                (x - 9 * math.cos(a + 0.5), y - 9 * math.sin(a + 0.5)),
            ]
            self.line(ends, color, width)

    def save(self, name):
        (ROOT / f"{name}.svg").write_text("\n".join(self.svg + ["</svg>"]) + "\n")
        self.image.save(ROOT / f"{name}.png", optimize=True)


def overview():
    c = Canvas(570)
    c.text(40, 26, "Give each task its own region.", 32, bold=True)
    c.text(40, 74, "Global task separation + local class discrimination + feature stability", 19, color=MUTED)
    c.box(30, 122, 520, 380, fill="#F7F9FC")
    c.box(650, 122, 520, 380, fill="#F7F9FC")
    c.text(290, 145, "Conventional contrastive learning", 21, bold=True, center=True)
    c.text(910, 145, "GPLASC", 21, bold=True, center=True)
    c.circle(290, 327, 128, fill="white", stroke="#BBCBDE", width=2)
    # Overlapping task distributions, intentionally schematic.
    for k, color in enumerate((BLUE, ORANGE, TEAL)):
        for i in range(12):
            a = i * 2.399 + k * 0.7
            r = 20 + (i % 4) * 22
            x = 290 + math.cos(a) * r + (k - 1) * 17
            y = 327 + math.sin(a) * r
            c.circle(x, y, 5, fill=color, stroke=color)
    c.text(290, 473, "Task distributions can overlap", 16, color=MUTED, center=True)
    c.circle(910, 327, 128, fill="white", stroke="#BBCBDE", width=2)
    centers = []
    for k, (color, tint, angle) in enumerate(((BLUE, "#EBF1FD", -90), (ORANGE, "#FCF1E5", 150), (TEAL, "#E6F5F1", 30))):
        a = math.radians(angle)
        x = 910 + math.cos(a) * 79
        y = 327 + math.sin(a) * 79
        centers.append((x, y))
        c.circle(x, y, 46, fill=tint, stroke=color)
        # Three class groups inside each allocated task region.
        for j in range(3):
            b = j * 2 * math.pi / 3 - 0.5
            gx = x + math.cos(b) * 24
            gy = y + math.sin(b) * 24
            for dx, dy in ((-4, -3), (4, -3), (0, 5)):
                c.circle(gx + dx, gy + dy, 3, fill=color, stroke=color)
        c.circle(x, y, 6, fill="white", stroke=color, width=2)
    c.text(910, 473, "Fixed task regions, adaptable class structure", 16, color=MUTED, center=True)
    c.line([(568, 318), (630, 318)], color=MUTED, arrow=True)
    c.text(600, 345, "allocate", 13, color=MUTED, center=True)
    for i, (color, label) in enumerate(((BLUE, "Task 1"), (ORANGE, "Task 2"), (TEAL, "Task 3"))):
        x = 40 + i * 128
        c.circle(x + 6, 534, 5, fill=color, stroke=color)
        c.text(x + 20, 525, label, 15, color=MUTED)
    c.text(700, 525, "Conceptual 2D view; not a learned embedding", 14, color=MUTED)
    c.save("method-overview")


def pipeline():
    c = Canvas(600)
    c.text(40, 26, "One encoder. Two training stages.", 30, bold=True)
    c.text(
        40, 72, "Geometry is learned before projection; contrastive relations are learned after it.", 18, color=MUTED
    )
    c.box(30, 120, 1140, 300, fill="#F7F9FC")
    c.text(50, 138, "01  REPRESENTATION LEARNING", 15, color=BLUE, bold=True)
    for x, y, w, h, title, sub in (
        (50, 210, 180, 88, "Current + replay", "Two augmented views"),
        (280, 210, 170, 88, "ResNet encoder", "512 / 2048 dimensions"),
        (500, 210, 170, 88, "Encoder features", "Shared representation"),
    ):
        c.box(x, y, w, h)
        c.text(x + w / 2, y + 18, title, 17, bold=True, center=True)
        c.text(x + w / 2, y + 50, sub, 13, color=MUTED, center=True)
    c.line([(230, 254), (280, 254)], color=MUTED, arrow=True)
    c.line([(450, 254), (500, 254)], color=MUTED, arrow=True)
    c.box(760, 174, 380, 84, fill="#E6F5F1", stroke="#B6DDD3")
    c.text(780, 191, "Region + feature losses", 18, color=TEAL, bold=True)
    c.text(780, 223, "Range hinge · prototype MSE · old-feature drift", 14, color=MUTED)
    c.line([(670, 238), (715, 238), (715, 216), (760, 216)], color=TEAL, arrow=True)
    c.box(500, 326, 170, 66, fill="#EBF1FD", stroke="#C8D9F4")
    c.text(585, 338, "Projection MLP", 16, color=BLUE, bold=True, center=True)
    c.text(585, 364, "128 dimensions", 13, color=MUTED, center=True)
    c.line([(585, 298), (585, 326)], color=BLUE, arrow=True)
    c.box(760, 316, 380, 76, fill="#EBF1FD", stroke="#C8D9F4")
    c.text(780, 332, "Contrastive + relation losses", 18, color=BLUE, bold=True)
    c.text(780, 361, "Asymmetric SupCon · instance relations", 14, color=MUTED)
    c.line([(670, 359), (760, 359)], color=BLUE, arrow=True)
    c.text(50, 341, "Previous-task model stays frozen.", 14, color=MUTED)
    c.text(50, 367, "Replay preserves old knowledge.", 14, color=MUTED)
    c.box(30, 444, 1140, 126, fill="#FCF7EF")
    c.text(50, 462, "02  LINEAR EVALUATION", 15, color=ORANGE, bold=True)
    for x, w, title in (
        (50, 250, "Frozen encoder"),
        (370, 250, "Train linear classifier"),
        (760, 380, "Class-IL / Task-IL accuracy"),
    ):
        c.box(x, 499, w, 48, stroke="#E8D9C1")
        c.text(x + w / 2, 514, title, 16, bold=True, center=True)
    c.line([(300, 523), (370, 523)], color=ORANGE, arrow=True)
    c.line([(620, 523), (760, 523)], color=ORANGE, arrow=True)
    c.save("training-pipeline")


if __name__ == "__main__":
    overview()
    pipeline()
