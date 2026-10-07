"""Synthetic SwitchChallenge benchmark: known-answer permutation puzzles rendered across
skins (plain / funnel / tiles), shape sets, themes, sizes, 1-3 choice rows and the
"shapes-under-numbers" variant, run through a2_tubes.analyse().

    python tools/switch_bench.py            # ~300 cases, accuracy by category
    python tools/switch_bench.py --n 600 -v

Complements tools/evaluate.py (which scores the real harvested site). This one stresses the
structure detection across other skins the challenge appears in (P&G funnels, tiled shapes,
annotations under boxes). A case is correct when analyse() reports a box combination that
composes to the rendered output ordering.
"""
import argparse
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import a2_tubes  # noqa: E402

_FONT_CANDS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
FONT = next((p for p in _FONT_CANDS if os.path.exists(p)), None)

SHAPES = {  # well-separated hues
    "square": (40, 40, 220), "triangle": (40, 220, 240), "circle": (70, 190, 70),
    "star": (210, 200, 60), "cross": (230, 120, 40), "hexagon": (200, 60, 150),
    "diamond": (190, 50, 200),
}
SHAPE_NAMES = list(SHAPES)


def _shape(img, name, cx, cy, r, color):
    if name == "circle":
        cv2.circle(img, (cx, cy), r, color, -1)
    elif name == "square":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, -1)
    elif name == "triangle":
        cv2.fillPoly(img, [np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)], color)
    elif name == "diamond":
        cv2.fillPoly(img, [np.array([[cx, cy - r], [cx - r, cy], [cx, cy + r], [cx + r, cy]], np.int32)], color)
    elif name == "cross":
        t = r // 2
        cv2.rectangle(img, (cx - t, cy - r), (cx + t, cy + r), color, -1)
        cv2.rectangle(img, (cx - r, cy - t), (cx + r, cy + t), color, -1)
    elif name == "star":
        pts = [[int(cx + (r if k % 2 == 0 else r * 0.45) * np.cos(-np.pi / 2 + k * np.pi / 5)),
                int(cy + (r if k % 2 == 0 else r * 0.45) * np.sin(-np.pi / 2 + k * np.pi / 5))] for k in range(10)]
        cv2.fillPoly(img, [np.array(pts, np.int32)], color)
    elif name == "hexagon":
        pts = [[int(cx + r * np.cos(np.pi / 6 + k * np.pi / 3)), int(cy + r * np.sin(np.pi / 6 + k * np.pi / 3))]
               for k in range(6)]
        cv2.fillPoly(img, [np.array(pts, np.int32)], color)


def _digits(img, box, cx, cy, px, color_bgr):
    pil = Image.fromarray(img[:, :, ::-1].copy())
    d = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT, px)
    pitch = int(px * 0.95)
    x = cx - pitch * len(box) / 2 + pitch / 2
    for ch in (str(v) for v in box):
        bb = d.textbbox((0, 0), ch, font=font)
        d.text((x - (bb[2] - bb[0]) / 2 - bb[0], cy - (bb[3] - bb[1]) / 2 - bb[1]), ch, font=font, fill=color_bgr[::-1])
        x += pitch
    img[:, :, :] = np.array(pil)[:, :, ::-1]


def _compose(boxes):
    idx = [1, 2, 3, 4]
    for b in boxes:
        idx = [idx[j - 1] for j in b]
    return tuple(idx)


def _layout(rng):
    n = rng.choice([1, 2, 3])
    choice_rows = set(rng.sample(range(n), rng.randint(1, n)))
    rows, correct = [], []
    for ri in range(n):
        if ri in choice_rows:
            uniq = []
            while len(uniq) < rng.choice([2, 3]):
                p = tuple(rng.sample(range(1, 5), 4))
                if p not in uniq:
                    uniq.append(p)
            rows.append([list(p) for p in uniq])
            correct.append(rng.randrange(len(uniq)))
        else:
            rows.append([rng.sample(range(1, 5), 4)])
            correct.append(0)
    return rows, _compose([rows[ri][correct[ri]] for ri in range(n)])


def render(style, rng):
    shapes = rng.sample(SHAPE_NAMES, 4)
    rows, comp = _layout(rng)
    bottom = [comp[i] - 1 for i in range(4)]
    dark = style["theme"] == "dark"
    bg = (28, 26, 24) if dark else (255, 255, 255)
    fg = (240, 240, 240) if dark else (30, 30, 30)
    tile = (60, 55, 50) if dark else (244, 245, 247)
    border = (90, 85, 80) if dark else (205, 208, 212)
    S = style["size"]
    W = 200 + 3 * int(S * 3.8)
    H = int(S * 2.4) * (len(rows) + 2) + 140
    img = np.full((H, W, 3), bg, np.uint8)
    r = int(S * 0.46)

    def shape_row(y, order):
        for c in range(4):
            cx = 100 + int((W - 200) * (c + 0.5) / 4)
            if style["skin"] == "tiles":
                cv2.rectangle(img, (cx - r - 8, y - r - 8), (cx + r + 8, y + r + 8), tile, -1)
            _shape(img, shapes[order[c]], cx, y, r, SHAPES[shapes[order[c]]])

    def funnel(y):
        if style["skin"] == "funnel":
            cv2.fillPoly(img, [np.array([[W // 2 - r, y - r // 2], [W // 2 + r, y - r // 2], [W // 2, y + r // 2]],
                                        np.int32)], (160, 120, 40))

    y = 80
    shape_row(y, list(range(4)))
    y += int(S * 1.6)
    funnel(y - int(S * 0.3))
    bw, bh = int(S * 3.0), int(S * 1.05)
    for row in rows:
        y += int(S * 1.4)
        span, k = W - 200, len(row)
        for bi, box in enumerate(row):
            cx = 100 + int(span * (bi + 0.5) / k)
            cv2.rectangle(img, (cx - bw // 2, y - bh // 2), (cx + bw // 2, y + bh // 2), tile, -1)
            cv2.rectangle(img, (cx - bw // 2, y - bh // 2), (cx + bw // 2, y + bh // 2), border, max(1, S // 40))
            _digits(img, box, cx, y, int(S * 0.74), fg)
            if style["under"]:
                for c in range(4):
                    nm = shapes[box[c] - 1]
                    _shape(img, nm, cx - bw // 2 + int(bw * (c + 0.5) / 4), y + bh // 2 + int(S * 0.26),
                           int(S * 0.13), SHAPES[nm])
    y += int(S * 1.5) + (int(S * 0.4) if style["under"] else 0)
    funnel(y - int(S * 0.2))
    y += int(S * 1.1)
    shape_row(y, bottom)
    return img, comp


def _picked_comp(res):
    try:
        return _compose([res["rows"][ri][res["choice"][ri]] for ri in range(len(res["choice"]))])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    if not FONT:
        sys.exit("no bold TTF font found for rendering")
    rng = random.Random(args.seed)
    stats = defaultdict(lambda: [0, 0])
    fails = []
    for _ in range(args.n):
        style = {"skin": rng.choice(["plain", "funnel", "tiles"]), "theme": rng.choice(["light", "dark"]),
                 "size": rng.choice([40, 60]), "under": rng.random() < 0.3}
        img, comp = render(style, rng)
        try:
            res = a2_tubes.analyse(img)
        except Exception:
            res = None
        ok = bool(res) and res.get("big", "?") != "?" and res.get("target") == comp and _picked_comp(res) == comp
        for dim, key in [("skin", style["skin"]), ("theme", style["theme"]), ("size", str(style["size"])),
                         ("under", str(style["under"])), ("ALL", "all")]:
            stats[(dim, key)][0] += ok
            stats[(dim, key)][1] += 1
        if not ok and args.v:
            fails.append((style, comp, res.get("target") if res else None))

    def line(k):
        c, t = stats[k]
        return f"{c / t * 100:5.1f}%  ({c}/{t})" if t else "n/a"

    print("\nOVERALL:", line(("ALL", "all")))
    for dim in ["skin", "theme", "size", "under"]:
        print(f"\n-- by {dim} --")
        for k in sorted((k for k in stats if k[0] == dim), key=lambda k: stats[k][0] / stats[k][1]):
            print(f"  {k[1]:<8} {line(k)}")
    if args.v and fails:
        print(f"\n{len(fails)} FAILURES")
        for st, comp, tgt in fails[:25]:
            print(f"  {st}  comp={comp} target={tgt}")


if __name__ == "__main__":
    main()
