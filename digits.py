"""Font-template digit recogniser (0-9) for isolated glyphs.

Templates are rendered at start-up from common UI fonts installed on Windows, so the
recogniser is not tied to one website. A glyph is compared by normalised correlation of its
tight-cropped shape plus its aspect ratio; the nearest template decides.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
FONT_FILES = [
    "segoeui.ttf", "segoeuil.ttf", "segoeuisl.ttf", "seguisb.ttf", "segoeuib.ttf",
    "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibril.ttf", "calibrib.ttf",
    "verdana.ttf", "tahoma.ttf", "trebuc.ttf", "corbel.ttf", "candara.ttf", "bahnschrift.ttf",
    "Roboto-Regular.ttf", "Roboto-Light.ttf", "Roboto-Medium.ttf", "Roboto-Thin.ttf",
    "OpenSans-Regular.ttf", "OpenSans-Semibold.ttf", "consola.ttf", "micross.ttf",
]
NORM_W, NORM_H = 20, 28
_templates = None


def normalise(binary: np.ndarray):
    """Tight-crop a 0/255 glyph mask -> (unit feature vector, aspect ratio) or None."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255.0
    aspect = crop.shape[1] / crop.shape[0]
    v = cv2.resize(crop, (NORM_W, NORM_H), interpolation=cv2.INTER_AREA).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n else v), aspect


def _render(font_path):
    font = ImageFont.truetype(font_path, 72)
    out = []
    for d in "0123456789":
        img = Image.new("L", (120, 120), 0)
        ImageDraw.Draw(img).text((20, 10), d, fill=255, font=font)
        feat = normalise((np.array(img) > 127).astype(np.uint8) * 255)
        if feat is not None:
            out.append((d, feat[0], feat[1]))
    return out


def templates():
    global _templates
    if _templates is None:
        _templates = []
        for f in FONT_FILES:
            path = os.path.join(FONT_DIR, f)
            if os.path.exists(path):
                _templates.extend(_render(path))
    return _templates


def classify(binary: np.ndarray):
    """-> (digit char, distance, margin to the best *other* digit) or (None, inf, 0)."""
    feat = normalise(binary)
    if feat is None:
        return None, float("inf"), 0.0
    v, aspect = feat
    best = {}
    for d, tv, ta in templates():
        dist = (1 - float(v @ tv)) + 0.6 * abs(np.log(aspect / ta))
        if dist < best.get(d, float("inf")):
            best[d] = dist
    ranked = sorted(best.items(), key=lambda t: t[1])
    if not ranked:
        return None, float("inf"), 0.0
    margin = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else 1.0
    return ranked[0][0], ranked[0][1], margin


def glyph_mask(gray: np.ndarray, box, dark: bool):
    """Binary mask (255 = ink) of the glyph inside box, polarity-normalised, Otsu-thresholded."""
    sub = gray[box.y:box.y + box.h, box.x:box.x + box.w]
    if sub.size == 0:
        return np.zeros((1, 1), np.uint8)
    if not dark:
        sub = 255 - sub
    _, th = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th
