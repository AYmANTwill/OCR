"""Theme-agnostic text localisation: ink -> glyphs -> words -> lines.

"Ink" is any pixel that differs strongly from its local background, so dark-on-light,
light-on-dark and text on coloured panels are all handled the same way.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from vision import Box, components

BG_KERNEL = 21
INK_DELTA = 40


@dataclass(frozen=True)
class Glyph:
    box: Box
    dark: bool  # dark ink on light background


@dataclass(frozen=True)
class Word:
    glyphs: tuple
    box: Box

    @property
    def dark(self):
        return self.glyphs[0].dark


def ink_masks(gray: np.ndarray):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (BG_KERNEL, BG_KERNEL))
    dark = cv2.subtract(cv2.dilate(gray, k), gray) > INK_DELTA
    light = cv2.subtract(gray, cv2.erode(gray, k)) > INK_DELTA
    return dark.astype(np.uint8) * 255, light.astype(np.uint8) * 255


def glyphs(gray: np.ndarray, min_h=8, max_h=160, region: Box = None, keep_dots=False):
    """Glyph-sized ink components (both polarities), in frame coordinates.

    keep_dots also keeps small square specks (the dot of '?', the dots of '÷').
    """
    ox, oy = (region.x, region.y) if region else (0, 0)
    sub = region.crop(gray) if region else gray
    out = []
    for mask, dark in zip(ink_masks(sub), (True, False)):
        for b, _, _ in components(mask, min_area=4):
            tall_enough = min_h <= b.h <= max_h and b.w <= 1.6 * b.h + 4
            flat_bar = b.h < min_h and max(6, 2.2 * b.h) <= b.w <= max_h  # '-', '=' bars, '_'
            dot = keep_dots and 2 <= b.w < min_h and 2 <= b.h < min_h and max(b.w, b.h) <= 2 * min(b.w, b.h)
            if tall_enough or flat_bar or dot:
                out.append(Glyph(Box(b.x + ox, b.y + oy, b.w, b.h), dark))
    return out


def union(boxes):
    x0 = min(b.x for b in boxes)
    y0 = min(b.y for b in boxes)
    x1 = max(b.x + b.w for b in boxes)
    y1 = max(b.y + b.h for b in boxes)
    return Box(x0, y0, x1 - x0, y1 - y0)


def lines(gls, gap_factor=2.4):
    """Group same-polarity glyphs into text lines (vertical overlap + horizontal proximity)."""
    result = []
    for dark in (True, False):
        open_lines = []
        for g in sorted((g for g in gls if g.dark == dark), key=lambda g: g.box.x):
            for ln in open_lines:
                ub = ln["box"]
                overlap = min(ub.y + ub.h, g.box.y + g.box.h) - max(ub.y, g.box.y)
                near = g.box.x - (ub.x + ub.w) <= gap_factor * max(ln["h"], g.box.h)
                if g.box.h < 0.5 * ln["h"]:
                    # dots and bars ('?' dot, '=' bars, the '_' of a gap box) may sit outside the box so far
                    cy = g.box.y + g.box.h / 2
                    same_line = ub.y - 0.3 * ln["h"] <= cy <= ub.y + ub.h + 0.8 * ln["h"]
                else:
                    same_line = overlap > 0.3 * min(ub.h, g.box.h)
                if near and same_line:
                    ln["glyphs"].append(g)
                    ln["box"] = union([ub, g.box])
                    ln["h"] = max(ln["h"], g.box.h)
                    break
            else:
                open_lines.append({"glyphs": [g], "box": g.box, "h": g.box.h})
        result.extend(Word(tuple(ln["glyphs"]), ln["box"]) for ln in open_lines)
    return result


def words(gls, gap_factor=0.6):
    """Split lines into words where the horizontal gap exceeds gap_factor * glyph height."""
    out = []
    for ln in lines(gls):
        cur = [ln.glyphs[0]]
        for g in ln.glyphs[1:]:
            h = max(max(x.box.h for x in cur), g.box.h)
            if g.box.x - max(x.box.x + x.box.w for x in cur) > gap_factor * h:
                out.append(Word(tuple(cur), union([x.box for x in cur])))
                cur = [g]
            else:
                cur.append(g)
        out.append(Word(tuple(cur), union([x.box for x in cur])))
    return out


def columns(word: Word):
    """Character columns: parts stacked in one column ('=', '?', 'i') count once. -> list of Box."""
    cols = []
    for g in sorted(word.glyphs, key=lambda g: g.box.x):
        if cols and g.box.x < cols[-1].x + cols[-1].w - 0.3 * g.box.w:
            cols[-1] = union([cols[-1], g.box])
        else:
            cols.append(g.box)
    return cols


def crop_for_ocr(img: np.ndarray, box: Box, dark: bool, target_h=64, pad_frac=0.35):
    """Crop, normalise to black text on white, upscale so the text is ~target_h tall."""
    pad = int(box.h * pad_frac) + 2
    y0, y1 = max(0, box.y - pad), min(img.shape[0], box.y + box.h + pad)
    x0, x1 = max(0, box.x - pad), min(img.shape[1], box.x + box.w + pad)
    g = img[y0:y1, x0:x1]
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if not dark:
        g = 255 - g
    scale = max(1.0, target_h / max(1, box.h))
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.copyMakeBorder(g, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
