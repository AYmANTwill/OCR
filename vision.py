"""Shared image helpers (pure numpy/OpenCV, BGR frames)."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> int:
        return self.w * self.h

    def crop(self, img: np.ndarray, pad: int = 0) -> np.ndarray:
        y0, x0 = max(self.y + pad, 0), max(self.x + pad, 0)
        return img[y0:self.y + self.h - pad, x0:self.x + self.w - pad]


def hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def in_hsv(img_hsv: np.ndarray, lo, hi) -> np.ndarray:
    return cv2.inRange(img_hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def components(mask: np.ndarray, min_area: int = 1):
    """Connected components -> list of (Box, pixel_area, (cx, cy))."""
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            out.append((Box(int(x), int(y), int(w), int(h)), int(a), (float(cents[i][0]), float(cents[i][1]))))
    return out


def group_rows(items, key_y, tol):
    """Group items whose y (key_y(item)) are within tol, rows sorted top->bottom."""
    rows = []
    for it in sorted(items, key=key_y):
        if rows and abs(key_y(it) - key_y(rows[-1][-1])) <= tol:
            rows[-1].append(it)
        else:
            rows.append([it])
    return rows


def grid_spacing(values, min_gap):
    """Estimate regular grid pitch from 1-D centroid positions."""
    vals = sorted(values)
    centers = []
    for v in vals:
        if centers and v - centers[-1][-1] < min_gap:
            centers[-1].append(v)
        else:
            centers.append([v])
    c = [sum(g) / len(g) for g in centers]
    diffs = [b - a for a, b in zip(c, c[1:])]
    if not diffs:
        return None
    base = min(diffs)
    # Use only diffs close to the smallest pitch (skips empty rows/cols).
    near = [d for d in diffs if d < base * 1.5]
    return sum(near) / len(near)


def hue_peaks(hues: np.ndarray, min_share=0.01, min_sep=10):
    """Dominant hues (OpenCV 0-179 scale) among the given pixels, strongest first."""
    hist = np.bincount(hues.ravel(), minlength=180).astype(np.float64)
    smooth = sum(np.roll(hist, k) for k in range(-3, 4))
    peaks = []
    for h in np.argsort(smooth)[::-1]:
        if smooth[h] < min_share * hues.size:
            break
        if all(min(abs(h - p), 180 - abs(h - p)) > min_sep for p in peaks):
            peaks.append(int(h))
    return peaks


def hue_distance(h: np.ndarray, peak: int) -> np.ndarray:
    d = np.abs(h.astype(np.int16) - peak)
    return np.minimum(d, 180 - d)


def hue_mask(hue: np.ndarray, peak: int, tol: int) -> np.ndarray:
    """uint8 mask (255) of pixels whose hue is within tol of peak (circular), via a lookup table."""
    lut = np.zeros(256, np.uint8)
    for v in range(180):
        d = abs(v - peak)
        if min(d, 180 - d) <= tol:
            lut[v] = 255
    return cv2.LUT(hue, lut)
