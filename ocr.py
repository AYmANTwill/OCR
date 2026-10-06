"""OCR backends (Windows.Media.Ocr + Tesseract) with batched reading of many small crops.

Batching matters: Tesseract runs as a subprocess (~100 ms per call), so every crop of a
frame is stacked into ONE image, read once per engine, and the text is mapped back.
"""
import os
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    import pytesseract

    _TESS = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_TESS):
        pytesseract.pytesseract.tesseract_cmd = _TESS
    else:
        pytesseract = None
except ImportError:
    pytesseract = None

try:
    import winocr
except ImportError:
    winocr = None

STACK_GAP = 40


def stack(crops):
    """Stack grayscale crops vertically on white. -> (image, [(y0, y1), ...])"""
    width = max(c.shape[1] for c in crops)
    parts, spans, y = [], [], 0
    for c in crops:
        parts.append(np.hstack([c, np.full((c.shape[0], width - c.shape[1]), 255, np.uint8)]))
        spans.append((y, y + c.shape[0]))
        parts.append(np.full((STACK_GAP, width), 255, np.uint8))
        y += c.shape[0] + STACK_GAP
    return np.vstack(parts), spans


def _by_span(items, spans):
    """items: [(cy, x, text)] -> one left-to-right string per span."""
    buckets = defaultdict(list)
    for cy, x, text in items:
        for i, (y0, y1) in enumerate(spans):
            if y0 <= cy <= y1:
                buckets[i].append((x, text))
                break
    return ["".join(t for _, t in sorted(buckets[i])) for i in range(len(spans))]


def win_read_batch(crops):
    """Windows OCR, one call for all crops. -> list[str] aligned with crops ('' if unread)."""
    if winocr is None or not crops:
        return [""] * len(crops)
    img, spans = stack(crops)
    res = winocr.recognize_pil_sync(Image.fromarray(img).convert("RGB"), "en")
    items = [(w["bounding_rect"]["y"] + w["bounding_rect"]["height"] / 2, w["bounding_rect"]["x"], w["text"])
             for line in res["lines"] for w in line["words"]]
    return _by_span(items, spans)


def tess_read_batch(crops, whitelist, psm=6):
    """Tesseract, one call for all crops. -> list[str] aligned with crops."""
    if pytesseract is None or not crops:
        return [""] * len(crops)
    img, spans = stack(crops)
    cfg = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    d = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    items = [(d["top"][i] + d["height"][i] / 2, d["left"][i], t.strip())
             for i, t in enumerate(d["text"]) if t.strip()]
    return _by_span(items, spans)


def available():
    return {"winocr": winocr is not None, "tesseract": pytesseract is not None}
