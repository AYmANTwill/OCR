"""SwitchChallenge (assessment #2), theme- and layout-agnostic.

Structure used (no colours or positions hard-coded):
  * two rows of n coloured symbols (same set) = input (top) and output (bottom);
  * between them, rows of number boxes; every box is a permutation of 1..n;
  * the input flows through the rows top -> bottom; a row with several boxes is a choice.
The answer is the one choice per row whose composition maps the top row onto the bottom row.
"""
from itertools import combinations, product

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import ocr
import textseg
from vision import Box, components, group_rows, hsv, hue_mask, hue_peaks

SAT_MIN, VAL_MIN = 90, 80
_read_cache: dict = {}


# ---------------------------------------------------------------- symbols
def _symbol_candidates(frame):
    h = hsv(frame)
    sat = cv2.inRange(h, (0, SAT_MIN, VAL_MIN), (180, 255, 255))
    hues = h[:, :, 0][sat > 0]
    if hues.size < 50:
        return []
    cands = []
    for peak in hue_peaks(hues):
        mask = cv2.bitwise_and(sat, hue_mask(h[:, :, 0], peak, 12))
        for b, area, _ in components(mask, min_area=40):
            if b.h >= 8 and 0.6 < b.w / b.h < 1.6:
                shape = cv2.resize(b.crop(mask), (24, 24), interpolation=cv2.INTER_AREA) > 127
                cands.append({"box": b, "hue": peak, "shape": shape, "fill": area / b.area, "area": area})
    return _dedupe(cands)


def _overlap(a, b):
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    return ix * iy / max(1, min(a.area, b.area))


def _dedupe(cands):
    """One shape can split across two neighbouring hue peaks: keep the bigger part."""
    kept = []
    for c in sorted(cands, key=lambda c: -c["area"]):
        if all(_overlap(c["box"], k["box"]) < 0.5 for k in kept):
            kept.append(c)
    return kept


def _rows_of_symbols(cands):
    """Horizontal runs of >= 3 similar-sized symbols spaced closely."""
    runs = []
    for row in group_rows(cands, key_y=lambda c: c["box"].cy, tol=6):
        row = sorted(row, key=lambda c: c["box"].x)
        cur = [row[0]]
        for c in row[1:]:
            ref = cur[-1]["box"]
            similar = 0.7 < c["box"].h / ref.h < 1.43
            close = c["box"].x - (ref.x + ref.w) <= 2.5 * ref.h
            if similar and close and abs(c["box"].cy - ref.cy) < 0.4 * ref.h:
                cur.append(c)
            else:
                if len(cur) >= 3:
                    runs.append(cur)
                cur = [c]
        if len(cur) >= 3:
            runs.append(cur)
    return runs


def _sym_cost(a, b):
    dh = min(abs(a["hue"] - b["hue"]), 180 - abs(a["hue"] - b["hue"])) / 90
    iou = np.logical_and(a["shape"], b["shape"]).sum() / max(1, np.logical_or(a["shape"], b["shape"]).sum())
    return dh * 3 + (1 - iou)


def find_symbol_rows(frame):
    """-> (top_run, bottom_run) or None. Both runs hold the same n symbols."""
    runs = _rows_of_symbols(_symbol_candidates(frame))
    best = None
    for top in runs:
        for bot in runs:
            if len(bot) != len(top) or bot[0]["box"].cy <= top[0]["box"].cy + 3 * top[0]["box"].h:
                continue
            if sorted(c["hue"] for c in top) != sorted(c["hue"] for c in bot):
                continue
            # input and output rows are vertically aligned
            score = abs(np.mean([c["box"].cx for c in top]) - np.mean([c["box"].cx for c in bot]))
            if best is None or score < best[0]:
                best = (score, top, bot)
    return (best[1], best[2]) if best else None


def symbol_permutation(top, bottom):
    """1-based: bottom[i] is top[perm[i] - 1]. Optimal assignment (Hungarian), O(n^3)."""
    cost = np.array([[_sym_cost(b, t) for t in top] for b in bottom])
    rows, cols = linear_sum_assignment(cost)
    perm = [0] * len(bottom)
    for r, c in zip(rows, cols):
        perm[int(r)] = int(c) + 1
    return tuple(perm)


# ---------------------------------------------------------------- number boxes
def _is_perm(s, n):
    return len(s) == n and sorted(s) == [str(i) for i in range(1, n + 1)]


def find_number_words(gray, top, bottom, n):
    sym_h = int(np.median([c["box"].h for c in top]))
    y0 = max(c["box"].y + c["box"].h for c in top) + sym_h // 2
    y1 = min(c["box"].y for c in bottom) - sym_h // 2
    if y1 - y0 < sym_h:
        return []
    region = Box(0, y0, gray.shape[1], y1 - y0)
    gls = textseg.glyphs(gray, min_h=max(8, int(sym_h * 0.35)), max_h=int(sym_h * 2.2), region=region)
    return [w for w in textseg.words(gls) if len(textseg.columns(w)) == n]


def read_words(gray, words, n):
    """-> candidate readings per word (tuples of ints), best first; [] when unreadable."""
    crops = [textseg.crop_for_ocr(gray, w.box, w.dark) for w in words]
    keys = [hash(c.tobytes()) for c in crops]
    todo = [i for i, k in enumerate(keys) if k not in _read_cache]
    if todo:
        sub = [crops[i] for i in todo]
        win = ocr.win_read_batch(sub)
        tess = ocr.tess_read_batch(sub, "123456789"[:n])
        for i, a, b in zip(todo, win, tess):
            cands = []
            for t in (a, b):
                digits = "".join(ch for ch in t if ch.isdigit())
                if _is_perm(digits, n) and tuple(map(int, digits)) not in cands:
                    cands.append(tuple(map(int, digits)))
            _read_cache[keys[i]] = cands
        if len(_read_cache) > 4000:
            _read_cache.clear()
    return [_read_cache[k] for k in keys]


def compose_all(rows):
    """rows (top->bottom) of candidate perms -> {choice: resulting perm}."""
    n = len(rows[0][0])
    out = {}
    for choice in product(*[range(len(r)) for r in rows]):
        idx = list(range(1, n + 1))
        for r, c in zip(rows, choice):
            idx = [idx[j - 1] for j in r[c]]
        out[choice] = tuple(idx)
    return out


MAX_SUBSTITUTIONS = 2     # boxes allowed to use their 2nd-best reading
MAX_CHOICES = 81          # cap on row combinations (real layouts have <= 3 x 3 = 9)
MAX_TRIES = 120           # cap on reading-alternative sets tried


def _alternatives(word_rows):
    """Reading choices per box, fewest substitutions first, bounded (never a full product)."""
    flat = [(ri, bi) for ri, r in enumerate(word_rows) for bi in range(len(r))]
    has_alt = [i for i, (ri, bi) in enumerate(flat) if len(word_rows[ri][bi]) > 1]
    for k in range(MAX_SUBSTITUTIONS + 1):
        for subs in combinations(has_alt, k):
            yield flat, set(subs)


def solve_rows(word_rows, target):
    """word_rows: rows of boxes, each box a list of candidate readings (best first).

    Tries every box's best reading first, then up to MAX_SUBSTITUTIONS alternatives.
    -> (choice, readings used, number of fitting choices) or (None, None, 0).
    """
    n_choices = 1
    for r in word_rows:
        n_choices *= len(r)
    if n_choices > MAX_CHOICES:
        return None, None, 0
    for tries, (flat, subs) in enumerate(_alternatives(word_rows)):
        if tries >= MAX_TRIES:
            break
        rows = [[None] * len(r) for r in word_rows]
        for i, (ri, bi) in enumerate(flat):
            rows[ri][bi] = word_rows[ri][bi][1 if i in subs else 0]
        hits = [c for c, p in compose_all(rows).items() if p == target]
        if hits:
            return hits[0], rows, len(hits)
    return None, None, 0


def _position(i, k):
    if k == 3:
        return ("(left)", "(middle)", "(right)")[i]
    return f"(#{i + 1} from left)" if k > 1 else ""


def _s(p):
    return "".join(map(str, p))


def analyse(frame):
    rows = find_symbol_rows(frame)
    if rows is None:
        return None
    top, bottom = rows
    n = len(top)
    target = symbol_permutation(top, bottom)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    words = find_number_words(gray, top, bottom, n)
    if not words:
        return None
    boxes = [(w, r) for w, r in zip(words, read_words(gray, words, n)) if r]
    if not boxes:
        return {"mode": "A2 Switch", "lines": ["numbers not readable - ctrl+alt+d to dump"], "big": "?",
                "target": target}
    med_h = float(np.median([w.box.h for w, _ in boxes]))
    grouped = [sorted(g, key=lambda t: t[0].box.cx)
               for g in group_rows(boxes, key_y=lambda t: t[0].box.cy, tol=med_h * 0.8)]
    choice, used, n_hits = solve_rows([[r for _, r in g] for g in grouped], target)
    lines = [f"output order {_s(target)}, {len(grouped)} row(s)"]
    if choice is None:
        lines.append("read: " + " / ".join(" ".join(_s(r[0]) for _, r in g) for g in grouped))
        return {"mode": "A2 Switch", "lines": lines + ["no combination fits"], "big": "?", "target": target}
    picks = [(ri, used[ri][c], _position(c, len(g))) for ri, (g, c) in enumerate(zip(grouped, choice)) if len(g) > 1]
    picks = picks or [(0, used[0][choice[0]], "")]
    for ri, p, pos in picks:
        lines.append(f"{'row %d: ' % (ri + 1) if len(grouped) > 1 else ''}{_s(p)} {pos}".rstrip())
    if n_hits > 1:
        lines.append(f"warning: {n_hits} combinations fit")
    return {"mode": "A2 Switch", "lines": lines, "big": "  +  ".join(_s(p) for _, p, _ in picks),
            "target": target, "choice": list(choice), "rows": [[list(p) for p in r] for r in used]}
