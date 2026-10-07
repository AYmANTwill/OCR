"""DigitChallenge (assessment #1), theme- and layout-agnostic.

Pipeline (structure first, OCR last):
  1. ink -> glyph parts -> text lines; the equation is the line with an '=' (two equal bars).
  2. every character column is classified by GEOMETRY: '?' (hook + dot), '=' (two bars),
     '÷' (bar + 2 dots), '-' (mid bar), '_' (baseline bar = gap box marker), '+' (centre cross),
     '×' (diagonal cross), '(' / ')' (tall, curved, below baseline).
  3. the remaining columns are digits, read by font templates (digits.py).
Rules from the test: operands are distinct digits 1-9.

A second, skin-robust reader runs when the column path above cannot solve the line (new
"practice" look-and-feels draw the answer slots as filled/empty pills or bare boxes, so there
is no '?' ink to segment, and thin '+' signs between boxes fall under the ink threshold).
That reader (`robust_tokens`) never reads the operands at all: it scans the band for the
operator/paren glyphs only, rebuilds the exact slot layout from arithmetic grammar
(#operands = #binary-operators + 1), and reads just the right-hand target number with OCR.
"""
from collections import Counter

import cv2
import numpy as np

import digits
import equation
import ocr
import textseg
from vision import Box, components, group_rows

_cache: dict = {}
_rhs_cache: dict = {}
_last_good = {"skeleton": None, "result": None}
_BINOPS = {"+", "-", "*", "/"}


# ------------------------------------------------------------------ geometry
def _parts_in(col, parts):
    return sorted((p for p in parts if p.x >= col.x - 1 and p.x + p.w <= col.x + col.w + 1), key=lambda b: b.y)


def _is_flat(p):
    return p.w >= 2.2 * p.h


def _is_dot(p, ref_h):
    return max(p.w, p.h) <= 0.3 * ref_h and max(p.w, p.h) <= 2 * min(p.w, p.h) + 1


def _is_frame(p, ps, H):
    """A cell's selection/hover box: a large ~square outline that encloses the glyph.

    The highlighted gap box has a saturated border that reaches the ink threshold (faint
    unselected borders do not), so it shows up as one big part around the '?'. Drop it.
    """
    if H <= 0 or p.h < 1.3 * H or p.w < 0.6 * p.h:
        return False
    return any(q is not p and q.x >= p.x - 1 and q.x + q.w <= p.x + p.w + 1
               and q.y >= p.y - 1 and q.y + q.h <= p.y + p.h + 1 for q in ps)


def _line_geometry(cols, parts):
    """-> (H, mid): digit height and text middle, measured on the number right of '='.

    The right-hand side is always a number, so it is the one reliable size reference.
    """
    eq_cols = [c for c in cols if len(_parts_in(c, parts)) == 2 and all(_is_flat(p) for p in _parts_in(c, parts))]
    if not eq_cols:
        return 0.0, 0.0
    eq = eq_cols[-1]
    rhs = [p for p in parts if p.x > eq.x + eq.w and not _is_flat(p) and p.h > 0.3 * eq.w]
    if not rhs:
        return 0.0, 0.0
    H = float(np.median([p.h for p in rhs]))
    mid = float(np.median([p.y + p.h / 2 for p in rhs]))
    return H, mid


def _cross_kind(mask):
    """'+' if the centre row and column are both filled (axis arms), 'x' if the corner cells
    carry the arms instead. The corner-vs-edge comparison for 'x' is robust to stroke weight
    and small size, where sampling the exact diagonal pixel is not. -> '+'/'*'/None."""
    h, w = mask.shape
    if h < 5 or w < 5:
        return None
    m = mask > 0
    band = max(1, int(round(min(h, w) * 0.12)))
    row = m[h // 2 - band:h // 2 + band + 1, :].any(axis=0).mean()
    col = m[:, w // 2 - band:w // 2 + band + 1].any(axis=1).mean()
    corners_q = (m[:h // 4, :w // 4].mean() + m[:h // 4, -(w // 4):].mean()
                 + m[-(h // 4):, :w // 4].mean() + m[-(h // 4):, -(w // 4):].mean())
    if row > 0.85 and col > 0.85 and corners_q < 0.25:
        return "+"
    mf = m.astype(np.float32)

    def dens(y0, y1, x0, x1):
        sub = mf[y0:y1, x0:x1]
        return float(sub.mean()) if sub.size else 0.0

    t, l = max(1, h // 3), max(1, w // 3)
    if dens(t, h - t, l, w - l) < 0.2 or row > 0.85:   # a '*' crosses at centre, not on the axes
        return None
    edge = (dens(0, t, l, w - l) + dens(h - t, h, l, w - l)
            + dens(t, h - t, 0, l) + dens(t, h - t, w - l, w)) / 4
    corner = (dens(0, t, 0, l) + dens(0, t, w - l, w)
              + dens(h - t, h, 0, l) + dens(h - t, h, w - l, w)) / 4
    if corner > 0.3 and corner > edge * 1.4:
        return "*"
    return None


def _paren_kind(mask):
    """'(' if the middle bulges left of the ends, ')' if right."""
    h, w = mask.shape
    xs = np.arange(w)

    def cx(rows):
        sub = mask[rows]
        tot = sub.sum()
        return (sub.sum(axis=0) * xs).sum() / tot if tot else w / 2
    third = max(1, h // 3)
    ends = (cx(slice(0, third)) + cx(slice(h - third, h))) / 2
    mid = cx(slice(third, h - third))
    if mid < ends - 0.12 * w:
        return "("
    if mid > ends + 0.12 * w:
        return ")"
    return None


def classify_column(col, parts, ink, H, mid):
    """-> one of '?', '=', '/', '-', '_', '+', '*', '(', ')', or None (= digit, needs OCR)."""
    ps = _parts_in(col, parts)
    if len(ps) > 1:
        kept = [p for p in ps if not _is_frame(p, ps, H)]
        if kept:
            ps = kept
    # cursor bar under the selected gap: the lowest part, flat, with a real glyph above it
    if len(ps) > 1 and _is_flat(ps[-1]) and any(not _is_flat(p) and not _is_dot(p, H) for p in ps[:-1]):
        ps = ps[:-1]
    flat = [p for p in ps if _is_flat(p)]
    dots = [p for p in ps if _is_dot(p, H)]
    if len(ps) == 2 and len(flat) == 2 and abs(ps[0].w - ps[1].w) < 0.35 * max(p.w for p in ps):
        return "="
    if len(ps) == 3 and len(flat) == 1 and len(dots) == 2:
        return "/"
    if len(ps) == 2 and not flat and ps[1] in dots and ps[1].y > ps[0].y + 0.6 * ps[0].h:
        return "?"
    if len(ps) != 1:
        return None
    p = ps[0]
    if _is_flat(p):
        return "_" if p.y + p.h / 2 > mid + 0.3 * H else "-"
    mask = ink[p.y:p.y + p.h, p.x:p.x + p.w]
    if p.h >= 0.95 * H and p.w < 0.55 * p.h:
        kind = _paren_kind(mask)
        if kind:
            return kind
    if p.h < 0.85 * H and 0.6 < p.w / p.h < 1.6:
        return _cross_kind(mask)
    return None


# ------------------------------------------------------------------ reading
def find_equation_lines(gray):
    gls = textseg.glyphs(gray, min_h=10, max_h=220, keep_dots=True)
    out = []
    for ln in textseg.lines(gls, gap_factor=2.0):
        parts = [g.box for g in ln.glyphs]
        if len(parts) < 4:
            continue
        cols = textseg.columns(ln)
        H, mid = _line_geometry(cols, parts)
        if H < 8:
            continue
        x0, y0 = ln.box.x, ln.box.y
        sub = ln.box.crop(gray)
        dark_ink, light_ink = textseg.ink_masks(sub)
        ink = np.zeros_like(gray, dtype=np.uint8)
        ink[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = dark_ink if ln.dark else light_ink
        shapes = [classify_column(c, parts, ink, H, mid) for c in cols]
        if "=" in shapes:
            out.append((ln, cols, shapes, H, mid))
    return out


def read_equation(gray, ln, cols, shapes):
    """Tokens for the line: structural classes + template-recognised digits."""
    key = (hash(ln.box.crop(gray).tobytes()), tuple(shapes))
    if key not in _cache:
        tokens = []
        for c, s in zip(cols, shapes):
            if s in ("?", "_"):
                tokens.append(equation.GAP)
            elif s is not None:
                tokens.append(s)
            else:
                d, _, _ = digits.classify(digits.glyph_mask(gray, c, ln.dark))
                tokens.append(d if d is not None else "#")
        if len(_cache) > 500:
            _cache.clear()
        _cache[key] = tokens
    return _cache[key]


def skeleton(toks):
    """Equation with every left-hand operand blanked: identical before and after typing."""
    eq = toks.index("=")
    return tuple("?" if (i < eq and (t is equation.GAP or t.isdigit())) else t for i, t in enumerate(toks))


def solve_tokens(toks):
    """Digits typed into gaps on the left-hand side are kept and excluded from the other gaps
    (operands are distinct). If they make the equation impossible, solve as if all were empty.
    -> (solutions, tokens actually solved)."""
    eq = toks.index("=")
    typed = {int(t) for t in toks[:eq] if isinstance(t, str) and t.isdigit()}
    sols = equation.solve(toks, pool=[d for d in equation.DIGITS if d not in typed])
    if sols or not typed:
        return sols, toks
    blank = [equation.GAP if i < eq and isinstance(t, str) and t.isdigit() else t for i, t in enumerate(toks)]
    return equation.solve(blank), blank


def _fmt(tokens):
    out = ""
    for t in tokens:
        t = "?" if t is equation.GAP else t
        out += t if (t.isdigit() and out[-1:].isdigit()) else (" " + t if out else t)
    return out


# ------------------------------------------------------------------ robust (skin-agnostic) reader
def _otsu_mask(gray, box, dark):
    sub = gray[box.y:box.y + box.h, box.x:box.x + box.w]
    if sub.size == 0:
        return np.zeros((1, 1), np.uint8)
    if not dark:
        sub = 255 - sub
    _, th = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def _is_filled_slot(mask):
    """A solid, convex blob = an answer pill/box (filled slot), not an operator or digit.

    Digits have internal holes/strokes (low fill); '+'/'×' are sparse; '(' ')' are thin
    crescents. A filled pill is near-solid and near-convex, which none of those are.
    """
    xs = np.nonzero(mask)[1]
    if len(xs) == 0:
        return False
    area = len(xs)
    fill = area / (mask.shape[0] * mask.shape[1])
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
    solidity = area / hull if hull > 0 else 0.0
    return fill > 0.6 and solidity > 0.85


def _scan_ops_band(gray, x_lo, x_hi, y0, y1, H, mid):
    """Operator/paren glyphs in a rectangle, by geometry, both ink polarities. Filled-pill
    slots are suppressed (a solid pill must never read as ')' or a digit) and a horizontal
    bar counts as '-' only at mid-height (a baseline bar is a gap-box marker, not a minus).
    -> list of signs left-to-right, e.g. ['(', '*', '*', ')', '+'].
    """
    y0 = max(0, int(y0))
    x_lo = max(0, int(x_lo))
    x_hi = min(gray.shape[1], int(x_hi))
    if x_hi - x_lo < 3 or y1 - y0 < 3:
        return []
    hits = []
    for mask_pol, dark in zip(textseg.ink_masks(gray[y0:int(y1), x_lo:x_hi]), (True, False)):
        for b, _, _ in components(mask_pol, min_area=15):
            box = Box(b.x + x_lo, b.y + y0, b.w, b.h)
            kind = None
            if _is_flat(box):                                  # a short wide bar: a minus is
                # one such bar at mid-height, about a digit wide (a '-' bar is itself solid,
                # so it is tested before the filled-slot guard which would swallow it as a
                # pill; the width bound rejects UI dividers / underlines / progress bars)
                at_mid = abs((box.y + box.h / 2) - mid) < 0.35 * H
                kind = "-" if at_mid and 0.25 * H < box.w < 1.8 * H else None
            else:
                th = _otsu_mask(gray, box, dark)
                if _is_filled_slot(th):
                    continue
                if box.h < 0.3 * H or box.h > 2.6 * H:
                    continue
                if box.h < 0.95 * H and 0.55 < box.w / box.h < 1.7:
                    kind = _cross_kind(th)
                if kind is None and box.h >= 0.6 * H and box.w < 0.55 * box.h:
                    kind = _paren_kind(th)
            if kind in ("+", "*", "-", "/", "(", ")"):
                hits.append((box.x, kind))
    hits.sort()
    out = []
    for x, k in hits:
        if out and k == out[-1][1] and abs(x - out[-1][0]) < 0.6 * H:
            continue  # same glyph seen in both polarities
        out.append((x, k))
    return [k for _, k in out]


def scan_operators(gray, ln, cols, shapes, H, mid):
    """Operators/parens left of '=' on a found line (used by the found-line robust path)."""
    eqx = cols[shapes.index("=")].x
    y0 = ln.box.y - 0.4 * H
    y1 = ln.box.y + ln.box.h + 0.4 * H
    return _scan_ops_band(gray, 0, eqx, y0, y1, H, mid)


def grammar_tokens(skeleton_signs):
    """Rebuild the full left-hand token list from the operator/paren skeleton alone, placing
    one GAP per operand factor. Valid because every operand is a single digit (test rule),
    so #operands = #binary-operators + 1.
    """
    toks, expect_factor = [], True
    for t in skeleton_signs:
        if expect_factor and t != "(":
            toks.append(equation.GAP)
            expect_factor = False
        toks.append(t)
        if t == "(" or t in _BINOPS:
            expect_factor = True
        elif t == ")":
            expect_factor = False
    if expect_factor:
        toks.append(equation.GAP)
    return toks


def _glyph_feature(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255.0
    aspect = crop.shape[1] / crop.shape[0]
    v = cv2.resize(crop, (20, 28), interpolation=cv2.INTER_AREA).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n else v), aspect


class _Keypad:
    """Digit recogniser (1..9) whose templates are taken from the on-screen keypad, so it
    reads in the puzzle's exact font/theme. Empty when no keypad is found."""

    def __init__(self, feats):
        self._t = feats  # [(value, unit_vec, aspect)]

    def __bool__(self):
        return bool(self._t)

    def classify(self, mask):
        feat = _glyph_feature(mask)
        if feat is None or not self._t:
            return None, float("inf"), 0.0
        v, aspect = feat
        best = {}
        for d, tv, ta in self._t:
            dist = (1 - float(v @ tv)) + 0.6 * abs(np.log(aspect / ta))
            if dist < best.get(d, float("inf")):
                best[d] = dist
        ranked = sorted(best.items(), key=lambda t: t[1])
        margin = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else 1.0
        return ranked[0][0], ranked[0][1], margin


def _find_keypad(gray, equation_band=None):
    """Detect the 1..n keypad (the most grid-regular cluster of >=6 equal-height single
    digits) and build a template per value, labelled 1,2,3,... in reading order."""
    cand = []
    for mask, dark in zip(textseg.ink_masks(gray), (True, False)):
        for b, area, _ in components(mask, min_area=12):
            if b.h < 10 or b.h > 260 or not (0.12 <= b.w / b.h <= 0.95) or area < 0.12 * b.area:
                continue
            if equation_band is not None and (b.y + b.h > equation_band[0] and b.y < equation_band[1]):
                continue
            cand.append((b, dark))
    # height clusters, largest first
    used = [False] * len(cand)
    order = sorted(range(len(cand)), key=lambda i: cand[i][0].h)
    groups = []
    for i in order:
        if used[i]:
            continue
        ref = cand[i][0].h
        grp = [k for k in order if not used[k] and 0.8 <= cand[k][0].h / ref <= 1.25]
        for k in grp:
            used[k] = True
        groups.append([cand[k] for k in grp])
    groups.sort(key=lambda g: -len(g))
    for grp in groups:
        if len(grp) < 6:
            continue
        boxes = [b for b, _ in grp]
        rows = group_rows(boxes, key_y=lambda b: b.cy, tol=0.6 * np.median([b.h for b in boxes]))
        rows = [sorted(r, key=lambda b: b.x) for r in rows]
        counts = [len(r) for r in rows]
        if max(counts) < 2:
            continue
        pitches = []
        for r in rows:
            xs = [b.cx for b in r]
            pitches += [q - p for p, q in zip(xs, xs[1:])]
        reg = 1.0 - min(1.0, float(np.std(pitches)) / (float(np.mean(pitches)) + 1e-6)) if len(pitches) >= 2 else 1.0
        if reg < 0.5:
            continue
        dark_of = {id(b): d for b, d in grp}
        ordered = [b for r in rows for b in r][:9]
        feats = []
        for i, b in enumerate(ordered):
            f = _glyph_feature(_otsu_mask(gray, b, dark_of[id(b)]))
            if f is not None:
                feats.append((str(i + 1), f[0], f[1]))
        if len(feats) >= 6:
            return _Keypad(feats)
    return _Keypad([])


def _read_number(gray, comps, dark, crop, keypad):
    """Read a run of digit components as a number. The keypad (same font) reads 1..9 per
    component; OCR (any font, handles 0) is the fallback and tie-break. When the keypad reads
    every component confidently it wins, since cross-font OCR is what misreads single digits.
    """
    if keypad and comps:
        kp, confident = "", True
        for c in sorted(comps, key=lambda b: b.x):
            d, dist, marg = keypad.classify(_otsu_mask(gray, c, dark))
            if d and marg > 0.05 and dist < 0.95:
                kp += d
            else:
                confident = False
                break
        # the keypad reads one digit per real component, so when it is confident on all of
        # them its digit count is authoritative — OCR sometimes hallucinates an extra digit.
        if confident and kp:
            return kp
    return _ocr_number(crop)


def _ocr_number(crop):
    """Read a number crop robustly: Windows OCR + Tesseract at several page-segmentation
    modes, then vote (most agreed reading; ties broken by length). Cuts single-digit
    misreads that any one engine/mode makes.
    """
    key = hash(crop.tobytes())
    if key in _rhs_cache:
        return _rhs_cache[key]
    reads = list(ocr.win_read_batch([crop]))
    for psm in (7, 8, 6, 10):
        reads += ocr.tess_read_batch([crop], "0123456789", psm=psm)
    cands = [d for d in ("".join(c for c in t if c.isdigit()) for t in reads) if d]
    out = ""
    if cands:
        cnt = Counter(cands)
        out = max(cnt.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
    if len(_rhs_cache) > 500:
        _rhs_cache.clear()
    _rhs_cache[key] = out
    return out


def _read_rhs_number(gray, ln, cols, eqi, keypad=None):
    """Target number right of '=' on a found line (keypad + OCR; template fallback)."""
    rhs = cols[eqi + 1:]
    if not rhs:
        return ""
    x0 = min(c.x for c in rhs)
    x1 = max(c.x + c.w for c in rhs)
    y0 = min(c.y for c in rhs)
    y1 = max(c.y + c.h for c in rhs)
    crop = textseg.crop_for_ocr(gray, Box(x0, y0, x1 - x0, y1 - y0), ln.dark)
    out = _read_number(gray, rhs, ln.dark, crop, keypad)
    if out:
        return out
    for c in rhs:
        d, _, _ = digits.classify(digits.glyph_mask(gray, c, ln.dark))
        out += d if d is not None else ""
    return out


def robust_tokens(gray, ln, cols, shapes, H, mid, keypad=None):
    """Skin-agnostic token list for one '='-bearing line, or None if it cannot be rebuilt."""
    if "=" not in shapes:
        return None
    eqi = shapes.index("=")
    signs = scan_operators(gray, ln, cols, shapes, H, mid)
    rhs = _read_rhs_number(gray, ln, cols, eqi, keypad)
    if not rhs:
        return None
    toks = grammar_tokens(signs) + ["="] + list(rhs)
    return toks if toks.count("=") == 1 and equation.GAP in toks else None


# ------------------------------------------------- '='-anchored reader (empty / blank slots)
def _find_equals(gray):
    """'=' candidates anywhere on screen: two stacked horizontal bars of similar width,
    vertically close and x-aligned. -> [(Box, dark, bar_width)], widest bar first.

    This anchors reading when the answer slots carry no ink (empty pills / bare boxes), so
    the glyphs are too few and too far apart for line grouping to find an equation line.
    """
    bars = []
    for mask, dark in zip(textseg.ink_masks(gray), (True, False)):
        for b, _, _ in components(mask, min_area=8):
            if b.w >= 8 and b.w >= 2.0 * b.h and b.h <= 0.6 * b.w:
                bars.append((b, dark))
    eqs = []
    for i, (a, da) in enumerate(bars):
        for j in range(i + 1, len(bars)):
            b, db = bars[j]
            if da != db or abs(a.w - b.w) > 0.45 * max(a.w, b.w):
                continue
            xov = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
            if xov < 0.55 * min(a.w, b.w):
                continue
            if not (0.25 * a.w <= abs(a.cy - b.cy) <= 1.7 * a.w):
                continue
            x0, y0 = min(a.x, b.x), min(a.y, b.y)
            box = Box(x0, y0, max(a.x + a.w, b.x + b.w) - x0, max(a.y + a.h, b.y + b.h) - y0)
            eqs.append((box, da, a.w))
    eqs.sort(key=lambda t: -t[2])
    return eqs


def _rhs_right_of(gray, eqbox, dark, keypad=None):
    """The number immediately right of an '=' box. -> (digits, H, mid) or ('', 0, cy)."""
    cx = eqbox.x + eqbox.w
    cand = []
    for mask, dk in zip(textseg.ink_masks(gray), (True, False)):
        if dk != dark:
            continue
        for b, _, _ in components(mask, min_area=20):
            if b.x < cx - 2 or abs(b.cy - eqbox.cy) > 1.6 * eqbox.w:
                continue
            if b.h < 0.6 * eqbox.w or b.h > 4.0 * eqbox.w or b.w > 2.0 * b.h + 4:
                continue
            cand.append(b)
    cand.sort(key=lambda b: b.x)
    grp = []
    for b in cand:
        if not grp or b.x - (grp[-1].x + grp[-1].w) <= 1.6 * max(grp[-1].h, b.h):
            grp.append(b)
        else:
            break
    if not grp:
        return "", 0.0, float(eqbox.cy)
    H = float(np.median([b.h for b in grp]))
    mid = float(np.median([b.cy for b in grp]))
    x0 = min(b.x for b in grp)
    x1 = max(b.x + b.w for b in grp)
    y0 = min(b.y for b in grp)
    y1 = max(b.y + b.h for b in grp)
    crop = textseg.crop_for_ocr(gray, Box(x0, y0, x1 - x0, y1 - y0), dark)
    return _read_number(gray, grp, dark, crop, keypad), H, mid


def robust_from_equals(gray, keypad=None):
    """Read + solve by anchoring on the '=' glyph, for skins whose slots carry no ink.
    -> (tokens, solutions) or (None, None)."""
    for eqbox, dark, _ in _find_equals(gray):
        rhs, H, mid = _rhs_right_of(gray, eqbox, dark, keypad)
        if not rhs or H < 6:
            continue
        signs = _scan_ops_band(gray, 0, eqbox.x, mid - 1.6 * H, mid + 1.6 * H, H, mid)
        toks = grammar_tokens(signs) + ["="] + list(rhs)
        if toks.count("=") != 1 or equation.GAP not in toks:
            continue
        sols = equation.solve(toks)
        if sols:
            return toks, sols
    return None, None


def _result(toks, sols):
    shown = ["?" if t is equation.GAP else t for t in toks]
    _last_good.update(skeleton=skeleton(toks), result=sols)
    lines = [f"Eq: {_fmt(toks)}", f"{len(sols)}{'+' if len(sols) >= 8 else ''} solution(s)"]
    if len(sols) > 1:
        lines.append("alt: " + " | ".join(" ".join(map(str, s)) for s in sols[1:4]))
    return {"mode": "A1 Digit", "lines": lines, "big": "  ".join(map(str, sols[0])),
            "tokens": shown, "solution": list(sols[0])}


def analyse(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = find_equation_lines(gray)
    found.sort(key=lambda t: -t[3])  # the task equation is the largest '=' line
    # the on-screen keypad gives same-font digit templates for reading the target number.
    # Exclude the equation's own row so its digits do not join the keypad cluster and shift
    # the reading-order labels; when no line is found, take the row from the '=' glyph.
    if found:
        band = (found[0][0].box.y, found[0][0].box.y + found[0][0].box.h)
    else:
        eqs = _find_equals(gray)
        band = (eqs[0][0].cy - 1.6 * eqs[0][2], eqs[0][0].cy + 1.6 * eqs[0][2]) if eqs else None
    keypad = _find_keypad(gray, band)
    # pass 1: original column/template reader (keeps the proven Aon / official behaviour),
    # but the right-hand target is always read with the keypad + OCR, since a font the bundled
    # templates do not cover would otherwise misread it into a solvable-but-wrong equation.
    for ln, cols, shapes, H, mid in found:
        toks = read_equation(gray, ln, cols, shapes)
        if toks.count("=") != 1:
            continue
        eqi = toks.index("=")
        # defer to the robust reader when the geometric scan sees an operator the column
        # reader missed (a thin '+' lost, or a pill read as a digit so its operator vanished).
        # Only a scan operator *beyond* pass-1's set signals an under-read; a scan that finds
        # fewer must not veto pass-1, since the scan can itself miss a glyph.
        p1_ops = [t for t in toks[:eqi] if t in ("+", "-", "*", "/", "(", ")")]
        if Counter(scan_operators(gray, ln, cols, shapes, H, mid)) - Counter(p1_ops):
            continue
        rhs = _read_rhs_number(gray, ln, cols, eqi, keypad)
        if rhs:
            toks = toks[:eqi + 1] + list(rhs)
        if "#" in toks:
            continue
        sols, solved = solve_tokens(toks)
        if sols:
            res = _result(solved, sols)
            res["tokens"] = ["?" if t is None else t for t in solved]
            return res
    # pass 2a: skin-robust reader on a found line (pills / boxes / thin operators / cartoon fonts)
    for ln, cols, shapes, H, mid in found:
        rt = robust_tokens(gray, ln, cols, shapes, H, mid, keypad)
        if rt is not None:
            sols = equation.solve(rt)
            if sols:
                return _result(rt, sols)
    # pass 2b: '='-anchored reader for empty-slot skins (no ink to form a line)
    toks, sols = robust_from_equals(gray, keypad)
    if sols:
        return _result(toks, sols)
    # pass 3: nothing solved — report the best read for the user / ctrl+alt+d dump
    if not found:
        return None
    ln, cols, shapes, _, _ = found[0]
    toks = read_equation(gray, ln, cols, shapes)
    shown = ["?" if t is None else t for t in toks]
    if _last_good["result"] and "=" in toks and _last_good["skeleton"] == skeleton(toks):
        return {"mode": "A1 Digit", "lines": ["all gaps filled", f"Eq: {_fmt(toks)}"],
                "big": "  ".join(map(str, _last_good["result"][0])), "tokens": shown}
    return {"mode": "A1 Digit", "lines": [f"Eq: {_fmt(toks)}", "no valid solution (misread?)"],
            "big": "?", "tokens": shown}
