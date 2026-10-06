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
import cv2
import numpy as np

import digits
import equation
import ocr
import textseg
from vision import Box, components

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
    """'+' if centre row and centre column are both filled, 'x' if diagonals are, else None."""
    h, w = mask.shape
    if h < 5 or w < 5:
        return None
    m = mask > 0
    band = max(1, int(round(min(h, w) * 0.12)))
    row = m[h // 2 - band:h // 2 + band + 1, :].any(axis=0).mean()
    col = m[:, w // 2 - band:w // 2 + band + 1].any(axis=1).mean()
    corners = m[:h // 4, :w // 4].mean() + m[:h // 4, -(w // 4):].mean() + m[-(h // 4):, :w // 4].mean() + m[-(h // 4):, -(w // 4):].mean()
    if row > 0.85 and col > 0.85 and corners < 0.25:
        return "+"
    ys = np.linspace(0, h - 1, 12).astype(int)
    d1 = np.mean([m[y, min(w - 1, int(y * w / h))] for y in ys])
    d2 = np.mean([m[y, max(0, w - 1 - int(y * w / h))] for y in ys])
    if d1 > 0.8 and d2 > 0.8 and row < 0.85:
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


def scan_operators(gray, ln, cols, shapes, H, mid):
    """Operator/paren glyphs left of '=', found by geometry independently of column
    segmentation (thin '+' signs between boxes are lost by the column path). Filled-pill
    slots are suppressed so a solid pill is never mistaken for ')' or a digit.
    -> list of signs in left-to-right order, e.g. ['(', '*', '*', ')', '+'].
    """
    eqi = shapes.index("=")
    eqx = cols[eqi].x
    y0 = max(0, int(ln.box.y - 0.4 * H))
    y1 = int(ln.box.y + ln.box.h + 0.4 * H)
    hits = []
    for mask_pol, dark in zip(textseg.ink_masks(gray[y0:y1, :eqx]), (True, False)):
        for b, _, _ in components(mask_pol, min_area=20):
            box = Box(b.x, b.y + y0, b.w, b.h)
            if box.h < 0.3 * H or box.h > 2.4 * H:
                continue
            th = _otsu_mask(gray, box, dark)
            if _is_filled_slot(th):
                continue
            kind = None
            if box.h < 0.95 * H and 0.6 < box.w / box.h < 1.6:
                kind = _cross_kind(th)
            if kind is None and box.h >= 0.7 * H and box.w < 0.5 * box.h:
                kind = _paren_kind(th)
            if kind is None and _is_flat(box) and abs((box.y + box.h / 2) - mid) < 0.3 * H:
                kind = "-"
            if kind in ("+", "*", "-", "/", "(", ")"):
                hits.append((box.x, kind))
    hits.sort()
    out = []
    for x, k in hits:
        if out and k == out[-1][1] and abs(x - out[-1][0]) < 0.6 * H:
            continue  # same glyph seen in both polarities
        out.append((x, k))
    return [k for _, k in out]


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


def _read_rhs_number(gray, ln, cols, eqi):
    """The target number right of '=' (can be multi-digit and contain 0). OCR first (handles
    any font/weight), then the font-template recogniser as a fallback when no engine is present.
    """
    rhs = cols[eqi + 1:]
    if not rhs:
        return ""
    x0 = min(c.x for c in rhs)
    x1 = max(c.x + c.w for c in rhs)
    y0 = min(c.y for c in rhs)
    y1 = max(c.y + c.h for c in rhs)
    crop = textseg.crop_for_ocr(gray, Box(x0, y0, x1 - x0, y1 - y0), ln.dark)
    key = hash(crop.tobytes())
    if key in _rhs_cache:
        return _rhs_cache[key]
    out = ""
    for text in ocr.win_read_batch([crop]) + ocr.tess_read_batch([crop], "0123456789"):
        digs = "".join(ch for ch in text if ch.isdigit())
        if digs:
            out = digs
            break
    else:
        for c in rhs:
            d, _, _ = digits.classify(digits.glyph_mask(gray, c, ln.dark))
            out += d if d is not None else ""
    if len(_rhs_cache) > 500:
        _rhs_cache.clear()
    _rhs_cache[key] = out
    return out


def robust_tokens(gray, ln, cols, shapes, H, mid):
    """Skin-agnostic token list for one '='-bearing line, or None if it cannot be rebuilt."""
    if "=" not in shapes:
        return None
    eqi = shapes.index("=")
    signs = scan_operators(gray, ln, cols, shapes, H, mid)
    rhs = _read_rhs_number(gray, ln, cols, eqi)
    if not rhs:
        return None
    toks = grammar_tokens(signs) + ["="] + list(rhs)
    return toks if toks.count("=") == 1 and equation.GAP in toks else None


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
    if not found:
        return None
    found.sort(key=lambda t: -t[3])  # the task equation is the largest '=' line
    # pass 1: original column/template reader (keeps the proven Aon / official behaviour)
    for ln, cols, shapes, _, _ in found:
        toks = read_equation(gray, ln, cols, shapes)
        if toks.count("=") != 1 or "#" in toks:
            continue
        sols, solved = solve_tokens(toks)
        if sols:
            res = _result(solved, sols)
            res["tokens"] = ["?" if t is None else t for t in solved]
            return res
    # pass 2: skin-robust reader (pills / boxes / thin operators / cartoon fonts)
    for ln, cols, shapes, H, mid in found:
        rt = robust_tokens(gray, ln, cols, shapes, H, mid)
        if rt is None:
            continue
        sols = equation.solve(rt)
        if sols:
            return _result(rt, sols)
    # pass 3: nothing solved — report the best read for the user / ctrl+alt+d dump
    ln, cols, shapes, _, _ = found[0]
    toks = read_equation(gray, ln, cols, shapes)
    shown = ["?" if t is None else t for t in toks]
    if _last_good["result"] and "=" in toks and _last_good["skeleton"] == skeleton(toks):
        return {"mode": "A1 Digit", "lines": ["all gaps filled", f"Eq: {_fmt(toks)}"],
                "big": "  ".join(map(str, _last_good["result"][0])), "tokens": shown}
    return {"mode": "A1 Digit", "lines": [f"Eq: {_fmt(toks)}", "no valid solution (misread?)"],
            "big": "?", "tokens": shown}
