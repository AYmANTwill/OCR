"""DigitChallenge (assessment #1), theme- and layout-agnostic.

Pipeline (structure first, OCR last):
  1. ink -> glyph parts -> text lines; the equation is the line with an '=' (two equal bars).
  2. every character column is classified by GEOMETRY: '?' (hook + dot), '=' (two bars),
     '÷' (bar + 2 dots), '-' (mid bar), '_' (baseline bar = gap box marker), '+' (centre cross),
     '×' (diagonal cross), '(' / ')' (tall, curved, below baseline).
  3. the remaining columns are digits, read by font templates (digits.py).
Rules from the test: operands are distinct digits 1-9.
"""
import cv2
import numpy as np

import equation
import digits
import textseg

_cache: dict = {}
_last_good = {"skeleton": None, "result": None}


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


def analyse(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = find_equation_lines(gray)
    if not found:
        return None
    found.sort(key=lambda t: -t[3])  # the task equation is the largest '=' line
    for ln, cols, shapes, _, _ in found:
        toks = read_equation(gray, ln, cols, shapes)
        if toks.count("=") != 1 or "#" in toks:
            continue
        sols, solved = solve_tokens(toks)
        shown = ["?" if t is None else t for t in solved]
        if sols:
            _last_good.update(skeleton=skeleton(toks), result=sols)
            lines = [f"Eq: {_fmt(solved)}", f"{len(sols)}{'+' if len(sols) >= 8 else ''} solution(s)"]
            if len(sols) > 1:
                lines.append("alt: " + " | ".join(" ".join(map(str, s)) for s in sols[1:4]))
            return {"mode": "A1 Digit", "lines": lines, "big": "  ".join(map(str, sols[0])),
                    "tokens": shown, "solution": list(sols[0])}
        if _last_good["result"] and _last_good["skeleton"] == skeleton(toks):
            return {"mode": "A1 Digit", "lines": ["all gaps filled", f"Eq: {_fmt(toks)}"],
                    "big": "  ".join(map(str, _last_good["result"][0])), "tokens": shown}
        return {"mode": "A1 Digit", "lines": [f"Eq: {_fmt(toks)}", "no valid solution (misread?)"],
                "big": "?", "tokens": shown}
    return None
