"""Assessment #3 screen reader: panel tasks (symmetry / rotation / add-subtract) + dot memory.

The panel is found by structure (largest filled single-colour rectangle), not by a fixed colour.
"""
import time

import cv2
import numpy as np

from vision import Box, components, grid_spacing, group_rows, hsv, hue_mask, hue_peaks, in_hsv


PANEL_MIN_SHARE = 0.06   # of the screen
PANEL_RECTANGULARITY = 0.85


def find_panel(frame):
    """Largest filled rectangle of one saturated colour (any hue). -> (Box, dark-shade mask) or (None, None).

    Task boxes are the same hue but darker; white elements are holes, ignored by the outer contour.
    """
    h = hsv(frame)
    sat = (h[:, :, 1] >= 80) & (h[:, :, 2] >= 60)
    min_area = int(frame.shape[0] * frame.shape[1] * PANEL_MIN_SHARE)
    if np.count_nonzero(sat) < min_area:
        return None, None
    best = None
    for peak in hue_peaks(h[:, :, 0][sat], min_share=0.1):
        same = cv2.bitwise_and(sat.astype(np.uint8) * 255, hue_mask(h[:, :, 0], peak, 10))
        closed = cv2.morphologyEx(same, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, hh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area >= min_area and area >= PANEL_RECTANGULARITY * w * hh and (best is None or area > best[0]):
                v = h[y:y + hh, x:x + w, 2][same[y:y + hh, x:x + w] > 0]
                dark = (same > 0) & (h[:, :, 2] < 0.8 * float(np.median(v)))
                best = (area, Box(x, y, w, hh), dark.astype(np.uint8) * 255, peak)
    if best is None:
        return None, None
    return best[1], best[2]


def find_dark_boxes(dark, panel):
    sub = cv2.morphologyEx(panel.crop(dark), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return [Box(b.x + panel.x, b.y + panel.y, b.w, b.h)
            for b, a, _ in components(sub, min_area=int(panel.area * 0.012))
            if 0.4 < b.w / b.h < 2.5 and a > 0.6 * b.area]


def white_mask(img):
    return in_hsv(hsv(img), (0, 0, 205), (180, 70, 255))


def _blobs(mask, min_area=4):
    """Square-ish filled blobs -> list of (cx, cy, area, Box). Blobs cut by the crop edge are
    page background leaking in at the panel's rounded border, not task elements."""
    H, W = mask.shape[:2]
    return [(cx, cy, a, b) for b, a, (cx, cy) in components(mask, min_area=min_area)
            if 0.55 < b.w / b.h < 1.8 and a > 0.55 * b.area
            and b.x > 0 and b.y > 0 and b.x + b.w < W and b.y + b.h < H]


def _size_classes(areas):
    lo, hi = min(areas), max(areas)
    if hi / max(lo, 1) < 2.5:
        return [1] * len(areas)
    cut = (lo * hi) ** 0.5
    return [2 if a > cut else 1 for a in areas]


def _grid_matrix(blobs, classes, origin, pitch, shape=None):
    (x0, y0), (px, py) = origin, pitch
    cells = [(int(round((b[1] - y0) / py)), int(round((b[0] - x0) / px)), c) for b, c in zip(blobs, classes)]
    if shape is None:
        shape = (max(r for r, _, _ in cells) + 1, max(c for _, c, _ in cells) + 1)
    m = np.zeros(shape, np.int8)
    for r, c, v in cells:
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            m[r, c] = v
    return m


def _pitch(blobs):
    side = max(b[3].w for b in blobs)
    return (grid_spacing([b[0] for b in blobs], min_gap=side * 0.5),
            grid_spacing([b[1] for b in blobs], min_gap=side * 0.5))


def _drop_text_row(rows):
    """The question text sits above the grid; drop a first row that doesn't fit the grid."""
    if len(rows) < 3:
        return rows
    gaps = [b[0][1] - a[0][1] for a, b in zip(rows, rows[1:])]
    typical = float(np.median(gaps[1:]))
    usual_len = int(np.median([len(r) for r in rows]))
    if abs(gaps[0] - typical) > 0.25 * typical or len(rows[0]) != usual_len:
        return rows[1:]
    return rows


def symmetry_task(frame, panel):
    blobs = _blobs(white_mask(panel.crop(frame)))
    if len(blobs) < 8:
        return None
    side = max(b[3].w for b in blobs)
    rows = [r for r in group_rows(blobs, key_y=lambda b: b[1], tol=side * 0.5) if len(r) >= 2]
    blobs = [b for r in _drop_text_row(rows) for b in r]
    if len(blobs) < 8:
        return None
    classes = _size_classes([b[2] for b in blobs])
    if len(set(classes)) < 2:
        return None  # uniform holes -> this is the dot screen, not a symmetry task
    px, py = _pitch(blobs)
    if not px or not py:
        return None
    m = _grid_matrix(blobs, classes, (min(b[0] for b in blobs), min(b[1] for b in blobs)), (px, py))
    diff = int(np.count_nonzero(m != np.fliplr(m))) // 2
    return {"mode": "A3 R1 Mirror?", "lines": [f"grid {m.shape[0]}x{m.shape[1]}, mismatches {diff}"],
            "big": "YES" if diff == 0 else "NO"}


def _box_matrix(box, blobs, classes, pitch):
    n_c = max(1, int(round(box.w / pitch[0])))
    n_r = max(1, int(round(box.h / pitch[1])))
    cw, ch = box.w / n_c, box.h / n_r
    if not blobs:
        return np.zeros((n_r, n_c), np.int8)
    return _grid_matrix(blobs, classes, (cw / 2, ch / 2), (cw, ch), shape=(n_r, n_c))


def rotation_task(frame, a, b):
    ba = _blobs(white_mask(a.crop(frame)))
    bb = _blobs(white_mask(b.crop(frame)))
    if len(ba) < 3 or len(bb) < 3:
        return None
    px, py = _pitch(ba + bb)
    if not px or not py:
        return None
    joint = _size_classes([x[2] for x in ba + bb])
    ma = _box_matrix(a, ba, joint[:len(ba)], (px, py))
    mb = _box_matrix(b, bb, joint[len(ba):], (px, py))

    def best(cands):
        return min(int(np.count_nonzero(c != mb)) if c.shape == mb.shape else 999 for c in cands)

    rot = best([np.rot90(ma, k) for k in range(4)])
    mir = best([np.rot90(np.fliplr(ma), k) for k in range(4)])
    note = "ambiguous (pattern symmetric)" if rot == mir else f"rot diff {rot} / mirror diff {mir}"
    return {"mode": "A3 R2 Rotated?", "lines": [f"grid {ma.shape[0]}x{ma.shape[1]}", note],
            "big": "YES" if rot <= mir else "NO"}


def _line_mask(frame, box, size=64):
    m = white_mask(box.crop(frame, pad=2))
    keep = np.zeros_like(m)
    min_len = 0.18 * min(box.w, box.h)
    for b, _, _ in components(m, min_area=3):
        if max(b.w, b.h) >= min_len:  # drop the faint grid dots, keep strokes
            keep[b.y:b.y + b.h, b.x:b.x + b.w] |= m[b.y:b.y + b.h, b.x:b.x + b.w]
    return cv2.resize(keep, (size, size), interpolation=cv2.INTER_AREA) > 60


def _operator(frame, a, b):
    x0, x1 = a.x + a.w, b.x
    if x1 - x0 < 4:
        return "+"
    comps = components(white_mask(frame[a.y:a.y + a.h, x0:x1]), min_area=3)
    if not comps:
        return "+"
    bx = max(comps, key=lambda c: c[1])[0]
    return "+" if bx.h > 0.5 * bx.w else "-"


def combine_task(frame, boxes):
    by_y = sorted(boxes, key=lambda b: b.y)
    a, b = sorted(by_y[:2], key=lambda b: b.x)
    c = by_y[2]
    op = _operator(frame, a, b)
    ma, mb, mc = (_line_mask(frame, x) for x in (a, b, c))
    k = np.ones((5, 5), np.uint8)
    if op == "+":
        expected = ma | mb
    else:
        expected = ma & ~cv2.dilate(mb.astype(np.uint8), k).astype(bool)
    de = cv2.dilate(expected.astype(np.uint8), k).astype(bool)
    dc = cv2.dilate(mc.astype(np.uint8), k).astype(bool)
    miss = np.count_nonzero(mc & ~de) + np.count_nonzero(expected & ~dc)
    score = miss / max(1, np.count_nonzero(expected) + np.count_nonzero(mc))
    return {"mode": f"A3 R3 A {op} B = C?", "lines": [f"mismatch {score:.0%}"],
            "big": "YES" if score < 0.08 else "NO"}


class DotTracker:
    """Remembers dot positions (normalised to the panel) in the order they appear.

    A screen showing holes but no dot is the recall screen: the dots stay displayed
    while you answer, and the next dot that appears starts a fresh sequence.
    """

    MOVE_EPS = 0.05
    RECALL_CONFIRM_S = 0.8  # holes-without-dot must persist this long (dot screens can render holes first)
    MIN_HOLES = 6

    def __init__(self):
        self.dots = []
        self.holes = []
        self.dot_hsv = None  # set by the calibrate hotkey
        self.recall = False
        self._visible = False
        self._hi_visible = False
        self._no_dot_since = None
        self._aon_no_hi_since = None
        self._new_trial = False

    def reset(self):
        self.dots, self._visible, self.recall, self._new_trial = [], False, False, False
        self._hi_visible = False
        self._aon_no_hi_since = None

    def aon_see(self, rel, board_present, now=None):
        """Aon rendering: one orange-ringed dot is highlighted per 'memorise' board for a few
        seconds, with plain task screens in between (no board). `rel` is the highlighted dot's
        position normalised to the board, or None. A new dot is recorded on the rising edge of a
        highlight or when it jumps; the recall screen is the board shown without a highlight."""
        now = time.time() if now is None else now
        if rel is not None:
            self._no_dot_since = None
            if self._new_trial:
                self.reset()
            moved = bool(self.dots) and (abs(rel[0] - self.dots[-1][0]) + abs(rel[1] - self.dots[-1][1]) > self.MOVE_EPS)
            if not self._hi_visible or moved:
                self.dots.append(rel)
            self._hi_visible = True
        else:
            self._hi_visible = False
            if board_present and self.dots:
                self._aon_no_hi_since = self._aon_no_hi_since or now
                if now - self._aon_no_hi_since >= self.RECALL_CONFIRM_S:
                    self.recall, self._new_trial = True, True
            elif not board_present:
                self._aon_no_hi_since = None

    def _dot_mask(self, h):
        if self.dot_hsv is not None:
            hue, sat, val = (int(v) for v in self.dot_hsv)
            return in_hsv(h, (max(0, hue - 8), max(0, sat - 70), max(0, val - 70)), (min(180, hue + 8), 255, 255))
        sat = h[:, :, 1] >= 80
        peaks = hue_peaks(h[:, :, 0][sat], min_share=0.2) if np.count_nonzero(sat) else []
        panel_like = (hue_mask(h[:, :, 0], peaks[0], 12) > 0) & sat if peaks else np.zeros(h.shape[:2], bool)
        white = in_hsv(h, (0, 0, 205), (180, 70, 255)) > 0
        return ((~(panel_like | white)) & (h[:, :, 1] >= 60)).astype(np.uint8) * 255

    def detect(self, frame, panel):
        """-> (dot position or None, list of hole positions), all normalised to the panel."""
        sub = panel.crop(frame)
        h = hsv(sub)
        side = min(panel.w, panel.h)
        min_area = int((side * 0.025) ** 2)
        mask = cv2.morphologyEx(self._dot_mask(h), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cands = [(b, a, c) for b, a, c in components(mask, min_area=min_area)
                 if 0.6 < b.w / b.h < 1.6 and a > 0.5 * b.area and b.w < side * 0.3]
        holes = [(cx / panel.w, cy / panel.h) for b, a, (cx, cy) in components(white_mask(sub), min_area=min_area)
                 if 0.75 < b.w / b.h < 1.33 and a > 0.65 * b.area and b.w < side * 0.3]
        if not cands:
            return None, holes
        _, _, (cx, cy) = max(cands, key=lambda t: t[1])
        return (cx / panel.w, cy / panel.h), holes

    def update(self, pos, holes=(), now=None):
        now = time.time() if now is None else now
        if holes:
            self.holes = list(holes) + ([pos] if pos else [])
        if pos is None:
            self._visible = False
            if len(holes) >= self.MIN_HOLES and self.dots:
                self._no_dot_since = self._no_dot_since or now
                if now - self._no_dot_since >= self.RECALL_CONFIRM_S:
                    self.recall, self._new_trial = True, True
            else:
                self._no_dot_since = None
            return
        self._no_dot_since = None
        if self._new_trial:
            self.reset()
        moved = bool(self.dots) and abs(pos[0] - self.dots[-1][0]) + abs(pos[1] - self.dots[-1][1]) > self.MOVE_EPS
        if not self._visible or moved:
            self.dots.append(pos)
        self._visible = True

    def describe(self):
        def where(p):
            return f"{'top' if p[1] < 0.5 else 'bottom'}-{'left' if p[0] < 0.5 else 'right'}"
        return [f"{i + 1}: {where(p)} ({p[0]:.2f},{p[1]:.2f})" for i, p in enumerate(self.dots)]


# --------------------------------------------------------------- Aon / assess.ly rendering
# The real P&G (Aon smartPredict) gridChallenge uses low-saturation panels: a medium-gray dot
# board, darker-gray task squares, and a black dot ringed in orange for the highlight. The
# colour-panel path above never fires on it, so these detectors handle that look directly.
AON_BOARD_LO, AON_BOARD_HI = 150, 225   # board grey value band


def _aon_highlight(frame):
    """The highlighted dot (black centre, orange/red ring). -> (cx, cy, Box) or None."""
    h = hsv(frame)
    m = ((h[:, :, 0] <= 22) & (h[:, :, 1] >= 110) & (h[:, :, 2] >= 110)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cands = [(b, a, c) for b, a, c in components(m, min_area=60) if 0.5 < b.w / b.h < 2.0]
    if not cands:
        return None
    b, _, (cx, cy) = max(cands, key=lambda t: t[1])
    return cx, cy, b


def _aon_board(frame):
    """The medium-gray dot board (memorise / recall screens). -> Box or None."""
    h = hsv(frame)
    gray = (h[:, :, 1] < 45) & (h[:, :, 2] >= AON_BOARD_LO) & (h[:, :, 2] <= AON_BOARD_HI)
    m = cv2.morphologyEx(gray.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = 0.05 * frame.shape[0] * frame.shape[1]
    best = None
    for c in cnts:
        x, y, w, hh = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if a >= area_min and a >= 0.8 * w * hh and w > hh and (best is None or a > best[0]):
            best = (a, Box(x, y, w, hh))
    return best[1] if best else None


def _aon_dots(frame, board):
    """All dot centres on the board, normalised. Dots are darker-gray discs on the gray board."""
    sub = board.crop(frame)
    h = hsv(sub)
    dark = ((h[:, :, 1] < 60) & (h[:, :, 2] < AON_BOARD_LO - 20)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    side = min(board.w, board.h)
    out = []
    for b, a, (cx, cy) in components(dark, min_area=int((side * 0.02) ** 2)):
        if 0.6 < b.w / b.h < 1.6 and a > 0.55 * b.area and b.w < side * 0.2:
            out.append((cx / board.w, cy / board.h))
    return out


def _where(p):
    col = ("left", "centre", "right")[min(2, int(p[0] * 3))]
    row = ("top", "middle", "bottom")[min(2, int(p[1] * 3))]
    return f"{row}-{col}"


# -------- Aon interference tasks (shown between memorise boards): symmetry / rotation / arithmetic
# These sit on a LIGHTER grey panel than the memorise board, so _aon_board never claims them and
# they are routed here instead. Each is answered by structure, verified on the real captures.
AON_TASK_LO, AON_TASK_HI = 226, 245   # the task-screen panel (lighter than the 190-grey board)


def _aon_task_panel(frame):
    """The light-grey rounded panel that carries an interference task. -> Box or None."""
    h = hsv(frame)
    gray = (h[:, :, 1] < 50) & (h[:, :, 2] >= AON_TASK_LO) & (h[:, :, 2] <= AON_TASK_HI)
    m = cv2.morphologyEx(gray.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = 0.05 * frame.shape[0] * frame.shape[1]
    best = None
    for c in cnts:
        x, y, w, hh = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if a >= area_min and a >= 0.7 * w * hh and (best is None or a > best[0]):
            best = (a, Box(x, y, w, hh))
    return best[1] if best else None


def _aon_elements(frame, panel):
    """Dark lattice marks (small dots + big squares) on a symmetry/rotation panel.
    -> [(cx, cy, area, Box)] in frame coordinates."""
    h = hsv(panel.crop(frame))
    dark = ((h[:, :, 1] < 90) & (h[:, :, 2] < 180)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    side = min(panel.w, panel.h)
    out = []
    for b, a, (cx, cy) in components(dark, min_area=int((side * 0.008) ** 2)):
        if 0.45 < b.w / b.h < 2.2 and a > 0.5 * b.area and b.w < side * 0.25:
            out.append((cx + panel.x, cy + panel.y, a, Box(b.x + panel.x, b.y + panel.y, b.w, b.h)))
    return out


def _squares(els):
    """1 for a big (filled square) mark, 0 for a small (empty) dot. None if only one size exists."""
    cls = _size_classes([e[2] for e in els])
    if len(set(cls)) < 2:
        return None
    return [1 if c == 2 else 0 for c in cls]


def _aon_symmetry(els, sq):
    px, py = _pitch(els)
    if not px or not py:
        return None
    m = _grid_matrix(els, sq, (min(e[0] for e in els), min(e[1] for e in els)), (px, py))
    cols = m.shape[1]
    if cols < 4 or cols % 2:
        return None
    diff = int(np.count_nonzero(m[:, :cols // 2] != np.fliplr(m[:, cols // 2:])))
    return {"mode": "Grid · symmetrical?", "big": "YES" if diff == 0 else "NO",
            "lines": [f"grid {m.shape[0]}x{cols}, mirror mismatch {diff}"]}


def _aon_rotation(els, sq):
    xc = sorted(e[0] for e in els)
    gap, at = max((xc[k + 1] - xc[k], k) for k in range(len(xc) - 1))
    px = grid_spacing([e[0] for e in els], min_gap=20)
    if not px or gap < 1.6 * px:
        return None                       # one continuous grid -> not a two-figure rotation task
    thr = (xc[at] + xc[at + 1]) / 2
    left = [(e, s) for e, s in zip(els, sq) if e[0] < thr]
    right = [(e, s) for e, s in zip(els, sq) if e[0] >= thr]
    if min(len(left), len(right)) < 6 or abs(len(left) - len(right)) > 0.3 * len(els):
        return None
    el, sl = [e for e, _ in left], [s for _, s in left]
    er, sr = [e for e, _ in right], [s for _, s in right]
    ml = _grid_matrix(el, sl, (min(e[0] for e in el), min(e[1] for e in el)), _pitch(el))
    mr = _grid_matrix(er, sr, (min(e[0] for e in er), min(e[1] for e in er)), _pitch(er))

    def closest(cands):
        return min((int(np.count_nonzero(c != mr)) if c.shape == mr.shape else 999) for c in cands)

    rot = closest([np.rot90(ml, k) for k in range(4)])
    mir = closest([np.rot90(np.fliplr(ml), k) for k in range(4)])
    yes = rot <= 2 and rot <= mir
    note = "ambiguous (symmetric)" if rot == mir else f"rot {rot} / mirror {mir}"
    return {"mode": "Grid · rotated & identical?", "big": "YES" if yes else "NO",
            "lines": [f"{ml.shape[0]}x{ml.shape[1]} · {note}"]}


def _aon_white_boxes(frame, panel):
    """The white figure boxes of an arithmetic task (A, B, C). -> list of Box."""
    h = hsv(panel.crop(frame))
    w = ((h[:, :, 1] < 40) & (h[:, :, 2] >= 244)).astype(np.uint8) * 255
    w = cv2.morphologyEx(w, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    side = min(panel.w, panel.h)
    out = []
    for b, a, _ in components(w, min_area=int((side * 0.15) ** 2)):
        if 0.75 < b.w / b.h < 1.3 and a > 0.8 * b.area:
            out.append(Box(b.x + panel.x, b.y + panel.y, b.w, b.h))
    return out


def _fig_lattice_n(frame, box):
    """How many dots per side (grid is N x N). Dots are small round dark marks; strokes are long,
    so they are filtered out. -> N (0 if too few dots to tell)."""
    h = hsv(box.crop(frame))
    side = box.w
    xs, ys = [], []
    for b, a, (cx, cy) in components((h[:, :, 2] < 150).astype(np.uint8) * 255, min_area=int((0.02 * side) ** 2)):
        if 0.55 < b.w / b.h < 1.8 and max(b.w, b.h) < 0.16 * side and a > 0.55 * b.area:
            xs.append(cx)
            ys.append(cy)

    def n_lines(vals):
        vals = sorted(vals)
        return 1 + sum(vals[i] - vals[i - 1] > 0.10 * side for i in range(1, len(vals))) if vals else 0

    return max(n_lines(xs), n_lines(ys))


def _fig_edges(frame, box, n):
    """Which unit segments of the n x n dot lattice are drawn in one figure box. -> set of names.
    Named H row col (horizontal), V (vertical), D ('\\' diagonal), A ('/' diagonal). Sampled strictly
    between dots; diagonals sampled off-centre so a crossing X keeps both arms distinct."""
    m = (hsv(box.crop(frame))[:, :, 2] < 150).astype(np.uint8)
    fr = [(i + 0.5) / n for i in range(n)]
    rad = max(6, int(box.w * 0.02))

    def dark(px, py):
        win = m[max(0, py - rad):py + rad + 1, max(0, px - rad):px + rad + 1]
        return win.size and win.mean() > 0.30

    def pt(r, c):
        return (fr[c] * box.w, fr[r] * box.h)

    def seg(a, b, ts, need):
        p, q = pt(*a), pt(*b)
        return sum(bool(dark(int(p[0] + (q[0] - p[0]) * t), int(p[1] + (q[1] - p[1]) * t))) for t in ts) >= need

    hv, dg, e = (0.3, 0.4, 0.5, 0.6, 0.7), (0.25, 0.35, 0.65, 0.75), set()
    for r in range(n):
        for c in range(n - 1):
            if seg((r, c), (r, c + 1), hv, 4):
                e.add(f"H{r}{c}")
    for c in range(n):
        for r in range(n - 1):
            if seg((r, c), (r + 1, c), hv, 4):
                e.add(f"V{r}{c}")
    for r in range(n - 1):
        for c in range(n - 1):
            if seg((r, c), (r + 1, c + 1), dg, 4):
                e.add(f"D{r}{c}")
            if seg((r, c + 1), (r + 1, c), dg, 4):
                e.add(f"A{r}{c}")
    return e


def _fig_operator(frame, a, b):
    x0, x1 = a.x + a.w, b.x
    if x1 - x0 < 6:
        return "+"
    comps = components(((hsv(frame[min(a.y, b.y):max(a.y + a.h, b.y + b.h), x0:x1])[:, :, 2] < 150)
                        ).astype(np.uint8) * 255, min_area=20)
    if not comps:
        return "+"
    bx = max(comps, key=lambda c: c[1])[0]
    return "+" if bx.h > 0.55 * bx.w else "-"


def _aon_arithmetic(frame, boxes):
    boxes = sorted(boxes, key=lambda b: -b.area)[:3]
    top = sorted(boxes, key=lambda b: b.y)
    a, b = sorted(top[:2], key=lambda x: x.x)
    c = top[2]
    n = min(max(max(_fig_lattice_n(frame, x) for x in (a, b, c)), 3), 6)  # all three share one lattice
    op = _fig_operator(frame, a, b)
    ea, eb, ec = _fig_edges(frame, a, n), _fig_edges(frame, b, n), _fig_edges(frame, c, n)
    got = (ea | eb) if op == "+" else (ea - eb)
    miss = got ^ ec
    return {"mode": f"Grid · A {op} B = C?", "big": "YES" if not miss else "NO",
            "lines": [f"{n}x{n} · A{op}B={''.join(sorted(got)) or '-'} vs C={''.join(sorted(ec)) or '-'}",
                      f"mismatch {len(miss)}"]}


def _aon_task(frame):
    """An interference task between memorise boards (not a dot board). -> result dict or None."""
    panel = _aon_task_panel(frame)
    if panel is None:
        return None
    boxes = _aon_white_boxes(frame, panel)
    if len(boxes) >= 3:
        return _aon_arithmetic(frame, boxes)
    els = _aon_elements(frame, panel)
    if len(els) < 20:
        return None
    sq = _squares(els)
    if sq is None:
        return None
    return _aon_rotation(els, sq) or _aon_symmetry(els, sq)


def _aon_analyse(frame, tracker):
    """Aon gridChallenge. -> result dict or None (not an Aon screen)."""
    board = _aon_board(frame)
    hi = _aon_highlight(frame)
    if board is not None and hi is not None:          # memorise screen
        rel = ((hi[0] - board.x) / board.w, (hi[1] - board.y) / board.h)
        tracker.aon_see(rel, board_present=True)
        n = len(tracker.dots)
        return {"mode": "Grid memorise", "big": f"{n}: {_where(rel)}",
                "lines": ["remembered so far:"] + [f"{i + 1}. {_where(p)}" for i, p in enumerate(tracker.dots)],
                "dots": list(tracker.dots), "holes": []}
    if board is not None:                             # board, no highlight -> recall / gap
        tracker.aon_see(None, board_present=True)
        if tracker.recall and tracker.dots:
            actual = _aon_dots(frame, board)
            picks = []
            for p in tracker.dots:
                near = min(actual, key=lambda q: (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2, default=p)
                picks.append(near)
            return {"mode": "Grid RECALL", "big": f"click {len(picks)} dots",
                    "lines": [f"{i + 1}. {_where(p)}" for i, p in enumerate(picks)],
                    "dots": picks, "holes": actual}
        n = len(tracker.dots)
        return {"mode": "Grid", "big": f"{n} memorised",
                "lines": [f"{i + 1}. {_where(p)}" for i, p in enumerate(tracker.dots)],
                "dots": list(tracker.dots), "holes": []}
    task = _aon_task(frame)                            # no board: an interference task, or not Aon
    if task is not None:
        task["dots"], task["holes"] = list(tracker.dots), []
        if tracker.dots:
            task["lines"] = task["lines"] + [f"({len(tracker.dots)} dots held in memory)"]
    return task


def analyse(frame, tracker: DotTracker):
    aon = _aon_analyse(frame, tracker)
    if aon is not None:
        return aon
    panel, dark = find_panel(frame)
    if panel is None:
        return None
    boxes = find_dark_boxes(dark, panel)
    if len(boxes) >= 3:
        result = combine_task(frame, sorted(boxes, key=lambda b: -b.area)[:3])
    elif len(boxes) == 2:
        result = rotation_task(frame, *sorted(boxes, key=lambda b: b.x))
    else:
        result = symmetry_task(frame, panel)
    if result is None:
        pos, holes = tracker.detect(frame, panel)
        tracker.update(pos, holes)
        n = len(tracker.dots)
        big = f"RECALL: click {n} in order" if tracker.recall else f"{n} dot(s) memorised"
        result = {"mode": "A3 Dots", "lines": [], "big": big}
    else:
        tracker.update(None)
    result["dots"] = list(tracker.dots)
    result["holes"] = list(tracker.holes)
    result["lines"] = result["lines"] + tracker.describe()
    return result
