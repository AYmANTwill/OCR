"""Run: python -m pytest test_samples.py -q"""
import os

import cv2
import numpy as np

import a1_equation
import a2_tubes
import a3_memory
import equation
from equation import GAP

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
ROUNDS = {"R1": (15, 228), "R2": (250, 465), "R3": (482, 698)}


def _img(name):
    return cv2.imread(os.path.join(HERE, name))


def _round(name):
    x0, x1 = ROUNDS[name]
    return cv2.resize(_img("a3_rounds.png")[30:300, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)


# ---- equation logic -------------------------------------------------------
def test_single_gap():
    assert equation.solve(["3", "+", GAP, "=", "7"], range(1, 10)) == [(4,)]


def test_two_gaps_lists_all_pairs():
    sols = equation.solve([GAP, "+", GAP, "=", "7"], range(1, 10))
    assert set(sols) == {(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)}


def test_precedence_and_multiplication_sign():
    assert (5, 3) in equation.solve([GAP, "+", "2", "x", GAP, "=", "11"], range(1, 10))


def test_multi_digit_and_division():
    assert equation.solve([GAP, GAP, "-", "5", "=", "7"], range(1, 10)) == [(1, 2)]
    assert equation.solve(["8", "÷", GAP, "=", "4"], range(1, 10)) == [(2,)]


def test_repeats_fallback_when_unique_impossible():
    tokens = [GAP, "+", GAP, "=", "4"]
    assert (2, 2) not in equation.solve(tokens, range(1, 10))
    assert (2, 2) in equation.solve(tokens, range(1, 10), unique=False)


# ---- assessment 1 on real screenshots -------------------------------------
def _valid(tokens_text, sol):
    n, holds = equation.compile_equation(equation.tokenize(tokens_text))
    return len(sol) == n and len(set(sol)) == n and holds(sol)


def test_a1_official_style_keeps_typed_digit():
    res = a1_equation.analyse(_img("a1_equation.png"))
    assert res["tokens"] == ["3", "+", "?", "=", "7"]
    assert res["big"] == "4"  # the 3 already typed stays, 4 completes it


def test_a1_practice_site_user_screen():
    res = a1_equation.analyse(_img("practice_digit_user.png"))
    assert res["tokens"] == ["?", "+", "?", "=", "1", "5"]
    assert _valid("? + ? = 15", res["solution"])


def test_a1_stale_answer_never_shown_for_another_equation():
    a1_equation.analyse(_img("practice_digit_user.png"))  # solves ? + ? = 15
    f = _img("a1_equation.png").copy()
    f[118:128, 310:340] = 236  # erase the '_' gap marker
    res = a1_equation.analyse(f)
    # The robust reader may re-derive a fresh (correct, low-target) equation for this frame,
    # but it must NEVER carry over the previous "= 15" answer. Guard the real property: the
    # stale practice filling (any pair summing to 15) is never shown for this different frame.
    assert res is None or res["big"] == "?" or sum(res.get("solution") or []) != 15


# ---- assessment 2 ---------------------------------------------------------
def test_compose_series():
    rows = [[(2, 3, 4, 1)], [(1, 2, 3, 4), (4, 3, 2, 1)]]
    assert a2_tubes.compose_all(rows) == {(0, 0): (2, 3, 4, 1), (0, 1): (1, 4, 3, 2)}


def test_a2_official_style():
    assert a2_tubes.analyse(_img("a2_tubes.png"))["big"] == "3241"


def test_a2_solve_unknown_operator():
    # operator-conversion ('inverted'): input + 1324 + x + 3241 = output; recover x.
    x = [2, 4, 1, 3]
    target = a2_tubes.compose_chain([[1, 3, 2, 4], x, [3, 2, 4, 1]])
    assert a2_tubes.solve_unknown(target, [[1, 3, 2, 4], None, [3, 2, 4, 1]]) == [tuple(x)]
    assert a2_tubes.solve_unknown((2, 1, 4, 3), [None]) == [(2, 1, 4, 3)]


def test_a2_practice_site_user_screen():
    res = a2_tubes.analyse(_img("practice_switch_user.png"))
    assert res["big"] == "2431" and res["target"] == (2, 4, 3, 1)


def test_a2_swapped_symbols_change_answer():
    f = _img("a2_tubes.png").copy()
    _, bottom = a2_tubes.find_symbol_rows(f)
    a, b = bottom[2]["box"], bottom[3]["box"]  # swap circle and square -> 3214, not offered
    s = max(a.w, a.h, b.w, b.h) + 6
    pa = f[int(a.cy) - s // 2:int(a.cy) + s // 2, int(a.cx) - s // 2:int(a.cx) + s // 2].copy()
    pb = f[int(b.cy) - s // 2:int(b.cy) + s // 2, int(b.cx) - s // 2:int(b.cx) + s // 2].copy()
    f[int(a.cy) - s // 2:int(a.cy) + s // 2, int(a.cx) - s // 2:int(a.cx) + s // 2] = pb
    f[int(b.cy) - s // 2:int(b.cy) + s // 2, int(b.cx) - s // 2:int(b.cx) + s // 2] = pa
    res = a2_tubes.analyse(f)
    assert res["target"] == (3, 2, 1, 4) and res["big"] == "?"


# ---- assessment 3 ---------------------------------------------------------
def test_a3_samples_match_official_answers():
    for name, want in {"R1": "NO", "R2": "YES", "R3": "YES"}.items():
        assert a3_memory.analyse(_round(name), a3_memory.DotTracker())["big"] == want, name


def test_a3_r1_made_symmetric_says_yes():
    f = _round("R1")
    panel, _ = a3_memory.find_panel(f)
    sub = panel.crop(f)
    mid = sub.shape[1] // 2
    sub[:, mid:mid * 2] = sub[:, :mid][:, ::-1]
    assert a3_memory.symmetry_task(f, panel)["big"] == "YES"


def test_a3_r2_mirrored_says_no():
    f = _round("R2")
    panel, dark = a3_memory.find_panel(f)
    a, b = sorted(a3_memory.find_dark_boxes(dark, panel), key=lambda x: x.x)
    f[b.y:b.y + b.h, b.x:b.x + b.w] = cv2.resize(np.ascontiguousarray(np.fliplr(a.crop(f))), (b.w, b.h))
    assert a3_memory.rotation_task(f, a, b)["big"] == "NO"


def test_a3_r3_broken_sum_says_no():
    f = _round("R3")
    panel, dark = a3_memory.find_panel(f)
    boxes = a3_memory.find_dark_boxes(dark, panel)
    c = max(boxes, key=lambda x: x.y)
    f[c.y + 4:c.y + c.h - 4, c.x + c.w // 2:c.x + c.w - 4] = f[c.y + 6, c.x + 6].copy()
    assert a3_memory.combine_task(f, boxes)["big"] == "NO"


def test_detectors_do_not_cross_fire():
    assert a1_equation.analyse(_img("a2_tubes.png")) is None
    assert a2_tubes.analyse(_img("a1_equation.png")) is None
    assert a3_memory.analyse(_img("a2_tubes.png"), a3_memory.DotTracker()) is None
    assert a1_equation.analyse(_img("practice_switch_user.png")) is None
    assert a2_tubes.analyse(_img("practice_digit_user.png")) is None
    for name in ("practice_switch_user.png", "practice_digit_user.png"):
        assert a3_memory.analyse(_img(name), a3_memory.DotTracker()) is None
    for r in ROUNDS:
        assert a1_equation.analyse(_round(r)) is None
        assert a2_tubes.analyse(_round(r)) is None


def test_dot_tracker_orders_and_dedupes():
    t = a3_memory.DotTracker()
    for pos in [(0.8, 0.8), (0.8, 0.8), None, (0.2, 0.3), None, (0.2, 0.3), (0.6, 0.1)]:
        t.update(pos)
    # same hole shown twice (with a task in between) is a separate dot
    assert t.dots == [(0.8, 0.8), (0.2, 0.3), (0.2, 0.3), (0.6, 0.1)]


# ---- assessment 3 dot screens (real screenshot) ---------------------------
def _dot_screen(with_dot=True):
    f = _img("a3_dots.png").copy()
    if not with_dot:
        cv2.circle(f, (419, 292), 26, f[60, 60].tolist(), -1)  # paint the dot over with panel blue
    return f


def test_dot_found_at_right_place_with_hole_layout():
    f = _dot_screen()
    panel, _ = a3_memory.find_panel(f)
    pos, holes = a3_memory.DotTracker().detect(f, panel)
    assert abs(pos[0] - 0.657) < 0.03 and abs(pos[1] - 0.64) < 0.03
    assert len(holes) == 15  # counted by hand on the screenshot
    assert a1_equation.analyse(f) is None and a2_tubes.analyse(f) is None


def test_recall_screen_keeps_dots_then_next_dot_starts_new_trial():
    t = a3_memory.DotTracker()
    holes = [(i / 10, 0.5) for i in range(8)]
    t.update((0.2, 0.2), holes, now=0.0)
    t.update(None, [], now=1.0)              # task screen
    t.update((0.7, 0.6), holes, now=2.0)
    t.update(None, holes, now=3.0)           # holes, no dot ...
    assert not t.recall                      # ... not yet confirmed (could be a dot screen loading)
    t.update(None, holes, now=4.0)
    assert t.recall and t.dots == [(0.2, 0.2), (0.7, 0.6)]  # dots still shown while recalling
    t.update((0.5, 0.5), holes, now=10.0)    # first dot of the next trial
    assert t.dots == [(0.5, 0.5)] and not t.recall


def test_dot_screen_end_to_end():
    t = a3_memory.DotTracker()
    res = a3_memory.analyse(_dot_screen(), t)
    assert res["mode"] == "A3 Dots" and len(res["dots"]) == 1 and len(res["holes"]) == 16  # 15 holes + the dot's own hole
    empty = _dot_screen(with_dot=False)
    a3_memory.analyse(empty, t)
    t._no_dot_since -= 1.0                   # simulate time passing on the recall screen
    assert a3_memory.analyse(empty, t)["big"].startswith("RECALL")


def test_a3_aon_memorise_real_capture():
    # real Aon / assess.ly gridChallenge frame (gray board, orange-ringed dot bottom-right)
    res = a3_memory.analyse(_img("a3_aon_memorise.jpg"), a3_memory.DotTracker())
    assert res["mode"] == "Grid memorise" and "bottom-right" in res["big"]


# ---- full benchmark from the practice site (tools/harvest.py) ---------------
def test_benchmark_every_harvested_task():
    import glob
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import evaluate
    for game in ("switch", "digit"):
        if glob.glob(os.path.join(os.path.dirname(HERE), "dataset", game, "*.png")):
            good, total = evaluate.run(game, verbose=True)
            assert good == total, f"{game}: {good}/{total}"
