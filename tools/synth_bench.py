"""Synthetic DigitChallenge benchmark: known-answer equations rendered across the full matrix
of expression structure x answer-slot style x theme/colours x font x size, run through
a1_equation.analyse(). Proves layout/colour/condition robustness without a live screen.

    python tools/synth_bench.py              # ~400 cases, accuracy by category
    python tools/synth_bench.py --n 800 -v   # more cases, list every failure
    python tools/synth_bench.py --dump fails # also write failing frames as PNGs

Fonts are discovered from the OS; the equation is always drawn in one font and the on-screen
1..9 keypad in the same one, exactly as the real skins do. A case counts as correct when
analyse() returns any filling that satisfies the rendered equation.
"""
import argparse
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import a1_equation  # noqa: E402
import equation  # noqa: E402

_FONT_GLOBS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
    "/usr/share/fonts/opentype/inter/Inter-Bold.otf",
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/verdana.ttf", "C:/Windows/Fonts/tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
FONTS = {os.path.splitext(os.path.basename(p))[0]: p for p in _FONT_GLOBS if os.path.exists(p)}

# bg, fg, accent (selected slot), faint (empty pill)
THEMES = {
    "light": ((255, 255, 255), (28, 28, 32), (255, 150, 40), (236, 236, 238)),
    "dark": ((26, 27, 32), (236, 238, 243), (96, 165, 250), (44, 46, 54)),
    "yellow": ((255, 250, 232), (74, 62, 30), (250, 200, 60), (250, 238, 190)),
    "blue": ((234, 242, 255), (18, 40, 92), (92, 132, 240), (212, 226, 252)),
    "green": ((235, 250, 238), (18, 70, 42), (70, 200, 120), (210, 240, 220)),
    "pink": ((255, 238, 244), (120, 24, 70), (240, 90, 150), (250, 216, 230)),
    "lowcon": ((245, 245, 245), (150, 150, 150), (200, 200, 200), (238, 238, 238)),
    "navy": ((18, 24, 48), (210, 224, 255), (255, 196, 70), (34, 42, 72)),
}
SLOTS = ["qmark", "underscore", "empty_box", "filled_pill", "empty_pill", "pills_mixed", "blank"]
SIZES = [34, 48, 70]
TEMPLATES = [
    (["S", "+", "S"], "(%s+%s)"), (["S", "-", "S"], "(%s-%s)"), (["S", "*", "S"], "(%s*%s)"),
    (["S", "+", "S", "+", "S"], "(%s+%s+%s)"), (["S", "+", "S", "-", "S"], "(%s+%s-%s)"),
    (["S", "-", "S", "+", "S"], "(%s-%s+%s)"), (["S", "*", "S", "+", "S"], "(%s*%s+%s)"),
    (["(", "S", "*", "S", ")", "+", "S"], "((%s*%s)+%s)"),
    (["(", "S", "*", "S", ")", "-", "S"], "((%s*%s)-%s)"),
    (["(", "S", "*", "S", "*", "S", ")", "+", "S"], "((%s*%s*%s)+%s)"),
    (["(", "S", "+", "S", ")", "*", "S"], "((%s+%s)*%s)"),
]
OP_GLYPH = {"+": "+", "-": "\u2212", "*": "\u00d7", "(": "(", ")": ")"}


def _pick(symbols, eval_str, rng):
    n = sum(1 for s in symbols if s == "S")
    for _ in range(400):
        vals = rng.sample(range(1, 10), n)
        rhs = eval(eval_str % tuple(vals))
        if isinstance(rhs, int) and rhs != 0:   # targets may be negative (e.g. '= -6')
            return vals, rhs
    return None, None


def render(template, style, rng):
    symbols, eval_str = template
    vals, rhs = _pick(symbols, eval_str, rng)
    if vals is None:
        return None
    bg, fg, accent, faint = THEMES[style["theme"]]
    font = ImageFont.truetype(FONTS[style["font"]], style["size"])
    S = style["size"]
    slot_w, slot_h, pad = int(S * 0.9), int(S * 1.25), int(S * 0.45)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def tok_w(sym):
        return slot_w if sym == "S" else int(probe.textlength(OP_GLYPH.get(sym, sym), font=font))

    toks = symbols + ["="] + list(str(rhs))
    widths = [tok_w(t) for t in toks]
    W = sum(widths) + pad * (len(toks) - 1) + 2 * S
    img = Image.new("RGB", (W, int(S * 5.2)), bg)
    d = ImageDraw.Draw(img)
    y_mid, x, slot_idx = int(S * 1.1), S, 0
    for t, w in zip(toks, widths):
        if t == "S":
            x0, y0, st = x, y_mid - slot_h // 2, style["slot"]
            selected = (st == "pills_mixed" and slot_idx == 0) or st == "filled_pill"
            empty = (st == "pills_mixed" and slot_idx > 0) or st == "empty_pill"
            rect = [x0, y0, x0 + slot_w, y0 + slot_h]
            if st == "empty_box":
                d.rounded_rectangle(rect, radius=int(S * 0.18), outline=fg, width=max(2, S // 22))
            elif selected:
                d.rounded_rectangle(rect, radius=int(S * 0.28), fill=accent)
            elif empty:
                d.rounded_rectangle(rect, radius=int(S * 0.28), fill=faint)
            elif st == "qmark":
                qw = d.textlength("?", font=font)
                d.text((x0 + (slot_w - qw) / 2, y_mid - S * 0.62), "?", fill=fg, font=font)
            elif st == "underscore":
                yb = y_mid + slot_h // 2 - 3
                d.line([x0 + int(S * 0.1), yb, x0 + slot_w - int(S * 0.1), yb], fill=fg, width=max(2, S // 16))
            slot_idx += 1
        else:
            g = OP_GLYPH.get(t, t)
            bb = font.getbbox(g)
            d.text((x, y_mid - (bb[1] + bb[3]) / 2), g, fill=fg, font=font)
        x += w + pad

    ky, kcell = int(S * 2.3), int(S * 1.05)
    for i in range(9):
        r, c = divmod(i, 3)
        cx, cyk = S + c * int(kcell * 1.25), ky + r * int(kcell * 0.9)
        d.rounded_rectangle([cx, cyk, cx + kcell, cyk + int(kcell * 0.8)], radius=int(S * 0.15),
                            fill=faint if style["theme"] != "light" else (245, 246, 248))
        dg = str(i + 1)
        bb = font.getbbox(dg)
        d.text((cx + (kcell - d.textlength(dg, font=font)) / 2, cyk + int(kcell * 0.4) - (bb[1] + bb[3]) / 2),
               dg, fill=fg, font=font)

    gt = [None if s == "S" else s for s in symbols] + ["="] + list(str(rhs))
    return np.array(img)[:, :, ::-1].copy(), gt, rhs


def _valid(gt_tokens, solution):
    try:
        n, holds = equation.compile_equation(gt_tokens)
    except Exception:
        return False
    return solution is not None and len(solution) == n and holds(tuple(solution))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dump", default="")
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    if not FONTS:
        sys.exit("no fonts found; install DejaVu/Inter or run on a machine with system fonts")
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
    rng = random.Random(args.seed)
    stats = defaultdict(lambda: [0, 0])
    fails = []
    for i in range(args.n):
        tmpl = rng.choice(TEMPLATES)
        style = {"theme": rng.choice(list(THEMES)), "font": rng.choice(list(FONTS)),
                 "size": rng.choice(SIZES), "slot": rng.choice(SLOTS)}
        out = render(tmpl, style, rng)
        if out is None:
            continue
        bgr, gt, rhs = out
        try:
            sol = (a1_equation.analyse(bgr) or {}).get("solution")
        except Exception:
            sol = None
        ok = _valid(gt, sol)
        for dim, key in [("slot", style["slot"]), ("theme", style["theme"]), ("font", style["font"]),
                         ("size", str(style["size"])), ("struct", " ".join(tmpl[0])),
                         ("sign", "neg" if rhs < 0 else "pos"), ("ALL", "all")]:
            stats[(dim, key)][0] += ok
            stats[(dim, key)][1] += 1
        if not ok:
            fails.append((i, style, gt, rhs, sol))
            if args.dump:
                cv2.imwrite(f"{args.dump}/fail_{i:04d}_{style['slot']}_{style['theme']}_{style['font']}_{style['size']}.png", bgr)

    def line(k):
        c, t = stats[k]
        return f"{c / t * 100:5.1f}%  ({c}/{t})" if t else "n/a"

    print("\nOVERALL:", line(("ALL", "all")), f"  [{len(FONTS)} fonts]")
    for dim in ["sign", "slot", "theme", "font", "size", "struct"]:
        print(f"\n-- by {dim} --")
        for k in sorted((k for k in stats if k[0] == dim), key=lambda k: stats[k][0] / stats[k][1]):
            print(f"  {k[1]:<26} {line(k)}")
    if args.v and fails:
        print(f"\n{len(fails)} FAILURES")
        for i, st, gt, rhs, sol in fails:
            g = " ".join("?" if t is None else str(t) for t in gt)
            print(f"  #{i:04d} {st['slot']:<11} {st['theme']:<7} {st['font']:<18} {st['size']}  [{g}] got={sol}")


if __name__ == "__main__":
    main()
