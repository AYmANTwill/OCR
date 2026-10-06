"""Score the vision solvers on the harvested benchmark.

    python tools/evaluate.py            # all games
    python tools/evaluate.py switch -v  # one game, list every failure

A switch task counts as correct when the boxes the overlay tells you to click are exactly the
ones the page's own data says are right. A digit task counts as correct when the digits shown
satisfy the page's equation (any valid filling is accepted by the test).
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time
from collections import defaultdict

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import a1_equation  # noqa: E402
import a2_tubes  # noqa: E402
import equation  # noqa: E402


def switch_expected(truth):
    ans = truth["answers"][0]
    picks = [r[c] for r, c in zip(truth["rows"], ans) if len(r) > 1] or [truth["rows"][0][ans[0]]]
    return "  +  ".join("".join(map(str, p)) for p in picks)


def check_switch(img, truth):
    res = a2_tubes.analyse(img)
    got = res["big"] if res else None
    return got == switch_expected(truth), got, switch_expected(truth)


def check_digit(img, truth):
    res = a1_equation.analyse(img)
    sol = tuple(res["solution"]) if res and "solution" in res else None
    n, holds = equation.compile_equation(equation.tokenize(truth["text"]))
    ok = sol is not None and len(sol) == n and len(set(sol)) == n and holds(sol)
    return ok, (res or {}).get("tokens"), truth["text"]


def run(game, verbose, only=""):
    checker = {"switch": check_switch, "digit": check_digit}[game]
    stats = defaultdict(lambda: [0, 0])
    times, fails = [], []
    for png in sorted(glob.glob(os.path.join(ROOT, "dataset", game, "*.png"))):
        if only and only not in os.path.basename(png):
            continue
        with open(png[:-4] + ".json", encoding="utf-8") as fh:
            truth = json.load(fh)
        img = cv2.imread(png)
        t0 = time.perf_counter()
        ok, got, want = checker(img, truth)
        times.append((time.perf_counter() - t0) * 1000)
        name = os.path.basename(png)[:-4]
        for key in (name.split("_")[0], truth.get("variant", ""), "ALL"):
            stats[key][0] += ok
            stats[key][1] += 1
        if not ok:
            fails.append((name, got, want))
    good, total = stats["ALL"]
    print(f"\n=== {game}: {good}/{total} correct ({100 * good / max(1, total):.1f} %)")
    for k, (g, t) in sorted(stats.items()):
        if k and k != "ALL":
            print(f"  {k:12s} {g:3d}/{t:<3d}")
    if times:
        print(f"  time/frame (cold OCR): median {statistics.median(times):.0f} ms, max {max(times):.0f} ms")
    for name, got, want in fails if verbose else fails[:8]:
        print(f"  FAIL {name}: got {got!r}  want {want!r}")
    return good, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", default=["switch", "digit"])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--only", default="", help="only files whose name contains this, e.g. @1.25")
    args = ap.parse_args()
    for g in args.games:
        run(g, args.verbose, args.only)


if __name__ == "__main__":
    main()
