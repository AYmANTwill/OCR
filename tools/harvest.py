"""Build a labelled benchmark from switch-challenge-pratice.org (headless browser).

Each task -> dataset/<game>/<theme>_<variant>_<n>.png  +  same name .json holding the ground
truth read from the page's DOM. The tool then answers the task to reach the next one.

    python tools/harvest.py digit  --n 40
    python tools/harvest.py switch --n 8          (per layout variant)
"""
import argparse
import json
import os
import sys
from itertools import product

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import equation  # noqa: E402

SITE = "https://switch-challenge-pratice.org/exercise/Aon"
SWITCH_VARIANTS = {
    "1row": "One row exercise",
    "2row_first": "Two row exercise with options on first row",
    "2row_last": "Two Row exercise with options on last row",
    "3row_first": "Three row exercise with options on first row",
    "3row_mid": "Three row exercise with options on middle row",
    "3row_last": "Three row exercise with options on last row",
    "2row_both": "Two row exercise with options on both Row",
}

SWITCH_DOM_JS = r"""
() => {
  const root = document.querySelector('[data-testid^=WindowView] foreignObject');
  const sym = svg => (svg.getAttribute('fill') || '') + ':' +
      (svg.firstElementChild ? svg.firstElementChild.tagName + (svg.firstElementChild.getAttribute('d') || '').slice(0, 12) : '');
  const figs = [...root.querySelectorAll('.figureContainer')].filter(f => f.querySelector('svg'));
  const symbols = f => [...f.querySelectorAll('svg')].map(sym);
  const buttons = [...root.querySelectorAll('button')].map(b => {
    const r = b.getBoundingClientRect();
    return {seq: [...b.querySelectorAll('p')].map(p => +p.innerText.trim()), x: r.x + r.width / 2, y: r.y + r.height / 2};
  });
  return {top: symbols(figs[0]), bottom: symbols(figs[figs.length - 1]), buttons};
}
"""


def _goto(pg, url, attempts=4):
    for k in range(attempts):
        try:
            pg.goto(url, wait_until="domcontentloaded", timeout=45000)
            pg.wait_for_selector("#Random", timeout=30000)
            return
        except Exception as exc:  # flaky host: retry, then give up loudly
            print(f"  retry {k + 1}/{attempts} after: {str(exc).splitlines()[0]}", flush=True)
    raise RuntimeError(f"could not load {url}")


def open_test(pg, game, variant=None):
    _goto(pg, f"{SITE}/{game}/settings")
    close = pg.get_by_role("button", name="Close")
    if close.count():
        close.first.click()  # dismiss the cookie banner without accepting
    pg.click("#Random")
    if variant:
        for v in SWITCH_VARIANTS.values():
            el = pg.locator(f'[id="{v}"]')
            if (el.get_attribute("aria-checked") == "true") != (v == variant):
                el.click()
    pg.get_by_role("button", name="Start").first.click()
    pg.wait_for_url("**/test", timeout=15000)
    pg.wait_for_timeout(900)


def click_next(pg):
    pg.locator("[data-testid^=Simple-Navigation-NextButton]:not([disabled]):visible").first.click(timeout=5000)
    pg.wait_for_timeout(700)


def apply(seq, items):
    return [items[j - 1] for j in seq]


def switch_rows(dom):
    """Option boxes grouped into rows (top->bottom), each row sorted left->right."""
    rows = []
    for b in sorted(dom["buttons"], key=lambda b: b["y"]):
        if rows and abs(b["y"] - rows[-1][-1]["y"]) < 20:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [sorted(r, key=lambda b: b["x"]) for r in rows]


def switch_truth(dom):
    """Every choice (one box per row) that maps the top symbols onto the bottom ones."""
    rows = [[b["seq"] for b in r] for r in switch_rows(dom)]
    hits = []
    for choice in product(*[range(len(r)) for r in rows]):
        items = list(dom["top"])
        for r, c in zip(rows, choice):
            items = apply(r[c], items)
        if items == dom["bottom"]:
            hits.append(list(choice))
    return rows, hits


def _save(out, name, png_page, truth):
    png_page.screenshot(path=os.path.join(out, name + ".png"))
    with open(os.path.join(out, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(truth, fh, ensure_ascii=False)


def harvest_switch(pg, out, theme, n):
    for key, label in SWITCH_VARIANTS.items():
        open_test(pg, "switchChallenge", label)
        i = 0
        while i < n:
            name = f"{theme}_{key}_{i:02d}"
            try:
                dom = pg.evaluate(SWITCH_DOM_JS)
                rows, hits = switch_truth(dom)
                if not hits:
                    raise RuntimeError(f"no answer for {rows}")
                _save(out, name, pg, {"game": "switch", "variant": key, "top": dom["top"],
                                      "bottom": dom["bottom"], "rows": rows, "answers": hits})
                print(name, rows, hits, "" if len(hits) == 1 else "  !! several answers", flush=True)
                # Click by (row, column): two rows can hold boxes with identical digits.
                for row_btns, c in zip(switch_rows(dom), hits[0]):
                    if len(row_btns) > 1:
                        pg.mouse.click(row_btns[c]["x"], row_btns[c]["y"])
                click_next(pg)
                i += 1
            except Exception as exc:
                print(f"  {name}: {str(exc).splitlines()[0]} -> restarting this layout", flush=True)
                open_test(pg, "switchChallenge", label)


def harvest_digit(pg, out, theme, n):
    open_test(pg, "digitChallenge")
    i = 0
    while i < n:
        name = f"{theme}_digit_{i:02d}"
        try:
            text = pg.eval_on_selector("[data-testid^=WindowView] button",
                                       "b => b.parentElement.innerText.replace(/\s+/g,' ').trim()")
            sols = equation.solve(equation.tokenize(text), limit=50)
            if not sols:
                raise RuntimeError(f"no solution for {text!r}")
            _save(out, name, pg, {"game": "digit", "text": text, "solutions": [list(s) for s in sols]})
            print(name, text, sols[:3], flush=True)
            for d in sols[0]:
                pg.get_by_role("button", name=str(d), exact=True).click()
            click_next(pg)
            i += 1
        except Exception as exc:
            print(f"  {name}: {str(exc).splitlines()[0]} -> restarting", flush=True)
            open_test(pg, "digitChallenge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", choices=["digit", "switch"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--themes", default="light,dark")
    ap.add_argument("--viewport", default="1920x1080", help="CSS pixels, e.g. 1366x768")
    ap.add_argument("--dpr", type=float, default=1.0, help="device pixel ratio (Windows scaling 125%% -> 1.25)")
    args = ap.parse_args()
    out = os.path.join(ROOT, "dataset", args.game)
    os.makedirs(out, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for theme in args.themes.split(","):
            w, h = (int(v) for v in args.viewport.split("x"))
            ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=args.dpr,
                                      color_scheme=theme)
            pg = ctx.new_page()
            tag = theme if (w, h, args.dpr) == (1920, 1080, 1.0) else f"{theme}-{w}x{h}@{args.dpr:g}"
            (harvest_digit if args.game == "digit" else harvest_switch)(pg, out, tag, args.n)
            ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
