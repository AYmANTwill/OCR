# Assessment Overlay (EMINES DataLab challenge)

Watches the screen and shows the answer in a small click-through box in a corner of the screen.
Everything is found by **structure**, not by fixed colours or positions, so it works on the
official Aon look, the switch-challenge-pratice.org look, light and dark themes, any window size.

| Game | How it is read | What it shows |
|---|---|---|
| A1 DigitChallenge | the text line holding an `=`; each character classified by shape (`?` hook+dot, `=`, `+`, `×`, `÷`, `−`, `(` `)`, `_` gap); digits by font templates. When a skin draws the answer slots as filled/empty pills or bare boxes (no `?` ink) or hides thin `+` signs, a fallback reader scans the band for the operator/paren glyphs only, rebuilds the slot layout from grammar (`#operands = #binary-operators + 1`) and reads just the target number with OCR | digits to type, left→right (distinct 1–9) + alternatives |
| A2 SwitchChallenge | two rows of the same coloured symbols (input/output); every number box between them, grouped in rows; boxes read by 2 OCR engines, only valid permutations kept | the box(es) to click, e.g. `2134` or `2134 + 2143` for two choice rows |
| A3 R1 "Is it symmetrical?" | largest single-colour panel (any colour); element grid | YES / NO |
| A3 R2 "Rotated but identical?" | both grids, 4 rotations vs mirror | YES / NO |
| A3 R3 "Correct?" (A ± B = C) | line drawings, + / − sign | YES / NO |
| A3 dot screens | holes + coloured dot | mini-map of the hole layout with dots numbered in order; "RECALL" on the answer screen |

SwitchChallenge rule used (verified on every harvested level): the input goes through the rows
of boxes top → bottom; a box `abcd` puts input symbol `a` first, `b` second, …; a row with
several boxes is where you choose. All 7 practice layouts are covered.

## Run
    start_overlay.bat           (double-click)
    python overlay.py           (--monitor 2 for a second screen, --fps 8)

Hotkeys: `ctrl+alt+p` pause · `ctrl+alt+r` reset dots · `ctrl+alt+c` set dot colour (hover a dot first)
· `ctrl+alt+d` save screenshot + result to `dumps/` · `ctrl+alt+m` move corner · `ctrl+alt+q` quit.

## Benchmark (real levels from switch-challenge-pratice.org, answers from the page itself)
    python tools/harvest.py switch --n 8         # 7 layouts x light/dark
    python tools/harvest.py digit  --n 40
    python tools/harvest.py digit --viewport 1366x768 --dpr 1.25   # other screen sizes / Windows scaling
    python tools/evaluate.py -v                  # accuracy per layout/theme + timing

## Tests
    python -m pytest test_samples.py -q          # unit tests + official screenshots + full benchmark

## Phone display (no window on the laptop)

For a shared screen, run the **invisible mode**: it draws nothing on the laptop and sends the
answer to your phone over the same Wi-Fi.

    start_phone.bat            (double-click — no window appears)
    pythonw headless.py        (same thing)

On launch a one-time dialog shows the address (also saved to `phone_url.txt`, e.g.
`http://192.168.x.x:8765`). Open that on your phone; the page updates a few times a second and
shows a "scan N" heartbeat so you can see it is live. Phone and laptop must be on the same
network (hotspot is fine). Hotkeys still work with no window focused: `ctrl+alt+q` quit,
`ctrl+alt+p` pause, `ctrl+alt+r` reset dots, `ctrl+alt+d` save a screenshot to `dumps/`.

`start_overlay.bat` (the on-laptop corner box) stays available for solo practice.

> Intended for the EMINES DataLab exercise on the limits of online assessment, run only on the
> challenge machine with their authorization. Not for use in a real hiring or exam test.

### Over a USB-C cable (no Wi-Fi needed)

Easiest, nothing to install on the laptop — use the phone's own USB tethering:
1. Plug the Samsung into the laptop with the USB-C cable.
2. On the phone: Settings > Connections > Mobile Hotspot and Tethering > **USB tethering** = ON.
3. Run `start_phone.bat`. The dialog (and `phone_url.txt`) now lists a **"USB cable"** address
   like `http://192.168.42.x:8765`. Open that in the phone's browser.

Alternative (advanced, survives any network lockdown): install Android platform-tools, enable
Developer options > USB debugging on the phone, then run `pythonw headless.py --usb`. The phone
opens `http://localhost:8765` and the traffic goes straight down the cable (adb reverse).
