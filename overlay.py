"""Live corner overlay: watches the screen and shows the answer for the current assessment.

Hotkeys:  ctrl+alt+p pause/resume   ctrl+alt+r reset dots   ctrl+alt+c dot colour = pixel under mouse
          ctrl+alt+d dump frame     ctrl+alt+m move corner  ctrl+alt+q quit
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import logging
import os
import queue
import threading
import time
import tkinter as tk
import traceback

import cv2
import keyboard
import mss
import numpy as np

import a1_equation
import a2_tubes
import a3_memory

W, H, MARGIN = 400, 250, 16
MAP_W, MAP_H = 160, 110
THUMB = (320, 180)          # change detection works on this thumbnail
PIXEL_DELTA = 18            # a thumbnail pixel "changed" if it moved by more than this
MIN_CHANGED_PIXELS = 4      # ~ one clicked box / one new dot is enough
RESCAN_IDLE_S = 1.0         # nothing detected: look again this often even if the screen is static
RESCAN_FOUND_S = 2.5        # something detected: refresh this often anyway
HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(HERE, "dumps")
log = logging.getLogger("overlay")

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except (AttributeError, OSError):
    pass


class Engine(threading.Thread):
    def __init__(self, monitor_idx, fps, out_q):
        super().__init__(daemon=True)
        self.monitor_idx, self.period, self.out_q = monitor_idx, 1.0 / fps, out_q
        self.tracker = a3_memory.DotTracker()
        self.paused = False
        self.mask_rect = (0, 0, 0, 0)
        self.last_frame = None
        self.last_result = None
        self.monitor = None
        self.scans = 0
        self._prev_small = None
        self._last_scan = 0.0
        self._order = ["a2", "a1", "a3"]
        self._detectors = {
            "a1": a1_equation.analyse,
            "a2": a2_tubes.analyse,
            "a3": lambda f: a3_memory.analyse(f, self.tracker),
        }

    def _analyse(self, frame):
        for name in list(self._order):
            res = self._detectors[name](frame)
            if res is not None:
                self._order.remove(name)
                self._order.insert(0, name)  # sticky: try the current assessment first
                return res
        return {"mode": "waiting", "lines": ["no assessment detected"], "big": "…"}

    def _should_scan(self, small, now):
        if self._prev_small is None:
            return True
        changed = int(np.count_nonzero(cv2.absdiff(small, self._prev_small) > PIXEL_DELTA))
        if changed >= MIN_CHANGED_PIXELS:
            return True
        idle = self.last_result is None or self.last_result.get("mode") == "waiting"
        return now - self._last_scan >= (RESCAN_IDLE_S if idle else RESCAN_FOUND_S)

    def _tick(self, sct):
        frame = cv2.cvtColor(np.asarray(sct.grab(self.monitor)), cv2.COLOR_BGRA2BGR)
        x, y, w, h = self.mask_rect
        frame[max(0, y):y + h, max(0, x):x + w] = 255  # hide our own window from the detectors
        self.last_frame = frame
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA)
        now = time.time()
        if not self._should_scan(small, now):
            return
        self._prev_small, self._last_scan = small, now
        res = self._analyse(frame)
        self.scans += 1
        res["ms"] = int((time.time() - now) * 1000)
        res["scan"] = self.scans
        res["at"] = now
        self.last_result = res
        log.info("scan %d %s %r %d ms", self.scans, res.get("mode"), res.get("big"), res["ms"])
        self.out_q.put(res)

    def run(self):
        with getattr(mss, "MSS", mss.mss)() as sct:
            self.monitor = sct.monitors[self.monitor_idx]
            log.info("capturing monitor %s", self.monitor)
            while True:
                t0 = time.time()
                if not self.paused:
                    try:
                        self._tick(sct)
                    except Exception as exc:  # keep the overlay alive and show the error
                        log.exception("analysis failed")
                        self.out_q.put({"mode": "error", "lines": [repr(exc)[:120], "see overlay.log"],
                                        "big": "ERR", "at": time.time()})
                time.sleep(max(0.0, self.period - (time.time() - t0)))


class Overlay:
    CORNERS = ["br", "bl", "tl", "tr"]

    def __init__(self, engine, out_q):
        self.engine, self.q = engine, out_q
        self._shown = None
        self.corner = 0
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.9)
        self.root.configure(bg="#111")
        self.mode = tk.Label(self.root, fg="#9cf", bg="#111", font=("Segoe UI", 10, "bold"), anchor="w")
        self.big = tk.Label(self.root, fg="#5f5", bg="#111", font=("Segoe UI", 20, "bold"), anchor="w")
        self.info = tk.Label(self.root, fg="#ddd", bg="#111", font=("Consolas", 9), anchor="nw", justify="left")
        self.canvas = tk.Canvas(self.root, width=MAP_W, height=MAP_H, bg="#123", highlightthickness=0)
        self.mode.place(x=8, y=4, width=W - 16)
        self.big.place(x=8, y=24, width=W - 16)
        self.info.place(x=8, y=70, width=W - MAP_W - 24, height=H - 76)
        self.canvas.place(x=W - MAP_W - 8, y=H - MAP_H - 8)
        self._place()
        self.root.update_idletasks()
        self._click_through()
        self._hotkeys()
        self.root.after(50, self._poll)

    def _place(self):
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        c = self.CORNERS[self.corner]
        x = sw - W - MARGIN if "r" in c else MARGIN
        y = sh - H - MARGIN - 48 if "b" in c else MARGIN
        self.root.geometry(f"{W}x{H}+{x}+{y}")
        mon = self.engine.monitor or {"left": 0, "top": 0}
        self.engine.mask_rect = (x - mon["left"], y - mon["top"], W, H)

    def _click_through(self):
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
        GWL_EXSTYLE, LAYERED, TRANSPARENT, TOOLWINDOW = -20, 0x80000, 0x20, 0x80
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | LAYERED | TRANSPARENT | TOOLWINDOW)

    def _hotkeys(self):
        keyboard.add_hotkey("ctrl+alt+p", self._toggle_pause)
        keyboard.add_hotkey("ctrl+alt+r", self.engine.tracker.reset)
        keyboard.add_hotkey("ctrl+alt+c", self._calibrate_dot)
        keyboard.add_hotkey("ctrl+alt+d", self._dump)
        keyboard.add_hotkey("ctrl+alt+m", lambda: self.root.after(0, self._next_corner))
        keyboard.add_hotkey("ctrl+alt+q", lambda: self.root.after(0, self.root.destroy))

    def _toggle_pause(self):
        self.engine.paused = not self.engine.paused
        self.q.put({"mode": "paused" if self.engine.paused else "running", "lines": [],
                    "big": "PAUSED" if self.engine.paused else "▶"})

    def _next_corner(self):
        self.corner = (self.corner + 1) % 4
        self._place()

    def _calibrate_dot(self):
        frame, mon = self.engine.last_frame, self.engine.monitor
        if frame is None or mon is None:
            return
        pt = wt.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        x, y = pt.x - mon["left"], pt.y - mon["top"]
        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            px = frame[y:y + 1, x:x + 1]
            self.engine.tracker.dot_hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
            self.q.put({"mode": "dot colour set", "lines": [f"BGR {px[0, 0].tolist()}"], "big": "colour OK"})

    def _dump(self):
        frame = self.engine.last_frame
        if frame is None:
            return
        os.makedirs(DUMP_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(DUMP_DIR, f"{stamp}.png"), frame)
        with open(os.path.join(DUMP_DIR, f"{stamp}.txt"), "w", encoding="utf-8") as fh:
            fh.write(repr(self.engine.last_result))
        self.q.put({"mode": "dumped", "lines": [f"dumps/{stamp}.png"], "big": "saved"})

    def _draw_map(self, holes, dots):
        """Mini-map of the panel: grey = holes, pink numbered = dots in the order shown."""
        self.canvas.delete("all")
        for x, y in holes:
            self.canvas.create_oval(x * MAP_W - 5, y * MAP_H - 5, x * MAP_W + 5, y * MAP_H + 5, fill="#567", outline="")
        for i, (x, y) in enumerate(dots):
            cx, cy = x * MAP_W, y * MAP_H
            self.canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill="#f3a", outline="white")
            self.canvas.create_text(cx, cy, text=str(i + 1), fill="white", font=("Segoe UI", 9, "bold"))

    def _poll(self):
        try:
            while True:
                res = self.q.get_nowait()
                self._shown = res
                self.big.config(text=str(res.get("big", "")))
                self.info.config(text="\n".join(res.get("lines", []))[:600])
                if "dots" in res:
                    self._draw_map(res.get("holes", []), res["dots"])
        except queue.Empty:
            pass
        res = self._shown
        if res is not None:
            age = time.time() - res.get("at", time.time())
            meta = f"   scan {res['scan']} · {res['ms']} ms · {age:.0f}s ago" if "scan" in res else ""
            paused = "  [PAUSED]" if self.engine.paused else ""
            self.mode.config(text=f"{res.get('mode', '')}{meta}{paused}")
        self.root.after(250, self._poll)


def _single_instance():
    """Named mutex: a second launch must not run alongside an older copy of the overlay."""
    ctypes.windll.kernel32.CreateMutexW(None, False, "AssessmentOverlaySingleInstance")
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        ctypes.windll.user32.MessageBoxW(
            None, "The overlay is already running. Press Ctrl+Alt+Q to stop it, then start it again.",
            "Assessment overlay", 0x40)
        raise SystemExit(0)


def main():
    _single_instance()
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary)")
    ap.add_argument("--fps", type=float, default=6.0)
    args = ap.parse_args()
    logging.basicConfig(filename=os.path.join(HERE, "overlay.log"), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", filemode="w")
    log.info("overlay starting: monitor=%s fps=%s", args.monitor, args.fps)
    q = queue.Queue()
    engine = Engine(args.monitor, args.fps, q)
    engine.start()
    time.sleep(0.3)  # let the engine read monitor geometry before the window is placed
    Overlay(engine, q).root.mainloop()


if __name__ == "__main__":
    main()
