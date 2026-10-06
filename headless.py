"""Invisible mode: analyse the screen, draw NOTHING on the laptop, mirror results to a phone.

Start it (no window appears):   pythonw headless.py
On your phone (same Wi-Fi) open the URL it shows (also saved to phone_url.txt).

Hotkeys (work with no window focused):
  ctrl+alt+q quit   ctrl+alt+p pause/resume   ctrl+alt+r reset dots   ctrl+alt+d save a screenshot+result
"""
import argparse
import ctypes
import logging
import os
import queue
import threading
import time

import cv2
import keyboard

from overlay import DUMP_DIR, HERE, Engine
from phone import PhoneServer

log = logging.getLogger("headless")


def save_dump(engine):
    frame = engine.last_frame
    if frame is None:
        return
    os.makedirs(DUMP_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(DUMP_DIR, f"{stamp}.png"), frame)
    with open(os.path.join(DUMP_DIR, f"{stamp}.txt"), "w", encoding="utf-8") as fh:
        fh.write(repr(engine.last_result))
    log.info("dump saved: %s.png", stamp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", type=int, default=1, help="screen to read (1 = primary)")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--usb", action="store_true", help="also try adb-reverse over the USB cable")
    ap.add_argument("--quiet", action="store_true", help="do not pop the one-time URL dialog")
    args = ap.parse_args()

    logging.basicConfig(filename=os.path.join(HERE, "overlay.log"), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", filemode="w")
    log.info("headless starting: monitor=%s fps=%s port=%s", args.monitor, args.fps, args.port)

    q = queue.Queue()
    engine = Engine(args.monitor, args.fps, q)
    engine.start()

    server = PhoneServer(args.port)
    server.start()
    from phone import adb_reverse, all_urls
    urls = all_urls(args.port)
    if args.usb:
        tunnel = adb_reverse(args.port)
        if tunnel:
            urls.insert(0, ("USB (adb tunnel)", tunnel))
            log.info("adb reverse active: %s", tunnel)
    with open(os.path.join(HERE, "phone_url.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(f"{label}: {u}" for label, u in urls) + "\n")
    for label, u in urls:
        log.info("phone URL [%s]: %s", label, u)
    if not args.quiet:
        body = ("On your phone, open one of these (try them top to bottom):\n\n"
                + "\n".join(f"  {label}:  {u}" for label, u in urls)
                + "\n\nUSB cable: on the phone turn ON Settings > Connections >\n"
                  "Mobile Hotspot and Tethering > USB tethering, then use the 'USB cable' line.\n\n"
                  "Ctrl+Alt+Q quits. These are also in phone_url.txt.")
        # One-time setup dialog so you know the address; dismiss it before projecting.
        threading.Thread(target=lambda: ctypes.windll.user32.MessageBoxW(
            None, body, "Phone display ready", 0x40), daemon=True).start()

    stop = threading.Event()
    keyboard.add_hotkey("ctrl+alt+q", stop.set)
    keyboard.add_hotkey("ctrl+alt+p", lambda: setattr(engine, "paused", not engine.paused))
    keyboard.add_hotkey("ctrl+alt+r", engine.tracker.reset)
    keyboard.add_hotkey("ctrl+alt+d", lambda: save_dump(engine))

    while not stop.is_set():
        try:
            server.update(q.get(timeout=0.5))
        except queue.Empty:
            pass
    server.stop()
    log.info("headless stopped by hotkey")


if __name__ == "__main__":
    main()
