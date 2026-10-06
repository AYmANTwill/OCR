# Phone display over the USB-C cable — step by step

Shows the reader's answer on your phone over the USB-C cable (no Wi-Fi, nothing drawn on
the laptop). Tested working on the Samsung S26 Ultra — its USB tethering uses the
`10.225.55.x` subnet.

---

## Each time you use it (about 1 minute)

**1. Plug the phone into the laptop** with the USB-C cable (both ends C is fine).

**2. On the phone, turn ON USB tethering:**
   - Settings → Connections → Mobile Hotspot and Tethering → **USB tethering = ON**
   - First time after plugging in, it's most reliable to toggle it **OFF, wait 3 sec, then ON**.
   - (You do **NOT** need USB debugging. Tethering alone is enough.)

**3. On the laptop, run the fixer as administrator:**
   - Right-click **`usb_fix.bat`** → **Run as administrator**
   - Press a key when it asks, wait ~10 seconds.
   - It prints a line like:
     ```
     Laptop USB address is: 10.225.55.153
     ...
        http://10.225.55.153:8765
     ```
   - It then starts the display server automatically (no window appears — that's normal).

**4. On the phone's browser, open the address it printed:**
   ```
   http://10.225.55.153:8765
   ```
   > ⚠️ The address can change between sessions. Always use the one the batch just printed,
   > not a memorised one.

**5. Check it's live:** the phone shows **"waiting… / no assessment detected"** and a
   **"scan N"** counter that keeps going up. That means the link works. When a challenge is
   on the laptop screen, the answer appears in big green text.

---

## While it's running

Hotkeys work on the laptop even with no window focused:

| Keys | Action |
|---|---|
| `Ctrl+Alt+Q` | **quit** the server |
| `Ctrl+Alt+P` | pause / resume |
| `Ctrl+Alt+R` | reset dot memory (A3 dot screens) |
| `Ctrl+Alt+D` | save a screenshot + result to `dumps/` |

For the Discord broadcast: share the **phone's** screen in the call, with that browser page open.

---

## Turning it off

- Press **`Ctrl+Alt+Q`** on the laptop, **or**
- Close the phone browser tab and unplug — then on the laptop, to go back to normal Wi-Fi
  behaviour on the USB port: it's already on automatic (DHCP), so nothing else to do.

---

## If something doesn't work

**Phone says "This site can't be reached" / address unreachable**
- The address changed. Re-run `usb_fix.bat` and use the new printed address.
- Make sure you typed `http://` and the `:8765` at the end.

**Batch says "The phone did not give an address"**
- USB tethering isn't actually active. On the phone toggle it **OFF then ON**, re-run the batch.
- Try a different USB-C cable/port (some cables are charge-only).

**Page loads but never leaves "waiting…"**
- That's fine if no challenge is on screen. If a challenge IS showing and it still waits,
  the reader may be on the wrong monitor — launch with `pythonw headless.py --monitor 2`.

**Page still won't load after a good connection** (rare — Windows firewall on a new network)
- Tell Claude; the one-line fix is:
  `netsh advfirewall firewall add rule name="phone8765" dir=in action=allow protocol=TCP localport=8765`

---

## Backup method (only if USB tethering ever refuses)

`adb reverse` makes the phone always use the same `http://localhost:8765`, regardless of
subnet — but it needs Android platform-tools installed and USB debugging ON. Ask Claude to
set it up if you want it; otherwise the tethering method above is simpler.
