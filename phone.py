"""Tiny local web server that mirrors the reader's latest result to a phone on the same Wi-Fi.

No dependencies (stdlib http.server). The phone opens http://<laptop-ip>:<port> and the page
polls /state a few times a second. The laptop screen itself draws nothing in headless mode.
"""
import json
import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SAMSUNG_USB_SUBNET = "192.168.42."  # Samsung USB-tethering hands the PC an address here

PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Answer</title><style>
 :root{color-scheme:dark}
 body{margin:0;background:#0b0f14;color:#e8edf2;font-family:system-ui,Segoe UI,Roboto,sans-serif;
      -webkit-user-select:none;user-select:none}
 #wrap{min-height:100vh;display:flex;flex-direction:column;justify-content:center;padding:5vw;box-sizing:border-box}
 #mode{font-size:4vw;color:#8bb4ff;letter-spacing:.5px}
 #big{font-size:22vw;font-weight:800;line-height:1.05;margin:2vw 0;word-break:break-word;color:#5ef08a}
 #lines{font-size:4.5vw;color:#cfd8e3;white-space:pre-wrap;line-height:1.5}
 #foot{position:fixed;left:0;right:0;bottom:0;padding:2vw 4vw;font-size:3.2vw;color:#5b6b7d;
       display:flex;justify-content:space-between;background:#0b0f14cc}
 .stale #big{color:#f0a35e}.stale #dot{background:#f0a35e}
 #dot{width:2.6vw;height:2.6vw;border-radius:50%;background:#5ef08a;display:inline-block;margin-right:1.5vw}
</style></head><body><div id=wrap>
 <div id=mode>waiting…</div><div id=big>—</div><div id=lines></div></div>
 <div id=foot><span><span id=dot></span><span id=stat>connecting…</span></span><span id=age></span></div>
<script>
let miss=0;
async function tick(){
 try{
  const r=await fetch('/state',{cache:'no-store'});const s=await r.json();miss=0;
  document.getElementById('mode').textContent=s.mode||'';
  document.getElementById('big').textContent=s.big||'—';
  document.getElementById('lines').textContent=(s.lines||[]).join('\\n');
  const age=s.at?(Date.now()/1000-s.at):999;
  document.getElementById('age').textContent='scan '+(s.scan||0)+' · '+Math.round(age)+'s';
  document.getElementById('stat').textContent=s.ms!=null?(s.ms+' ms'):'live';
  document.body.classList.toggle('stale',age>6);
 }catch(e){miss++;document.getElementById('stat').textContent='disconnected';
  if(miss>3)document.body.classList.add('stale');}
}
setInterval(tick,400);tick();
</script></body></html>"""


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def all_urls(port):
    """Every address the phone might reach the laptop on. -> list of (label, url), best first.

    Over a USB-C cable with Samsung "USB tethering" on, the laptop gets a 192.168.42.x address;
    that one is listed first. Wi-Fi/LAN addresses follow. 'this PC' is for adb-reverse.
    """
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            if info[0] == socket.AF_INET:
                ips.add(info[4][0])
    except OSError:
        pass
    ips.add(lan_ip())
    ips = {ip for ip in ips if ip != "0.0.0.0" and not ip.startswith("169.254")}
    out = []
    for ip in sorted(ips, key=lambda x: (not x.startswith(SAMSUNG_USB_SUBNET), x)):
        label = "USB cable" if ip.startswith(SAMSUNG_USB_SUBNET) else "Wi-Fi / LAN"
        out.append((label, f"http://{ip}:{port}"))
    out.append(("this PC only", f"http://localhost:{port}"))
    return out


def adb_reverse(port):
    """If Android platform-tools (adb) are installed and a phone is attached with USB debugging,
    tunnel the phone's localhost:<port> to this laptop over the cable. -> url or None."""
    exe = shutil.which("adb")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "reverse", f"tcp:{port}", f"tcp:{port}"],
                           capture_output=True, timeout=10)
        return f"http://localhost:{port}" if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _safe(res):
    """Keep only JSON-friendly fields (results may hold numpy ints)."""
    if not res:
        return {}
    out = {"mode": str(res.get("mode", "")), "big": str(res.get("big", "")),
           "lines": [str(x) for x in res.get("lines", [])]}
    for k in ("scan", "ms", "at"):
        if k in res:
            try:
                out[k] = float(res[k])
            except (TypeError, ValueError):
                pass
    return out


class PhoneServer:
    def __init__(self, port=8765):
        self.port = port
        self._state = {}
        self._lock = threading.Lock()
        self._httpd = None

    def update(self, res):
        with self._lock:
            self._state = _safe(res)

    def _handler(self):
        server = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, ctype):
                data = body.encode() if isinstance(body, str) else body
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.startswith("/state"):
                    with server._lock:
                        self._send(json.dumps(server._state), "application/json")
                else:
                    self._send(PAGE, "text/html; charset=utf-8")

        return H

    def start(self):
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), self._handler())
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return f"http://{lan_ip()}:{self.port}"

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
