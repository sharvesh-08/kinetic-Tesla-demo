"""Launch the complete MALE UAV pipeline and dashboard.

Start Redis 5 or newer first, then run ``python run_demo.py``. This entry point
delegates to ``start_male_uav.py`` so Layers 2–7 are started with the API/UI.
"""

from __future__ import annotations

import os
import socket
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_HEALTH = "http://127.0.0.1:8000/health"
UI_HEALTH = "http://127.0.0.1:8000/"


def wait_for(url: str, process: subprocess.Popen, timeout_s: float = 40.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    body = response.read()
                    if url == API_HEALTH:
                        import json
                        if json.loads(body).get("simulation_running") is not True:
                            time.sleep(0.4)
                            continue
                    if url == UI_HEALTH and b'id="root"' not in body:
                        time.sleep(0.4)
                        continue
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.4)
    return False


def main() -> int:
    orchestrator = ROOT / "start_male_uav.py"
    if not orchestrator.is_file():
        print(f"Cannot find full-stack launcher: {orchestrator}", file=sys.stderr)
        return 2

    occupied = []
    for port in (8000,):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                occupied.append(port)
    if occupied:
        print(f"Ports {', '.join(map(str, occupied))} are already in use. Stop the existing demo services, then retry.",
              file=sys.stderr)
        return 2

    child = subprocess.Popen([sys.executable, str(orchestrator)], cwd=ROOT, env=os.environ.copy())
    try:
        if not wait_for(API_HEALTH, child):
            print("The MALE pipeline did not start. Check that Redis 5+ is running and inspect the launcher error above.",
                  file=sys.stderr)
            return child.poll() or 1
        if not wait_for(UI_HEALTH, child):
            print("The API started, but the dashboard did not become ready. Inspect the launcher output above.",
                  file=sys.stderr)
            return child.poll() or 1

        url = "http://127.0.0.1:8000"
        print(f"Full MALE UAV pipeline and dashboard are ready: {url}", flush=True)
        print("Pipeline services: Layers 2–7, MALE API, and React dashboard. Press Ctrl+C to stop.", flush=True)
        webbrowser.open(url)
        return child.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
