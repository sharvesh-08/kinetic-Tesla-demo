"""All-in-one launcher for MALE UAV Digital Twin.

Starts Redis (if not already running), Layers 2-7, the FastAPI backend,
serves the React frontend, and opens the operations dashboard in your browser.
"""

from __future__ import annotations

import atexit
import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

# Resolve base directories
ROOT = Path(__file__).resolve().parent
if not (ROOT / "app" / "main.py").is_file():
    nested = ROOT / "MALE-UAV-DESIGN-1"
    if (nested / "app" / "main.py").is_file():
        ROOT = nested

SOURCE = ROOT if (ROOT / "app" / "main.py").is_file() else ROOT / "MALE-UAV-DESIGN"
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6380/15")
API_HEALTH = "http://127.0.0.1:8000/health"
UI_URL = "http://127.0.0.1:8000"

processes: list[tuple[str, subprocess.Popen]] = []
redis_process: subprocess.Popen | None = None
stopping = False


def get_redis_port() -> int:
    try:
        from urllib.parse import urlparse
        p = urlparse(REDIS_URL).port
        if p:
            return p
    except Exception:
        pass
    return 6380


def find_redis_executable() -> str | None:
    """Locate redis-server executable on Windows or Unix."""
    found = shutil.which("redis-server")
    if found:
        return found
    user_profile = os.environ.get("USERPROFILE", "")
    candidates = glob.glob(
        os.path.join(
            user_profile,
            "AppData", "Local", "Microsoft", "WinGet", "Packages",
            "*redis*", "**", "redis-server.exe"
        ),
        recursive=True
    )
    if candidates and os.path.isfile(candidates[0]):
        return candidates[0]
    return None


def check_redis_status(port: int | None = None) -> str:
    """Check Redis readiness: 'ready', 'loading', or 'down'."""
    target_port = port or get_redis_port()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.5)
            sock.connect(("127.0.0.1", target_port))
            sock.sendall(b"*1\r\n$4\r\nPING\r\n")
            resp = sock.recv(1024)
            if b"LOADING" in resp:
                return "loading"
            return "ready"
    except OSError:
        return "down"


def ensure_redis() -> bool:
    """Ensure Redis is running and finished loading datasets."""
    global redis_process
    port = get_redis_port()
    status = check_redis_status(port)
    if status == "ready":
        print(f"[OK] Redis is already running and ready on port {port}.")
        return True

    if status == "down":
        exe = find_redis_executable()
        if not exe:
            try:
                import fakeredis  # noqa: F401
                print(f"Starting embedded fakeredis server on port {port}...")
                env = os.environ.copy()
                env["REDIS_URL"] = REDIS_URL
                env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SOURCE), str(ROOT), env.get("PYTHONPATH", "")) if p)
                redis_process = subprocess.Popen(
                    [sys.executable, "-m", "tools.embedded_redis", str(port)],
                    cwd=str(ROOT),
                    env=env,
                )
            except Exception as err:
                print(f"! Warning: redis-server executable not found and fakeredis failed: {err}", file=sys.stderr)
                print("  Please ensure Redis is installed and running.", file=sys.stderr)
                return False
        else:
            print(f"Starting Redis server from: {exe}...")
            redis_process = subprocess.Popen([exe, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for Redis to come up and finish loading any existing dataset
    for i in range(120):
        if redis_process and redis_process.poll() is not None:
            print(f"! Redis process terminated prematurely with code {redis_process.poll()}", file=sys.stderr)
            return False
        status = check_redis_status(port)
        if status == "ready":
            print(f"[OK] Redis server is ready on port {port}.", flush=True)
            return True
        if status == "loading" and i % 4 == 0:
            print("  Redis is loading dataset into memory, please wait...", flush=True)
        time.sleep(0.5)
    print("! Warning: Redis did not become ready after startup attempt.", file=sys.stderr)
    return False


def port_in_use(port: int = 8000) -> bool:
    """Check whether another process already owns the dashboard port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def dashboard_already_running() -> bool:
    try:
        with urllib.request.urlopen(API_HEALTH, timeout=1.5) as response:
            return response.status == 200 and json.loads(response.read()).get("simulation_running") is True
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def stop(*_):
    """Gracefully stop all child processes."""
    global stopping
    if stopping:
        return
    stopping = True
    if not processes and redis_process is None:
        return
    print("\nShutting down all MALE UAV services...", flush=True)

    # Terminate pipeline and API
    for name, proc in reversed(processes):
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass

    # Terminate Redis if we started it
    if redis_process and redis_process.poll() is None:
        try:
            redis_process.terminate()
            redis_process.wait(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            try:
                redis_process.kill()
            except OSError:
                pass

    print("All services stopped.", flush=True)


def wait_for_health(timeout_s: float = 35.0) -> bool:
    """Poll API health until simulation is running and dashboard is served."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if stopping:
            return False
        # Check if any child exited prematurely
        for name, proc in processes:
            if proc.poll() is not None:
                print(f"! {name} exited unexpectedly with code {proc.poll()}", file=sys.stderr)
                return False
        try:
            with urllib.request.urlopen(API_HEALTH, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if data.get("simulation_running") is True:
                        return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    atexit.register(stop)

    print("=" * 60)
    print("  MALE UAV Digital Twin - All-in-One Launcher")
    print("=" * 60)

    # Always rebuild so the server cannot serve a stale frontend bundle.
    frontend_dist = ROOT / "frontend" / "dist" / "index.html"
    frontend_dir = ROOT / "frontend"
    if shutil.which("node") and (frontend_dir / "node_modules" / "vite" / "bin" / "vite.js").is_file():
        print("Building React frontend dashboard...", flush=True)
        subprocess.run(["node", str(frontend_dir / "node_modules" / "vite" / "bin" / "vite.js"), "build"],
                       cwd=str(frontend_dir), check=True)
    elif not frontend_dist.is_file():
        print("! Error: frontend/dist is missing. Install frontend dependencies and Node.js.", file=sys.stderr)
        return 1
    else:
        print("! Node/Vite unavailable; serving the existing frontend build.", file=sys.stderr)

    if port_in_use(8000):
        if dashboard_already_running():
            print("! An old MALE-UAV dashboard process is already serving port 8000.", file=sys.stderr)
            print("  It cannot load updated Python pipeline code. Stop that run with Ctrl+C in its terminal, then rerun python3 run.py.", file=sys.stderr)
        else:
            print("! Port 8000 is already in use by another process. Stop it before starting this dashboard.", file=sys.stderr)
        return 1

    # Start Redis after detecting a stale API, so a blocked restart does not
    # accidentally launch duplicate pipeline consumers against the same Redis.
    if not ensure_redis():
        print("! Startup stopped: Redis must be ready before launching the pipeline.", file=sys.stderr)
        return 1

    # 3. Configure environment
    env = os.environ.copy()
    env["FRONTEND_PREBUILT"] = "1"
    env["REDIS_URL"] = REDIS_URL
    env.setdefault("FEATURE_REGISTRY_PATH", str(ROOT / "configs" / "feature_registry_v4.yaml"))
    env.setdefault("LAYER6_ARTIFACT", str(ROOT / "models" / "layer6-male-uav-v4-synthetic-dev.joblib"))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SOURCE), str(ROOT), env.get("PYTHONPATH", "")) if p)
    env["OMP_NUM_THREADS"] = "1"

    # 4. Define services
    commands = [
        (f"Layer {num}", [sys.executable, "-m", f"layer{num}.main"], ROOT)
        for num in range(2, 8)
    ]
    commands.append(
        ("MALE-UAV API & UI", [
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", os.environ.get("API_HOST", "127.0.0.1"),
            "--port", "8000"
        ], SOURCE)
    )

    # 5. Launch all services
    print("Starting Pipeline Layers 2-7 and FastAPI dashboard...", flush=True)
    for name, cmd, cwd in commands:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env)
        processes.append((name, proc))
        time.sleep(0.1)

    # 6. Wait for dashboard readiness
    print("Waiting for digital twin simulation to initialize...", flush=True)
    if not wait_for_health():
        print("! Startup failed or timed out. Check logs above.", file=sys.stderr)
        stop()
        return 1

    print("\n" + "=" * 60)
    print(f"[OK] Dashboard is LIVE: {UI_URL}")
    print("  - Three.js 3D Viewer:       http://127.0.0.1:8000/engine/index.html")
    print("  - Interactive API Docs:     http://127.0.0.1:8000/docs")
    print("  - Telemetry Stream:         http://127.0.0.1:8000/api/dashboard/state")
    print("=" * 60)
    print("Press Ctrl+C to stop all services.\n", flush=True)

    # 7. Open browser automatically
    webbrowser.open(UI_URL)

    # 8. Monitor processes
    try:
        while not stopping:
            for name, proc in processes:
                code = proc.poll()
                if code is not None:
                    print(f"! {name} terminated with code {code}", file=sys.stderr)
                    return code
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
