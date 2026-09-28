"""Build the React console and start the API with pipeline Layers 2–7."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import redis


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT if (ROOT / "app" / "main.py").is_file() else ROOT / "MALE-UAV-DESIGN"
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/15")
processes: list[subprocess.Popen] = []
stopping = False


def stop(*_):
    global stopping
    stopping = True
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def main():
    frontend = ROOT / "frontend"
    if os.environ.get("FRONTEND_PREBUILT") == "1" or ((frontend / "dist" / "index.html").is_file() and shutil.which("npm") is None):
        if not (frontend / "dist" / "index.html").is_file():
            raise RuntimeError("The prebuilt React dashboard is missing.")
    else:
        if not (frontend / "node_modules").is_dir():
            raise RuntimeError("Install frontend dependencies first: npm ci --prefix frontend")
        # Invoke Vite through Node so copied workspaces do not depend on the
        # executable bit of node_modules/.bin/vite.
        subprocess.run(["node", str(frontend / "node_modules" / "vite" / "bin" / "vite.js"), "build"],
                       cwd=frontend, check=True)
    server = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2, protocol=2)
    try:
        version = server.info("server")["redis_version"]
        major, minor = (int(piece) for piece in version.split(".")[:2])
        if (major, minor) < (5, 0):
            raise RuntimeError(f"Redis {version} is too old; start Redis 5.0+ and set REDIS_URL")
        if not server.ping():
            raise RuntimeError("Redis is not responding")
    finally:
        server.close()
    artifact = Path(os.environ.get("LAYER6_ARTIFACT", str(ROOT / "models/layer6-male-uav-v4-synthetic-dev.joblib")))
    if not artifact.is_file():
        raise RuntimeError(f"Layer 6 artifact is missing: {artifact}")
    env = os.environ.copy()
    env["REDIS_URL"] = REDIS_URL
    env.setdefault("FEATURE_REGISTRY_PATH", str(ROOT / "configs/feature_registry_v4.yaml"))
    env.setdefault("LAYER6_ARTIFACT", str(artifact))
    env["PYTHONPATH"] = os.pathsep.join(part for part in (str(SOURCE), str(ROOT), env.get("PYTHONPATH", "")) if part)
    env.setdefault("OMP_NUM_THREADS", "1")
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    commands = [(f"Layer {number}", [sys.executable, "-m", f"layer{number}.main"], ROOT) for number in range(2, 8)]
    commands.extend([
        ("MALE-UAV API", [sys.executable, "-m", "uvicorn", "app.main:app", "--host", os.environ.get("API_HOST", "127.0.0.1"), "--port", os.environ.get("LAYER8_PORT", "8000")], SOURCE),
    ])
    try:
        for name, command, cwd in commands:
            process = subprocess.Popen(command, cwd=cwd, env=env)
            processes.append(process)
            print(f"{name}: process {process.pid}", flush=True)
        print("MALE UAV dashboard: http://127.0.0.1:8000", flush=True)
        print("Pipeline API: http://127.0.0.1:8000/api/pipeline", flush=True)
        while not stopping:
            for name, process in zip((item[0] for item in commands), processes):
                code = process.poll()
                if code is not None:
                    raise RuntimeError(f"{name} exited with code {code}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop()


if __name__ == "__main__":
    main()
