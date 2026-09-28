"""Exercise the actual publisher, all eight services and API on isolated ports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import redis

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=35)
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix='fixed5s-pipeline-'))
    redis_port = free_port()
    api_port = free_port()
    while api_port == redis_port:
        api_port = free_port()
    env = {**os.environ, 'REDIS_URL': f'redis://127.0.0.1:{redis_port}/0',
           'REDIS_PORT': str(redis_port), 'LAYER8_PORT': str(api_port),
           'GROQ_API_KEY': ''}
    client = redis.Redis.from_url(env['REDIS_URL'], decode_responses=True, socket_timeout=2)
    processes = []
    print(f'Isolated logs: {directory}', flush=True)
    try:
        with (directory / 'redis.log').open('w') as redis_log, (directory / 'pipeline.log').open('w') as pipeline_log:
            redis_proc = subprocess.Popen(['redis-server', '--bind', '127.0.0.1', '--port', str(redis_port),
                                           '--save', '', '--appendonly', 'no', '--dir', str(directory)],
                                          stdout=redis_log, stderr=subprocess.STDOUT)
            processes.append(redis_proc)
            for _ in range(100):
                try:
                    if client.ping():
                        break
                except redis.exceptions.ConnectionError:
                    time.sleep(0.05)
            else:
                raise RuntimeError('Isolated Redis did not start')
            subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_layer4_service', '-q'],
                           cwd=ROOT, env=env, check=True)
            env['FRONTEND_PREBUILT'] = '1'
            launcher = subprocess.Popen([sys.executable, 'start_male_uav.py'], cwd=ROOT, env=env,
                                        stdout=pipeline_log, stderr=subprocess.STDOUT)
            processes.append(launcher)
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                if launcher.poll() is not None:
                    raise RuntimeError(f'Launcher exited early; inspect {directory / "pipeline.log"}')
                time.sleep(0.25)
            streams = ['engine:telemetry:ecu', 'engine:synced:frames',
                       'engine:synced:ekf:frames', 'engine:physics:predictions', 'engine:ekf:residuals',
                       'engine:features:windows', 'engine:layer6:assessments', 'engine:risk:rul',
                       'engine:dashboard:telemetry']
            counts = {name: client.xlen(name) for name in streams}
            assert all(counts.values()), counts
            def records(stream: str) -> list[dict]:
                return [json.loads(fields['payload']) for _, fields in client.xrange(stream)]
            windows = records('engine:features:windows')
            assessments = records('engine:layer6:assessments')
            risks = records('engine:risk:rul')
            assert len(windows) >= 3, counts
            assert all(w['window_length_s'] == 5 and w['features'] for w in windows)
            assert 0 < risks[-1]['engine_risk_score'] < 30, risks[-1]
            assert all(w['window_end_ns'] - w['window_start_ns'] == 5_000_000_000 for w in windows)
            assert all(b['window_end_ns'] - a['window_end_ns'] == 2_500_000_000 for a, b in zip(windows, windows[1:]))
            assert assessments and all(a['classification_available'] for a in assessments)
            for assessment in assessments:
                black_box = assessment['black_box']
                assert len(black_box['fault_flags']) == 8
                assert len(black_box['class_probabilities']) == 9
                assert all(0 <= item['confidence'] <= 1 for item in black_box['fault_flags'].values())
                assert 0 <= black_box['anomaly_score'] <= 1
                assert black_box['shap_top_features']
            with urllib.request.urlopen(f'http://127.0.0.1:{api_port}/api/dashboard/state', timeout=5) as response:
                dashboard = json.load(response)
            assert dashboard['telemetry'] and dashboard['layer6'], dashboard
            assert dashboard['telemetry']['residual_source']['iat'] == 'estimated_physics', dashboard['telemetry']['residual_source']
            assert dashboard['telemetry']['iat_expected'] is not None
            with urllib.request.urlopen(f'http://127.0.0.1:{api_port}/api/pipeline', timeout=5) as response:
                pipeline = json.load(response)
            displayed_risk = pipeline['assessment']['risk_fusion']['engine_risk_score']
            assert 0 < displayed_risk < 30, pipeline
            advisory_request = urllib.request.Request(
                f'http://127.0.0.1:{api_port}/api/dashboard/maintenance-advisory',
                data=b'{}', headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(advisory_request, timeout=5) as response:
                advisory = json.load(response)
            assert advisory['fault_class'] == pipeline['assessment']['lightgbm']['predicted_class']
            assert advisory['suggestions']
            print(json.dumps({'stream_counts': counts,
                              'risk': risks[-1]['engine_risk_score'],
                              'api_risk': displayed_risk,
                              'maintenance_fault_class': advisory['fault_class'],
                              'maintenance_shap_count': len(advisory['shap_evidence']),
                              'action': risks[-1]['recommended_action'],
                              'dashboard_layer6_available': bool(dashboard['layer6'])}, indent=2), flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        client.close()


if __name__ == '__main__':
    main()
