"""Generate labeled *development* windows from the updated MALE engine.

Uses the real simulator, telemetry adapter, Layer 3 ONNX models, Layer 4 EKF,
and Layer 5 v4 feature registry. Synthetic fault adjustments are explicitly
marked and are not evidence of flight diagnostic performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT if (ROOT / "app" / "simulator.py").is_file() else ROOT / "MALE-UAV-DESIGN"))

from app.simulator import EngineSimulator
from layer1.adapter import to_layer1_events
from layer3.inference import PhysicsInference
from layer4.filter import FederatedEKF
from layer5.registry import FeatureRegistry
from layer5.config import load_config as load_windowing_config
from layer5.windowing import AdaptiveWindowAssembler
from layer6.model import CLASSES


def perturb(snapshot: dict, fault: str, progress: float) -> dict:
    """Fault-conditioned sensor scenarios layered on the existing simulator."""
    t = dict(snapshot)
    if fault == "misfire":
        t.update(vibration=t["vibration"] + 2.5, egt=t["egt"] - 40,
                 rpm=t["rpm"] - 150, injection_timing=t["injection_timing"] - 4)
    elif fault == "injector_abnormality":
        t.update(fuel_pressure=max(5, t["fuel_pressure"] - 20),
                 injection_duration=max(1, t["injection_duration"] - 1.1),
                 fuel_flow=max(0, t["fuel_flow"] - 2.5), egt=t["egt"] + 35)
    elif fault == "cooling_degradation":
        t.update(cht=t["cht"] + 10 + 45*progress,
                 oil_temp=t["oil_temp"] + 10 + 16*progress)
    elif fault == "lubrication_issue":
        t.update(oil_press=max(0, t["oil_press"] - 18 - 12*progress),
                 oil_temp=t["oil_temp"] + 8 + 17*progress)
    elif fault == "sensor_drift_failure":
        t["egt"] += 40*progress
    elif fault == "combustion_instability":
        t.update(egt=t["egt"] + 30*math.sin(snapshot["timestamp"]*2),
                 vibration=t["vibration"] + 1.2 + .3*math.sin(snapshot["timestamp"]*3),
                 injection_timing=t["injection_timing"] + 3*math.sin(snapshot["timestamp"]*2))
    elif fault == "overheating_trend":
        t.update(egt=t["egt"] + 25 + 80*progress,
                 cht=t["cht"] + 5 + 35*progress,
                 oil_temp=t["oil_temp"] + 5 + 12*progress)
    elif fault == "abnormal_vibration":
        t["vibration"] += 1.5 + 2*progress
    return t


def rows_for_sortie(fault: str, index: int, seconds: float, registry: FeatureRegistry):
    sortie_id = f"SYN-{fault}-{index:03d}"
    engine_serial = f"SYN-{index:03d}"
    sim = EngineSimulator(dt=.5, seed=4242+index*37+list(CLASSES).index(fault)*997)
    physics, ekf = PhysicsInference(), FederatedEKF()
    windows = AdaptiveWindowAssembler(config=replace(load_windowing_config(), schema_version=registry.version), registry=registry)
    start_ns = 1_800_000_000_000_000_000 + (list(CLASSES).index(fault)*100+index)*1_000_000_000_000
    result = []
    for tick in range(int(seconds / sim.dt)):
        original = sim.step()
        adjusted = perturb(original, fault, (tick+1)/(seconds/sim.dt))
        adjusted["sensor_timestamp_ns"] = str(start_ns + tick*500_000_000)
        adjusted["engine_serial"] = engine_serial
        adjusted["sortie_id"] = sortie_id
        adjusted["operating_hours"] = adjusted["timestamp"] / 3600.0
        events = to_layer1_events(adjusted, sample_rate_hz=2,
                                  session_id=sortie_id)
        values = {event["sensor"]: event["value"] for event in events}
        frame = {"master_timestamp_ns": adjusted["sensor_timestamp_ns"],
                 "values": values,
                 "channel_valid": {name: True for name in values},
                 "channel_new_sample": {name: True for name in values},
                 "sync_quality": 1., "timing_estimated": False,
                 "context": events[0]["context"]}
        assessment_input = ekf.process(physics.process(frame)).to_dict()
        for window in windows.process(assessment_input):
            if window.features["history_seconds_available_60s"] < 60:
                continue
            result.append({
                "sortie_id": sortie_id, "engine_serial": engine_serial,
                "run_id": sortie_id, "ground_truth_label": fault,
                "fault_class": fault, "fault_active": fault != "nominal",
                "fault_labels": "" if fault == "nominal" else fault,
                "fault_onset_ns": start_ns, "fault_severity": 1.,
                "window_start_ns": window.window_start_ns,
                "window_end_ns": window.window_end_ns,
                "feature_schema_hash": window.feature_schema_hash,
                "feature_schema_version": window.feature_schema_version,
                "source_kind": "updated_male_simulator_plus_explicit_synthetic_fault",
                "data_status": "synthetic_development_unverified",
                "healthy_verified": False,
                **window.features,
            })
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"data/layer6/male_uav_v4")
    parser.add_argument("--runs-per-class", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=75)
    args = parser.parse_args()
    if args.runs_per_class < 3 or args.seconds < 65:
        raise ValueError("Need at least three independent sorties per class and 65 seconds per sortie")
    registry = FeatureRegistry(ROOT/"configs/feature_registry_v4.yaml")
    groups = {name: [rows_for_sortie(name, i, args.seconds, registry)
                     for i in range(args.runs_per_class)] for name in CLASSES}
    training = [row for runs in groups.values() for run in runs[:-1] for row in run]
    holdout = [row for runs in groups.values() for row in runs[-1]]
    nominal = [row for run in groups["nominal"][:-1] for row in run]
    fields = [name for name in training[0] if name not in registry.feature_names] + list(registry.feature_names)
    write_csv(args.output/"train.csv", training, fields)
    write_csv(args.output/"holdout.csv", holdout, fields)
    write_csv(args.output/"nominal_unverified.csv", nominal, fields)
    report = {"schema_version": registry.version, "feature_count": len(registry.feature_names),
              "feature_schema_hash": registry.schema_hash(registry.feature_names),
              "training_rows": len(training), "holdout_rows": len(holdout),
              "nominal_candidates": len(nominal), "class_counts": dict(Counter(r["fault_class"] for r in training)),
              "provenance": "simulator-derived, synthetic fault adjustments, not flight validated",
              "splits": "last independent sortie of each class held out; training tool separates calibration by sortie"}
    (args.output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
