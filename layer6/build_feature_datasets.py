"""Build Layer 6 datasets by passing raw observed/truth runs through Layers 4 and 5.

The supplied source is synthetic and has only pre-fault nominal segments; its nominal
windows are candidates, not verified healthy sorties. The output files retain that label.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from layer4.filter import FederatedEKF
from layer5.registry import FeatureRegistry
from layer5.windowing import AdaptiveWindowAssembler

CHANNEL_MAP = {
    "egt_degC": ("obs_exhaust_gas_temp_c", "truth_exhaust_gas_temp_c"),
    "cht_degC": ("obs_cylinder_head_temp_c", "truth_cylinder_head_temp_c"),
    "map_kPa": ("obs_manifold_pressure_kpa", "truth_manifold_pressure_kpa"),
    "fuel_flow_Lph": ("obs_fuel_flow_lph", "truth_fuel_flow_lph"),
    "oil_press_kPa": ("obs_oil_pressure_kpa", "truth_oil_pressure_kpa"),
    "oil_temp_degC": ("obs_oil_temp_c", "truth_oil_temp_c"),
    "vibration_rms_g": ("obs_vibration_g", "truth_vibration_g"),
}


def _layer4_item(row: dict[str, str], ekf: FederatedEKF, run_id: str, healthy_prior: dict[str, float]) -> dict[str, Any]:
    values: dict[str, float] = {}
    predicted: dict[str, float] = {}
    for channel, (observed, truth) in CHANNEL_MAP.items():
        values[channel] = float(row[observed])
        predicted[channel] = healthy_prior[channel]
    timestamp = int(row["master_timestamp_ns"])
    result = ekf.process({
        "master_timestamp_ns": timestamp, "dt_s": 0.01, "values": values, "predicted": predicted,
        "channel_valid": {name: True for name in values},
        "channel_new_sample": {name: True for name in values}, "sync_quality": 1.0,
        "timing_estimated": False,
        "context": {"engine_serial": "SIMULATED", "sortie_id": run_id,
                    "operating_context": {"source": "fault_injected_dataset",
                                          "altitude_m": float(row.get("altitude_m", 0))}},
    }).to_dict()
    for name in ("rpm", "throttle_pct", "altitude_m"):
        if row.get(name) not in (None, ""):
            result.setdefault("telemetry", {})[name] = float(row[name])
    result["ground_truth_label"] = row["fault_class"]
    result["fault_labels"] = row.get("fault_labels") or row["fault_class"]
    result["fault_active"] = row["fault_active"] == "1"
    return result


def _row(window: dict[str, Any], run_id: str, label: str, fault_active: bool,
         maintenance_outcome: str, fault_labels: str = "") -> dict[str, Any]:
    return {
        "sortie_id": run_id, "engine_serial": "SIMULATED",
        "window_start_ns": int(window["window_start_ns"]), "window_end_ns": int(window["window_end_ns"]),
        "ground_truth_label": label, "fault_active": fault_active,
        "fault_labels": fault_labels,
        "maintenance_outcome": maintenance_outcome,
        "feature_schema_hash": window["feature_schema_hash"], "features": window["features"],
        "feature_valid": window["feature_valid"],
    }


def build(source: Path, *, require_all_classes: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry = FeatureRegistry()
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[row["run_id"]].append(row)
    classified: list[dict[str, Any]] = []
    nominal_candidates: list[dict[str, Any]] = []
    for run_id, rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row["master_timestamp_ns"]))
        onset = int(rows[0]["fault_start_timestamp_ns"])
        fault_class = rows[0]["fault_class"]
        prefix = [row for row in rows if int(row["master_timestamp_ns"]) < onset and row["fault_active"] == "0"]
        if not prefix:
            raise ValueError(f"No pre-fault baseline for {run_id}")
        # Source truth is fault-affected too. The baseline uses pre-onset values
        # only; this is a development healthy prior, not the deployed PINN.
        healthy_prior = {channel: statistics.median(float(row[truth]) for row in prefix)
                         for channel, (_, truth) in CHANNEL_MAP.items()}
        # Bootstrap contiguous pre-fault blocks into a full-duration nominal
        # simulation. Avoid label-dependent fake missing tails. Keep the source
        # run ID so all derived windows stay together in the held-out split.
        rng = random.Random(run_id)
        nominal_ekf = FederatedEKF()
        nominal_assembler = AdaptiveWindowAssembler(registry=registry)
        block_start = 0
        for index, source_row in enumerate(rows):
            if index % 25 == 0:
                block_start = rng.randrange(max(1, len(prefix) - 24))
            simulated = dict(prefix[min(block_start + index % 25, len(prefix) - 1)])
            simulated["master_timestamp_ns"] = source_row["master_timestamp_ns"]
            simulated["fault_class"] = "nominal"
            simulated["fault_active"] = "0"
            for window in nominal_assembler.process(_layer4_item(simulated, nominal_ekf, run_id, healthy_prior)):
                if window.features["history_seconds_available_60s"] >= 60.0:
                    nominal_candidates.append(_row(window.to_dict(), run_id, "nominal", False,
                                                   "synthetic_prefault_block_replay"))

        ekf = FederatedEKF()
        assembler = AdaptiveWindowAssembler(registry=registry)
        for row in rows:
            output = assembler.process(_layer4_item(row, ekf, run_id, healthy_prior))
            for window in output:
                # Both the 5-second and 60-second blocks must be wholly after fault onset.
                if int(window.window_start_ns) >= onset + 60_000_000_000:
                    classified.append(_row(window.to_dict(), run_id, fault_class, True,
                                           "fault_injected_source", rows[0].get("fault_labels") or fault_class))
    if not classified or not nominal_candidates:
        raise RuntimeError("Source did not yield both post-onset fault windows and nominal candidates")
    classes = {row["ground_truth_label"] for row in classified}
    if require_all_classes and len(classes) != 8:
        raise RuntimeError(f"Expected all 8 fault classes after Layer 5 assembly, received {sorted(classes)}")
    if not all(row["features"]["history_seconds_available_60s"] >= 60.0 for row in classified):
        raise RuntimeError("Source runs are too short to train the complete 60-second fault feature block")
    training_rows = classified + nominal_candidates
    if require_all_classes and {row["ground_truth_label"] for row in training_rows} != {"nominal", "misfire", "injector_abnormality", "cooling_degradation",
            "lubrication_issue", "sensor_drift_failure", "combustion_instability",
            "overheating_trend", "abnormal_vibration"}:
        raise RuntimeError("Layer 5 training data does not contain nominal and all required fault labels")
    return training_rows, nominal_candidates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = FeatureRegistry()
    columns = ["sortie_id", "engine_serial", "window_start_ns", "window_end_ns", "ground_truth_label", "fault_labels",
               "fault_active", "maintenance_outcome", "feature_schema_hash", *registry.feature_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in columns if key in row},
                             **row["features"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--nominal-output", type=Path, required=True)
    parser.add_argument("--allow-partial-class-coverage", action="store_true",
                        help="For held-out scenario evaluation only; training still requires all eight classes")
    args = parser.parse_args()
    training, nominal = build(args.source, require_all_classes=not args.allow_partial_class_coverage)
    write_csv(args.training_output, training)
    write_csv(args.nominal_output, nominal)
    print(json.dumps({"training_windows": len(training), "fault_classes": sorted({r["ground_truth_label"] for r in training}),
                      "nominal_candidate_windows": len(nominal), "nominal_provenance": "synthetic block replay of pre-fault measurements; constant pre-fault prior; electrical channels absent; not validated on deployed PINNs",
                      "schema": training[0]["feature_schema_hash"]}, indent=2))


if __name__ == "__main__":
    main()
