"""Build Layer 5 v3 development windows from the supplied native-unit CSVs.

The source files have no timestamps, sortie IDs, or verified unit contract. Each
trace below is therefore a new, explicitly synthetic trajectory anchored to
empirical source rows. All frames go through the real Layer 4 EKF and Layer 5
window assembler. Outputs are development data only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from layer4.filter import FederatedEKF
from layer5.registry import FeatureRegistry
from layer5.windowing import AdaptiveWindowAssembler

RATE_HZ = 100
DT_NS = 10_000_000
WINDOW_S = 5
HOP_S = 2.5
HISTORY_S = 60
DURATION_S = 75
ONSET_S = 5
FAULT_CLASSES = (
    "misfire", "injector_abnormality", "cooling_degradation", "lubrication_issue",
    "sensor_drift_failure", "combustion_instability", "overheating_trend", "abnormal_vibration",
)
SOURCE_LABELS = {
    "MISFIRE": "misfire", "INJECTOR_ABNORMALITY": "injector_abnormality",
    "LUBRICATION_ISSUE": "lubrication_issue", "SENSOR_DRIFT_FAILURE": "sensor_drift_failure",
    "COMBUSTION_INSTABILITY": "combustion_instability", "OVERHEATING_TREND": "overheating_trend",
    "ABNORMAL_VIBRATION": "abnormal_vibration", "NOMINAL": "nominal",
    "NORMAL": "nominal",
}
CHANNEL_COLUMNS = {
    "egt_degC": "egt", "cht_degC": "cht", "map_kPa": "map",
    "fuel_flow_Lph": "fuel_flow", "oil_press_kPa": "oil_press",
    "oil_temp_degC": "oil_temp", "vibration_rms_g": "vibration",
    "batt_volts": "battery_voltage", "alt_amps": "alternator_current",
}
CONTEXT_COLUMNS = ("rpm", "throttle", "engine_load", "altitude", "iat")
SOURCE_COLUMNS = (
    "rpm", "throttle", "engine_load", "egt", "cht", "oil_temp", "map", "oil_press",
    "fuel_flow", "iat", "vibration", "battery_voltage", "alternator_current",
    "electrical_health", "injection_timing", "injection_duration", "fuel_pressure",
    "altitude", "baro_pressure", "air_density_ratio", "egt_residual", "cht_residual",
    "oil_press_residual", "vibration_residual", "label",
)


def read_source(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
        if first.startswith("version https://git-lfs.github.com/spec/"):
            raise ValueError(f"{path} is a Git LFS pointer, not a hydrated CSV")
        handle.seek(0)
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{path} must be the supplied sensor CSV with a 'label' column")
        rows: list[dict[str, Any]] = []
        for row_num, raw in enumerate(reader, start=2):
            if not raw.get("label"):
                continue
            row: dict[str, Any] = {"label": str(raw["label"]).strip().upper(), "source_row_id": row_num}
            for column in SOURCE_COLUMNS:
                if column in {"label", "egt_residual", "cht_residual", "oil_press_residual", "vibration_residual"}:
                    continue
                value = raw.get(column)
                if value in (None, ""):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                if math.isfinite(number):
                    row[column] = number
            if all(column in row for column in CHANNEL_COLUMNS.values()) and all(column in row for column in CONTEXT_COLUMNS):
                rows.append(row)
    if not rows:
        raise ValueError(f"No usable sensor rows found in {path}")
    return rows


def nearest_nominal(anchor: dict[str, Any], candidates: list[dict[str, Any]],
                    scales: dict[str, float]) -> dict[str, Any]:
    keys = ("rpm", "throttle", "engine_load", "altitude")
    def distance(row: dict[str, Any]) -> float:
        return sum(((float(row[k]) - float(anchor[k])) / scales[k]) ** 2 for k in keys)
    return min(candidates, key=distance)


def regime(row: dict[str, Any]) -> str:
    if float(row["altitude"]) > 1000:
        return "high_altitude_native"
    if float(row["engine_load"]) >= 80 or float(row["throttle"]) >= 0.8:
        return "high_power_native"
    if float(row["rpm"]) < 1800:
        return "low_power_native"
    return "cruise_native"


def subsystem(fault: str) -> str:
    return {
        "misfire": "combustion", "injector_abnormality": "air_path",
        "cooling_degradation": "thermal", "lubrication_issue": "lubrication",
        "sensor_drift_failure": "sensor", "combustion_instability": "combustion",
        "overheating_trend": "thermal", "abnormal_vibration": "vibration",
    }.get(fault, "none")


def choose_severity(rng: random.Random) -> float:
    band = rng.choice(((0.05, 0.25), (0.25, 0.60), (0.60, 1.00)))
    return rng.uniform(*band)


def trace_frames(*, run_id: str, sortie_id: str, engine_serial: str,
                 label: str, anchor: dict[str, Any], nominal: dict[str, Any],
                 severity: float, run_start_ns: int, rng: random.Random,
                 native_scales: dict[str, float]) -> Any:
    ekf = FederatedEKF()
    assembler = AdaptiveWindowAssembler()
    onset_ns = run_start_ns + ONSET_S * RATE_HZ * DT_NS if label != "nominal" else 0
    class_delta = {channel: float(anchor[src]) - float(nominal[src])
                   for channel, src in CHANNEL_COLUMNS.items()}
    if label == "cooling_degradation":
        # This class is absent from the user's fault CSV. Create a transparent,
        # native-scale cooling signature from nominal source variability.
        class_delta = {name: 0.0 for name in CHANNEL_COLUMNS}
        class_delta["cht_degC"] = 2.0 * native_scales["cht"]
        class_delta["oil_temp_degC"] = 1.2 * native_scales["oil_temp"]
    phases = {name: rng.uniform(0, 2 * math.pi) for name in CHANNEL_COLUMNS}
    previous_noise = {name: 0.0 for name in CHANNEL_COLUMNS}
    for sample in range(DURATION_S * RATE_HZ):
        timestamp_ns = run_start_ns + sample * DT_NS
        t_s = sample / RATE_HZ
        elapsed = t_s - ONSET_S
        active = label != "nominal" and elapsed >= 0
        ramp = max(0.0, min(1.0, elapsed / 15.0)) if active else 0.0
        observed: dict[str, float] = {}
        predicted: dict[str, float] = {}
        for channel, source_column in CHANNEL_COLUMNS.items():
            center = float(nominal[source_column])
            amp = max(abs(center) * 0.002, native_scales[source_column] * 0.03, 1e-6)
            oscillation = amp * math.sin((2.0 * math.pi * t_s / 37.0) + phases[channel])
            previous_noise[channel] = 0.96 * previous_noise[channel] + rng.gauss(0.0, amp * 0.08)
            healthy_prediction = center + oscillation + previous_noise[channel]
            effect = class_delta[channel] * severity * ramp
            if label == "misfire" and active and channel in {"egt_degC", "vibration_rms_g"}:
                pulse = 1.0 if int(elapsed * 3) % 5 == 0 else 0.0
                effect += (-2.0 if channel == "egt_degC" else 2.0) * native_scales[source_column] * severity * pulse
            if label == "combustion_instability" and active and channel in {"egt_degC", "vibration_rms_g"}:
                effect += native_scales[source_column] * severity * 0.5 * math.sin(elapsed * 2.3)
            predicted[channel] = healthy_prediction
            observed[channel] = healthy_prediction + effect
        values = {
            "rpm": float(nominal["rpm"]), "throttle_pct": float(nominal["throttle"]),
            "altitude_m": float(nominal["altitude"]), **observed,
        }
        envelope = {
            "master_timestamp_ns": timestamp_ns, "dt_s": 0.01, "values": values,
            "predicted": predicted, "channel_valid": {key: True for key in CHANNEL_COLUMNS},
            "channel_new_sample": {key: True for key in CHANNEL_COLUMNS},
            "sync_quality": 1.0, "timing_estimated": True,
            "context": {"engine_serial": engine_serial, "sortie_id": sortie_id,
                        "operating_context": {"altitude_m": float(nominal["altitude"]),
                                              "ambient_temperature_native": float(nominal["iat"]),
                                              "power_band": "high" if float(nominal["engine_load"]) >= 80 else "medium" if float(nominal["engine_load"]) >= 50 else "low",
                                              "regime_source": "inferred_from_native_source_values",
                                              "data_status": "development_data_units_unverified"}},
            "ground_truth_label": label, "fault_labels": "" if label == "nominal" else label,
            "fault_active": active, "fault_onset_ns": onset_ns,
            "fault_severity": severity if label != "nominal" else 0.0,
            "operating_regime": regime(nominal), "ambient_temperature_native": float(nominal["iat"]),
            "altitude_native": float(nominal["altitude"]),
            "power_band": "high" if float(nominal["engine_load"]) >= 80 else "medium" if float(nominal["engine_load"]) >= 50 else "low",
            "engine_serial": engine_serial, "sortie_id": sortie_id, "run_id": run_id,
            "source_anchor_row_id": int(anchor["source_row_id"]),
            "source_nominal_row_id": int(nominal["source_row_id"]),
        }
        ekf_output = ekf.process(envelope).to_dict()
        ekf_output.update({key: envelope[key] for key in (
            "ground_truth_label", "fault_labels", "fault_active", "fault_onset_ns", "fault_severity",
            "operating_regime", "ambient_temperature_native", "altitude_native", "power_band",
            "engine_serial", "sortie_id", "run_id", "source_anchor_row_id", "source_nominal_row_id",
        )})
        # FeatureWindow metadata uses telemetry/context from its source envelope.
        ekf_output["telemetry"].update({"rpm": values["rpm"], "throttle_pct": values["throttle_pct"],
                                        "altitude_m": values["altitude_m"]})
        for window in assembler.process(ekf_output):
            # Fault rows need 60 s after onset before their 5 s window begins.
            if label != "nominal" and window.window_start_ns < onset_ns + HISTORY_S * 1_000_000_000:
                continue
            if label == "nominal" and window.features["history_seconds_available_60s"] < HISTORY_S:
                continue
            yield window, envelope


def write_dataset(path: Path, rows: list[dict[str, Any]], registry: FeatureRegistry) -> None:
    metadata = [
        "engine_serial", "sortie_id", "run_id", "window_start_ns", "window_end_ns",
        "ground_truth_label", "fault_class", "fault_labels", "fault_active", "fault_onset_ns",
        "fault_severity", "subsystem", "operating_regime", "ambient_temperature_native",
        "altitude_native", "power_band", "maintenance_outcome", "data_status",
        "source_kind", "source_anchor_row_id", "source_nominal_row_id", "feature_schema_version",
        "feature_schema_hash", "feature_valid_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata + list(registry.feature_names), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **row["features"]})


def data_dictionary(path: Path) -> None:
    units = {
        "label": "categorical label; source spelling preserved in provenance",
        "egt_residual": "unknown; source column intentionally excluded from generated Layer 4 features",
        "cht_residual": "unknown; source column intentionally excluded from generated Layer 4 features",
        "oil_press_residual": "unknown; source column intentionally excluded from generated Layer 4 features",
        "vibration_residual": "unknown; source column intentionally excluded from generated Layer 4 features",
    }
    channel_cols = set(CHANNEL_COLUMNS.values())
    for column in SOURCE_COLUMNS:
        if column not in units:
            units[column] = "unknown (synthetic native unit; no conversion applied)" if column in channel_cols | set(CONTEXT_COLUMNS) else "unknown (copied only as source provenance/context where used)"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_column", "native_unit", "use_in_generation"])
        writer.writeheader()
        for column in SOURCE_COLUMNS:
            use = "used as Layer 4 observed/predicted source anchor; no conversion" if column in channel_cols else "used as trace context" if column in CONTEXT_COLUMNS else "not used as Layer 5 model feature source"
            writer.writerow({"source_column": column, "native_unit": units[column], "use_in_generation": use})


def build(args: argparse.Namespace) -> dict[str, Any]:
    fault_rows = read_source(args.fault_csv)
    healthy_rows = read_source(args.nominal_csv)
    nominal_rows = [row for row in healthy_rows if SOURCE_LABELS.get(row["label"]) == "nominal"]
    nominal_rows.extend(row for row in fault_rows if SOURCE_LABELS.get(row["label"]) == "nominal")
    if not nominal_rows:
        raise ValueError("No nominal source rows found")
    by_fault: dict[str, list[dict[str, Any]]] = {fault: [] for fault in FAULT_CLASSES if fault != "cooling_degradation"}
    for row in fault_rows:
        label = SOURCE_LABELS.get(row["label"])
        if label in by_fault:
            by_fault[label].append(row)
    absent = [fault for fault in by_fault if not by_fault[fault]]
    if absent:
        raise ValueError(f"No source records for required classes: {absent}")
    numerical_context = ("rpm", "throttle", "engine_load", "altitude")
    scales = {key: max(statistics.pstdev(float(row[key]) for row in nominal_rows), 1e-6) for key in numerical_context}
    channel_scales = {key: max(statistics.pstdev(float(row[key]) for row in nominal_rows), 1e-6)
                      for key in set(CHANNEL_COLUMNS.values())}
    registry = FeatureRegistry()
    rng = random.Random(args.seed)
    lightgbm_rows: list[dict[str, Any]] = []
    isolation_candidates: list[dict[str, Any]] = []
    run_counter = 0
    fault_counts: dict[str, int] = {}

    def consume_trace(label: str, anchor: dict[str, Any], nominal: dict[str, Any], replica: int,
                      source_kind: str) -> int:
        nonlocal run_counter
        run_counter += 1
        trace_id = f"SRCDEV-{label}-{replica:03d}-{run_counter:05d}"
        engine_serial = f"SYNTH-ENGINE-{(run_counter - 1) % args.synthetic_engines + 1:03d}"
        run_start = args.epoch_ns + run_counter * 1_000_000_000_000
        severity = choose_severity(rng) if label != "nominal" else 0.0
        trace_rng = random.Random(args.seed + run_counter * 7919)
        produced = 0
        for window, source_envelope in trace_frames(
            run_id=trace_id, sortie_id=trace_id, engine_serial=engine_serial, label=label,
            anchor=anchor, nominal=nominal, severity=severity, run_start_ns=run_start,
            rng=trace_rng, native_scales=channel_scales,
        ):
            fault_active = label != "nominal"
            row = {
                "engine_serial": engine_serial, "sortie_id": trace_id, "run_id": trace_id,
                "window_start_ns": int(window.window_start_ns), "window_end_ns": int(window.window_end_ns),
                "ground_truth_label": label, "fault_class": label,
                "fault_labels": "" if label == "nominal" else label, "fault_active": str(fault_active).lower(),
                "fault_onset_ns": int(run_start + ONSET_S * 1_000_000_000) if fault_active else "",
                "fault_severity": severity, "subsystem": subsystem(label),
                "operating_regime": regime(nominal),
                "ambient_temperature_native": float(nominal["iat"]), "altitude_native": float(nominal["altitude"]),
                "power_band": "high" if float(nominal["engine_load"]) >= 80 else "medium" if float(nominal["engine_load"]) >= 50 else "low",
                "maintenance_outcome": "synthetic_fault_injected_unverified" if fault_active else "unverified_nominal_candidate",
                "data_status": "development_data_units_unverified", "source_kind": source_kind,
                "source_anchor_row_id": int(anchor["source_row_id"]),
                "source_nominal_row_id": int(nominal["source_row_id"]),
                "feature_schema_version": window.feature_schema_version,
                "feature_schema_hash": window.feature_schema_hash,
                "feature_valid_json": json.dumps(window.feature_valid, separators=(",", ":")),
                "features": window.features,
            }
            lightgbm_rows.append(row)
            if not fault_active:
                isolation_candidates.append(row)
            produced += 1
        return produced

    # Source fault classes: each generated trace has an independent ID and anchor.
    for fault in FAULT_CLASSES:
        for replica in range(args.runs_per_class):
            if fault == "cooling_degradation":
                anchor = rng.choice(nominal_rows)
                baseline = nearest_nominal(anchor, nominal_rows, scales)
                source_kind = "synthetic_cooling_injection_on_nominal_source_row"
            else:
                anchor = rng.choice(by_fault[fault])
                baseline = nearest_nominal(anchor, nominal_rows, scales)
                source_kind = "source_conditioned_fault_class_empirical_anchor"
            count = consume_trace(fault, anchor, baseline, replica, source_kind)
            fault_counts[fault] = fault_counts.get(fault, 0) + count
    # Separate unverified nominal candidate traces; they are not healthy-certified.
    for replica in range(args.nominal_runs):
        anchor = rng.choice(nominal_rows)
        baseline = nearest_nominal(anchor, nominal_rows, scales)
        consume_trace("nominal", anchor, baseline, replica, "unverified_nominal_source_candidate")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_dataset(out / "lightgbm_layer5_v3_development.csv", lightgbm_rows, registry)
    write_dataset(out / "isolation_forest_layer5_v3_unverified_nominal_candidates.csv", isolation_candidates, registry)
    data_dictionary(out / "source_data_dictionary.csv")
    report = {
        "data_status": "development_data_units_unverified",
        "feature_schema_version": registry.version,
        "feature_count": len(registry.feature_names),
        "feature_schema_hash": registry.schema_hash(list(registry.feature_names)),
        "rate_hz": RATE_HZ, "sample_period_ns": DT_NS, "window_seconds": WINDOW_S,
        "hop_seconds": HOP_S, "history_seconds": HISTORY_S,
        "duration_seconds": DURATION_S, "fault_onset_seconds": ONSET_S,
        "fault_windows_by_class": fault_counts,
        "lightgbm_development_rows": len(lightgbm_rows),
        "nominal_candidate_rows": len(isolation_candidates),
        "nominal_training_eligibility": "not verified; do not use as production Isolation Forest population",
        "source_fault_class_absent": "cooling_degradation; those scenarios are injected from nominal source rows and tagged synthetic",
        "residual_columns_used_as_features": False,
        "source_fault_rows": sum(len(rows) for rows in by_fault.values()),
        "source_nominal_rows": len(nominal_rows),
        "synthetic_trace_count": run_counter,
        "run_id_policy": "unique per independently generated trace; no window crosses a trace",
    }
    (out / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault-csv", type=Path, required=True)
    parser.add_argument("--nominal-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/layer6/source_conditioned_v3"))
    parser.add_argument("--runs-per-class", type=int, default=5)
    parser.add_argument("--nominal-runs", type=int, default=20)
    parser.add_argument("--synthetic-engines", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-ns", type=int, default=1_800_000_000_000_000_000)
    args = parser.parse_args()
    if args.runs_per_class < 3 or args.nominal_runs < 1 or args.synthetic_engines < 1:
        parser.error("Use at least 3 independent runs per class, 1 nominal run, and 1 synthetic engine")
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
