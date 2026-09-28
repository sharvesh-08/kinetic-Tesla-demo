"""Convert raw synthetic telemetry rows into the Layer 5/6 feature contract.

This converter is for development data whose source contains no timestamps or
Layer 4 EKF outputs. It preserves supplied residuals and marks derived residuals
as proxy features in the companion report.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from layer5.registry import FeatureRegistry

CHANNELS = {
    "egt_degC": "egt", "cht_degC": "cht", "map_kPa": "map",
    "fuel_flow_Lph": "fuel_flow", "oil_press_kPa": "oil_press",
    "oil_temp_degC": "oil_temp", "vibration_rms_g": "vibration",
    "batt_volts": "battery_voltage", "alt_amps": "alternator_current",
}
SUPPLIED_RESIDUALS = {
    "egt_degC": "egt_residual", "cht_degC": "cht_residual",
    "oil_press_kPa": "oil_press_residual", "vibration_rms_g": "vibration_residual",
}
LABELS = {
    "ABNORMAL_VIBRATION": "abnormal_vibration", "COMBUSTION_INSTABILITY": "combustion_instability",
    "INJECTOR_ABNORMALITY": "injector_abnormality", "LUBRICATION_ISSUE": "lubrication_issue",
    "MISFIRE": "misfire", "NOMINAL": "nominal", "OVERHEATING_TREND": "overheating_trend",
    "SENSOR_DRIFT_FAILURE": "sensor_drift_failure",
}


def convert(source: Path, output: Path, report: Path, window: int = 500, hop: int = 250) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("source CSV is empty")
    registry = FeatureRegistry()
    values = {channel: [float(row[raw]) for row in rows] for channel, raw in CHANNELS.items()}
    baselines = {channel: statistics.median(series) for channel, series in values.items()}
    scales = {}
    for channel, series in values.items():
        deviations = [abs(value - baselines[channel]) for value in series]
        scales[channel] = max(statistics.median(deviations) * 1.4826, 1e-9)

    feature_names = list(registry.feature_names)
    metadata = ["sortie_id", "engine_serial", "window_start_ns", "window_end_ns",
                "ground_truth_label", "fault_active", "fault_onset_ns", "fault_severity",
                "maintenance_outcome", "feature_schema_hash"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata + feature_names)
        writer.writeheader()
        written = 0
        for start in range(0, len(rows) - window + 1, hop):
            block = rows[start:start + window]
            raw_labels = {LABELS.get(row["label"], row["label"].lower()) for row in block}
            if len(raw_labels) != 1:
                continue  # never train a fault class on a mixed transition window
            label = next(iter(raw_labels))
            residuals = {}
            for channel, raw in CHANNELS.items():
                if channel in SUPPLIED_RESIDUALS:
                    residuals[channel] = [float(row[SUPPLIED_RESIDUALS[channel]]) for row in block]
                else:
                    # Development proxy only: Layer 4 should replace this with EKF residuals.
                    residuals[channel] = [float(row[raw]) - baselines[channel] for row in block]
            features = {}
            for channel in CHANNELS:
                sequence = residuals[channel]
                sample = sequence[::10]
                mean = statistics.fmean(sequence)
                max_abs = max(abs(value) for value in sequence)
                xmean = (len(sequence) - 1) / 2
                denom = sum((index - xmean) ** 2 for index in range(len(sequence)))
                slope = sum((index - xmean) * (value - mean) for index, value in enumerate(sequence)) / denom * 100
                nis = [(value / scales[channel]) ** 2 for value in sequence]
                for index, value in enumerate(sample):
                    features[registry.raw_name(channel, index)] = value
                features[registry.summary_name(channel, "mean")] = mean
                features[registry.summary_name(channel, "max_abs")] = max_abs
                features[registry.summary_name(channel, "slope")] = slope
                features[registry.summary_name(channel, "nis_mean")] = statistics.fmean(nis)
                features[registry.summary_name(channel, "nis_max")] = max(nis)
                features[registry.summary_name(channel, "bias_mean")] = mean
                features[registry.summary_name(channel, "availability")] = 1.0
            start_ns = start * 10_000_000
            record = {"sortie_id": f"synthetic-{label}-{start // hop:05d}",
                      "engine_serial": "synthetic-engine", "window_start_ns": start_ns,
                      "window_end_ns": start_ns + 5_000_000_000, "ground_truth_label": label,
                      "fault_active": str(label != "nominal").lower(), "fault_onset_ns": "",
                      "fault_severity": "", "maintenance_outcome": "healthy" if label == "nominal" else "unknown",
                      "feature_schema_hash": registry.schema_hash(feature_names)}
            record.update(features)
            writer.writerow(record)
            written += 1
    report.write_text(json.dumps({"source_rows": len(rows), "output_windows": written,
                                  "window_samples": window, "hop_samples": hop,
                                  "assumption": "100 Hz rows; supplied four residuals preserved; five residual channels are raw-minus-global-median development proxies",
                                  "proxy_channels": [channel for channel in CHANNELS if channel not in SUPPLIED_RESIDUALS],
                                  "feature_count": len(feature_names), "labels": sorted(LABELS.values())}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    convert(args.source, args.output, args.report)
