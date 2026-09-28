"""Pattern-preserving joint augmentation of Layer 5 v3 window records.

This expands existing processed windows for development experiments only. It
does not create new independent aircraft sorties or verified healthy runs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from layer5.registry import FeatureRegistry

FAULT_CLASSES = (
    "misfire", "injector_abnormality", "cooling_degradation", "lubrication_issue",
    "sensor_drift_failure", "combustion_instability", "overheating_trend", "abnormal_vibration",
)
META_COLUMNS = [
    "engine_serial", "sortie_id", "run_id", "window_start_ns", "window_end_ns",
    "ground_truth_label", "fault_class", "fault_labels", "fault_active", "fault_onset_ns",
    "fault_severity", "subsystem", "operating_regime", "ambient_temperature_native",
    "altitude_native", "power_band", "maintenance_outcome", "data_status", "source_kind",
    "source_anchor_row_id", "source_nominal_row_id", "source_parent_run_id",
    "augmentation_method", "healthy_verified", "feature_schema_version", "feature_schema_hash",
    "feature_valid_json",
]


def _read(path: Path, registry: FeatureRegistry) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        features = [name for name in reader.fieldnames if name not in META_COLUMNS]
        rows: list[dict[str, Any]] = []
        for raw in reader:
            if raw.get("feature_schema_version") != registry.version:
                raise ValueError(f"{path} contains a non-v3 feature window")
            registry.require_schema(features, raw["feature_schema_hash"])
            record = {key: raw.get(key, "") for key in META_COLUMNS}
            record["features"] = {name: float(raw[name]) for name in features}
            record["feature_valid_json"] = raw.get("feature_valid_json", "{}")
            rows.append(record)
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows, features


def _slope(values: list[float], sample_period_s: float) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) * sample_period_s / 2.0
    mean_y = statistics.fmean(values)
    numerator = sum(((i * sample_period_s) - mean_x) * (value - mean_y) for i, value in enumerate(values))
    denominator = sum(((i * sample_period_s) - mean_x) ** 2 for i in range(n))
    return numerator / denominator if denominator else 0.0


def _augment_features(source: dict[str, float], label: str, severity: float,
                      nominal_scales: dict[str, float], rng: random.Random,
                      registry: FeatureRegistry, donor_severity: float) -> dict[str, float]:
    result = dict(source)
    # One shared latent term and a small channel term perturb complete residual
    # trajectories together; raw positions are never shuffled independently.
    shared = rng.gauss(0.0, 1.0)
    for channel in registry.channels:
        raw_names = [registry.raw_name(channel, i) for i in range(registry.RAW_SAMPLES)]
        raw = [float(source[name]) for name in raw_names]
        channel_scale = max(abs(float(source[registry.summary_name(channel, "mean")])),
                            float(source[registry.summary_name(channel, "max_abs")]), 1e-6)
        severity_gain = (severity / max(donor_severity, 0.05)) if label != "nominal" else 1.0
        gain = min(1.25, max(0.75, severity_gain)) * rng.uniform(0.96, 1.04)
        offset = shared * channel_scale * 0.008 + rng.gauss(0.0, channel_scale * 0.002)
        drift = rng.gauss(0.0, channel_scale * 0.001)
        augmented = [value * gain + offset + drift * ((i - 24.5) / 24.5) for i, value in enumerate(raw)]
        for name, value in zip(raw_names, augmented, strict=True):
            result[name] = value
        result[registry.summary_name(channel, "mean")] = statistics.fmean(augmented)
        result[registry.summary_name(channel, "max_abs")] = max(abs(value) for value in augmented)
        result[registry.summary_name(channel, "slope")] = _slope(augmented, 0.1)
        result[registry.summary_name(channel, "nis_mean")] = max(0.0, float(source[registry.summary_name(channel, "nis_mean")]) * gain * gain)
        result[registry.summary_name(channel, "nis_max")] = max(0.0, float(source[registry.summary_name(channel, "nis_max")]) * gain * gain)
        result[registry.summary_name(channel, "bias_mean")] = float(source[registry.summary_name(channel, "bias_mean")]) * gain + offset
        result[registry.summary_name(channel, "availability")] = float(source[registry.summary_name(channel, "availability")])
        if channel in registry.long_channels:
            for statistic in registry.long_summaries:
                name = registry.long_name(channel, statistic)
                old = float(source[name])
                if statistic == "mean":
                    result[name] = old * gain + offset
                elif statistic == "slope":
                    result[name] = old * gain + drift / 60.0
                elif statistic == "nis_mean" or statistic == "nis_max":
                    result[name] = max(0.0, old * gain * gain)
                elif statistic == "bias_mean":
                    result[name] = old * gain + offset
                else:
                    result[name] = old

    # Add the absent cooling class by shifting paired thermal residual histories
    # together, with a slow 60-second trend. Scale comes from nominal-window
    # Layer 5 residual variation, never from an invented column value.
    if label == "cooling_degradation":
        for channel, multiplier in (("cht_degC", 1.0), ("oil_temp_degC", 0.65)):
            amplitude = nominal_scales[channel] * severity * multiplier
            raw_names = [registry.raw_name(channel, i) for i in range(registry.RAW_SAMPLES)]
            augmented = [result[name] + amplitude * (0.35 + 0.65 * i / max(len(raw_names) - 1, 1))
                         for i, name in enumerate(raw_names)]
            for name, value in zip(raw_names, augmented, strict=True):
                result[name] = value
            result[registry.summary_name(channel, "mean")] = statistics.fmean(augmented)
            result[registry.summary_name(channel, "max_abs")] = max(abs(value) for value in augmented)
            result[registry.summary_name(channel, "slope")] = _slope(augmented, 0.1)
            long_mean = registry.long_name(channel, "mean")
            long_slope = registry.long_name(channel, "slope")
            long_nis = registry.long_name(channel, "nis_mean")
            nis_delta = (amplitude / max(nominal_scales[channel], 1e-6)) ** 2 * 0.25
            result[registry.summary_name(channel, "nis_mean")] += nis_delta
            result[registry.summary_name(channel, "nis_max")] += 2.0 * nis_delta
            result[long_mean] += amplitude * 0.55
            result[long_slope] += amplitude / 60.0
            result[long_nis] += nis_delta
    return result


def _nominal_channel_scales(rows: list[dict[str, Any]], registry: FeatureRegistry) -> dict[str, float]:
    scales: dict[str, float] = {}
    for channel in ("cht_degC", "oil_temp_degC"):
        means = [abs(float(row["features"][registry.summary_name(channel, "mean")])) for row in rows]
        max_abs = [float(row["features"][registry.summary_name(channel, "max_abs")]) for row in rows]
        scale = max(statistics.median(max_abs), statistics.pstdev(means), 1e-3)
        scales[channel] = scale
    return scales


def _target_counts(total: int, nominal_count: int, fault_count: int) -> dict[str, int]:
    classes = ["nominal", *FAULT_CLASSES]
    weights = {"nominal": nominal_count, **{fault: fault_count for fault in FAULT_CLASSES}}
    weight_sum = sum(weights.values())
    exact = {key: total * value / weight_sum for key, value in weights.items()}
    counts = {key: math.floor(value) for key, value in exact.items()}
    left = total - sum(counts.values())
    for key in sorted(classes, key=lambda item: exact[item] - counts[item], reverse=True)[:left]:
        counts[key] += 1
    return counts


def _build_file(rows_by_class: dict[str, list[dict[str, Any]]], output: Path,
                target_counts: dict[str, int], registry: FeatureRegistry,
                features: list[str], rng: random.Random, prefix: str,
                nominal_scales: dict[str, float], nominal_only: bool = False) -> dict[str, int]:
    if any(not rows_by_class.get(label) for label in target_counts):
        missing = [label for label in target_counts if not rows_by_class.get(label)]
        raise ValueError(f"Missing donor windows for classes: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    header = META_COLUMNS + features
    counts: dict[str, int] = defaultdict(int)
    epoch = 1_800_000_000_000_000_000
    opener = gzip.open if output.suffix == ".gz" else open
    with opener(output, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        row_index = 0
        for label, count in target_counts.items():
            for _ in range(count):
                donor = rng.choice(rows_by_class[label])
                row_index += 1
                row = {key: donor.get(key, "") for key in META_COLUMNS}
                original_severity = float(donor.get("fault_severity") or 0.0)
                if label == "nominal":
                    severity = 0.0
                else:
                    # Keep the donor's severity band and modestly vary within it.
                    severity = min(1.0, max(0.05, original_severity * rng.uniform(0.92, 1.08)))
                feature_values = _augment_features(donor["features"], label, severity,
                                                   nominal_scales, rng, registry, original_severity)
                run_id = f"{prefix}-{row_index:06d}"
                start_ns = epoch + row_index * 10_000_000_000
                end_ns = start_ns + 5_000_000_000
                active = label != "nominal"
                row.update({
                    "engine_serial": f"AUGMENTED-ENGINE-{(row_index - 1) % 32 + 1:03d}",
                    "sortie_id": run_id, "run_id": run_id,
                    "window_start_ns": start_ns, "window_end_ns": end_ns,
                    "ground_truth_label": label, "fault_class": label,
                    "fault_labels": "" if label == "nominal" else label,
                    "fault_active": str(active).lower(),
                    "fault_onset_ns": start_ns - 60_000_000_000 if active else "",
                    "fault_severity": severity,
                    "maintenance_outcome": "unverified_nominal_candidate" if label == "nominal" else "synthetic_fault_augmented_unverified",
                    "data_status": "development_data_units_unverified_augmented",
                    "source_kind": "joint_layer5_window_augmentation",
                    "source_parent_run_id": donor.get("run_id", ""),
                    "augmentation_method": "whole-window correlated residual perturbation; source sequence order retained",
                    "healthy_verified": "false",
                    "feature_schema_version": registry.version,
                    "feature_schema_hash": donor["feature_schema_hash"],
                })
                writer.writerow({**row, **feature_values})
                counts[label] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightgbm-source", type=Path, required=True)
    parser.add_argument("--nominal-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/layer6/source_conditioned_v3"))
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if args.rows < len(FAULT_CLASSES) + 1:
        parser.error("--rows must allow at least one row for each fault class and nominal")

    registry = FeatureRegistry()
    lightgbm_source, features = _read(args.lightgbm_source, registry)
    nominal_source, nominal_features = _read(args.nominal_source, registry)
    if features != nominal_features:
        raise ValueError("LightGBM and nominal source feature orders differ")
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in lightgbm_source:
        by_class[str(row["ground_truth_label"])].append(row)
    nominal_rows = [row for row in nominal_source if row["ground_truth_label"] == "nominal"]
    if not nominal_rows:
        raise ValueError("No nominal donor windows available")
    # Preserve the supplied raw-row proportions: 85k unverified nominal rows
    # (50k NORMAL + 35k NOMINAL) versus 35k per fault class, including a
    # synthetic 35k-equivalent cooling class for the target distribution.
    balanced_counts = _target_counts(args.rows, nominal_count=85_000, fault_count=35_000)
    rng = random.Random(args.seed)
    nominal_scales = _nominal_channel_scales(nominal_rows, registry)
    report = {
        "data_status": "development_data_units_unverified_augmented",
        "feature_schema_version": registry.version,
        "feature_count": len(features),
        "feature_schema_hash": registry.schema_hash(features),
        "requested_rows_per_output": args.rows,
        "row_generation": "joint feature-window augmentation; rows are not independent sorties",
        "lightgbm_class_counts": _build_file(
            by_class, args.output_dir / "lightgbm_layer5_v3_50000_development_augmented.csv.gz",
            balanced_counts, registry, features, rng, "AUG-LGBM", nominal_scales),
        "isolation_forest_candidate_count": _build_file(
            {"nominal": nominal_rows}, args.output_dir / "isolation_forest_layer5_v3_50000_unverified_candidates_augmented.csv.gz",
            {"nominal": args.rows}, registry, features, rng, "AUG-IF", nominal_scales, nominal_only=True),
        "isolation_forest_eligibility": "all 50,000 rows remain unverified nominal candidates; do not use as verified healthy training data",
        "cooling_degradation": "synthetic 60-second thermal rise in correlated CHT and oil-temperature residual sequences; absent from supplied fault labels",
        "pattern_preservation": "augmentation samples complete donor windows; raw 50-point sequences retain temporal order; shared perturbations preserve cross-feature structure; summaries are recomputed or transformed consistently",
        "parent_windows_lightgbm": len(lightgbm_source),
        "parent_windows_nominal": len(nominal_rows),
    }
    (args.output_dir / "augmentation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
