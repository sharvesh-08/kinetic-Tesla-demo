"""Train Layer 6 from named Layer 5 FeatureWindow CSV records."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from layer6.artifact import Layer6Artifact, build_metadata, save_artifact
from layer6.model import Layer6Models
from layer5.registry import FeatureRegistry

META = {
    "sortie_id", "run_id", "engine_serial", "window_start_ns", "window_end_ns",
    "ground_truth_label", "fault_class", "fault_active", "fault_onset_ns", "fault_severity",
    "fault_labels", "maintenance_outcome", "feature_schema_hash", "feature_schema_version",
    "subsystem", "operating_regime", "ambient_temperature_native", "altitude_native", "power_band",
    "data_status", "source_kind", "source_anchor_row_id", "source_nominal_row_id", "feature_valid_json",
    "source_parent_run_id", "augmentation_method", "healthy_verified",
}


def load(path: Path) -> list[dict]:
    registry = FeatureRegistry()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        result = []
        for raw in csv.DictReader(handle):
            features = {key: float(value) for key, value in raw.items() if key not in META and value not in (None, "")}
            registry.require_schema(list(features), raw["feature_schema_hash"])
            result.append({**{key: raw.get(key) for key in META},
                           "fault_active": str(raw.get("fault_active", "false")).casefold() in {"true", "1"},
                           "fault_labels": [label for label in str(raw.get("fault_labels") or raw["ground_truth_label"]).split("|") if label != "nominal"],
                           "features": features, "feature_valid": {key: True for key in features}})
    return result


def split_fingerprint(rows: list[dict]) -> str:
    payload = "\n".join(sorted(f"{_split_group(row)}:{row['window_start_ns']}" for row in rows))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_group(row: dict) -> str:
    """Keep augmented windows from one parent trace in the same data split."""
    return str(row.get("source_parent_run_id") or row.get("sortie_id"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--healthy-dataset", type=Path, required=True)
    parser.add_argument("--version", default="layer6-8fault-v3-synthetic-dev-1")
    parser.add_argument("--n-estimators", type=int, default=250,
                        help="LightGBM boosting rounds (the LightGBM equivalent of epochs; default: 250)")
    parser.add_argument("--allow-synthetic-nominal", action="store_true",
                        help="Use explicitly unverified synthetic nominal windows for a synthetic-dev artifact")
    args = parser.parse_args()
    rows = load(args.dataset)
    if any(row["features"]["history_seconds_available_60s"] < 60.0 for row in rows):
        raise ValueError("Training requires complete 60-second feature history; regenerate longer independent sorties")
    seed = 42
    runs_by_class: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        runs_by_class[row["ground_truth_label"]].add(_split_group(row))
    rng = random.Random(seed)
    calibration_ids: set[str] = set()
    for label, run_ids in sorted(runs_by_class.items()):
        choices = sorted(run_ids)
        if len(choices) < 2:
            raise ValueError(f"Fault {label} needs at least two independent sorties for calibration")
        rng.shuffle(choices)
        calibration_ids.update(choices[:max(1, round(0.2 * len(choices)))])
    train_rows = [row for row in rows if _split_group(row) not in calibration_ids]
    calibration_rows = [row for row in rows if _split_group(row) in calibration_ids]
    train_groups = {_split_group(row) for row in train_rows}
    calibration_groups = {_split_group(row) for row in calibration_rows}
    if train_groups & calibration_groups:
        raise AssertionError("Parent trace leakage across classifier and calibration split")
    healthy_rows = [row for row in load(args.healthy_dataset) if _split_group(row) in train_groups]
    if {_split_group(row) for row in healthy_rows} & calibration_groups:
        raise AssertionError("Nominal candidate from a calibration parent trace entered Isolation Forest")
    model = Layer6Models.fit(train_rows, calibration_rows, healthy_rows, model_version=args.version, seed=seed,
                             n_estimators=args.n_estimators,
                             allow_synthetic_nominal=args.allow_synthetic_nominal)
    metadata = build_metadata(model, training_dataset=args.dataset, healthy_dataset=args.healthy_dataset,
                              calibration_split_sha256=split_fingerprint(calibration_rows), training_rows=len(train_rows),
                              calibration_rows=len(calibration_rows), healthy_rows=len(healthy_rows), seed=seed)
    destination = save_artifact(args.artifact, Layer6Artifact(metadata=metadata, model=model))
    print(f"Saved {destination}; train={len(train_rows)} calibration={len(calibration_rows)} "
          f"nominal_candidates={len(healthy_rows)} independent_parent_group_split=true")


if __name__ == "__main__":
    main()
