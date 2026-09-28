"""Run-level evaluation of all eight calibrated fault heads on untouched sorties."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .artifact import load_artifact
from .model import FAULT_CLASSES


def evaluate(artifact_path: Path, dataset_path: Path) -> dict:
    artifact = load_artifact(artifact_path)
    model = artifact.model
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["ground_truth_label"] in FAULT_CLASSES:
                grouped[row["sortie_id"]].append(row)
    counts = {name: {"true_positive_runs": 0, "false_negative_runs": 0,
                     "false_positive_runs": 0, "true_negative_runs": 0} for name in FAULT_CLASSES}
    per_regime: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"correct_primary_runs": 0, "runs": 0})
    simultaneous = {"runs": 0, "all_true_flags_detected": 0, "extra_flags": 0}
    for run_id, rows in grouped.items():
        truth = rows[0]["ground_truth_label"]
        true_labels = set((rows[0].get("fault_labels") or truth).split("|"))
        outputs = []
        for row in rows:
            features = {name: float(row[name]) for name in model.feature_names}
            outputs.append(model.predict({"window_start_ns": int(row["window_start_ns"]),
                                          "window_end_ns": int(row["window_end_ns"]),
                                          "features": features,
                                          "feature_valid": {name: True for name in features},
                                          "feature_schema_hash": row["feature_schema_hash"]}))
        mean_probability = {name: sum(output.class_probabilities[name] for output in outputs) / len(outputs)
                            for name in FAULT_CLASSES}
        primary = max(mean_probability, key=mean_probability.get)
        regime = next((name for name in ("cruise", "climb", "hot_day", "high_altitude")
                       if f"-{name}-" in run_id), "unknown")
        per_regime[regime]["runs"] += 1
        per_regime[regime]["correct_primary_runs"] += int(primary == truth)
        flagged_labels = {name for name in FAULT_CLASSES if mean_probability[name] >= model.fault_thresholds[name]}
        if len(true_labels) > 1:
            simultaneous["runs"] += 1
            simultaneous["all_true_flags_detected"] += int(true_labels <= flagged_labels)
            simultaneous["extra_flags"] += len(flagged_labels - true_labels)
        for name in FAULT_CLASSES:
            flagged = mean_probability[name] >= model.fault_thresholds[name]
            counts[name]["true_positive_runs" if flagged else "false_negative_runs"] += int(name in true_labels)
            counts[name]["false_positive_runs" if flagged else "true_negative_runs"] += int(name not in true_labels)
    return {"artifact_version": artifact.metadata.model_version,
            "holdout_sorties": len(grouped), "by_fault": counts,
            "by_regime": dict(per_regime), "simultaneous": simultaneous,
            "scope": "Independent synthetic generator seeds only; not flight validation."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("test_dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.artifact, args.test_dataset)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
