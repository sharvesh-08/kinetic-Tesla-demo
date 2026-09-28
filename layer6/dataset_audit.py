"""Audit a raw fault-injection source before it is admitted to Layer 6 training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_CLASSES = {
    "misfire", "injector_abnormality", "cooling_degradation", "lubrication_issue",
    "sensor_drift_failure", "combustion_instability", "overheating_trend", "abnormal_vibration",
}
REQUIRED_OBSERVATIONS = {
    "obs_exhaust_gas_temp_c", "obs_cylinder_head_temp_c", "obs_manifold_pressure_kpa",
    "obs_fuel_flow_lph", "obs_oil_pressure_kpa", "obs_oil_temp_c", "obs_vibration_g",
}
REQUIRED_TRUTHS = {name.replace("obs_", "truth_") for name in REQUIRED_OBSERVATIONS}


def audit(path: str | Path) -> dict[str, Any]:
    """Return a deliberately conservative suitability report for a source CSV."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        classes: Counter[str] = Counter()
        active: Counter[str] = Counter()
        runs: defaultdict[str, set[str]] = defaultdict(set)
        severity: defaultdict[str, list[float]] = defaultdict(list)
        rows = 0
        for row in reader:
            rows += 1
            label = row.get("fault_class", "")
            classes[label] += 1
            runs[label].add(row.get("run_id", ""))
            if row.get("fault_active") == "1":
                active[label] += 1
                severity[label].append(float(row["fault_severity"]))
    missing = sorted((REQUIRED_OBSERVATIONS | REQUIRED_TRUTHS) - columns)
    present = set(classes)
    # Pre-fault records are useful nominal candidates but are not independent healthy sorties.
    report = {
        "source": str(path), "rows": rows, "fault_classes": dict(sorted(classes.items())),
        "active_rows": dict(sorted(active.items())),
        "sorties_per_fault_class": {key: len(value) for key, value in sorted(runs.items())},
        "active_severity_range": {key: [min(value), max(value)] for key, value in sorted(severity.items()) if value},
        "missing_required_raw_columns": missing,
        "missing_fault_classes": sorted(REQUIRED_CLASSES - present),
        "has_explicit_nominal_sortie_label": "nominal" in present,
        "not_training_ready_reasons": [
            "The source contains raw observed/truth telemetry, not Layer 5 FeatureWindow records.",
            "Pre-fault segments are nominal candidates, but no independent healthy sortie label is present.",
            "Electrical residual channels (battery voltage and alternator current) are absent.",
            "No operating-regime, aircraft/engine serial, environment, maintenance, flight-hour, cycle-count, or RUL labels are present.",
        ],
    }
    report["source_only_usable_for"] = "synthetic-generator calibration and pipeline smoke tests"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.source_csv)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
