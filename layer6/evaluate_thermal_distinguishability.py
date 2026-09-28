"""Evaluate cooling versus thermal-load signatures on independent synthetic runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .generate_fault_traces import REGIMES

LABELS = {"cooling_degradation": 0, "overheating_trend": 1}
SETS = {
    "cht_only": ("cht_degC",),
    "cht_egt": ("cht_degC", "egt_degC"),
    "cht_oil_temp": ("cht_degC", "oil_temp_degC"),
    "cht_egt_oil_temp": ("cht_degC", "egt_degC", "oil_temp_degC"),
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["ground_truth_label"] in LABELS]


def _regime(run_id: str) -> str:
    matches = [name for name in REGIMES if f"-{name}-" in run_id]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify operating regime from {run_id}")
    return matches[0]


def evaluate(train: Path, test: Path) -> dict:
    training = _read(train)
    holdout = _read(test)
    if {row["sortie_id"] for row in training} & {row["sortie_id"] for row in holdout}:
        raise ValueError("Training and holdout sorties overlap")
    report: dict = {"train_sorties": len({row["sortie_id"] for row in training}),
                    "holdout_sorties": len({row["sortie_id"] for row in holdout}),
                    "sensor_sets": {}}
    for set_name, channels in SETS.items():
        names = [f"resid_{channel}_{stat}_60s" for channel in channels for stat in ("mean", "slope")]
        x_train = np.asarray([[float(row[name]) for name in names] for row in training])
        y_train = np.asarray([LABELS[row["ground_truth_label"]] for row in training])
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(x_train, y_train)
        per_regime: dict[str, dict] = {}
        for regime in REGIMES:
            rows = [row for row in holdout if _regime(row["sortie_id"]) == regime]
            grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                grouped[row["sortie_id"]].append(row)
            truth = []
            prediction = []
            for run_rows in grouped.values():
                probabilities = model.predict_proba(np.asarray([[float(row[name]) for name in names] for row in run_rows]))[:, 1]
                truth.append(LABELS[run_rows[0]["ground_truth_label"]])
                prediction.append(int(float(np.mean(probabilities)) >= 0.5))
            per_regime[regime] = {
                "holdout_sorties": len(grouped),
                "cooling_correct": sum(t == 0 and p == 0 for t, p in zip(truth, prediction)),
                "overheating_correct": sum(t == 1 and p == 1 for t, p in zip(truth, prediction)),
                "separable_on_this_holdout": bool(truth and all(t == p for t, p in zip(truth, prediction))),
            }
        report["sensor_sets"][set_name] = per_regime
    report["scope"] = "Synthetic generator smoke test only; one or a few runs per regime cannot establish field identifiability."
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.train, args.test)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
