"""Fit versioned PI-RUL log-rate priors from fleet degradation trajectories.

Input must contain multiple independent trajectories with known degradation state
and failure/removal outcome. The short classifier trace generator is insufficient.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def fit(source: Path, destination: Path, *, minimum_flights: int = 10) -> dict:
    grouped: defaultdict[tuple[str, str], list[tuple[float, float, bool]]] = defaultdict(list)
    provenance_labels: set[str] = set()
    onsets: defaultdict[str, dict[str, float]] = defaultdict(dict)
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            provenance_labels.add(row.get("source_kind") or "external")
            subsystem = row["subsystem"]
            flight_id = row["flight_id"]
            if row.get("fault_onset_hours") not in (None, ""):
                onsets[subsystem][flight_id] = float(row["fault_onset_hours"])
            grouped[(subsystem, flight_id)].append((float(row["operating_hours"]),
                                                     float(row["degradation_state"]),
                                                     str(row["failure_or_removal"]).lower() in {"1", "true", "yes"}))
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pi_rul_priors.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rates: defaultdict[str, list[float]] = defaultdict(list)
    failures: defaultdict[str, int] = defaultdict(int)
    for (subsystem, flight_id), samples in grouped.items():
        samples.sort()
        onset = onsets[subsystem].get(flight_id, samples[0][0])
        usable = [item for item in samples if item[0] >= onset and item[1] < 1.0]
        if len(usable) < 3 or usable[-1][0] <= usable[0][0]:
            continue
        x = np.asarray([item[0] for item in usable])
        y = np.asarray([item[1] for item in usable])
        rate = float(np.polyfit(x - x[0], y, 1)[0])
        if rate <= 0 or not math.isfinite(rate):
            continue
        rates[subsystem].append(rate)
        failures[subsystem] += int(any(item[2] for item in samples))
    for subsystem, policy in config["subsystems"].items():
        values = rates[subsystem]
        if len(values) < minimum_flights or failures[subsystem] < 1:
            raise ValueError(f"{subsystem} needs at least {minimum_flights} independent flights and one observed failure/removal")
        logs = np.log(values)
        policy["prior_rate_median_per_hour"] = float(np.exp(np.mean(logs)))
        policy["prior_rate_log_sigma"] = float(max(np.std(logs, ddof=1), 0.05))
        if onsets[subsystem]:
            onset_values = list(onsets[subsystem].values())
            policy["prior_onset_min_hours"] = float(min(onset_values))
            policy["prior_onset_max_hours"] = float(max(onset_values))
    config["provenance"] = "synthetic_fleet_fit" if provenance_labels == {"synthetic"} else "fleet_trajectory_fit"
    config["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    config["source_flights_by_subsystem"] = {name: len(rates[name]) for name in config["subsystems"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = fit(args.source, args.destination)
    print({"provenance": result["provenance"], "flights": result["source_flights_by_subsystem"]})


if __name__ == "__main__":
    main()
