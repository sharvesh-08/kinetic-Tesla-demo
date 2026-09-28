"""Create deterministic, independent-sortie inputs for the physics fault injector.

This intentionally generates *scenario labels*, not fake independent ML features.  The
scenarios must be run through the physics simulator and Layers 4–5 before model training.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from .model import CLASSES


REGIMES = ("cold_start", "idle", "taxi", "climb", "cruise", "descent", "high_power_transient", "hot_day", "cold_day", "high_altitude")
BANDS = {"early": (0.05, 0.25), "moderate": (0.25, 0.60), "severe": (0.60, 1.00)}


def generate(output: Path, *, sorties_per_cell: int = 100, seed: int = 42) -> int:
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario_id", "sortie_id", "engine_serial", "fault_class", "fault_active", "fault_severity",
              "severity_band", "fault_onset_s", "duration_s", "operating_regime", "ambient_temperature_c",
              "altitude_m", "power_band", "sensor_noise_seed", "maintenance_outcome"]
    total = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        # Nominal scenarios are explicitly independent and are the only candidates for IF.
        cells = [("nominal", "none", regime) for regime in REGIMES]
        cells += [(fault, band, regime) for fault in CLASSES if fault != "nominal" for band in BANDS for regime in REGIMES]
        for fault, band, regime in cells:
            for replica in range(sorties_per_cell):
                total += 1
                hot = regime == "hot_day"; cold = regime == "cold_day"; high = regime == "high_altitude"
                writer.writerow({
                    "scenario_id": f"L6-{total:06d}", "sortie_id": f"SYN-{total:06d}",
                    "engine_serial": f"ENG-{rng.randrange(1, 51):03d}", "fault_class": fault,
                    "fault_active": str(fault != "nominal").lower(),
                    "fault_severity": 0.0 if fault == "nominal" else round(rng.uniform(*BANDS[band]), 4),
                    "severity_band": band, "fault_onset_s": 0.0 if fault == "nominal" else round(rng.uniform(60, 600), 2),
                    "duration_s": round(rng.uniform(900, 3600), 1), "operating_regime": regime,
                    "ambient_temperature_c": round(rng.uniform(30, 45) if hot else rng.uniform(-15, 2) if cold else rng.uniform(5, 32), 1),
                    "altitude_m": round(rng.uniform(3500, 6500) if high else rng.uniform(0, 2500), 1),
                    "power_band": "high" if regime in {"climb", "high_power_transient"} else "low" if regime in {"idle", "taxi", "descent"} else "medium",
                    "sensor_noise_seed": rng.randrange(1, 2**31),
                    "maintenance_outcome": "healthy" if fault == "nominal" else "repair_required",
                })
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/layer6_synthetic_scenarios.csv"))
    parser.add_argument("--sorties-per-cell", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"Wrote {generate(args.output, sorties_per_cell=args.sorties_per_cell, seed=args.seed)} scenario rows to {args.output}")


if __name__ == "__main__":
    main()
