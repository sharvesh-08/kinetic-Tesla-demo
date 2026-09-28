"""Generate provisional, physics-shaped raw fault traces for development only.

The output is raw observed/truth telemetry. It must pass through Layers 4 and 5
before training. These traces are not flight evidence or validated engine models.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from .model import FAULT_CLASSES

REGIMES = {
    "cruise": (3000.0, 55.0, 1000.0, 0.0),
    "climb": (4100.0, 85.0, 2600.0, 8.0),
    "hot_day": (3200.0, 65.0, 1300.0, 16.0),
    "high_altitude": (3700.0, 75.0, 5200.0, -3.0),
}
BASE = {
    "exhaust_gas_temp_c": 760.0,
    "cylinder_head_temp_c": 165.0,
    "manifold_pressure_kpa": 72.0,
    "fuel_flow_lph": 18.0,
    "oil_pressure_kpa": 340.0,
    "oil_temp_c": 95.0,
    "vibration_g": 0.22,
}
FIELDS = ["run_id", "master_timestamp_ns", "fault_start_timestamp_ns", "fault_class",
          "fault_labels", "fault_active", "fault_severity", "operating_regime", "rpm", "throttle_pct", "altitude_m",
          *(f"obs_{name}" for name in BASE), *(f"truth_{name}" for name in BASE)]


def _effects(fault: str, elapsed_s: float, severity: float) -> dict[str, float]:
    if elapsed_s <= 0:
        return {}
    if "+" in fault:
        combined: dict[str, float] = {}
        for component in fault.split("+"):
            for name, value in _effects(component, elapsed_s, severity).items():
                combined[name] = combined.get(name, 0.0) + value
        return combined
    ramp = 1 - math.exp(-elapsed_s / 18.0)
    slow = 1 - math.exp(-elapsed_s / 35.0)
    if fault == "cooling_degradation":
        # Reduced heat rejection: CHT and oil temperature rise together; EGT stays near baseline.
        return {"cylinder_head_temp_c": 38 * severity * ramp,
                "oil_temp_c": 25 * severity * slow}
    if fault == "overheating_trend":
        # Additional combustion heat: the same CHT rise can have a different cause.
        # EGT responds strongly while oil response remains small.
        return {"exhaust_gas_temp_c": 75 * severity * ramp,
                "cylinder_head_temp_c": 38 * severity * ramp,
                "oil_temp_c": 5 * severity * slow}
    if fault == "misfire":
        pulse = 1.0 if int(elapsed_s * 3) % 5 == 0 else 0.0
        return {"exhaust_gas_temp_c": -95 * severity * pulse,
                "vibration_g": 0.25 * severity * pulse}
    if fault == "injector_abnormality":
        return {"fuel_flow_lph": 5 * severity * ramp,
                "exhaust_gas_temp_c": -35 * severity * ramp}
    if fault == "lubrication_issue":
        return {"oil_pressure_kpa": -85 * severity * ramp,
                "oil_temp_c": 16 * severity * slow,
                "vibration_g": 0.13 * severity * slow}
    if fault == "sensor_drift_failure":
        return {"oil_pressure_kpa": 45 * severity * elapsed_s / 65.0}
    if fault == "combustion_instability":
        return {"exhaust_gas_temp_c": 38 * severity * math.sin(elapsed_s * 2.3),
                "vibration_g": 0.08 * severity * abs(math.sin(elapsed_s * 2.3))}
    if fault == "abnormal_vibration":
        return {"vibration_g": 0.45 * severity * ramp}
    return {}


def generate(path: Path, *, runs_per_class_regime: int = 4, duration_s: int = 100,
             onset_s: int = 25, rate_hz: int = 100, seed: int = 42,
             run_prefix: str = "SYN", faults: tuple[str, ...] = FAULT_CLASSES) -> int:
    if duration_s - onset_s < 65 or rate_hz != 100:
        raise ValueError("Require 100 Hz and at least 65 post-onset seconds for complete long windows")
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for fault in faults:
            if any(component not in FAULT_CLASSES for component in fault.split("+")):
                raise ValueError(f"Unknown fault combination {fault}")
            for regime, (rpm, throttle, altitude, thermal_offset) in REGIMES.items():
                for replica in range(runs_per_class_regime):
                    count += 1
                    run_id = f"{run_prefix}-{fault}-{regime}-{replica:03d}"
                    severity = rng.uniform(0.55, 1.0)
                    offset = rng.uniform(-2.0, 2.0)
                    for sample in range(duration_s * rate_hz):
                        t = sample / rate_hz
                        truth = dict(BASE)
                        truth["exhaust_gas_temp_c"] += thermal_offset * 1.8 + offset
                        truth["cylinder_head_temp_c"] += thermal_offset + offset
                        truth["oil_temp_c"] += thermal_offset * 0.35
                        truth["fuel_flow_lph"] *= throttle / 55.0
                        truth["manifold_pressure_kpa"] *= throttle / 55.0
                        effects = _effects(fault, t - onset_s, severity)
                        observed = {name: value + effects.get(name, 0.0) + rng.gauss(0, 0.003 * max(abs(value), 1.0))
                                    for name, value in truth.items()}
                        writer.writerow({"run_id": run_id, "master_timestamp_ns": sample * 10_000_000,
                                         "fault_start_timestamp_ns": onset_s * 1_000_000_000,
                                         "fault_class": fault.split("+")[0],
                                         "fault_labels": "|".join(fault.split("+")),
                                         "fault_active": int(t >= onset_s),
                                         "fault_severity": severity, "operating_regime": regime,
                                         "rpm": rpm, "throttle_pct": throttle, "altitude_m": altitude,
                                         **{f"obs_{name}": value for name, value in observed.items()},
                                         **{f"truth_{name}": value for name, value in truth.items()}})
    return count


def generate_fleet_priors(path: Path, *, runs_per_subsystem: int = 50, seed: int = 42,
                          run_prefix: str = "FLEET") -> int:
    """Emit independent synthetic degradation trajectories with failure outcomes.

    This deliberately simple constant-rate process exercises the prior-fit
    workflow. It does not represent demonstrated aero-engine run life.
    """
    if runs_per_subsystem < 10:
        raise ValueError("At least ten independent runs per subsystem are required")
    rng = random.Random(seed)
    subsystems = ("lubrication", "mechanical", "electrical", "combustion", "thermal")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["flight_id", "subsystem", "operating_hours",
                                                     "degradation_state", "failure_or_removal",
                                                     "fault_onset_hours", "severity", "source_kind"])
        writer.writeheader()
        for subsystem in subsystems:
            for replica in range(runs_per_subsystem):
                rate = rng.lognormvariate(math.log(0.035), 0.75)
                onset = rng.uniform(0.0, 2.0)
                initial = rng.uniform(0.0, 0.1)
                flight_id = f"{run_prefix}-{subsystem}-{replica:04d}"
                for hour in range(49):
                    state = min(initial + rate * max(hour - onset, 0.0), 1.0)
                    failed = state >= 1.0
                    writer.writerow({"flight_id": flight_id, "subsystem": subsystem,
                                     "operating_hours": hour, "degradation_state": state,
                                     "failure_or_removal": int(failed),
                                     "fault_onset_hours": onset, "severity": rate,
                                     "source_kind": "synthetic"})
                    if failed:
                        break
    return runs_per_subsystem * len(subsystems)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--runs-per-class-regime", type=int, default=4)
    parser.add_argument("--duration-s", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-prefix")
    parser.add_argument("--faults", default=",".join(FAULT_CLASSES),
                        help="Comma-separated classes or plus-joined simultaneous faults")
    parser.add_argument("--fleet-prior-output", type=Path,
                        help="Generate synthetic 48-hour degradation trajectories for prior-fit smoke tests")
    parser.add_argument("--fleet-runs-per-subsystem", type=int, default=50)
    args = parser.parse_args()
    if args.output:
        print(f"Generated {generate(args.output, runs_per_class_regime=args.runs_per_class_regime, duration_s=args.duration_s, seed=args.seed, run_prefix=args.run_prefix or 'SYN', faults=tuple(args.faults.split(',')))} raw runs")
    if args.fleet_prior_output:
        print(f"Generated {generate_fleet_priors(args.fleet_prior_output, runs_per_subsystem=args.fleet_runs_per_subsystem, seed=args.seed, run_prefix=args.run_prefix or 'FLEET')} fleet degradation runs")
    if not args.output and not args.fleet_prior_output:
        parser.error("Provide a raw trace output path or --fleet-prior-output")


if __name__ == "__main__":
    main()
