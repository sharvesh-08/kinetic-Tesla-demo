"""Derive PINN-only state inputs from MALE UAV simulator telemetry.

These values are generated inside Layer 3 and never written back into the
protected simulator or represented as measured sensors.  They are provisional
features that can be replaced by a trained estimator when labelled data is
available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EstimatedInput:
    value: float
    method: str
    measured_inputs: tuple[str, ...]


class Layer3InputEstimator:
    """Create missing PINN state inputs from measured simulator channels."""

    def estimate(self, measured: dict[str, float]) -> dict[str, EstimatedInput]:
        throttle = float(measured.get("throttle_pct", 50.0))
        rpm = float(measured.get("rpm", 2500.0))
        fuel_flow = float(measured.get("fuel_flow_Lph", 10.0))
        injection_ms = float(measured.get("injection_duration_ms", 4.2))
        oil_temp = float(measured.get("oil_temp_degC", 90.0))
        battery_v = float(measured.get("batt_volts", 28.0))
        return {
            "injection_timing_degBTDC": EstimatedInput(
                float(measured.get("injection_timing_degBTDC", 22.5)),
                "provisional_speed_timing_map", ("rpm",)),
            "mixture_afr": EstimatedInput(
                float(np.clip(14.7 - 0.04 * (injection_ms - 4.2) - 0.015 * (fuel_flow - 10.0), 10.0, 18.0)),
                "provisional_regression_feature", ("injection_duration_ms", "fuel_flow_Lph")),
            "cowl_flap_pct": EstimatedInput(
                float(np.clip(100.0 - throttle, 0.0, 100.0)),
                "provisional_control_proxy", ("throttle_pct",)),
            "battery_soc_pct": EstimatedInput(
                float(np.clip((battery_v - 22.0) / 8.0 * 100.0, 0.0, 100.0)),
                "provisional_voltage_regression", ("batt_volts",)),
            "friction_torque_Nm": EstimatedInput(
                float(max(0.0, 8.0 + 0.003 * max(0.0, rpm - 1000.0) + 0.05 * max(0.0, oil_temp - 90.0))),
                "provisional_thermal_speed_regression", ("rpm", "oil_temp_degC")),
        }



def apply_estimates(state: dict[str, float]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Fill only absent PINN inputs and return auditable provenance."""
    enriched = dict(state)
    provenance: dict[str, dict[str, Any]] = {}
    for name, estimate in Layer3InputEstimator().estimate(state).items():
        if name not in enriched:
            enriched[name] = estimate.value
            provenance[name] = {
                "value": estimate.value,
                "method": estimate.method,
                "measured_inputs": list(estimate.measured_inputs),
                "measured": False,
                "training_required": True,
            }
    return enriched, provenance
