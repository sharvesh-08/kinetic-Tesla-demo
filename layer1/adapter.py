"""MALE engine snapshots -> validated, individually timestamped sensor events.

Conversions happen at this boundary, never in the engine equations. Extra
channels are transported without changing the input order of trained models.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from .schema import TELEMETRY_CHANNELS
from .service import validate_ecu_event


@dataclass(frozen=True)
class ParameterSpec:
    channel: str
    unit: str
    minimum: float
    maximum: float
    default: float
    scale: float = 1.0
    offset: float = 0.0


PARAMETERS = {
    "rpm": ParameterSpec("rpm", "RPM", 0, 5800, 3200),
    "throttle": ParameterSpec("throttle_pct", "ratio", 0, 1, .65, 100),
    "egt": ParameterSpec("egt_degC", "degC", -40, 1200, 650),
    "cht": ParameterSpec("cht_degC", "degC", -40, 600, 180),
    "oil_temp": ParameterSpec("oil_temp_degC", "degC", -40, 280, 95),
    "map": ParameterSpec("map_kPa", "inHg", 5, 45, 24, 3.386389),
    "oil_press": ParameterSpec("oil_press_kPa", "PSI", 0, 120, 50, 6.894757),
    "fuel_flow": ParameterSpec("fuel_flow_Lph", "L/h", 0, 80, 10),
    "iat": ParameterSpec("Tm_K_k", "degC", -60, 150, 30, 1, 273.15),
    "vibration": ParameterSpec("vibration_rms_g", "g RMS", 0, 30, .45),
    "ambient_temp": ParameterSpec("ambient_temp_degC", "degC", -60, 70, 25),
    "battery_voltage": ParameterSpec("batt_volts", "V", 0, 32, 28),
    "alternator_current": ParameterSpec("alt_amps", "A (positive output)", 0, 80, 45),
    "electrical_health": ParameterSpec("electrical_health_pct", "%", 0, 100, 100),
    "injection_timing": ParameterSpec("injection_timing_degBTDC", "degBTDC", 0, 50, 22.5),
    "injection_duration": ParameterSpec("injection_duration_ms", "ms", 0, 20, 4.2),
    "fuel_pressure": ParameterSpec("fuel_pressure_kPa", "PSI", 0, 120, 42, 6.894757),
}


def parameter_catalog() -> dict[str, dict[str, Any]]:
    return {name: asdict(spec) for name, spec in PARAMETERS.items()}


def validate_overrides(flags: dict, values: dict) -> None:
    unknown = (set(flags) | set(values)) - PARAMETERS.keys()
    if unknown:
        raise ValueError(f"Unknown override parameters: {sorted(unknown)}")
    for name, enabled in flags.items():
        if not isinstance(enabled, bool):
            raise ValueError(f"{name}: override flag must be boolean")
        if enabled and name not in values:
            raise ValueError(f"{name}: enabled override requires a value")
    for name, value in values.items():
        spec = PARAMETERS[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name}: expected a finite number")
        if not spec.minimum <= value <= spec.maximum:
            raise ValueError(f"{name}: expected {spec.minimum}..{spec.maximum} {spec.unit}")


def to_layer1_events(snapshot: dict[str, Any], *, sample_rate_hz: float = 2.0,
                     session_id: str | None = None) -> list[dict[str, Any]]:
    """Missing readings stay absent; invalid present readings fail validation.

    Per-channel timestamps/rates may be supplied by asynchronous sensor sources.
    A simulator snapshot uses its actual tick rate, not a fabricated native rate.
    """
    timestamp = snapshot.get("sensor_timestamp_ns", time.time_ns())
    values: dict[str, Any] = {}
    for source, spec in PARAMETERS.items():
        value = snapshot.get(source)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"{source}: expected a numeric sensor reading")
            values[spec.channel] = value * spec.scale + spec.offset
        elif snapshot.get(spec.channel) is not None:
            values[spec.channel] = snapshot[spec.channel]
    if snapshot.get("baro_pressure") is not None:
        values["ambient_press_kPa"] = snapshot["baro_pressure"] * 3.386389
    elif snapshot.get("ambient_press_kPa") is not None:
        values["ambient_press_kPa"] = snapshot["ambient_press_kPa"]
    if snapshot.get("altitude_m") is not None:
        values["altitude_m"] = snapshot["altitude_m"]
    elif snapshot.get("altitude") is not None:
        values["altitude_m"] = snapshot["altitude"] * .3048
    if snapshot.get("air_density_ratio") is not None:
        values["air_density_ratio"] = snapshot["air_density_ratio"]
    context = {name: snapshot[name] for name in (
        "engine_serial", "sortie_id", "operating_hours", "context_provenance",
        "atmospheric_profile", "control_revision", "override_values", "injected_fault",
    ) if snapshot.get(name) is not None}
    if session_id is not None:
        context["session_id"] = session_id
        context.setdefault("sortie_id", session_id)
    context.setdefault("engine_serial", "MALE-UAV-SIM")
    if isinstance(snapshot.get("timestamp"), (int, float)):
        context.setdefault("operating_hours", float(snapshot["timestamp"]) / 3600.0)
    context["parameter_provenance"] = dict(snapshot.get("provenance") or {})
    context["operating_context"] = {"altitude_m": values.get("altitude_m"),
                                    "ambient_temp_degC": values.get("ambient_temp_degC"),
                                    "source": "MALE-UAV simulator"}
    rates = snapshot.get("channel_rate_hz") or {}
    timestamps = snapshot.get("channel_timestamp_ns") or {}
    validity = snapshot.get("channel_valid") or {}
    return [validate_ecu_event({
        "sensor": name, "value": value, "valid": validity.get(name, True),
        "sensor_timestamp_ns": timestamps.get(name, timestamp),
        "sample_rate_hz": rates.get(name, sample_rate_hz), "context": context,
    }) for name, value in values.items()]
