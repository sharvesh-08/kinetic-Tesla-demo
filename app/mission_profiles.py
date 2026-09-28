"""
Predefined UAV Aero-Piston Engine Mission Profiles.

Provides flight mission operational profiles that dictate throttle settings
and engine baseline speed over simulation time t.
"""

from typing import Callable, Dict, Any
import numpy as np


def mission_endurance(t: float) -> dict[str, float]:
    """
    Endurance Mission Profile.
    Long-duration, steady-state flight profile with smooth, low-amplitude throttle sweeps.
    """
    throttle_raw = 0.5 + 0.1 * np.sin(0.02 * t)
    throttle = float(np.clip(throttle_raw, 0.2, 0.9))
    rpm_base = 2000.0 + 2500.0 * throttle
    return {
        "throttle": throttle,
        "rpm_base": float(rpm_base),
    }


def mission_high_altitude(t: float) -> dict[str, float]:
    """
    High-Altitude Mission Profile.
    Higher operational power demands with frequent climb/descent throttle modulations.
    """
    throttle_raw = 0.45 + 0.2 * np.sin(0.03 * t)
    throttle = float(np.clip(throttle_raw, 0.2, 0.9))
    rpm_base = 2000.0 + 2200.0 * throttle
    return {
        "throttle": throttle,
        "rpm_base": float(rpm_base),
    }


def mission_nominal_cruise(t: float) -> dict[str, float]:
    """Nominal Cruise Flight Profile — Healthy engine, EHI ~ 100%."""
    throttle_raw = 0.6 + 0.05 * np.sin(0.015 * t)
    throttle = float(np.clip(throttle_raw, 0.2, 0.9))
    rpm_base = 2000.0 + 2500.0 * throttle
    return {"throttle": throttle, "rpm_base": float(rpm_base), "fault_mode": "NONE"}


def mission_thermal_overload(t: float) -> dict[str, float]:
    """Thermal Overload Profile — Overheating fault scenario after t=30s."""
    throttle_raw = 0.75 + 0.1 * np.sin(0.02 * t)
    throttle = float(np.clip(throttle_raw, 0.3, 0.95))
    rpm_base = 2000.0 + 2500.0 * throttle
    fault_mode = "OVERHEATING" if t > 30.0 else "NONE"
    return {"throttle": throttle, "rpm_base": float(rpm_base), "fault_mode": fault_mode}


def mission_lubrication_issue(t: float) -> dict[str, float]:
    """Lubrication Issue Profile — Oil breakdown scenario after t=30s."""
    throttle_raw = 0.65 + 0.08 * np.sin(0.025 * t)
    throttle = float(np.clip(throttle_raw, 0.2, 0.9))
    rpm_base = 2000.0 + 2500.0 * throttle
    fault_mode = "LUBRICATION_ISSUE" if t > 30.0 else "NONE"
    return {"throttle": throttle, "rpm_base": float(rpm_base), "fault_mode": fault_mode}


def mission_exhaust_leak(t: float) -> dict[str, float]:
    """Exhaust Leak Profile — Exhaust leak scenario after t=30s."""
    throttle_raw = 0.6 + 0.05 * np.sin(0.02 * t)
    throttle = float(np.clip(throttle_raw, 0.2, 0.9))
    rpm_base = 2000.0 + 2500.0 * throttle
    fault_mode = "EXHAUST_LEAK" if t > 30.0 else "NONE"
    return {"throttle": throttle, "rpm_base": float(rpm_base), "fault_mode": fault_mode}


# Central Registry of Mission Profiles
MISSION_PROFILES: dict[str, Callable[[float], dict[str, float]]] = {
    "endurance": mission_nominal_cruise,
    "high_altitude": mission_thermal_overload,
    "cruise": mission_nominal_cruise,
    "nominal_cruise": mission_nominal_cruise,
    "thermal_overload": mission_thermal_overload,
    "lubrication_issue": mission_lubrication_issue,
    "exhaust_leak": mission_exhaust_leak,
}


def get_mission_profile(name: str) -> Callable[[float], dict[str, float]]:
    """
    Retrieve a mission profile function by name.
    
    Raises ValueError if the profile name is unknown.
    """
    key = name.lower().strip()
    if key not in MISSION_PROFILES:
        raise ValueError(
            f"Unknown mission profile '{name}'. "
            f"Available profiles: {list(MISSION_PROFILES.keys())}"
        )
    return MISSION_PROFILES[key]
