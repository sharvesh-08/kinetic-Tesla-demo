"""Shared, policy-normalized physical indicator evidence for Layer 7."""
from __future__ import annotations

import math
from typing import Any


def physical_indicator_signals(
    indicators: dict[str, Any], policy: dict[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Return each subsystem's strongest normalized sensor/degradation signal.

    Residuals are deviations from the physics model, so either sign is evidence
    and is normalized by absolute magnitude. Other indicators are normalized in
    their configured failure direction against the configured threshold.
    """
    by_subsystem = {name: 0.0 for name in policy["subsystem_consequence_weight"]}
    detail: dict[str, dict[str, Any]] = {}
    for name, raw_value in indicators.items():
        entry = policy["indicators"].get(name)
        if not entry or isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            continue
        threshold = float(policy["failure_thresholds"][entry["threshold_key"]])
        direction = float(entry["failure_direction"])
        if "residual" in name.lower():
            signal = abs(value) / max(abs(threshold), 1e-9)
        elif direction > 0:
            signal = value / max(abs(threshold), 1e-9)
        else:
            healthy = float(entry.get("healthy_reference", 1.0))
            span = healthy - threshold
            signal = (healthy - value) / span if span > 0 else 0.0
        signal = min(1.0, max(0.0, signal))
        subsystem = entry["subsystem"]
        by_subsystem[subsystem] = max(by_subsystem[subsystem], signal)
        detail[name] = {
            "value": value,
            "threshold": threshold,
            "subsystem": subsystem,
            "normalized_signal": signal,
        }
    return by_subsystem, detail
