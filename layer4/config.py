"""Validated configuration loader for Layer 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SubfilterConfig:
    """Diagonal covariance tuning for one federated sub-filter."""

    channels: tuple[str, ...]
    q: dict[str, float]
    r: dict[str, float]
    initial_p: dict[str, float]
    bias_q: dict[str, float]
    bias_r: dict[str, float]


@dataclass(frozen=True)
class EKFConfig:
    """Runtime and tuning values for the federated filter bank."""

    master_rate_hz: float
    missing_p_inflation: float
    innovation_gate_sigma: float
    rpm_rate_threshold: float
    throttle_rate_threshold: float
    q_inflation_factor: float
    drifting_bias_sigma: float
    failed_bias_sigma: float
    failed_residual_sigma: float
    minimum_bias_age_samples: int
    substitution_enabled: bool
    subfilters: dict[str, SubfilterConfig]


def _positive(mapping: dict[str, Any], key: str) -> float:
    value = float(mapping[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def load_config(path: str | Path | None = None) -> EKFConfig:
    """Load and validate the standalone EKF tuning YAML."""
    source = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "ekf_tuning.yaml"
    with source.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported ekf_tuning schema_version")
    health, transient = raw["sensor_health"], raw["transient"]
    filters: dict[str, SubfilterConfig] = {}
    seen: set[str] = set()
    for name, item in raw["subfilters"].items():
        channels = tuple(item["channels"])
        if not channels or seen.intersection(channels):
            raise ValueError(f"Subfilter {name} has no channels or duplicates another subfilter")
        seen.update(channels)
        matrices = {key: {channel: float(item[key][channel]) for channel in channels}
                    for key in ("q", "r", "initial_p", "bias_q", "bias_r")}
        if any(value <= 0 for matrix in matrices.values() for value in matrix.values()):
            raise ValueError(f"All covariance entries for {name} must be positive")
        filters[name] = SubfilterConfig(channels=channels, **matrices)
    return EKFConfig(
        master_rate_hz=_positive(raw, "master_rate_hz"),
        missing_p_inflation=_positive(raw, "missing_measurement_p_inflation"),
        innovation_gate_sigma=_positive(health, "innovation_gate_sigma"),
        rpm_rate_threshold=_positive(transient, "rpm_rate_threshold_per_s"),
        throttle_rate_threshold=_positive(transient, "throttle_rate_threshold_pct_per_s"),
        q_inflation_factor=_positive(transient, "q_inflation_factor"),
        drifting_bias_sigma=_positive(health, "drifting_bias_sigma"),
        failed_bias_sigma=_positive(health, "failed_bias_sigma"),
        failed_residual_sigma=_positive(health, "failed_residual_sigma"),
        minimum_bias_age_samples=int(_positive(health, "minimum_bias_age_samples")),
        substitution_enabled=bool(health["substitution_enabled"]),
        subfilters=filters,
    )
