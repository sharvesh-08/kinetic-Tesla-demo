"""Validated, policy-only configuration for Layer 7.

All safety policy values intentionally live in YAML; source code contains no operational
thresholds or consequence weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "layer7_risk_rul.yaml"
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"failure_thresholds", "subsystem_consequence_weight", "risk_fusion", "rul", "engine_state_thresholds",
                "risk_action_thresholds", "redline_limits", "authority", "maintenance", "indicators"}
    if config.get("schema_version") != 2 or not required.issubset(config):
        raise ValueError("Invalid Layer 7 configuration")
    state = config["engine_state_thresholds"]
    if not (0 < float(state["abnormal_fault_probability"]) < float(state["critical_fault_probability"]) < 1
            and 0 < float(state["abnormal_anomaly_score"]) < float(state["critical_anomaly_score"]) < 1):
        raise ValueError("Engine-state thresholds must be ordered within (0, 1)")
    if not config["authority"].get("advisory_only", False):
        raise ValueError("Layer 7 must remain advisory-only")
    if not 0 < float(config["rul"]["confidence_level"]) < 1:
        raise ValueError("RUL confidence level must be between zero and one")
    if int(config["rul"]["minimum_trend_points"]) < 3:
        raise ValueError("At least three observations are required for a trend")
    weights = config["risk_fusion"]
    weight_names = ("residual_weight", "shap_weight", "classifier_weight", "anomaly_weight", "trend_weight")
    if any(name not in weights or float(weights[name]) < 0 for name in weight_names):
        raise ValueError("Risk-fusion evidence weights must be present and nonnegative")
    if not 0.999 <= sum(float(weights[name]) for name in weight_names) <= 1.001:
        raise ValueError("Risk-fusion evidence weights must sum to one")
    for name, entry in config["indicators"].items():
        if entry["threshold_key"] not in config["failure_thresholds"] or entry["subsystem"] not in config["subsystem_consequence_weight"]:
            raise ValueError(f"Invalid indicator policy: {name}")
        if float(entry["failure_direction"]) not in (-1, 1):
            raise ValueError(f"Invalid failure direction for {name}")
    return config
