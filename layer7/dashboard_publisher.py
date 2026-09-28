"""Publish a dashboard-ready, timestamped view of the Layer 7 assessment."""

from __future__ import annotations

import json
from typing import Any
from redis.exceptions import ResponseError
from redis_limits import xadd_bounded
from .risk_evidence import physical_indicator_signals


def build_dashboard_payload(
    event: dict[str, Any],
    risk_output: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Combine Layer 7 results with their Layer 2/6 evidence without inventing data."""
    black_box = event.get("black_box") or {}
    decision_eligible = bool(event.get("decision_eligible", event.get("classification_available", bool(black_box))))
    class_probabilities = dict(black_box.get("class_probabilities") or {})
    raw_fault_probability = max(
        (_finite_or_zero(value) for name, value in class_probabilities.items() if name != "nominal"),
        default=0.0,
    )
    provisional_threshold = float(policy["engine_state_thresholds"].get("provisional_fault_probability", 0.60))
    provisional_model_evidence = not decision_eligible and raw_fault_probability >= provisional_threshold
    use_model_evidence = decision_eligible or provisional_model_evidence
    shap = dict(black_box.get("shap_subsystem") or {})
    if "vibration" in shap:
        shap["mechanical"] = _finite_or_zero(shap.get("mechanical")) + _finite_or_zero(shap["vibration"])
    inference_anomaly = _finite_or_none(black_box.get("anomaly_score"))
    anomaly = inference_anomaly if use_model_evidence else None
    confidence = _finite_or_none(black_box.get("confidence"))
    quality = _finite_or_none(event.get("data_quality_score"))
    nis_by_subsystem = dict(event.get("nis_by_subsystem") or {})
    trends = list(risk_output.get("trends") or [])
    bayesian_rul = risk_output.get("bayesian_rul") or {}
    rul_validated = bool(bayesian_rul.get("validated_for_decisions", False))
    fusion = policy["risk_fusion"]
    classifier_fault_probability = max(
        (_finite_or_zero(value) for name, value in class_probabilities.items() if name != "nominal"),
        default=0.0,
    ) if use_model_evidence else 0.0
    risk_shap = shap if use_model_evidence else {}
    physical_signals, physical_detail = physical_indicator_signals(
        dict(event.get("indicators") or {}), policy)
    trend_by_subsystem: dict[str, float] = {}
    for trend in trends:
        subsystem = trend.get("subsystem")
        if subsystem:
            trend_by_subsystem[subsystem] = max(
                trend_by_subsystem.get(subsystem, 0.0),
                1.0 if trend.get("direction") == "degrading" else 0.0,
            )

    risk_evidence: dict[str, Any] = {}
    for subsystem, score in (risk_output.get("subsystem_risk") or {}).items():
        shap_magnitude = abs(_finite_or_zero(risk_shap.get(subsystem)))
        shap_signal = shap_magnitude / (1.0 + shap_magnitude)
        nis_value = _finite_or_zero(nis_by_subsystem.get(subsystem))
        nis_signal = min(nis_value / float(fusion["nis_inconsistency_threshold"]), 1.0)
        trend_signal = trend_by_subsystem.get(subsystem, 0.0)
        trend_or_nis_signal = max(trend_signal, nis_signal)
        physical_signal = physical_signals.get(subsystem, 0.0)
        physical_channels = {name: detail for name, detail in physical_detail.items()
                             if detail["subsystem"] == subsystem}
        components = {
            "physical_residual": float(fusion["residual_weight"]) * physical_signal,
            "shap": float(fusion["shap_weight"]) * shap_signal,
            "classifier": float(fusion["classifier_weight"]) * classifier_fault_probability,
            "anomaly": float(fusion["anomaly_weight"]) * (_finite_or_zero(anomaly)),
            "trend": float(fusion["trend_weight"]) * trend_or_nis_signal,
        }
        raw_score = min(100.0, max(0.0, 100.0 * sum(components.values())))
        consequence_weight = float(policy["subsystem_consequence_weight"][subsystem])
        maximum_consequence_weight = max(float(item) for item in policy["subsystem_consequence_weight"].values())
        risk_evidence[subsystem] = {
            "shap_contribution": _finite_or_none(risk_shap.get(subsystem)),
            "shap_signal": shap_signal,
            "classifier_fault_probability": classifier_fault_probability,
            "isolation_forest_anomaly_score": anomaly,
            "degradation_trend_signal": trend_signal,
            "nis_value": nis_value,
            "nis_signal": nis_signal,
            "trend_or_nis_signal": trend_or_nis_signal,
            "physical_indicator_signal": physical_signal,
            "physical_indicator_details": physical_channels,
            "weights": {
                "physical_residual": float(fusion["residual_weight"]),
                "shap": float(fusion["shap_weight"]),
                "classifier": float(fusion["classifier_weight"]),
                "anomaly": float(fusion["anomaly_weight"]),
                "trend": float(fusion["trend_weight"]),
            },
            "weighted_components": components,
            "reconstructed_subsystem_risk_score": raw_score,
            "consequence_weight": consequence_weight,
            "consequence_weighted_engine_score": raw_score * consequence_weight / maximum_consequence_weight,
            "data_quality_score": quality,
            "risk_score": _finite_or_none(score),
        }

    predicted_class = black_box.get("predicted_class")
    action = risk_output.get("recommended_action")
    has_model_result = bool(event.get("inference_available", event.get("classification_available", bool(black_box)))) and bool(black_box)
    degrading = any(trend.get("direction") == "degrading" for trend in trends)
    if risk_output.get("engine_state") in {"normal", "abnormal", "critical"}:
        health_status = risk_output["engine_state"]
    elif action in {"IMMEDIATE", "ABORT_RECOMMEND"}:
        health_status = "critical"
    elif decision_eligible and (predicted_class not in (None, "nominal") or black_box.get("anomaly_detected")
          or black_box.get("novel_degradation") or degrading or action not in (None, "NORMAL")):
        health_status = "abnormal"
    elif decision_eligible and has_model_result:
        health_status = "normal"
    else:
        health_status = "unknown"

    active_faults = list(risk_output.get("active_faults") or [])
    fault_category = predicted_class
    if fault_category is None and active_faults:
        fault_category = max(active_faults, key=lambda item: _finite_or_zero(item.get("probability"))).get("fault_class")
    if fault_category is None:
        fault_category = "unknown" if not has_model_result else "none"

    return {
        "schema_version": 1,
        "session_id": event.get("session_id"),
        "control_revision": event.get("control_revision"),
        "parameter_provenance": event.get("parameter_provenance", {}),
        "engine_state": risk_output.get("engine_state"),
        "fault_flags": risk_output.get("fault_flags") or {},
        "model_training_scope": black_box.get("training_scope"),
        "timestamp_ns": int(event["timestamp_ns"]),
        "engine_serial": event.get("engine_serial"),
        "sortie_id": event.get("sortie_id"),
        "classification_available": has_model_result,
        "classification_unavailable_reason": event.get("classification_unavailable_reason"),
        "inference_available": has_model_result,
        "decision_eligible": decision_eligible,
        "provisional_model_evidence": provisional_model_evidence,
        "decision_eligibility_reason": event.get("decision_eligibility_reason"),
        "synchronized_sensor_values": dict(event.get("telemetry") or {}),
        "synchronized_sensor_frame": dict(event.get("layer2_frame") or {}),
        "layer2_frame_source": "engine:synced:frames",
        "sensor_channel_valid": dict((event.get("layer2_frame") or {}).get("channel_valid") or {}),
        "sensor_channel_source": dict((event.get("layer2_frame") or {}).get("channel_source") or {}),
        "sensor_channel_age_ms": dict((event.get("layer2_frame") or {}).get("channel_age_ms") or {}),
        "sensor_channel_rate_hz": dict((event.get("layer2_frame") or {}).get("channel_rate_hz") or {}),
        "sensor_values_timestamp_ns": event.get("telemetry_timestamp_ns"),
        "sensor_values_timestamp_semantics": "timestamp of the latest synchronized Layer 2 sample carried through the current feature window",
        "sensor_value_units": _channel_units(),
        "lightgbm": {
            "predicted_class": predicted_class,
            "class_probabilities": class_probabilities,
            "confidence": confidence,
            "feature_attributions": black_box.get("shap_top_features", []),
            "subsystem_attributions": shap,
            "model_version": black_box.get("model_version"),
        },
        "isolation_forest": {
            "anomaly_score": inference_anomaly,
            "anomaly_detected": black_box.get("anomaly_detected"),
            "novel_degradation": black_box.get("novel_degradation"),
            "model_version": black_box.get("model_version"),
        },
        "fault_category": fault_category,
        "active_faults": active_faults,
        "risk_fusion": {
            "engine_risk_score": _finite_or_none(risk_output.get("engine_risk_score")),
            "engine_risk_confidence": _finite_or_none(risk_output.get("engine_risk_confidence")),
            "subsystem_risk": dict(risk_output.get("subsystem_risk") or {}),
            "subsystem_risk_confidence": dict(risk_output.get("subsystem_risk_confidence") or {}),
            "dominant_subsystem": risk_output.get("dominant_subsystem"),
            "evidence_by_subsystem": risk_evidence,
            "fusion_policy": dict(fusion),
            "subsystem_consequence_weights": dict(policy["subsystem_consequence_weight"]),
            "risk_action_thresholds": dict(policy["risk_action_thresholds"]),
            "redline_limits": dict(policy["redline_limits"]),
            "redline_hits": list(risk_output.get("redline_hits") or []),
            "data_quality_score": quality,
        },
        "rul": {
            "hours": _finite_or_none(risk_output.get("rul_hours")),
            "interval_hours": risk_output.get("rul_interval_hours"),
            "confidence_level": _finite_or_none(risk_output.get("rul_confidence_level")),
            "assumed_profile": risk_output.get("rul_assumed_profile"),
            "reason_unknown": (risk_output.get("rul_reason_unknown") if risk_output.get("rul_hours") is None else
                               None if rul_validated else
                               "Simulation estimate shown; configured RUL prior is not validated for maintenance decisions."),
            "prior_provenance": bayesian_rul.get("prior_provenance"),
            "validated_for_decisions": rul_validated,
        },
        "degradation_indicators": dict(event.get("indicators") or {}),
        "degradation_trends": trends,
        "sensor_degradation_trends": {
            trend["indicator"]: trend for trend in trends if trend.get("indicator")
        },
        "engine_health_status": health_status,
        "recommended_action": action,
        "action_rationale": risk_output.get("action_rationale"),
        "maintenance_plan": list(risk_output.get("maintenance_plan") or []),
        "operating_hours": event.get("operating_hours"),
        "operating_context": dict(event.get("operating_context") or {}),
        "nis_by_subsystem": nis_by_subsystem,
        "model_versions": dict(risk_output.get("model_versions") or {}),
        "layer7_output": risk_output,
    }


async def publish_dashboard_payload(
    redis_client: Any,
    stream: str,
    payload: dict[str, Any],
) -> str:
    """Idempotently publish one dashboard record using its assessment timestamp."""
    entry_id = f"{int(payload['timestamp_ns']) // 1_000_000}-0"
    existing = await redis_client.xrange(stream, min=entry_id, max=entry_id, count=1)
    if not existing:
        published_id = await xadd_bounded(
            redis_client,
            stream,
            {"payload": json.dumps(payload, separators=(",", ":"), allow_nan=False)},
            id=entry_id,
        )
        if not published_id:
            raise ResponseError("Dashboard assessment write failed")
    return entry_id


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _finite_or_zero(value: Any) -> float:
    return _finite_or_none(value) or 0.0


def _channel_units() -> dict[str, str]:
    # Reuse the canonical Layer 1 channel dictionary so dashboard units cannot drift.
    from layer1.schema import TELEMETRY_CHANNELS

    return {name: spec.unit for name, spec in TELEMETRY_CHANNELS.items()}
