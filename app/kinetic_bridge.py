"""Read Layer 7 dashboard output into the existing MALE UAV API and UI."""
from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass

from layer1.adapter import PARAMETERS, to_layer1_events
from layer1.service import Layer1Service
from app.physics_model import predict_expected


@dataclass
class PipelineState:
    payload: dict

    def to_dict(self):
        return dict(self.payload)

    @property
    def timestamp(self):
        return self.payload["timestamp"]

    @property
    def anomaly_detected(self):
        return self.payload.get("anomaly") is True

    @property
    def diagnostic(self):
        return self.payload["diagnostic"]


def legacy_payload(telemetry: dict, assessment: dict, physics: dict, ekf: dict) -> dict:
    """Only expose values actually supplied by the new pipeline; null = unavailable."""
    payload = dict(telemetry)
    # Layer 4 carries the Layer 3 prior used for this exact residual timestamp.
    predicted = dict(physics.get("predicted") or {})
    if ekf.get("predicted"):
        predicted.update(ekf["predicted"])
    prediction_valid = physics.get("prediction_valid") or {}
    if ekf.get("predicted"):
        prediction_valid = {channel: channel in predicted for channel in predicted}
    residual = ekf.get("residual") or {}
    residual_source: dict[str, str] = {}
    for source in ("egt", "cht", "oil_temp", "map", "oil_press", "fuel_flow", "iat", "vibration"):
        spec = PARAMETERS[source]
        is_valid = prediction_valid.get(spec.channel, True) if prediction_valid else True
        expected = predicted.get(spec.channel) if is_valid else None
        drift = residual.get(spec.channel)
        measured = telemetry.get(source)
        payload[source + "_measured"] = measured
        exp_val = ((expected - spec.offset) / spec.scale) if expected is not None else None
        payload[source + "_expected"] = exp_val
        if drift is not None:
            payload[source + "_residual"] = drift / spec.scale
            residual_source[source] = "ekf"
        elif measured is not None and exp_val is not None:
            payload[source + "_residual"] = measured - exp_val
            residual_source[source] = "model_difference"
        else:
            payload[source + "_residual"] = None
        payload["res_" + ("oil" if source == "oil_temp" else source)] = payload[source + "_residual"]
    payload["residual_source"] = residual_source
    ml = assessment.get("lightgbm") or {}
    anomaly = assessment.get("isolation_forest") or {}
    rul = assessment.get("rul") or {}
    status = assessment.get("engine_health_status", "unknown")
    risk = assessment.get("risk_fusion") or {}
    decision_eligible = bool(assessment.get("decision_eligible", assessment.get("classification_available", False)))
    diagnostic = assessment.get("action_rationale") or assessment.get("classification_unavailable_reason") or "Waiting for a complete pipeline assessment."
    health = (100.0 - float(risk["engine_risk_score"])
              if isinstance(risk.get("engine_risk_score"), (int, float)) else None)
    redline_hits = risk.get("redline_hits") or []
    payload.update({
        "health": health,
        "health_index": health, "degradation": None,
        "health_status": status.upper(),
        "fault_class": (ml.get("predicted_class") or "UNAVAILABLE") if decision_eligible else "UNAVAILABLE",
        "fault_confidence": ml.get("confidence") if decision_eligible else None,
        "anomaly": anomaly.get("anomaly_detected"),
        "anomaly_detected": anomaly.get("anomaly_detected"),
        "anomaly_rule": bool(redline_hits),
        "anomaly_score_ml": anomaly.get("anomaly_score"),
        "anomaly_flag_ml": ("UNAVAILABLE" if anomaly.get("anomaly_detected") is None else
                            "ABNORMAL" if anomaly["anomaly_detected"] else "NORMAL"),
        "fault_suggestion": (ml.get("predicted_class") or "UNAVAILABLE") if decision_eligible else "UNAVAILABLE",
        "rul_hours": rul.get("hours"), "rul_reason_unknown": rul.get("reason_unknown"), "diagnostic": diagnostic,
        "rul_validated_for_decisions": rul.get("validated_for_decisions"),
        "residual_timestamp_ns": ekf.get("master_timestamp_ns"),
        "residual_measurements": layer2_frame_to_dashboard_values(ekf.get("layer2_frame") or {}),
        "explanation": {"main_indicator": (assessment.get("risk_fusion") or {}).get("dominant_subsystem", "Unavailable"),
                        "diagnostic_message": diagnostic, "severity": status.upper(),
                        "health_trend": "See per-indicator trends in Pipeline Results"},
        "pipeline": assessment,
        "model_status": physics.get("model_status", {}),
        "unsupported_metrics": {"degradation": "Legacy accumulated wear has no equivalent."},
    })
    return payload


def fill_estimated_residuals(payload: dict, sensor_frame: dict) -> None:
    """Fill display-only gaps from the dynamic demonstration physics model.

    These estimates never enter the EKF, feature windows, or risk fusion.
    Missing sensor measurements stay missing, and genuine predictions win.
    """
    values = sensor_frame.get("values") or {}
    valid = sensor_frame.get("channel_valid") or {}
    controls = ("rpm", "throttle_pct", "ambient_temp_degC", "altitude_m")
    if any(valid.get(name) is not True or not isinstance(values.get(name), (int, float))
           or not math.isfinite(values[name]) for name in controls[:3]):
        return
    altitude = values.get("altitude_m", 0.0)
    if not isinstance(altitude, (int, float)) or not math.isfinite(altitude):
        altitude = 0.0
    estimate = predict_expected(values["rpm"], values["throttle_pct"] / 100.0,
                                values["ambient_temp_degC"], altitude)
    estimates = {
        "egt": estimate.egt_expected, "cht": estimate.cht_expected,
        "oil_temp": estimate.oil_temp_expected, "map": estimate.map_expected,
        "oil_press": estimate.oil_press_expected,
        "fuel_flow": estimate.fuel_flow_expected,
        "iat": estimate.iat_expected,
        "vibration": estimate.vibration_expected,
    }
    sources = payload.setdefault("residual_source", {})
    for source, expected in estimates.items():
        if valid.get(PARAMETERS[source].channel) is not True:
            continue
        measured = payload.get(source + "_measured")
        if payload.get(source + "_expected") is not None or not isinstance(measured, (int, float)):
            continue
        if not math.isfinite(measured):
            continue
        payload[source + "_expected"] = expected
        payload[source + "_residual"] = measured - expected
        payload["res_" + ("oil" if source == "oil_temp" else source)] = measured - expected
        sources[source] = "estimated_physics"


def layer2_frame_to_dashboard_values(frame: dict) -> dict:
    """Adapt raw synchronized Layer 2 channels to the existing UI units."""
    values = frame.get("values") or {}
    result: dict = {}
    specs_by_channel = {spec.channel: (name, spec) for name, spec in PARAMETERS.items()}
    for channel, value in values.items():
        item = specs_by_channel.get(channel)
        if item is not None:
            name, spec = item
            result[name] = (value - spec.offset) / spec.scale
    if "ambient_temp_degC" in values:
        result["ambient_temp"] = values["ambient_temp_degC"]
    if "ambient_press_kPa" in values:
        result["baro_pressure"] = values["ambient_press_kPa"] / 3.386389
    if "altitude_m" in values:
        result["altitude"] = values["altitude_m"] * 3.28084
    if "air_density_ratio" in values:
        result["air_density_ratio"] = values["air_density_ratio"]
    context = frame.get("context") or {}
    if context.get("atmospheric_profile") is not None:
        result["atmospheric_profile"] = context["atmospheric_profile"]
    return result


def _layer7_estimates(assessment: dict) -> dict:
    """Map only Layer 7 estimates into dashboard fields; keep raw sensors separate."""
    risk = assessment.get("risk_fusion") or {}
    rul = assessment.get("rul") or {}
    health = (100.0 - float(risk["engine_risk_score"])
              if isinstance(risk.get("engine_risk_score"), (int, float)) else None)
    result = {"health": health, "health_index": health,
              "anomaly_rule": bool(risk.get("redline_hits") or []),
              "rul_hours": rul.get("hours"), "rul_reason_unknown": rul.get("reason_unknown"),
              "rul_validated_for_decisions": rul.get("validated_for_decisions"),
              "pipeline": assessment}
    for name, value in (risk.get("subsystem_risk") or {}).items():
        result["layer7_risk_" + name] = value
    return result


def _assessment_layer2_frame(assessment: dict, session_id: str) -> dict:
    frame = assessment.get("synchronized_sensor_frame") or {}
    context = frame.get("context") or {}
    frame_session = context.get("session_id")
    timestamp = int(frame.get("master_timestamp_ns") or 0)
    if (not frame or not frame_session or frame_session != session_id
            or time.time_ns() - timestamp > 15_000_000_000):
        return {}
    valid = frame.get("channel_valid") or {}
    values = frame.get("values") or {}
    frame["values"] = {name: value for name, value in values.items() if valid.get(name) is True}
    return frame


class KineticTwin:
    """Feed Layer 1 and read Layer 7 through Redis; no alternate prediction path."""
    def __init__(self, *, ambient_temp=25.0, history_maxlen=500, sample_rate_hz=2.0):
        self.layer1 = Layer1Service()
        self.redis = self.layer1.redis
        self.sample_rate_hz = sample_rate_hz
        self.history = deque(maxlen=history_maxlen)
        self.reset()

    def reset(self):
        self.session_id = uuid.uuid4().hex
        self.history.clear()
        self.step_count = 0

    async def close(self):
        await self.layer1.close()

    async def _latest(self, stream):
        try:
            rows = await self.redis.xrevrange(stream, count=1)
            if not rows:
                return {}
            item = json.loads(rows[0][1]["payload"])
            session = item.get("session_id") or (item.get("context") or {}).get("session_id")
            if session and self.session_id and session != self.session_id:
                return {}
            return item
        except Exception:
            return {}

    async def _latest_layer2_frame(self):
        """Return the current Layer 2 synchronized sensor frame for the UI."""
        try:
            rows = await self.redis.xrevrange("engine:synced:frames", count=1)
            if not rows:
                return {}
            frame = json.loads(rows[0][1]["payload"])
            context = frame.get("context") or {}
            session = context.get("session_id") or frame.get("session_id")
            if session and self.session_id and session != self.session_id:
                return {}
            return frame
        except Exception:
            return {}

    async def _latest_valid_ekf(self):
        """Find the latest EKF observation that actually updated a sensor."""
        try:
            rows = await self.redis.xrevrange("engine:ekf:residuals", count=100)
            for _, fields in rows:
                item = json.loads(fields["payload"])
                session = (item.get("context") or {}).get("session_id") or item.get("session_id")
                if session and self.session_id and session != self.session_id:
                    continue
                return item
        except Exception:
            pass
        return {}

    async def update(self, telemetry):
        snapshot = {**telemetry, "sortie_id": self.session_id}
        await self.layer1.publish_snapshot(to_layer1_events(snapshot, sample_rate_hz=self.sample_rate_hz,
                                                              session_id=self.session_id))
        assessment, physics, ekf, sensor_frame = await asyncio.gather(
            self._latest("engine:dashboard:telemetry"), self._latest("engine:physics:predictions"),
            self._latest_valid_ekf(), self._latest_layer2_frame())
        # Sensor readings shown by the dashboard are sourced only from Layer 2.
        # Simulator values remain upstream input and are never a display fallback.
        payload = legacy_payload({}, assessment, physics, ekf)
        # The API simulation loop advances its clock from PipelineState.timestamp.
        # Keep the simulator clock even while all displayed measurements are
        # sourced exclusively from the synchronized Layer 2 frame.
        payload["timestamp"] = telemetry.get("timestamp", self.step_count * (1.0 / self.sample_rate_hz))
        for name in PARAMETERS:
            payload[name] = None
            payload[name + "_measured"] = None
        if not sensor_frame:
            sensor_frame = _assessment_layer2_frame(assessment, self.session_id)
        frame_values = sensor_frame.get("values") or {}
        if frame_values:
            payload["layer2_sensor_frame"] = sensor_frame
            payload["sensor_values_timestamp_ns"] = sensor_frame.get("master_timestamp_ns")
            payload["sensor_channel_valid"] = sensor_frame.get("channel_valid", {})
            payload["sensor_channel_source"] = sensor_frame.get("channel_source", {})
            payload["sensor_channel_age_ms"] = sensor_frame.get("channel_age_ms", {})
            payload["sensor_channel_rate_hz"] = sensor_frame.get("channel_rate_hz", {})
            frame_ui_values = layer2_frame_to_dashboard_values(sensor_frame)
            payload.update(frame_ui_values)
            for key in frame_ui_values:
                payload[key + "_measured"] = payload[key]
            for source in ("egt", "cht", "oil_temp", "map", "oil_press", "fuel_flow", "iat", "vibration"):
                meas = payload.get(source + "_measured") if payload.get(source + "_measured") is not None else payload.get(source)
                exp = payload.get(source + "_expected")
                if payload.get(source + "_residual") is None and meas is not None and exp is not None:
                    payload[source + "_residual"] = round(meas - exp, 3)
                    payload["residual_source"][source] = "model_difference"
            fill_estimated_residuals(payload, sensor_frame)
            payload["channel_valid"] = sensor_frame.get("channel_valid", {})
            payload["channel_source"] = sensor_frame.get("channel_source", {})
            payload["channel_age_ms"] = sensor_frame.get("channel_age_ms", {})
            payload["channel_rate_hz"] = sensor_frame.get("channel_rate_hz", {})
        for channel, parameter_name in (("ambient_temp_degC", "ambient_temp"),
                                        ("ambient_press_kPa", "baro_pressure"),
                                        ("altitude_m", "altitude"),
                                        ("air_density_ratio", "air_density_ratio")):
            if (sensor_frame.get("channel_valid") or {}).get(channel) is not True:
                payload[parameter_name] = None
        for channel, is_valid in (sensor_frame.get("channel_valid") or {}).items():
            if not is_valid:
                spec_entry = next(((name, spec) for name, spec in PARAMETERS.items() if spec.channel == channel), None)
                if spec_entry:
                    name, _ = spec_entry
                    payload[name] = None
                    payload[name + "_measured"] = None
                    payload[name + "_expected"] = None
                    payload[name + "_residual"] = None
                    payload.get("residual_source", {}).pop(name, None)
        context = sensor_frame.get("context") or {}
        if context:
            payload["override_values"] = context.get("override_values", snapshot.get("override_values", {}))
        payload["control_revision"] = context.get("control_revision", snapshot.get("control_revision", 0))
        payload["parameter_provenance"] = context.get(
            "parameter_provenance", snapshot.get("provenance", {}))
        payload.update(_layer7_estimates(assessment))
        payload["data_flow_status"] = {
            "layer2_sensor_frame": "available" if sensor_frame else "not_embedded_or_stale",
            "layer7_assessment": "available" if assessment else "waiting_or_stale",
        }
        payload["session_id"] = self.session_id
        payload["pipeline_status"] = "available" if assessment else "waiting_or_stale"
        payload["assessment_control_revision"] = assessment.get("control_revision")
        payload["control_pending"] = (not assessment or
                                       payload["control_revision"] != assessment.get("control_revision"))
        if payload["control_pending"]:
            payload["pipeline_status"] = "control_revision_pending"
        state = PipelineState(payload)
        self.history.append(state)
        self.step_count += 1
        return state
