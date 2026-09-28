"""Dependency-aware inference over the five approved Layer 3 ONNX models."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from layer4.config import load_config as load_ekf_config

from .config import Layer3Config
from .input_estimator import apply_estimates


class Session(Protocol):
    def run(self, output_names: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: tuple[str, ...]
    outputs: tuple[str, ...]
    file_name: str


SPECS = (
    ModelSpec("mvem", ("Pm_Pa_k", "Tm_K_k", "rpm", "throttle_pct", "ambient_temp_K",
              "ambient_press_Pa", "injection_timing_degBTDC", "mixture_afr", "cowl_flap_pos", "dt_s"),
              ("Pm_Pa_k1", "throttle_mass_flow_kg_s", "air_mass_flow_kg_s", "fuel_mass_flow_kg_s"),
              "mvem_int8_weights.onnx"),
    ModelSpec("wiebe", ("rpm", "throttle_pct", "ambient_temp_K", "ambient_press_Pa", "Pm_Pa",
              "air_mass_flow_kg_s", "fuel_mass_flow_kg_s", "injection_timing_degBTDC", "mixture_afr", "cowl_flap_pos"),
              ("T_ind_Nm", "T_head_K", "T_exhaust_K", "wiebe_a", "wiebe_m", "burn_duration_deg", "theta_start_degBTDC"),
              "wiebe_int8_weights.onnx"),
    ModelSpec("battery_alternator", ("rpm", "throttle_pct", "ambient_temp_K", "T_oil_K_k", "soc_pct_k", "dt_s"),
              ("batt_volts", "alt_amps", "T_alt_Nm", "soc_pct_k1", "battery_current_A"),
              "battery_alternator_int8_weights.onnx"),
    ModelSpec("oil", ("rpm", "throttle_pct", "ambient_temp_K", "ambient_press_Pa", "T_head_K_k",
              "T_oil_K_k", "true_T_fric_Nm", "cowl_flap_pos", "dt_s"),
              ("T_oil_K_k1", "oil_press_kPa", "oil_viscosity_Pa_s"), "oil_int8_weights.onnx"),
    ModelSpec("vibration", ("rpm", "map_kPa", "ambient_press_kPa", "oil_temp_degC", "T_ind_Nm",
              "T_fric_Nm", "T_alt_Nm", "angular_accel_rad_s2"),
              ("A_1x_g_rms", "A_2x_mechanical_g_rms", "A_firing_g_rms", "A_2x_total_g_rms",
               "vibration_rms_g", "vib_kurtosis", "T_prop_Nm"), "vibration_int8_weights.onnx"),
)

PREDICTION_OWNERS = {
    "map_kPa": "mvem", "fuel_flow_Lph": "mvem", "cht_degC": "wiebe", "egt_degC": "wiebe",
    "oil_press_kPa": "oil", "oil_temp_degC": "oil", "vibration_rms_g": "vibration",
    "batt_volts": "battery_alternator", "alt_amps": "battery_alternator",
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class PhysicsInference:
    """Run models only when every physical input is valid and available."""

    def __init__(self, config: Layer3Config | None = None, sessions: dict[str, Session] | None = None) -> None:
        self.config = config or Layer3Config()
        self.specs = {spec.name: spec for spec in SPECS}
        self.sessions = sessions if sessions is not None else self._load_sessions()
        self.versions = self._model_versions()
        self.training_domains = {}
        for spec in SPECS:
            path = self._artifact_path(spec).with_name("normalization.json")
            self.training_domains[spec.name] = json.loads(path.read_text()) if path.is_file() else {}
        self.previous_timestamp_ns: int | None = None
        self.previous_rpm: float | None = None
        self.session_id = None
        self.nominal_prediction_bias: dict[str, float] = {}
        self._validate_prediction_coverage()

    def _artifact_path(self, spec: ModelSpec) -> Path:
        return self.config.artifact_root / spec.name / "artifacts" / spec.file_name

    def _load_sessions(self) -> dict[str, Session]:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Layer 3 runtime requires onnxruntime") from error
        sessions: dict[str, Session] = {}
        for spec in SPECS:
            artifact = self._artifact_path(spec)
            normalization = artifact.with_name("normalization.json")
            if not artifact.is_file() or not normalization.is_file():
                raise FileNotFoundError(f"missing Layer 3 artifact contract for {spec.name}")
            names = tuple(json.loads(normalization.read_text(encoding="utf-8")))
            if names != spec.features:
                raise ValueError(f"{spec.name} normalization feature order does not match runtime mapping")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            sessions[spec.name] = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
        return sessions

    def _model_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        for spec in SPECS:
            artifact = self._artifact_path(spec)
            versions[spec.name] = hashlib.sha256(artifact.read_bytes()).hexdigest()[:16] if artifact.is_file() else "injected"
        return versions

    def _validate_prediction_coverage(self) -> None:
        required = {channel for subfilter in load_ekf_config().subfilters.values() for channel in subfilter.channels}
        missing = required - set(PREDICTION_OWNERS)
        missing_models = {owner for channel, owner in PREDICTION_OWNERS.items() if channel in required and owner not in self.sessions}
        if missing or missing_models:
            raise ValueError(f"Layer 3 cannot cover EKF channels; unmapped={sorted(missing)}, models={sorted(missing_models)}")

    def _inputs(self, frame: dict[str, Any], dt_s: float) -> dict[str, float]:
        values, valid = frame.get("values", {}), frame.get("channel_valid", {})
        state = {name: float(value) for name, value in values.items() if valid.get(name) is True and _finite(value)}
        if "map_kPa" in state:
            state["Pm_Pa_k"] = state["Pm_Pa"] = state["map_kPa"] * 1000.0
        if "ambient_temp_degC" in state:
            state["ambient_temp_K"] = state["ambient_temp_degC"] + 273.15
        if "ambient_press_kPa" in state:
            state["ambient_press_Pa"] = state["ambient_press_kPa"] * 1000.0
        if "oil_temp_degC" in state:
            state["T_oil_K_k"] = state["oil_temp_degC"] + 273.15
        if "cht_degC" in state:
            state["T_head_K_k"] = state["cht_degC"] + 273.15
        if "cowl_flap_pct" in state:
            state["cowl_flap_pos"] = state["cowl_flap_pct"] / 100.0
        if "battery_soc_pct" in state:
            state["soc_pct_k"] = state["battery_soc_pct"]
        if "friction_torque_Nm" in state:
            state["true_T_fric_Nm"] = state["T_fric_Nm"] = state["friction_torque_Nm"]
        state["dt_s"] = dt_s
        if self.previous_rpm is not None and "rpm" in state and dt_s > 0:
            state["angular_accel_rad_s2"] = (state["rpm"] - self.previous_rpm) * 2.0 * math.pi / 60.0 / dt_s
        return state

    @staticmethod
    def _run(spec: ModelSpec, session: Session, state: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        missing = [name for name in spec.features if not _finite(state.get(name))]
        if missing:
            return {}, missing
        matrix = np.asarray([[state[name] for name in spec.features]], dtype=np.float32)
        raw = np.asarray(session.run(None, {"features": matrix})[0]).reshape(-1)
        if len(raw) != len(spec.outputs) or not np.isfinite(raw).all():
            raise ValueError(f"{spec.name} returned an invalid output tensor")
        return {name: float(value) for name, value in zip(spec.outputs, raw, strict=True)}, []

    def process(self, frame: dict[str, Any]) -> dict[str, Any]:
        session = (frame.get("context") or {}).get("session_id")
        if session and session != self.session_id:
            self.previous_timestamp_ns = self.previous_rpm = None
            self.nominal_prediction_bias.clear()
            self.session_id = session
        timestamp_ns = int(frame["master_timestamp_ns"])
        dt_s = 0.01 if self.previous_timestamp_ns is None else max((timestamp_ns - self.previous_timestamp_ns) / 1e9, 1e-6)
        state, derived_input_provenance = apply_estimates(self._inputs(frame, dt_s))
        state["cowl_flap_pos"] = state["cowl_flap_pct"] / 100.0
        state["soc_pct_k"] = state["battery_soc_pct"]
        state["true_T_fric_Nm"] = state["T_fric_Nm"] = state["friction_torque_Nm"]
        if "Tm_K_k" not in state and "ambient_temp_K" in state:
            state["Tm_K_k"] = state["ambient_temp_K"]
        status: dict[str, dict[str, Any]] = {}
        for spec in SPECS:
            try:
                state_model = dict(state)
                for name, bounds in self.training_domains[spec.name].items():
                    if _finite(state.get(name)):
                        state_model[name] = float(np.clip(state[name], float(bounds[0]), float(bounds[1])))
                result, missing = self._run(spec, self.sessions[spec.name], state_model)
                if missing:
                    status[spec.name] = {"available": False, "missing_inputs": missing}
                    continue
                state.update(result)
                status[spec.name] = {"available": True, "missing_inputs": []}
            except (TypeError, ValueError, RuntimeError) as error:
                status[spec.name] = {"available": False, "error": str(error)}

        predicted: dict[str, float] = {}
        candidates = {
            "map_kPa": state.get("Pm_Pa_k1", float("nan")) / 1000.0,
            "cht_degC": state.get("T_head_K", float("nan")) - 273.15,
            "egt_degC": state.get("T_exhaust_K", float("nan")) - 273.15,
            "oil_press_kPa": state.get("oil_press_kPa"),
            "oil_temp_degC": state.get("T_oil_K_k1", float("nan")) - 273.15,
            "vibration_rms_g": state.get("vibration_rms_g"),
            "batt_volts": state.get("batt_volts"),
            "alt_amps": state.get("alt_amps"),
        }
        if self.config.fuel_density_kg_L and _finite(state.get("fuel_mass_flow_kg_s")):
            candidates["fuel_flow_Lph"] = state["fuel_mass_flow_kg_s"] * 3600.0 / self.config.fuel_density_kg_L
        for channel, value in candidates.items():
            if _finite(value) and status.get(PREDICTION_OWNERS.get(channel, ""), {}).get("available"):
                predicted[channel] = float(value)
        # An implausible model output is unavailable evidence, not a perfect fit.
        bounds = {"cht_degC": (60.0, 220.0), "egt_degC": (350.0, 850.0),
                  "fuel_flow_Lph": (1.0, 45.0)}
        for channel, (low, high) in bounds.items():
            if channel in predicted and not low <= predicted[channel] <= high:
                del predicted[channel]
        # The ONNX models and simulator have different nominal baselines. Learn
        # their offset only during untouched Standard Mode, then freeze it during
        # overrides/faults so changes remain visible to EKF and Layer 7.
        context = frame.get("context") or {}
        nominal = (str(context.get("injected_fault") or "NONE").upper() == "NONE"
                   and not context.get("manual_override"))
        values = frame.get("values") or {}
        valid = frame.get("channel_valid") or {}
        for channel, raw_prediction in list(predicted.items()):
            observed = values.get(channel)
            if nominal and valid.get(channel) is True and _finite(observed):
                difference = float(observed) - raw_prediction
                prior = self.nominal_prediction_bias.get(channel)
                self.nominal_prediction_bias[channel] = (difference if prior is None
                                                         else prior + 0.02 * (difference - prior))
            if channel in self.nominal_prediction_bias:
                predicted[channel] = raw_prediction + self.nominal_prediction_bias[channel]

        output = dict(frame)
        output.update({
            "master_timestamp_ns": str(timestamp_ns), "dt_s": dt_s, "predicted": predicted,
            "prediction_valid": {channel: channel in predicted for channel in PREDICTION_OWNERS},
            "model_versions": self.versions, "model_status": status,
            "derived_input_provenance": derived_input_provenance,
            "physics_state": {key: value for key, value in state.items() if key not in frame.get("values", {})},
            "context": dict(frame.get("context") or {}),
        })
        self.previous_timestamp_ns = timestamp_ns
        if _finite(state.get("rpm")):
            self.previous_rpm = state["rpm"]
        return output
