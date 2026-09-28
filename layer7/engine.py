"""Auditable risk fusion, trend assessment, and trajectory-based RUL estimation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, sqrt
from typing import Any
import numpy as np

from .config import load_config
from .bayesian_rul import BayesianPIRUL
from .risk_evidence import physical_indicator_signals


@dataclass(frozen=True)
class TrendAssessment:
    indicator: str; subsystem: str; slope_per_op_hour: float | None
    slope_interval: tuple[float, float] | None; direction: str; change_point_detected: bool
    time_to_threshold_h: float | None; time_to_threshold_interval_h: tuple[float, float] | None
    n_points: int; horizon_h: float

@dataclass(frozen=True)
class FaultAssessment:
    fault_class: str; probability: float; confidence: float; subsystem: str

@dataclass(frozen=True)
class MaintenanceRecommendation:
    subsystem: str; task_id: str; task_text: str; urgency: str
    deadline_operating_hours: float | None; evidence: list[str]; rationale: str; confidence: float

@dataclass(frozen=True)
class ParamEstimate:
    value: float; interval: tuple[float, float]; n_points: int

@dataclass(frozen=True)
class RiskRULOutput:
    timestamp_ns: int; engine_risk_score: float; engine_risk_confidence: float
    subsystem_risk: dict[str, float]; subsystem_risk_confidence: dict[str, float]
    rul_hours: float | None; rul_interval_hours: tuple[float, float] | None
    rul_confidence_level: float; rul_assumed_profile: str; dominant_subsystem: str
    active_faults: list[FaultAssessment]; recommended_action: str; action_rationale: str
    novel_degradation_active: bool; trends: list[TrendAssessment]
    maintenance_plan: list[MaintenanceRecommendation]; health_parameters: dict[str, ParamEstimate]
    data_quality_score: float; model_versions: dict[str, str]; rul_reason_unknown: str | None = None
    engine_state: str | None = None
    fault_flags: dict[str, dict[str, float | bool]] | None = None
    bayesian_rul: dict[str, Any] | None = None
    redline_hits: list[str] | None = None

    def to_dict(self) -> dict[str, Any]: return asdict(self)


class Layer7Engine:
    """Stateful per-engine processor. Inputs are deliberately plain dictionaries."""
    def __init__(self, config_path: str | None = None, pi_rul_prior_path: str | None = None) -> None:
        self.config = load_config(config_path); self.history: dict[str, list[tuple[float, float]]] = {}
        self.previous_slopes: dict[str, float] = {}; self.escalation_counts: dict[str, int] = {}
        self.pi_rul = BayesianPIRUL(pi_rul_prior_path)

    def _trend(self, name: str) -> TrendAssessment:
        policy = self.config["indicators"][name]; samples = self.history.get(name, [])
        minimum = int(self.config["rul"]["minimum_trend_points"])
        horizon = samples[-1][0] - samples[0][0] if len(samples) > 1 else 0.0
        if len(samples) < minimum or horizon < float(self.config["rul"]["minimum_horizon_hours"]):
            return TrendAssessment(name, policy["subsystem"], None, None, "insufficient_history", False, None, None, len(samples), horizon)
        x, y = np.asarray([point[0] for point in samples]), np.asarray([point[1] for point in samples])
        design = np.column_stack((np.ones(len(x)), x - x.mean()))
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None); residual = y - design @ beta
        variance = max(float(residual @ residual) / max(len(x) - design.shape[1], 1), np.finfo(float).eps)
        slope_se = sqrt(variance / max(float(np.sum((x - x.mean()) ** 2)), np.finfo(float).eps))
        z = sqrt(2.0) * _erfinv(float(self.config["rul"]["confidence_level"]))
        slope, interval = float(beta[1]), (float(beta[1] - z * slope_se), float(beta[1] + z * slope_se))
        signed_slope = float(policy["failure_direction"]) * slope
        signed_interval = sorted((float(policy["failure_direction"]) * interval[0], float(policy["failure_direction"]) * interval[1]))
        if signed_interval[0] > 0: direction = "degrading"
        elif signed_interval[1] < 0: direction = "improving"
        else: direction = "stable"
        threshold = float(self.config["failure_thresholds"][policy["threshold_key"]]); current = float(y[-1]); direction_sign = float(policy["failure_direction"])
        rate = direction_sign * slope; distance = direction_sign * (threshold - current)
        t = distance / rate if rate > 0 and distance >= 0 else (0.0 if distance < 0 else None)
        changed = name in self.previous_slopes and abs(slope - self.previous_slopes[name]) > float(self.config["rul"]["slope_stable_z"]) * slope_se
        self.previous_slopes[name] = slope
        return TrendAssessment(name, policy["subsystem"], slope, interval, direction, changed, t, None, len(samples), horizon)

    def _stress(self, context: dict[str, float]) -> float:
        c, r = self.config["rul"]["stress_coefficients"], self.config["rul"]
        altitude = max(float(context.get("altitude_m", r["reference_altitude_m"])) - float(r["reference_altitude_m"]), 0.0)
        temp = max(float(context.get("ambient_temp_degC", r["reference_ambient_temp_degC"])) - float(r["reference_ambient_temp_degC"]), 0.0)
        return 1 + altitude / 1000 * float(c["altitude_factor_per_1000m"]) + temp / 10 * float(c["ambient_temp_factor_per_10C"]) + float(context.get("thermal_cycles_per_hour", 0)) * float(c["thermal_cycle_factor"])

    def _rul(self, trends: list[TrendAssessment], quality: float, context: dict[str, float]) -> tuple[float | None, tuple[float, float] | None, str | None]:
        usable = [trend for trend in trends if trend.direction == "degrading" and trend.slope_per_op_hour is not None]
        if not usable:
            minimum_points = int(self.config["rul"]["minimum_trend_points"])
            minimum_horizon = float(self.config["rul"]["minimum_horizon_hours"])
            history_ready = any(trend.n_points >= minimum_points and trend.horizon_h >= minimum_horizon
                                for trend in trends)
            if not history_ready:
                return None, None, (f"RUL is waiting for at least {minimum_points} trend samples across "
                                    f"{minimum_horizon:g} engine operating hours.")
            return None, None, "No currently degrading sensor trend; RUL cannot be estimated."
        samples, seed = int(self.config["rul"]["monte_carlo_samples"]), int(self.config["rul"]["random_seed"])
        rng, times = np.random.default_rng(seed), []
        z = sqrt(2.0) * _erfinv(float(self.config["rul"]["confidence_level"])); stress = self._stress(context)
        quality_factor = 1 + (1 - quality) * float(self.config["risk_fusion"]["uncertainty_quality_multiplier"])
        for trend in usable:
            width = (trend.slope_interval[1] - trend.slope_interval[0]) / (2 * z) * quality_factor
            slopes = rng.normal(trend.slope_per_op_hour, max(width, np.finfo(float).eps), samples)
            policy = self.config["indicators"][trend.indicator]; latest = self.history[trend.indicator][-1][1]
            threshold = float(self.config["failure_thresholds"][policy["threshold_key"]])
            sampled_thresholds = rng.normal(threshold, abs(threshold) * float(self.config["rul"]["threshold_uncertainty_fraction"]), samples)
            sampled_current = rng.normal(latest, max(abs(latest), abs(threshold)) * float(self.config["rul"]["measurement_noise_fraction"]) * quality_factor, samples)
            distance = float(policy["failure_direction"]) * (sampled_thresholds - sampled_current)
            rates = float(policy["failure_direction"]) * slopes * stress
            values = np.where(distance <= 0, 0.0, np.where(rates > 0, distance / rates, np.inf))
            times.append(values)
        weakest = np.min(np.vstack(times), axis=0); finite = weakest[np.isfinite(weakest)]
        if len(finite) < samples // 2: return None, None, "Trajectory uncertainty prevents a defensible RUL estimate."
        alpha = (1 - float(self.config["rul"]["confidence_level"])) / 2
        return float(np.median(finite)), (float(np.quantile(finite, alpha)), float(np.quantile(finite, 1 - alpha))), None

    def process(self, event: dict[str, Any]) -> RiskRULOutput:
        timestamp = int(event["timestamp_ns"])
        raw_hours = event.get("operating_hours")
        has_operating_hours = isinstance(raw_hours, (int, float)) and not isinstance(raw_hours, bool) and np.isfinite(raw_hours)
        op_hours = float(raw_hours) if has_operating_hours else None
        quality = float(np.clip(event.get("data_quality_score", 1.0), self.config["risk_fusion"]["minimum_data_quality"], 1.0))
        indicators = event.get("indicators", {})
        if op_hours is not None:
            for name, value in indicators.items():
                if name in self.config["indicators"] and value is not None: self.history.setdefault(name, []).append((op_hours, float(value)))
        trends = [self._trend(name) for name in self.config["indicators"] if name in self.history]
        inference_output = event.get("black_box", {})
        decision_eligible = bool(event.get("decision_eligible", event.get("classification_available", bool(inference_output))))
        raw_probabilities = inference_output.get("class_probabilities", {})
        raw_fault_signal = max((float(value) for name, value in raw_probabilities.items() if name != "nominal"), default=0.0)
        state_policy = self.config["engine_state_thresholds"]
        provisional_threshold = float(state_policy.get("provisional_fault_probability", 0.60))
        # Unverified development models stay explicitly ineligible for validated
        # operational decisions, but a strong simulated fault prediction must not
        # be silently rendered as a healthy engine in this digital twin.
        provisional_fault = not decision_eligible and raw_fault_signal >= provisional_threshold
        use_model_evidence = decision_eligible or provisional_fault
        black_box = inference_output if use_model_evidence else {}
        shap = dict(black_box.get("shap_subsystem", {}))
        if "vibration" in shap:
            shap["mechanical"] = float(shap.get("mechanical", 0.0)) + float(shap["vibration"])
        anomaly, confidence = float(black_box.get("anomaly_score", 0)), float(black_box.get("confidence", 0))
        class_probabilities = black_box.get("class_probabilities", {})
        fault_signal = max((float(value) for name, value in class_probabilities.items() if name != "nominal"), default=0.0)
        subsystem_risk: dict[str, float] = {}; subsystem_conf: dict[str, float] = {}; rf = self.config["risk_fusion"]
        physical_signal, physical_detail = physical_indicator_signals(indicators, self.config)
        for subsystem, weight in self.config["subsystem_consequence_weight"].items():
            magnitude = abs(float(shap.get(subsystem, 0))); shap_signal = magnitude / (1 + magnitude)
            related = [trend for trend in trends if trend.subsystem == subsystem]
            trend_signal = max((1.0 if trend.direction == "degrading" else 0.0 for trend in related), default=0.0)
            nis = float(event.get("nis_by_subsystem", {}).get(subsystem, 0)); nis_signal = min(nis / float(rf["nis_inconsistency_threshold"]), 1.0)
            trend_or_nis = max(trend_signal, nis_signal)
            evidence = (float(rf["residual_weight"]) * physical_signal.get(subsystem, 0.0)
                        + float(rf["shap_weight"]) * shap_signal
                        + float(rf["classifier_weight"]) * fault_signal
                        + float(rf["anomaly_weight"]) * anomaly
                        + float(rf["trend_weight"]) * trend_or_nis)
            observed_physical = any(item["subsystem"] == subsystem for item in physical_detail.values())
            observed_evidence = observed_physical or subsystem in event.get("nis_by_subsystem", {}) or bool(related) or bool(black_box)
            subsystem_risk[subsystem] = float(np.clip(100 * evidence, 0, 100))
            subsystem_conf[subsystem] = float(np.clip((max(confidence, quality) if observed_evidence else 0.0), 0, 1))
        redlines = event.get("telemetry", {}); limits = self.config["redline_limits"]
        rpm = redlines.get("rpm")
        stopped = isinstance(rpm, (int, float)) and not isinstance(rpm, bool) and np.isfinite(rpm) and rpm <= 0
        rpm_provenance = (event.get("parameter_provenance") or {}).get("rpm") or {}
        commanded_stop = stopped and (rpm_provenance.get("source") == "manual_simulator_override"
                                      and (event.get("layer2_frame") or {}).get("context", {}).get("override_values", {}).get("rpm") == 0)
        redline_hits = [key for key, test in {"oil_pressure": not stopped and redlines.get("oil_press_kPa", float("inf")) < float(limits["oil_press_min_kPa"]), "cht": redlines.get("cht_degC", -float("inf")) > float(limits["cht_max_degC"]), "egt": redlines.get("egt_degC", -float("inf")) > float(limits["egt_max_degC"]), "battery": redlines.get("batt_volts", float("inf")) < float(limits["batt_volts_min_V"]), "vibration": redlines.get("vibration_rms_g", -float("inf")) > float(limits["vib_rms_max_g"]), "rpm": redlines.get("rpm", -float("inf")) > float(limits["rpm_max"]), "engine_stall": stopped and not commanded_stop}.items() if test]
        catastrophic_indicators = [
            name for name, policy in self.config["indicators"].items()
            if name in indicators and indicators[name] is not None
            and float(policy["failure_direction"]) *
            (float(indicators[name]) - float(self.config["failure_thresholds"][policy["threshold_key"]])) >= 0
        ]
        catastrophic = bool(catastrophic_indicators)
        redline_subsystems = {
            "oil_pressure": "lubrication", "cht": "thermal", "egt": "combustion",
            "battery": "electrical", "vibration": "mechanical", "rpm": "mechanical",
            "engine_stall": "mechanical",
        }
        severe_evidence = bool(redline_hits or catastrophic)
        if severe_evidence:
            risk_floor = max(float(value) for value in self.config["risk_action_thresholds"].values())
            for subsystem in [redline_subsystems[name] for name in redline_hits] + [
                self.config["indicators"][name]["subsystem"] for name in catastrophic_indicators
            ]:
                subsystem_risk[subsystem] = max(subsystem_risk[subsystem], risk_floor)
        # Consequence-weighted max: one affected subsystem is never diluted by
        # healthy subsystems. Explicit redlines/catastrophic indicators retain
        # the configured immediate-action risk floor after consequence weighting.
        maximum_weight = max(float(value) for value in self.config["subsystem_consequence_weight"].values())
        engine_risk = max(subsystem_risk[name] * float(weight) / maximum_weight
                          for name, weight in self.config["subsystem_consequence_weight"].items())
        if severe_evidence:
            engine_risk = max(engine_risk, risk_floor)
        dominant = max(subsystem_risk, key=subsystem_risk.get)
        evidence_available = bool(physical_detail) or bool(event.get("nis_by_subsystem")) or bool(black_box)
        if redline_hits or catastrophic: action = "IMMEDIATE"
        elif not evidence_available: action = "INSUFFICIENT_DATA"
        else:
            threshold_action = next((action for action, threshold in sorted(self.config["risk_action_thresholds"].items(), key=lambda x: x[1], reverse=True) if engine_risk >= float(threshold)), "NORMAL")
            self.escalation_counts[threshold_action] = self.escalation_counts.get(threshold_action, 0) + 1
            action = threshold_action if threshold_action == "NORMAL" or self.escalation_counts[threshold_action] >= int(rf["risk_confirmation_windows"]) else "NORMAL"
        if black_box and fault_signal >= float(state_policy.get("provisional_fault_probability", 0.60)) and action == "NORMAL":
            action = "ADVISORY"
        bayesian_rul = None
        if event.get("rul_observations") and event.get("engine_serial") and event.get("sortie_id"):
            bayesian_rul = self.pi_rul.update(event)
            stress = self._stress(event.get("operating_context", {}))
            whole = dict(bayesian_rul["whole_engine"])
            whole["rul_median_hours"] = float(whole["rul_median_hours"]) / stress
            whole["rul_ci_low_hours"] = float(whole["rul_ci_low_hours"]) / stress
            whole["rul_ci_high_hours"] = float(whole["rul_ci_high_hours"]) / stress
            bayesian_rul = {**bayesian_rul, "whole_engine": whole,
                            "environmental_stress_factor": stress}
            rul = float(whole["rul_median_hours"])
            interval = (float(whole["rul_ci_low_hours"]), float(whole["rul_ci_high_hours"]))
            reason = None
        elif op_hours is None:
            rul, interval, reason = None, None, "Operating hours are unavailable; RUL cannot be estimated."
        else:
            rul, interval, reason = self._rul(trends, quality, event.get("operating_context", {}))
        fault_flags = dict(black_box.get("fault_flags") or {})
        faults: list[FaultAssessment] = []
        for name, flag in fault_flags.items():
            if not bool(flag.get("active")):
                continue
            per_class_shap = (black_box.get("shap_by_class") or {}).get(name, {})
            physical_shap = {key: value for key, value in per_class_shap.items() if key != "context"}
            subsystem = max(physical_shap, key=lambda key: abs(float(physical_shap[key])), default=dominant)
            if subsystem == "vibration":
                subsystem = "mechanical"
            if name == "sensor_drift_failure":
                subsystem = black_box.get("sensor_drift_bias_subsystem") or "unattributed_sensor"
            faults.append(FaultAssessment(name, float(flag["confidence"]), float(flag["confidence"]), subsystem))
        physical_evidence_available = bool(physical_detail) or bool(event.get("nis_by_subsystem"))
        if action == "IMMEDIATE" and bool(state_policy["immediate_action_forces_critical"]):
            engine_state: str | None = "critical"
        elif (black_box
              and fault_signal >= float(state_policy["critical_fault_probability"])
              and anomaly >= float(state_policy["critical_anomaly_score"])):
            engine_state = "critical"
        elif engine_risk >= 100 * float(state_policy["critical_fault_probability"]):
            engine_state = "critical"
        elif engine_risk >= 100 * float(state_policy["abnormal_fault_probability"]):
            engine_state = "abnormal"
        elif black_box:
            if (fault_signal >= float(state_policy["critical_fault_probability"])
                    and anomaly >= float(state_policy["critical_anomaly_score"])):
                engine_state = "critical"
            elif (fault_signal >= float(state_policy["abnormal_fault_probability"])
                  or anomaly >= float(state_policy["abnormal_anomaly_score"])):
                engine_state = "abnormal"
            else:
                engine_state = "normal"
        else:
            if engine_risk >= 100 * float(state_policy["critical_fault_probability"]):
                engine_state = "critical"
            elif engine_risk >= 100 * float(state_policy["abnormal_fault_probability"]):
                engine_state = "abnormal"
            else:
                engine_state = "normal" if physical_evidence_available else None


        plan: list[MaintenanceRecommendation] = []
        if action not in {"NORMAL", "INSUFFICIENT_DATA"}:
            entry = self.config["maintenance"]["catalogue"].get(dominant)
            lower = interval[0] if interval and (bayesian_rul is None or bayesian_rul["validated_for_decisions"]) else None
            margin = float(self.config["maintenance"]["rul_safety_margin_fraction"])
            urgency = "before_next_sortie" if action in {"IMMEDIATE", "ABORT_RECOMMEND", "POWER_DERATE", "MAINTENANCE_FLAG"} else "monitor"
            plan.append(MaintenanceRecommendation(dominant, entry["task_id"] if entry else "GENERIC-INSPECT", entry["task_text"] if entry else self.config["maintenance"]["catalogue_fallback"].replace("<subsystem>", dominant), urgency, lower * (1 - margin) if lower is not None else None, [f"risk={engine_risk:.1f}", f"action={action}"], f"{dominant} is the dominant explainable risk contributor.", subsystem_conf[dominant]))
        rationale = f"{dominant} risk {subsystem_risk[dominant]:.1f}; action {action}." + (f" Redline: {', '.join(redline_hits)}." if redline_hits else "")

        versions = {str(name): str(version) for name, version in event.get("upstream_model_versions", {}).items()}
        versions.update({"layer6": str(inference_output.get("model_version", "unavailable")), "layer7_policy": str(self.config["schema_version"])})

        covered_confidence = [value for value in subsystem_conf.values() if value > 0]
        risk_confidence = float(np.mean(covered_confidence)) if covered_confidence else 0.0
        return RiskRULOutput(timestamp, float(engine_risk), risk_confidence, subsystem_risk, subsystem_conf, rul, interval, float(self.config["rul"]["confidence_level"]), str(self.config["rul"]["assumed_profile"]), dominant, faults, action, rationale, bool(black_box.get("novel_degradation", False)), trends, plan, {}, quality, versions, reason, engine_state, fault_flags, bayesian_rul, redline_hits)


def _erfinv(value: float) -> float:
    """Numerically adequate inverse erf without adding a SciPy edge dependency."""
    a = 0.147; sign = 1 if value >= 0 else -1; logarithm = np.log(1 - value * value)
    return sign * sqrt(sqrt((2 / (np.pi * a) + logarithm / 2) ** 2 - logarithm / a) - (2 / (np.pi * a) + logarithm / 2))
