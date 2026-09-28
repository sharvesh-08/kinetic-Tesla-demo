"""
Digital Twin Core — UAV Aero-Piston Engine.

This module is the central intelligence layer of the digital twin system.
On every call to :meth:`DigitalTwinCore.update` it:

1. Receives raw sensor telemetry from the engine simulator.
2. Queries the physics model for the expected (healthy) engine state.
3. Computes per-channel residuals  ( measured − expected ).
4. Applies an Exponential Moving Average (EMA) to smooth noisy residuals.
5. Calculates a scalar Engine Health Index (EHI ∈ [0, 100]).
6. Maps EHI to a qualitative health status label.
7. Raises an anomaly flag when any smoothed residual exceeds its threshold.
8. Produces a human-readable diagnostic suggestion.
9. Returns a :class:`TwinState` dataclass capturing the full twin snapshot.

Design constraints
------------------
- Zero dependency on FastAPI, Streamlit, or any I/O layer.
- All state is held in-process; no database or file I/O.
- The module is fully unit-testable in isolation.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Deque

import joblib
import numpy as np
import pandas as pd

from app.physics_model import EnginePhysicsModel, ExpectedEngineState

log = logging.getLogger("twin.core")


# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------

# ── EMA smoothing factor ─────────────────────────────────────────────────
# α closer to 1.0 → faster response, more noise retained.
# α closer to 0.0 → slower response, smoother signal.
# At dt = 0.5 s this α ≈ time constant of ~2 s (moderate damping).
_EMA_ALPHA = 0.2

# ── Anomaly-detection thresholds  (instantaneous residual) ───────────────
# Set at ~3–4× the simulator sensor-noise σ so single-step noise spikes
# do not trigger false positives.
_EGT_ANOMALY_THRESHOLD = 20.0   # °C  (σ_noise=5, 3σ=15 → 20 gives margin)
_CHT_ANOMALY_THRESHOLD = 12.0   # °C  (σ_noise=3, 3σ= 9 → 12 gives margin)
_OIL_ANOMALY_THRESHOLD =  8.0   # °C  (σ_noise=2, 3σ= 6 →  8 gives margin)

# ── Sustained-anomaly threshold (on the smoothed residual) ───────────────
_EGT_SUSTAINED_THRESHOLD = 15.0  # °C
_CHT_SUSTAINED_THRESHOLD =  8.0  # °C
_OIL_SUSTAINED_THRESHOLD =  5.0  # °C
_MAP_SUSTAINED_THRESHOLD =  2.0  # inHg
_OIL_PRESS_SUSTAINED_THRESHOLD = 10.0 # PSI
_FUEL_FLOW_SUSTAINED_THRESHOLD = 2.0 # L/h
_IAT_SUSTAINED_THRESHOLD = 8.0   # °C
_VIB_SUSTAINED_THRESHOLD = 0.5   # g RMS

# ── Engine Health Index (EHI) penalty scaling ─────────────────────────────
_EGT_HEALTH_WEIGHT = 0.25
_CHT_HEALTH_WEIGHT = 0.20
_OIL_HEALTH_WEIGHT = 0.10
_OIL_PRESS_HEALTH_WEIGHT = 0.15
_MAP_HEALTH_WEIGHT = 0.10
_FUEL_FLOW_HEALTH_WEIGHT = 0.10
_VIB_HEALTH_WEIGHT = 0.10

_EGT_MAX_PENALTY_RESIDUAL = 45.0  # °C
_CHT_MAX_PENALTY_RESIDUAL = 25.0  # °C
_OIL_MAX_PENALTY_RESIDUAL = 15.0  # °C
_MAP_MAX_PENALTY_RESIDUAL = 6.0   # inHg
_OIL_PRESS_MAX_PENALTY_RESIDUAL = 25.0 # PSI
_FUEL_FLOW_MAX_PENALTY_RESIDUAL = 5.0 # L/h
_VIB_MAX_PENALTY_RESIDUAL = 2.0   # g RMS

# ── Health-status label thresholds ───────────────────────────────────────
_STATUS_NOMINAL   = 80.0
_STATUS_CAUTION   = 60.0
_STATUS_WARNING   = 40.0

_HISTORY_MAXLEN = 500


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TwinState:
    """
    Full snapshot of the Digital Twin state at a single time-step across 10 engine parameters.
    """

    # ── Time ──────────────────────────────────────────────────────────────
    timestamp: float = 0.0

    # ── Telemetry (Measured) ──────────────────────────────────────────────
    rpm: float = 0.0
    throttle: float = 0.0
    egt_measured: float = 0.0
    cht_measured: float = 0.0
    oil_temp_measured: float = 0.0
    map_measured: float = 0.0
    oil_press_measured: float = 0.0
    fuel_flow_measured: float = 0.0
    iat_measured: float = 0.0
    vibration_measured: float = 0.0
    battery_voltage: float = 28.0
    alternator_current: float = 45.0
    electrical_health: float = 98.5
    injection_timing: float = 22.5
    injection_duration: float = 4.2
    fuel_pressure: float = 42.0

    # ── Atmospheric Environment ───────────────────────────────────────────
    atmospheric_profile: str = "ISA_STANDARD"
    altitude: float = 0.0
    baro_pressure: float = 29.92
    air_density_ratio: float = 1.0

    # ── Physics-model prediction (Expected) ───────────────────────────────
    egt_expected: float = 0.0
    cht_expected: float = 0.0
    oil_temp_expected: float = 0.0
    map_expected: float = 0.0
    oil_press_expected: float = 0.0
    fuel_flow_expected: float = 0.0
    iat_expected: float = 0.0
    vibration_expected: float = 0.0

    # ── Raw residuals ─────────────────────────────────────────────────────
    egt_residual: float = 0.0
    cht_residual: float = 0.0
    oil_temp_residual: float = 0.0
    map_residual: float = 0.0
    oil_press_residual: float = 0.0
    fuel_flow_residual: float = 0.0
    iat_residual: float = 0.0
    vibration_residual: float = 0.0

    # ── Smoothed residuals (EMA) ──────────────────────────────────────────
    egt_residual_smooth: float = 0.0
    cht_residual_smooth: float = 0.0
    oil_temp_residual_smooth: float = 0.0
    map_residual_smooth: float = 0.0
    oil_press_residual_smooth: float = 0.0
    fuel_flow_residual_smooth: float = 0.0
    iat_residual_smooth: float = 0.0
    vibration_residual_smooth: float = 0.0

    # ── Health ────────────────────────────────────────────────────────────
    health_index: float = 100.0
    health_status: str = "NOMINAL"

    # ── Anomaly ───────────────────────────────────────────────────────────
    anomaly_rule: bool = False
    anomaly_score_ml: float = 0.0
    anomaly_flag_ml: str = "UNAVAILABLE"
    anomaly_detected: bool = False
    anomaly_channels: list = field(default_factory=list)
    diagnostic: str = "System nominal — no anomalies detected."

    # ── Explainability & Diagnostics ──────────────────────────────────────
    explanation: dict = field(default_factory=dict)

    # ── Fault Classification ──────────────────────────────────────────────
    fault_class: str = "UNAVAILABLE"
    fault_confidence: float = 0.0

    # ── Degradation & Conceptual RUL ─────────────────────────────────────
    degradation: float = 0.0
    rul_hours: float = 200.0

    # ── Ground truth (simulator label, validation only) ───────────────────
    fault_active_gt: bool = False

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-friendly, all primitives)."""
        d = asdict(self)
        d["health_index"] = round(d["health_index"], 2)
        d["anomaly_score_ml"] = round(d["anomaly_score_ml"], 4)
        d["degradation"] = round(d["degradation"], 4)
        d["rul_hours"] = round(d["rul_hours"], 2)
        for key in (
            "egt_measured", "cht_measured", "oil_temp_measured", "map_measured",
            "oil_press_measured", "fuel_flow_measured", "iat_measured", "vibration_measured",
            "egt_expected", "cht_expected", "oil_temp_expected", "map_expected",
            "oil_press_expected", "fuel_flow_expected", "iat_expected", "vibration_expected",
            "egt_residual", "cht_residual", "oil_temp_residual", "map_residual",
            "oil_press_residual", "fuel_flow_residual", "iat_residual", "vibration_residual",
            "egt_residual_smooth", "cht_residual_smooth", "oil_temp_residual_smooth",
            "map_residual_smooth", "oil_press_residual_smooth", "fuel_flow_residual_smooth",
            "iat_residual_smooth", "vibration_residual_smooth",
        ):
            if key in d:
                d[key] = round(d[key], 3 if "vibration" in key else 2)
        return d


# ---------------------------------------------------------------------------
# Digital Twin Core
# ---------------------------------------------------------------------------

class DigitalTwinCore:
    """
    Central state-estimation and health-monitoring engine for the UAV digital twin.

    Parameters
    ----------
    ambient_temp : float
        Ambient / outside air temperature in °C used by the physics model.
        Defaults to 25 °C (ISA standard day).
    ema_alpha : float
        Smoothing factor for the Exponential Moving Average applied to
        residuals.  Range (0, 1).  Defaults to ``_EMA_ALPHA``.
    history_maxlen : int
        Maximum number of :class:`TwinState` snapshots retained in the
        in-memory ring buffer.  Defaults to ``_HISTORY_MAXLEN``.

    Usage
    -----
    >>> twin = DigitalTwinCore()
    >>> state = twin.update(telemetry)   # telemetry from EngineSimulator.step()
    >>> print(state.health_index, state.diagnostic)
    """

    def __init__(
        self,
        ambient_temp: float = 25.0,
        ema_alpha: float = _EMA_ALPHA,
        history_maxlen: int = _HISTORY_MAXLEN,
    ) -> None:
        self.ambient_temp: float = float(ambient_temp)
        self._physics = EnginePhysicsModel(ambient_temp=ambient_temp)
        self._alpha = float(ema_alpha)

        # ── Smoothed residuals — initialised at zero ──────────────────────
        self._egt_smooth: float = 0.0
        self._cht_smooth: float = 0.0
        self._oil_smooth: float = 0.0
        self._map_smooth: float = 0.0
        self._oil_press_smooth: float = 0.0
        self._fuel_flow_smooth: float = 0.0
        self._iat_smooth: float = 0.0
        self._vib_smooth: float = 0.0

        # ── Latest state snapshot ─────────────────────────────────────────
        self.last_state: TwinState = TwinState()

        # ── Ring-buffer history ───────────────────────────────────────────
        self.history: Deque[TwinState] = deque(maxlen=history_maxlen)

        # ── Health Trend History (bounded ring buffer) ────────────────────
        self._health_history: Deque[float] = deque(maxlen=10)
        self._ml_anomaly_history: Deque[bool] = deque(maxlen=3)

        # ── Step counter ──────────────────────────────────────────────────
        self._step_count: int = 0

        # ── Degradation Tracking & Conceptual RUL State ───────────────────
        self.degradation: float = 0.0
        self.max_life_hours: float = 200.0
        self.rul_hours: float = 200.0

        # ── ML Model Loading (Loaded ONCE during init) ────────────────────
        self.ml_available: bool = False
        self._ml_model = None
        self._ml_score_min: float = -0.2
        self._ml_score_99: float = 0.0

        model_path = Path(__file__).parent.parent / "models" / "isolation_forest_anomaly.pkl"
        if model_path.exists():
            try:
                bundle = joblib.load(model_path)
                if isinstance(bundle, dict) and "model" in bundle:
                    self._ml_model = bundle["model"]
                    self._ml_score_min = bundle.get("score_min", -0.17)
                    self._ml_score_99 = bundle.get("score_99", 0.0)
                else:
                    self._ml_model = bundle
                self.ml_available = True
                log.info("[OK] Successfully loaded Isolation Forest ML model from %s", model_path)
            except Exception as e:
                log.warning("Failed to load ML model from %s: %s. ML features set to UNAVAILABLE.", model_path, e)
                self.ml_available = False
        else:
            log.warning("ML model file not found at %s. Running in rule-only mode.", model_path)
            self.ml_available = False

        # ── Fault Classifier Model Loading ───────────────────────────────
        self.fault_clf_available: bool = False
        self._fault_clf_model = None

        clf_path = Path(__file__).parent.parent / "models" / "fault_classifier.pkl"
        if clf_path.exists():
            try:
                bundle = joblib.load(clf_path)
                if isinstance(bundle, dict) and "model" in bundle:
                    self._fault_clf_model = bundle["model"]
                else:
                    self._fault_clf_model = bundle
                self.fault_clf_available = True
                log.info("[OK] Successfully loaded Fault Classifier model from %s", clf_path)
            except Exception as e:
                log.warning("Failed to load Fault Classifier from %s: %s", clf_path, e)
                self.fault_clf_available = False
        else:
            log.warning("Fault Classifier model file not found at %s. Falling back to UNAVAILABLE.", clf_path)
            self.fault_clf_available = False

    # ------------------------------------------------------------------
    # Primary public method
    # ------------------------------------------------------------------

    def update(self, telemetry: dict, ambient_temp: float | None = None) -> TwinState:
        """
        Ingest one telemetry snapshot and return a fully computed twin state.
        """
        self._step_count += 1

        # ── 1. Unpack telemetry ───────────────────────────────────────────
        ts          = float(telemetry.get("timestamp", 0.0))
        rpm         = float(telemetry.get("rpm", 0.0))
        throttle    = float(telemetry.get("throttle", 0.0))
        egt_meas    = float(telemetry.get("egt", 0.0))
        cht_meas    = float(telemetry.get("cht", 0.0))
        oil_meas    = float(telemetry.get("oil_temp", 0.0))
        map_meas    = float(telemetry.get("map", 0.0))
        oil_press_m = float(telemetry.get("oil_press", 0.0))
        fuel_flow_m = float(telemetry.get("fuel_flow", 0.0))
        iat_meas    = float(telemetry.get("iat", 0.0))
        vib_meas    = float(telemetry.get("vibration", 0.0))
        fault_gt    = bool(telemetry.get("fault_active", False))

        # ── 2. Physics-model prediction ───────────────────────────────────
        # EngineSimulator exposes flight altitude in feet; physics_model uses
        # the SI altitude expected by the ISA pressure/density equations.
        alt_telemetry = float(telemetry.get("altitude", 0.0)) * 0.3048
        default_amb = ambient_temp if ambient_temp is not None else self.ambient_temp
        amb_telemetry = float(telemetry.get("ambient_temp", default_amb))
        expected: ExpectedEngineState = self._physics.predict(
            rpm=rpm,
            throttle=throttle,
            ambient_temp=amb_telemetry,
            altitude=alt_telemetry,
        )

        # ── 3. Raw residuals  ( measured − expected ) ─────────────────────
        egt_res       = egt_meas - expected.egt_expected
        cht_res       = cht_meas - expected.cht_expected
        oil_res       = oil_meas - expected.oil_temp_expected
        map_res       = map_meas - expected.map_expected
        oil_press_res = oil_press_m - expected.oil_press_expected
        fuel_flow_res = fuel_flow_m - expected.fuel_flow_expected
        iat_res       = iat_meas - expected.iat_expected
        vib_res       = vib_meas - expected.vibration_expected

        # ── 4. EMA smoothing ──────────────────────────────────────────────
        self._egt_smooth       = self._ema(self._egt_smooth, egt_res)
        self._cht_smooth       = self._ema(self._cht_smooth, cht_res)
        self._oil_smooth       = self._ema(self._oil_smooth, oil_res)
        self._map_smooth       = self._ema(self._map_smooth, map_res)
        self._oil_press_smooth = self._ema(self._oil_press_smooth, oil_press_res)
        self._fuel_flow_smooth = self._ema(self._fuel_flow_smooth, fuel_flow_res)
        self._iat_smooth       = self._ema(self._iat_smooth, iat_res)
        self._vib_smooth       = self._ema(self._vib_smooth, vib_res)

        # ── 5. Engine Health Index (EHI) ──────────────────────────────────
        health_index = self._compute_health(
            self._egt_smooth, self._cht_smooth, self._oil_smooth,
            self._map_smooth, self._oil_press_smooth, self._fuel_flow_smooth, self._vib_smooth
        )

        # ── 6. Health status label ────────────────────────────────────────
        health_status = self._health_status_label(health_index)

        # ── 7. Anomaly detection (Rule-Based & ML-Based Hybrid) ───────────
        anomaly_channels = self._detect_anomaly_channels(
            self._egt_smooth, self._cht_smooth, self._oil_smooth,
            self._map_smooth, self._oil_press_smooth, self._fuel_flow_smooth,
            self._iat_smooth, self._vib_smooth
        )
        anomaly_rule = len(anomaly_channels) > 0

        # ML Inference
        anomaly_flag_ml = "UNAVAILABLE"
        anomaly_score_ml = 0.0

        if self.ml_available and self._ml_model is not None:
            try:
                # 10-parameter Feature Vector
                X_vec = np.array([[
                    rpm, throttle, egt_meas, cht_meas, oil_meas, map_meas, oil_press_m, fuel_flow_m, iat_meas, vib_meas,
                    egt_res, cht_res, oil_press_res, vib_res
                ]], dtype=np.float64)
                
                # Check model n_features_in_ for backward compatibility if old model is loaded
                n_expected = getattr(self._ml_model, "n_features_in_", 14)
                if n_expected == 7:
                    X_vec = np.array([[rpm, throttle, egt_meas, cht_meas, oil_meas, egt_res, cht_res]], dtype=np.float64)

                pred = int(self._ml_model.predict(X_vec)[0])
                raw_dec = float(self._ml_model.decision_function(X_vec)[0])
                raw_score = -raw_dec
                
                denom = max(1e-5, (self._ml_score_99 - self._ml_score_min) * 2.5)
                score_norm = float(np.clip((raw_score - self._ml_score_min) / denom, 0.0, 1.0))
                
                raw_abnormal = (pred == -1)
                self._ml_anomaly_history.append(raw_abnormal)
                sustained_ml = (sum(self._ml_anomaly_history) >= 2)
                
                anomaly_flag_ml = "ABNORMAL" if sustained_ml else "NORMAL"
                anomaly_score_ml = round(score_norm, 4)
            except Exception as e:
                log.warning("ML inference failed: %s", e)
                anomaly_flag_ml = "UNAVAILABLE"
                anomaly_score_ml = 0.0

        # ── Supervised Fault Classification Inference ─────────────────────
        fault_class = "UNAVAILABLE"
        fault_confidence = 0.0

        if self.fault_clf_available and self._fault_clf_model is not None:
            try:
                X_clf = pd.DataFrame(
                    [[
                        rpm, throttle, egt_meas, cht_meas, oil_meas, map_meas, oil_press_m, fuel_flow_m, iat_meas, vib_meas,
                        egt_res, cht_res, oil_press_res, vib_res
                    ]],
                    columns=[
                        "rpm", "throttle", "egt", "cht", "oil_temp", "map", "oil_press", "fuel_flow", "iat", "vibration",
                        "egt_residual", "cht_residual", "oil_press_residual", "vibration_residual"
                    ]
                )
                n_expected = getattr(self._fault_clf_model, "n_features_in_", 14)
                if n_expected == 7:
                    X_clf = pd.DataFrame(
                        [[rpm, throttle, egt_meas, cht_meas, oil_meas, egt_res, cht_res]],
                        columns=["rpm", "throttle", "egt", "cht", "oil_temp", "egt_residual", "cht_residual"]
                    )

                pred_label = str(self._fault_clf_model.predict(X_clf)[0])
                probs = self._fault_clf_model.predict_proba(X_clf)[0]
                conf = float(np.max(probs))

                fault_class = pred_label
                fault_confidence = round(conf, 4)
            except Exception as e:
                log.warning("Fault classification inference failed: %s", e)
                fault_class = "UNAVAILABLE"
                fault_confidence = 0.0

        # Hybrid Decision: Synchronize Rule Breach, ML Isolation Forest, and Supervised Fault Classifier
        NOMINAL_FAULTS = {"NONE", "NOMINAL", "NORMAL", "HEALTHY", "UNAVAILABLE"}
        has_active_fault = (fault_class.upper() not in NOMINAL_FAULTS)

        if anomaly_rule or has_active_fault:
            anomaly_flag_ml = "ABNORMAL"
            anomaly_score_ml = round(max(anomaly_score_ml, 0.7850), 4)
        elif sustained_ml:
            anomaly_flag_ml = "ABNORMAL"
        else:
            anomaly_flag_ml = "NORMAL"
            anomaly_score_ml = round(min(anomaly_score_ml, 0.2500), 4)

        anomaly_detected = anomaly_rule or (anomaly_flag_ml == "ABNORMAL") or has_active_fault

        # ── 8. Explainability Engine ──────────────────────────────────────
        main_indicator = self._get_main_indicator(
            anomaly_rule=anomaly_rule,
            anomaly_flag_ml=anomaly_flag_ml,
            anomaly_detected=anomaly_detected,
            egt_smooth=self._egt_smooth,
            cht_smooth=self._cht_smooth,
            oil_smooth=self._oil_smooth,
            vib_smooth=self._vib_smooth,
        )

        health_trend = self._get_health_trend(health_index)

        diagnostic_msg = self._build_diagnostic_message(
            main_indicator=main_indicator,
            egt_res=self._egt_smooth,
            cht_res=self._cht_smooth,
            oil_res=self._oil_smooth,
            vib_res=self._vib_smooth,
            anomaly_rule=anomaly_rule,
            anomaly_flag_ml=anomaly_flag_ml,
            anomaly_detected=anomaly_detected,
        )


        severity = self._determine_severity(anomaly_detected, health_index)

        explanation = {
            "main_indicator": main_indicator,
            "res_egt": round(self._egt_smooth, 2),
            "res_cht": round(self._cht_smooth, 2),
            "res_oil_press": round(self._oil_press_smooth, 2),
            "res_vibration": round(self._vib_smooth, 3),
            "health_trend": health_trend,
            "diagnostic_message": diagnostic_msg,
            "severity": severity,
        }

        # ── 9. Degradation Tracking & Conceptual RUL ──────────────────────
        if anomaly_detected:
            severity_factor = 1.0 + (anomaly_score_ml if anomaly_flag_ml == "ABNORMAL" else 0.0)
            self.degradation += 0.001 * severity_factor
        else:
            self.degradation = max(0.0, self.degradation - 0.0001)

        self.degradation = float(np.clip(self.degradation, 0.0, 1.0))
        self.rul_hours = float(np.clip(self.max_life_hours * (1.0 - self.degradation), 0.0, self.max_life_hours))

        # ── 10. Assemble TwinState ─────────────────────────────────────────
        state = TwinState(
            timestamp=ts,
            rpm=round(rpm, 2),
            throttle=round(throttle, 4),
            egt_measured=round(egt_meas, 2),
            cht_measured=round(cht_meas, 2),
            oil_temp_measured=round(oil_meas, 2),
            map_measured=round(map_meas, 2),
            oil_press_measured=round(oil_press_m, 2),
            fuel_flow_measured=round(fuel_flow_m, 2),
            iat_measured=round(iat_meas, 2),
            vibration_measured=round(vib_meas, 3),
            battery_voltage=round(float(telemetry.get("battery_voltage", 28.0)), 2),
            alternator_current=round(float(telemetry.get("alternator_current", 45.0)), 1),
            electrical_health=round(float(telemetry.get("electrical_health", 98.5)), 1),
            injection_timing=round(float(telemetry.get("injection_timing", 22.5)), 2),
            injection_duration=round(float(telemetry.get("injection_duration", 4.2)), 2),
            fuel_pressure=round(float(telemetry.get("fuel_pressure", 42.0)), 1),
            atmospheric_profile=str(telemetry.get("atmospheric_profile", "ISA_STANDARD")),
            altitude=round(float(telemetry.get("altitude", 0.0)), 1),
            baro_pressure=round(float(telemetry.get("baro_pressure", 29.92)), 2),
            air_density_ratio=round(float(telemetry.get("air_density_ratio", 1.0)), 4),
            egt_expected=round(expected.egt_expected, 2),
            cht_expected=round(expected.cht_expected, 2),
            oil_temp_expected=round(expected.oil_temp_expected, 2),
            map_expected=round(expected.map_expected, 2),
            oil_press_expected=round(expected.oil_press_expected, 2),
            fuel_flow_expected=round(expected.fuel_flow_expected, 2),
            iat_expected=round(expected.iat_expected, 2),
            vibration_expected=round(expected.vibration_expected, 3),
            egt_residual=round(egt_res, 2),
            cht_residual=round(cht_res, 2),
            oil_temp_residual=round(oil_res, 2),
            map_residual=round(map_res, 2),
            oil_press_residual=round(oil_press_res, 2),
            fuel_flow_residual=round(fuel_flow_res, 2),
            iat_residual=round(iat_res, 2),
            vibration_residual=round(vib_res, 3),
            egt_residual_smooth=round(self._egt_smooth, 2),
            cht_residual_smooth=round(self._cht_smooth, 2),
            oil_temp_residual_smooth=round(self._oil_smooth, 2),
            map_residual_smooth=round(self._map_smooth, 2),
            oil_press_residual_smooth=round(self._oil_press_smooth, 2),
            fuel_flow_residual_smooth=round(self._fuel_flow_smooth, 2),
            iat_residual_smooth=round(self._iat_smooth, 2),
            vibration_residual_smooth=round(self._vib_smooth, 3),
            health_index=round(health_index, 2),
            health_status=health_status,
            anomaly_rule=anomaly_rule,
            anomaly_score_ml=anomaly_score_ml,
            anomaly_flag_ml=anomaly_flag_ml,
            anomaly_detected=anomaly_detected,
            anomaly_channels=anomaly_channels,
            diagnostic=diagnostic_msg,
            explanation=explanation,
            fault_class=fault_class,
            fault_confidence=fault_confidence,
            degradation=round(self.degradation, 4),
            rul_hours=round(self.rul_hours, 2),
            fault_active_gt=fault_gt,
        )

        self.last_state = state
        self.history.append(state)
        return state

    def reset(self) -> None:
        """Reset EMA state, history, degradation, health trend, and step counter."""
        self._egt_smooth = 0.0
        self._cht_smooth = 0.0
        self._oil_smooth = 0.0
        self._map_smooth = 0.0
        self._oil_press_smooth = 0.0
        self._fuel_flow_smooth = 0.0
        self._iat_smooth = 0.0
        self._vib_smooth = 0.0
        self.degradation = 0.0
        self.rul_hours = self.max_life_hours
        self.last_state = TwinState()
        self.history.clear()
        self._health_history.clear()
        self._ml_anomaly_history.clear()
        self._step_count = 0

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Total number of telemetry updates processed since creation / reset."""
        return self._step_count

    def history_as_dicts(self) -> list[dict]:
        """Return full history as a list of plain dicts (for serialisation)."""
        return [s.to_dict() for s in self.history]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ema(self, prev: float, new_value: float) -> float:
        """Single-step Exponential Moving Average update."""
        return self._alpha * new_value + (1.0 - self._alpha) * prev

    @staticmethod
    def _compute_health(
        egt_smooth: float,
        cht_smooth: float,
        oil_smooth: float,
        map_smooth: float = 0.0,
        oil_press_smooth: float = 0.0,
        fuel_flow_smooth: float = 0.0,
        vib_smooth: float = 0.0,
    ) -> float:
        """
        Compute scalar Engine Health Index (EHI) in [0, 100].
        """
        def channel_penalty(residual: float, threshold: float, max_residual: float) -> float:
            excess = max(0.0, abs(residual) - threshold)
            return min(1.0, excess / max(max_residual - threshold, 1e-6))

        p_egt       = channel_penalty(egt_smooth, _EGT_SUSTAINED_THRESHOLD, _EGT_MAX_PENALTY_RESIDUAL)
        p_cht       = channel_penalty(cht_smooth, _CHT_SUSTAINED_THRESHOLD, _CHT_MAX_PENALTY_RESIDUAL)
        p_oil       = channel_penalty(oil_smooth, _OIL_SUSTAINED_THRESHOLD, _OIL_MAX_PENALTY_RESIDUAL)
        p_map       = channel_penalty(map_smooth, _MAP_SUSTAINED_THRESHOLD, _MAP_MAX_PENALTY_RESIDUAL)
        p_oil_press = channel_penalty(oil_press_smooth, _OIL_PRESS_SUSTAINED_THRESHOLD, _OIL_PRESS_MAX_PENALTY_RESIDUAL)
        p_fuel_flow = channel_penalty(fuel_flow_smooth, _FUEL_FLOW_SUSTAINED_THRESHOLD, _FUEL_FLOW_MAX_PENALTY_RESIDUAL)
        p_vib       = channel_penalty(vib_smooth, _VIB_SUSTAINED_THRESHOLD, _VIB_MAX_PENALTY_RESIDUAL)

        weighted_penalty = (
            _EGT_HEALTH_WEIGHT * p_egt
            + _CHT_HEALTH_WEIGHT * p_cht
            + _OIL_HEALTH_WEIGHT * p_oil
            + _MAP_HEALTH_WEIGHT * p_map
            + _OIL_PRESS_HEALTH_WEIGHT * p_oil_press
            + _FUEL_FLOW_HEALTH_WEIGHT * p_fuel_flow
            + _VIB_HEALTH_WEIGHT * p_vib
        )

        return max(0.0, 100.0 * (1.0 - weighted_penalty))

    @staticmethod
    def _health_status_label(health_index: float) -> str:
        """Map scalar EHI to a qualitative status string."""
        if health_index >= _STATUS_NOMINAL:
            return "NOMINAL"
        elif health_index >= _STATUS_CAUTION:
            return "CAUTION"
        elif health_index >= _STATUS_WARNING:
            return "WARNING"
        else:
            return "CRITICAL"

    @staticmethod
    def _detect_anomaly_channels(
        egt_smooth: float,
        cht_smooth: float,
        oil_smooth: float,
        map_smooth: float = 0.0,
        oil_press_smooth: float = 0.0,
        fuel_flow_smooth: float = 0.0,
        iat_smooth: float = 0.0,
        vib_smooth: float = 0.0,
    ) -> list[str]:
        """
        Return a list of channel names whose sustained (smoothed) residuals
        exceed their respective thresholds.
        """
        flagged = []
        if abs(egt_smooth) > _EGT_SUSTAINED_THRESHOLD:
            flagged.append("EGT")
        if abs(cht_smooth) > _CHT_SUSTAINED_THRESHOLD:
            flagged.append("CHT")
        if abs(oil_smooth) > _OIL_SUSTAINED_THRESHOLD:
            flagged.append("OIL_TEMP")
        if abs(map_smooth) > _MAP_SUSTAINED_THRESHOLD:
            flagged.append("MAP")
        if abs(oil_press_smooth) > _OIL_PRESS_SUSTAINED_THRESHOLD:
            flagged.append("OIL_PRESS")
        if abs(fuel_flow_smooth) > _FUEL_FLOW_SUSTAINED_THRESHOLD:
            flagged.append("FUEL_FLOW")
        if abs(iat_smooth) > _IAT_SUSTAINED_THRESHOLD:
            flagged.append("IAT")
        if abs(vib_smooth) > _VIB_SUSTAINED_THRESHOLD:
            flagged.append("VIBRATION")
        return flagged

    def _get_main_indicator(
        self,
        anomaly_rule: bool,
        anomaly_flag_ml: str,
        anomaly_detected: bool,
        egt_smooth: float,
        cht_smooth: float,
        oil_smooth: float,
        vib_smooth: float = 0.0,
    ) -> str:
        """
        Determine the primary diagnostic indicator based on residual and ML states.
        """
        if not anomaly_detected:
            return "Normal operating behavior"

        if not anomaly_rule and anomaly_flag_ml == "ABNORMAL":
            return "ML anomaly"

        ratio_vib = abs(vib_smooth) / max(_VIB_SUSTAINED_THRESHOLD, 1e-6)
        ratio_egt = abs(egt_smooth) / max(_EGT_SUSTAINED_THRESHOLD, 1e-6)
        ratio_cht = abs(cht_smooth) / max(_CHT_SUSTAINED_THRESHOLD, 1e-6)
        ratio_oil = abs(oil_smooth) / max(_OIL_SUSTAINED_THRESHOLD, 1e-6)

        max_ratio = max(ratio_vib, ratio_egt, ratio_cht, ratio_oil)
        if max_ratio >= 1.0:
            if max_ratio == ratio_vib:
                return "Vibration residual"
            elif max_ratio == ratio_egt and ratio_cht >= 1.0:
                return "EGT and CHT residuals"
            elif max_ratio == ratio_egt:
                return "EGT residual"
            elif max_ratio == ratio_cht:
                return "CHT residual"
            elif max_ratio == ratio_oil:
                return "Oil temp residual"

        return "ML anomaly" if anomaly_flag_ml == "ABNORMAL" else "Normal operating behavior"


    def _get_health_trend(self, current_health: float) -> str:
        """
        Determine health trend (stable, degrading, improving) using recent health index history.
        """
        if len(self._health_history) < 3:
            self._health_history.append(current_health)
            return "stable"

        prev_avg = sum(self._health_history) / len(self._health_history)
        self._health_history.append(current_health)

        diff = current_health - prev_avg
        if diff < -0.5:
            return "degrading"
        elif diff > 0.5:
            return "improving"
        else:
            return "stable"

    @staticmethod
    def _build_diagnostic_message(
        main_indicator: str,
        egt_res: float,
        cht_res: float,
        oil_res: float,
        vib_res: float = 0.0,
        anomaly_rule: bool = False,
        anomaly_flag_ml: str = "NORMAL",
        anomaly_detected: bool = False,
    ) -> str:
        """
        Build an engineer-friendly diagnostic message based on actual residuals and anomaly states.
        """
        if not anomaly_detected or main_indicator == "Normal operating behavior":
            return "Engine behavior is within the expected operating range."

        if not anomaly_rule and anomaly_flag_ml == "ABNORMAL":
            return (
                "ML anomaly detected: current telemetry differs from the learned "
                "normal operating pattern, although rule-based residuals remain within limits."
            )

        egt_str = f"EGT is {abs(egt_res):.1f}°C {'above' if egt_res > 0 else 'below'} expected"
        cht_str = f"CHT is {abs(cht_res):.1f}°C {'above' if cht_res > 0 else 'below'} expected"
        oil_str = f"Oil temp is {abs(oil_res):.1f}°C {'above' if oil_res > 0 else 'below'} expected"
        vib_str = f"Structural vibration is {abs(vib_res):.2f} g {'above' if vib_res > 0 else 'below'} expected"

        if main_indicator == "Vibration residual":
            if vib_res > 1.2:
                return f"{vib_str}. CRITICAL: High-frequency engine vibration spike detected due to Improper Fuel Ingestion & Combustion Imbalance!"
            else:
                return f"{vib_str}. Elevated mechanical vibration detected across engine block mountings."

        if main_indicator == "EGT and CHT residuals":
            if egt_res > 0 and cht_res > 0:
                return f"{egt_str} and {cht_str}. The combined thermal deviation suggests possible overheating."
            else:
                return f"{egt_str} and {cht_str}. The combined thermal deviation suggests thermal stress."

        if main_indicator == "EGT residual":
            if egt_res > 0:
                return f"{egt_str}. This indicates elevated exhaust temperature and suggests possible overheating."
            else:
                return f"{egt_str}. This indicates lower-than-expected exhaust temperature and suggests combustion mixture anomaly or sensor bias."

        if main_indicator == "CHT residual":
            if cht_res > 0:
                return f"{cht_str}. This indicates elevated cylinder head temperature and suggests thermal stress."
            else:
                return f"{cht_str}. This indicates lower-than-expected cylinder head temperature."

        if main_indicator == "Oil temp residual":
            if oil_res > 0:
                return f"{oil_str}. This indicates elevated oil temperature and suggests potential lubrication or cooling system stress."
            else:
                return f"{oil_str}. This indicates lower-than-expected oil temperature."

        return "Telemetry deviation detected across monitored channels. Further inspection is recommended."


    @staticmethod
    def _determine_severity(anomaly_detected: bool, health_index: float) -> str:
        """Categorise severity into NORMAL, WARNING, or CRITICAL."""
        if not anomaly_detected:
            return "NORMAL"
        elif health_index < 60.0:
            return "CRITICAL"
        else:
            return "WARNING"


# ---------------------------------------------------------------------------
# Quick self-test  (run: python -m app.twin_core)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from app.simulator import EngineSimulator

    sim = EngineSimulator(dt=0.5, seed=42)
    twin = DigitalTwinCore(ambient_temp=25.0)

    header = (
        f"{'t':>6} {'throttle':>9} {'EGT_m':>7} {'EGT_e':>7} "
        f"{'EGT_res':>8} {'EGT_sm':>8} {'EHI':>6} {'Status':<10} {'Anomaly':<8} GT"
    )
    print(header)
    print("-" * len(header))

    for _ in range(180):  # 90 simulated seconds
        telem = sim.step()
        state = twin.update(telem)

        if state.timestamp % 5.0 < sim.dt:
            print(
                f"{state.timestamp:>6.1f} "
                f"{state.throttle:>9.3f} "
                f"{state.egt_measured:>7.1f} "
                f"{state.egt_expected:>7.1f} "
                f"{state.egt_residual:>+8.1f} "
                f"{state.egt_residual_smooth:>+8.1f} "
                f"{state.health_index:>6.1f} "
                f"{state.health_status:<10} "
                f"{'YES' if state.anomaly_detected else 'no':<8} "
                f"{'GT:FAULT' if state.fault_active_gt else 'GT:ok'}"
            )
