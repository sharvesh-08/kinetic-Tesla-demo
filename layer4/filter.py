"""Small independent EKFs with separated slow sensor-bias estimators."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from layer4.config import EKFConfig, SubfilterConfig, load_config


@dataclass(frozen=True)
class EKFOutput:
    """Stable Layer 4 output contract consumed by Layer 5."""

    master_timestamp_ns: int
    residual: dict[str, float]
    residual_raw: dict[str, float]
    bias_estimate: dict[str, float]
    bias_rate: dict[str, float]
    transient_active: bool
    q_inflation_factor: float
    channel_valid: dict[str, bool]
    predicted: dict[str, float]
    residual_z: dict[str, float]
    sensor_health: dict[str, str]
    virtual_value: dict[str, float]
    measurement_updated: dict[str, bool]
    module_health: dict[str, Any]
    latency_us: float
    sync_quality: float
    timing_estimated: bool
    telemetry: dict[str, float]
    context: dict[str, Any]
    indicators: dict[str, float]
    nis_by_subsystem: dict[str, float]
    nis_by_channel: dict[str, float]
    model_versions: dict[str, str]
    layer2_frame: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass
class _ChannelState:
    p: float
    bias: float = 0.0
    bias_p: float = 1.0
    bias_rate: float = 0.0
    bias_samples: int = 0


class FederatedEKF:
    """Federated diagonal EKFs driven by Layer 3 healthy predictions.

    Layer 3 supplies the nonlinear state prediction. Each scalar diagonal update is
    equivalent to the corresponding independent block of a federated EKF and avoids
    a large joint covariance inversion.
    """

    def __init__(self, config: EKFConfig | None = None) -> None:
        self.config = config or load_config()
        self.states: dict[str, _ChannelState] = {}
        for sub in self.config.subfilters.values():
            for channel in sub.channels:
                self.states[channel] = _ChannelState(p=sub.initial_p[channel], bias_p=sub.bias_r[channel])
        self.last_consistent = {name: True for name in self.config.subfilters}
        self.previous_timestamp_ns: int | None = None
        self.previous_controls: dict[str, float] = {}
        self.session_id = None

    @staticmethod
    def _finite_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

    def _transient(self, timestamp_ns: int, values: dict[str, Any]) -> bool:
        active = False
        if self.previous_timestamp_ns is not None and timestamp_ns > self.previous_timestamp_ns:
            dt = (timestamp_ns - self.previous_timestamp_ns) / 1e9
            for channel, threshold in (("rpm", self.config.rpm_rate_threshold), ("throttle_pct", self.config.throttle_rate_threshold)):
                value, previous = values.get(channel), self.previous_controls.get(channel)
                if self._finite_number(value) and previous is not None and abs(float(value) - previous) / dt > threshold:
                    active = True
        for channel in ("rpm", "throttle_pct"):
            if self._finite_number(values.get(channel)):
                self.previous_controls[channel] = float(values[channel])
        self.previous_timestamp_ns = timestamp_ns
        return active

    def process(self, envelope: dict[str, Any]) -> EKFOutput:
        """Process one Layer 3 prediction envelope without mutating its input."""
        session = (envelope.get("context") or {}).get("session_id")
        if session and session != self.session_id:
            self.__init__(self.config)
            self.session_id = session
        started = time.perf_counter_ns()
        timestamp = int(envelope["master_timestamp_ns"])
        values = dict(envelope.get("values", {})); predicted_input = dict(envelope.get("predicted", {}))
        valid_input = dict(envelope.get("channel_valid", {})); new_input = dict(envelope.get("channel_new_sample", {}))
        transient = self._transient(timestamp, values)
        q_factor = self.config.q_inflation_factor if transient else 1.0
        dt = 1.0 / self.config.master_rate_hz
        if self.previous_timestamp_ns is not None:
            dt = max(1e-6, float(envelope.get("dt_s", dt)))

        predicted: dict[str, float] = {}
        raw: dict[str, float] = {}; innovation_s: dict[str, float] = {}; updateable: dict[str, bool] = {}
        channel_innovation_z: dict[str, float] = {}; sub_updated: dict[str, bool] = {}

        for name, sub in self.config.subfilters.items():
            updated = False
            for channel in sub.channels:
                state = self.states[channel]
                state.p += sub.q[channel] * q_factor
                prediction = predicted_input.get(channel)
                can_update = (valid_input.get(channel, False) is True and new_input.get(channel, False) is True
                              and self._finite_number(values.get(channel)) and self._finite_number(prediction))
                updateable[channel] = can_update
                if not can_update:
                    state.p *= self.config.missing_p_inflation
                    continue
                predicted[channel] = float(prediction)
                innovation = float(values[channel]) - float(prediction)
                s = state.p + sub.r[channel]
                raw[channel] = innovation; innovation_s[channel] = s
                channel_innovation_z[channel] = innovation / math.sqrt(s)
                updated = True
            sub_updated[name] = updated

        consistent: dict[str, bool] = {}
        for name in self.config.subfilters:
            if not sub_updated[name]:
                consistent[name] = self.last_consistent[name]
            else:
                relevant = [abs(channel_innovation_z[channel]) for channel in self.config.subfilters[name].channels
                            if channel in channel_innovation_z]
                consistent[name] = all(value <= self.config.innovation_gate_sigma for value in relevant)
                self.last_consistent[name] = consistent[name]
        channel_subsystem = {channel: name for name, sub in self.config.subfilters.items() for channel in sub.channels}
        residual: dict[str, float] = {}; residual_z: dict[str, float] = {}; bias: dict[str, float] = {}; rates: dict[str, float] = {}
        health: dict[str, str] = {}; virtual: dict[str, float] = {}

        for channel, innovation in raw.items():
            name = channel_subsystem[channel]; sub = self.config.subfilters[name]; state = self.states[channel]
            # Bias adaptation is gated by cross-domain consistency: isolated disagreement
            # is sensor-like; simultaneous subsystem disagreement remains degradation signal.
            peers_consistent = all(ok for peer, ok in consistent.items() if peer != name)
            isolated_sensor_disagreement = (not consistent[name]) and peers_consistent
            state.bias_p += sub.bias_q[channel]
            previous_bias = state.bias
            bias_reset = False
            if isolated_sensor_disagreement:
                gain_b = state.bias_p / (state.bias_p + sub.bias_r[channel])
                state.bias += gain_b * (innovation - state.bias)
                state.bias_p = (1.0 - gain_b) * state.bias_p
                state.bias_samples += 1
            elif not peers_consistent:
                # Once a second physical domain disagrees, the signature is no longer
                # isolated sensor drift. Remove any early bias attribution so the full
                # coupled degradation remains visible downstream.
                state.bias = 0.0
                state.bias_p = sub.bias_r[channel]
                state.bias_samples = 0
                bias_reset = True
            state.bias_rate = 0.0 if bias_reset else (state.bias - previous_bias) / dt
            corrected = innovation - state.bias
            s = innovation_s[channel]
            residual[channel] = corrected; residual_z[channel] = corrected / math.sqrt(s)
            bias[channel] = state.bias; rates[channel] = state.bias_rate
            # Measurement correction tracks the bias-corrected observation around the
            # Layer 3 prior; only covariance is retained because the next PINN prior is authoritative.
            gain = state.p / s
            state.p = max((1.0 - gain) * state.p, np.finfo(float).eps)
            bias_sigma = abs(state.bias) / math.sqrt(max(state.bias_p + sub.bias_r[channel], np.finfo(float).eps))
            if state.bias_samples >= self.config.minimum_bias_age_samples and (bias_sigma >= self.config.failed_bias_sigma or abs(residual_z[channel]) >= self.config.failed_residual_sigma):
                if self.config.substitution_enabled:
                    health[channel] = "substituted"; virtual[channel] = predicted[channel]
                else:
                    health[channel] = "failed"
            elif state.bias_samples >= self.config.minimum_bias_age_samples and bias_sigma >= self.config.drifting_bias_sigma:
                health[channel] = "drifting"
            else:
                health[channel] = "ok"

        for channel in self.states:
            if channel not in health:
                candidate = predicted_input.get(channel)
                if (not valid_input.get(channel, False) and self.config.substitution_enabled
                        and self._finite_number(candidate)):
                    health[channel] = "substituted"
                    virtual[channel] = float(candidate)
                    predicted.setdefault(channel, float(candidate))
                else:
                    health[channel] = "ok" if valid_input.get(channel, False) else "failed"
            bias.setdefault(channel, self.states[channel].bias); rates.setdefault(channel, self.states[channel].bias_rate)
        invalid = [channel for channel in self.states if not valid_input.get(channel, False)]
        missing_predictions = [channel for channel in self.states if valid_input.get(channel, False) and channel not in predicted_input]
        nis_by_subsystem = {
            name: float(np.mean([channel_innovation_z[channel] ** 2 for channel in sub.channels
                                 if channel in channel_innovation_z]))
            for name, sub in self.config.subfilters.items()
            if any(channel in channel_innovation_z for channel in sub.channels)
        }
        # Layer 7 trends use degradation magnitudes. Keep only values derived from
        # actual EKF updates; unavailable channels do not become zero evidence.
        indicator_names = {
            "oil_press_kPa": "oil_press_residual_kPa", "oil_temp_degC": "oil_temp_residual_degC",
            "cht_degC": "cht_residual_degC", "egt_degC": "egt_residual_degC",
            "map_kPa": "map_residual_kPa", "vibration_rms_g": "vib_A1x_residual_g",
        }
        indicators = {indicator_names[channel]: abs(value) for channel, value in residual.items()
                      if channel in indicator_names}
        latency_us = (time.perf_counter_ns() - started) / 1000.0
        return EKFOutput(
            master_timestamp_ns=timestamp, residual=residual, residual_raw=raw,
            bias_estimate=bias, bias_rate=rates,
            transient_active=transient, q_inflation_factor=q_factor,
            channel_valid={channel: bool(valid_input.get(channel, False)) for channel in self.states},
            predicted=predicted, residual_z=residual_z, sensor_health=health,
            virtual_value=virtual, measurement_updated=updateable,
            module_health={"status": "degraded" if invalid or missing_predictions else "ok",
                           "invalid_channels": invalid, "missing_predictions": missing_predictions},
            latency_us=latency_us,
            sync_quality=float(envelope.get("sync_quality", 0.0)),
            timing_estimated=bool(envelope.get("timing_estimated", True)),
            telemetry={name: float(value) for name, value in values.items()
                       if self._finite_number(value) and valid_input.get(name, False)},
            context=dict(envelope.get("context", {})), indicators=indicators,
            nis_by_subsystem=nis_by_subsystem,
            nis_by_channel={channel: value ** 2 for channel, value in channel_innovation_z.items()},
            model_versions={str(name): str(version) for name, version in envelope.get("model_versions", {}).items()},
            layer2_frame={
                "master_timestamp_ns": str(timestamp),
                "values": dict(envelope.get("values", {})),
                "channel_valid": dict(envelope.get("channel_valid", {})),
                "channel_new_sample": dict(envelope.get("channel_new_sample", {})),
                "channel_source": dict(envelope.get("channel_source", {})),
                "channel_age_ms": dict(envelope.get("channel_age_ms", {})),
                "channel_rate_hz": dict(envelope.get("channel_rate_hz", {})),
                "sync_quality": float(envelope.get("sync_quality", 0.0)),
                "context": dict(envelope.get("context", {})),
            },
        )
