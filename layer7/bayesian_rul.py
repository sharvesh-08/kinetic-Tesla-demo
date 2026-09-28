"""Per-flight particle PI-RUL estimator with explicit overlap handling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class _State:
    degradation: np.ndarray
    rate_per_hour: np.ndarray
    onset_hours: np.ndarray
    weights: np.ndarray
    last_timestamp_ns: int
    elapsed_hours: float = 0.0
    last_long_update_ns: int | None = None
    anomaly_decay: int = 0
    long_update_count: int = 0


class BayesianPIRUL:
    def __init__(self, prior_path: str | Path | None = None) -> None:
        path = Path(prior_path) if prior_path else Path(__file__).resolve().parents[1] / "configs" / "pi_rul_priors.yaml"
        self.config: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        if self.config.get("schema_version") != 1:
            raise ValueError("Unsupported PI-RUL prior schema")
        self.rng = np.random.default_rng(int(self.config["seed"]))
        self.states: dict[tuple[str, str], dict[str, _State]] = {}
        self.first_timestamp: dict[tuple[str, str], int] = {}

    def _initialize(self, timestamp_ns: int, policy: dict[str, Any]) -> _State:
        count = int(self.config["particles"])
        degradation = np.clip(self.rng.normal(float(self.config["initial_degradation_mean"]),
                                              float(self.config["initial_degradation_sigma"]), count), 0, 0.3)
        rate = self.rng.lognormal(math.log(float(policy["prior_rate_median_per_hour"])),
                                  float(policy["prior_rate_log_sigma"]), count)
        onset = self.rng.uniform(float(policy.get("prior_onset_min_hours", 0.0)),
                                 float(policy.get("prior_onset_max_hours", 0.0)) + 1e-12, count)
        return _State(degradation, rate, onset, np.full(count, 1 / count), timestamp_ns)

    @staticmethod
    def _quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
        order = np.argsort(values)
        sorted_values = values[order]
        cumulative = np.cumsum(weights[order])
        return float(np.interp(probability, cumulative, sorted_values))

    def _observation(self, features: dict[str, float], policy: dict[str, Any], *, long: bool) -> tuple[float, float, float | None] | None:
        suffix = "_60s" if long else ""
        observations: list[tuple[float, float, float | None]] = []
        for channel, scale in policy["channels"].items():
            mean_key = f"resid_{channel}_mean{suffix}"
            if mean_key not in features:
                continue
            availability = float(features.get(f"resid_{channel}_availability{suffix}", 0.0))
            if availability <= 0:
                continue
            mean = float(features[mean_key])
            bias = float(features.get(f"resid_{channel}_bias_mean{suffix}", 0.0))
            nis = float(features.get(f"resid_{channel}_nis_mean{suffix}", 1.0))
            physical = max(abs(mean) - abs(bias), 0.0) / float(scale)
            slope_key = f"resid_{channel}_slope_60s"
            rate_observation = abs(float(features[slope_key])) * 3600 / float(scale) if long and slope_key in features else None
            observations.append((min(physical, 2.0), max(nis, 0.0), rate_observation))
        if not observations:
            return None
        # Largest channel evidence drives a subsystem; NIS broadens measurement noise.
        return max(observations, key=lambda item: item[0])

    def _likelihood(self, state: _State, observed: float, nis: float, rate_observation: float | None,
                    *, long: bool, weight: float, history_seconds: float = 0.0) -> None:
        sigma_key = "long_observation_sigma" if long else "short_observation_sigma"
        sigma = float(self.config[sigma_key]) * math.sqrt(1 + nis / 10.0)
        expected_state = state.degradation
        active_duration = None
        if long and history_seconds > 0:
            window_hours = history_seconds / 3600.0
            active_duration = np.minimum(np.maximum(state.elapsed_hours - state.onset_hours, 0.0), window_hours)
            expected_state = state.degradation - state.rate_per_hour * active_duration * (
                1.0 - active_duration / (2.0 * window_hours))
        residual = (observed - expected_state) / sigma
        log_weights = np.log(np.maximum(state.weights, 1e-300)) - 0.5 * weight * residual ** 2
        if long and rate_observation is not None:
            expected_rate = state.rate_per_hour * active_duration / window_hours
            rate_sigma = float(self.config["long_rate_observation_sigma_per_hour"]) * math.sqrt(1 + nis / 10.0)
            log_weights -= 0.5 * weight * ((rate_observation - expected_rate) / rate_sigma) ** 2
        log_weights -= np.max(log_weights)
        state.weights = np.exp(log_weights)
        state.weights /= np.sum(state.weights)
        effective = 1.0 / float(np.sum(state.weights ** 2))
        if effective < len(state.weights) / 2:
            indices = self.rng.choice(len(state.weights), size=len(state.weights), p=state.weights)
            state.degradation = state.degradation[indices]
            state.rate_per_hour = state.rate_per_hour[indices]
            state.onset_hours = state.onset_hours[indices]
            state.weights.fill(1.0 / len(state.weights))

    def update(self, event: dict[str, Any]) -> dict[str, Any]:
        key = (str(event.get("engine_serial") or "unknown"), str(event.get("sortie_id") or "unknown"))
        timestamp_ns = int(event["timestamp_ns"])
        features = dict(event.get("rul_observations") or {})
        if key not in self.states:
            self.states[key] = {name: self._initialize(timestamp_ns, policy)
                                for name, policy in self.config["subsystems"].items()}
            self.first_timestamp[key] = timestamp_ns
            initial = True
        else:
            initial = False
        black_box = event.get("black_box") or {}
        anomaly = bool(black_box.get("anomaly_detected"))
        per_class_shap = black_box.get("shap_by_class") or {}
        active = black_box.get("fault_flags") or {}
        triggered = {max(per_class_shap.get(name, {}), key=lambda item: abs(per_class_shap[name][item]))
                     for name, flag in active.items() if flag.get("active") and per_class_shap.get(name)}
        triggered = {"mechanical" if name == "vibration" else "combustion" if name == "air_path" else name
                     for name in triggered}
        if anomaly and not triggered:
            triggered = set(self.config["subsystems"])
        intervals: dict[str, dict[str, float | int]] = {}
        engine_samples: list[np.ndarray] = []
        for name, policy in self.config["subsystems"].items():
            state = self.states[key][name]
            elapsed_hours = max((timestamp_ns - state.last_timestamp_ns) / 3_600_000_000_000, 0.0)
            if not initial and elapsed_hours > 0:
                previous_elapsed = state.elapsed_hours
                state.elapsed_hours += elapsed_hours
                active_delta = np.maximum(state.elapsed_hours - state.onset_hours, 0.0) - np.maximum(previous_elapsed - state.onset_hours, 0.0)
                state.degradation = np.clip(state.degradation + state.rate_per_hour * active_delta, 0, 2)
                if anomaly and name in triggered:
                    state.anomaly_decay = int(self.config["anomaly_decay_windows"])
                if state.anomaly_decay:
                    inflation = float(self.config["anomaly_process_noise_multiplier"])
                    state.rate_per_hour *= self.rng.lognormal(
                        0, float(self.config["anomaly_rate_log_sigma_per_window"]) * inflation,
                        len(state.weights))
                    state.anomaly_decay -= 1
                short_observation = self._observation(features, policy, long=False)
                if short_observation is not None:
                    self._likelihood(state, *short_observation, long=False,
                                     weight=float(self.config["short_likelihood_weight"]))
                history = float(features.get("history_seconds_available_60s", 0.0))
                spacing_ns = int(float(self.config["long_update_spacing_seconds"]) * 1e9)
                ready = history >= float(self.config["long_min_history_seconds"])
                fresh = (state.last_long_update_ns is None and
                         timestamp_ns - self.first_timestamp[key] >= int(float(self.config["long_min_history_seconds"]) * 1e9)) or (
                             state.last_long_update_ns is not None and timestamp_ns - state.last_long_update_ns >= spacing_ns)
                if ready and fresh:
                    long_observation = self._observation(features, policy, long=True)
                    if long_observation is not None:
                        self._likelihood(state, *long_observation, long=True,
                                         weight=min(history / 60.0, 1.0), history_seconds=history)
                        state.last_long_update_ns = timestamp_ns
                        state.long_update_count += 1
            state.last_timestamp_ns = timestamp_ns
            threshold = float(self.config["failure_state"])
            wait_for_onset = np.maximum(state.onset_hours - state.elapsed_hours, 0.0)
            rul = wait_for_onset + np.maximum(threshold - state.degradation, 0) / np.maximum(state.rate_per_hour, 1e-12)
            sampled = self.rng.choice(rul, size=len(rul), p=state.weights)
            engine_samples.append(sampled)
            intervals[name] = {
                "rul_median_hours": self._quantile(rul, state.weights, 0.5),
                "rul_ci_low_hours": self._quantile(rul, state.weights, 0.05),
                "rul_ci_high_hours": self._quantile(rul, state.weights, 0.95),
                "posterior_theta_mean": float(np.sum(state.weights * state.rate_per_hour)),
                "posterior_theta_var": float(np.sum(state.weights * (state.rate_per_hour - np.sum(state.weights * state.rate_per_hour)) ** 2)),
                "long_update_count": state.long_update_count,
            }
        engine_rul = np.min(np.vstack(engine_samples), axis=0)
        engine_low, engine_median, engine_high = (float(np.quantile(engine_rul, probability))
                                                  for probability in (0.05, 0.5, 0.95))
        return {"subsystems": intervals,
                "whole_engine": {"rul_median_hours": engine_median,
                                 "rul_ci_low_hours": engine_low, "rul_ci_high_hours": engine_high},
                "prior_provenance": self.config["provenance"],
                "validated_for_decisions": bool(self.config.get("validated_for_decisions", False))}
