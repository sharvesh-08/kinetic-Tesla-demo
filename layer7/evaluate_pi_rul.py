"""Evaluate first-hour PI-RUL interval coverage on held-out synthetic fleet runs."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .bayesian_rul import BayesianPIRUL


class GaussianRateBaseline:
    """Lightweight Gaussian approximation with a single mean onset time."""

    def __init__(self, policy: dict, config: dict) -> None:
        median = float(policy["prior_rate_median_per_hour"])
        log_sigma = float(policy["prior_rate_log_sigma"])
        rate_mean = median * np.exp(log_sigma ** 2 / 2)
        rate_var = (np.exp(log_sigma ** 2) - 1) * median ** 2 * np.exp(log_sigma ** 2)
        self.mean = np.asarray([float(config["initial_degradation_mean"]), rate_mean])
        self.covariance = np.diag([float(config["initial_degradation_sigma"]) ** 2, rate_var])
        self.onset = (float(policy.get("prior_onset_min_hours", 0)) +
                      float(policy.get("prior_onset_max_hours", 0))) / 2
        self.elapsed = 0.0
        self.config = config

    def update(self, seconds: float, short_mean: float, long_mean: float, slope_per_second: float,
               history_seconds: float, *, last_long_seconds: float | None) -> None:
        current_hours = seconds / 3600.0
        active_delta = max(current_hours - self.onset, 0.0) - max(self.elapsed - self.onset, 0.0)
        transition = np.asarray([[1.0, active_delta], [0.0, 1.0]])
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T
        self.elapsed = current_hours
        self._observe(short_mean, np.asarray([1.0, 0.0]),
                      float(self.config["short_observation_sigma"]) ** 2 /
                      float(self.config["short_likelihood_weight"]))
        if last_long_seconds is not None:
            weight = min(history_seconds / 60.0, 1.0)
            window_hours = history_seconds / 3600.0
            active_duration = min(max(current_hours - self.onset, 0.0), window_hours)
            mean_coefficient = -active_duration * (1 - active_duration / (2 * window_hours))
            self._observe(long_mean, np.asarray([1.0, mean_coefficient]),
                          float(self.config["long_observation_sigma"]) ** 2 / weight)
            active_fraction = active_duration / window_hours
            if active_fraction > 0:
                self._observe(slope_per_second * 3600, np.asarray([0.0, active_fraction]),
                              float(self.config["long_rate_observation_sigma_per_hour"]) ** 2 / weight)

    def _observe(self, value: float, observation: np.ndarray, variance: float) -> None:
        innovation_variance = float(observation @ self.covariance @ observation + variance)
        gain = self.covariance @ observation / innovation_variance
        self.mean += gain * (value - observation @ self.mean)
        self.covariance = (np.eye(2) - np.outer(gain, observation)) @ self.covariance

    def interval(self) -> tuple[float, float, float]:
        samples = np.random.default_rng(42).multivariate_normal(self.mean, self.covariance, 5000)
        positive_rate = np.maximum(samples[:, 1], 1e-6)
        rul = max(self.onset - self.elapsed, 0.0) + np.maximum(1.0 - samples[:, 0], 0.0) / positive_rate
        return tuple(float(np.quantile(rul, probability)) for probability in (0.05, 0.5, 0.95))


def evaluate(source: Path, *, runs_per_subsystem: int = 5) -> dict:
    grouped: defaultdict[tuple[str, str], list[tuple[float, float, bool]]] = defaultdict(list)
    truth: dict[tuple[str, str], tuple[float, float]] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["subsystem"], row["flight_id"])
            grouped[key].append(
                (float(row["operating_hours"]), float(row["degradation_state"]),
                 row["failure_or_removal"] == "1"))
            if key not in truth and row.get("severity") and row.get("fault_onset_hours"):
                truth[key] = (float(row["fault_onset_hours"]), float(row["severity"]))
    base = BayesianPIRUL().config
    selected: defaultdict[str, list[tuple[str, list[tuple[float, float, bool]]]]] = defaultdict(list)
    for (subsystem, flight_id), samples in sorted(grouped.items()):
        samples.sort()
        if len(selected[subsystem]) < runs_per_subsystem:
            selected[subsystem].append((flight_id, samples))
    report = {"subsystems": {}, "scope": "Held-out synthetic constant-rate fleet only; exact threshold time from generator parameters."}
    with tempfile.TemporaryDirectory() as directory:
        for subsystem, flights in selected.items():
            one = dict(base)
            one["subsystems"] = {subsystem: base["subsystems"][subsystem]}
            config_path = Path(directory) / f"{subsystem}.yaml"
            config_path.write_text(yaml.safe_dump(one), encoding="utf-8")
            channel, scale = next(iter(one["subsystems"][subsystem]["channels"].items()))
            coverage = []
            median_relative_error = []
            runtimes = []
            first_intervals = []
            gaussian_coverage = []
            gaussian_error = []
            gaussian_runtimes = []
            gaussian_widths = []
            particle_widths = []
            for flight_id, samples in flights:
                estimator = BayesianPIRUL(config_path)
                gaussian = GaussianRateBaseline(one["subsystems"][subsystem], one)
                hours = np.asarray([item[0] for item in samples])
                degradation = np.asarray([item[1] for item in samples])
                onset, true_rate = truth[(subsystem, flight_id)]
                failure_hour = onset + (1.0 - degradation[0]) / true_rate
                first = None
                particle_seconds = 0.0
                gaussian_seconds = 0.0
                last_gaussian_long = None
                for seconds in np.arange(5.0, 3600.1, 2.5):
                    state = float(np.interp(seconds / 3600.0, hours, degradation))
                    previous = float(np.interp(max(seconds - 60.0, 0.0) / 3600.0, hours, degradation))
                    slope_per_second = (state - previous) / min(seconds, 60.0)
                    short_mean = float(np.mean(np.interp(
                        np.linspace(max(seconds - 5.0, 0.0), seconds, 9) / 3600.0, hours, degradation)))
                    long_mean = float(np.mean(np.interp(
                        np.linspace(max(seconds - 60.0, 0.0), seconds, 25) / 3600.0, hours, degradation)))
                    features = {"history_seconds_available_60s": min(seconds, 60.0)}
                    for suffix in ("", "_60s"):
                        features.update({f"resid_{channel}_mean{suffix}": (long_mean if suffix else short_mean) * float(scale),
                                         f"resid_{channel}_bias_mean{suffix}": 0.0,
                                         f"resid_{channel}_nis_mean{suffix}": 1.0,
                                         f"resid_{channel}_availability{suffix}": 1.0})
                    features[f"resid_{channel}_slope_60s"] = slope_per_second * float(scale)
                    started = time.perf_counter()
                    result = estimator.update({"timestamp_ns": int(seconds * 1e9),
                                               "engine_serial": "SYNTHETIC", "sortie_id": flight_id,
                                               "rul_observations": features, "black_box": {}})
                    particle_seconds += time.perf_counter() - started
                    if seconds > 5.0:
                        long_ready = seconds >= 35.0 and (
                            last_gaussian_long is None or seconds - last_gaussian_long >= 60.0)
                        started = time.perf_counter()
                        gaussian.update(seconds, short_mean, long_mean, slope_per_second,
                                        min(seconds, 60.0), last_long_seconds=seconds if long_ready else None)
                        gaussian_seconds += time.perf_counter() - started
                        if long_ready:
                            last_gaussian_long = seconds
                    if first is None:
                        first = result["subsystems"][subsystem]
                runtimes.append(particle_seconds)
                gaussian_runtimes.append(gaussian_seconds)
                interval = result["subsystems"][subsystem]
                true_remaining = failure_hour - 1.0
                coverage.append(interval["rul_ci_low_hours"] <= true_remaining <= interval["rul_ci_high_hours"])
                median_relative_error.append(abs(interval["rul_median_hours"] - true_remaining) / true_remaining)
                first_intervals.append(first["rul_ci_high_hours"] - first["rul_ci_low_hours"])
                particle_widths.append(interval["rul_ci_high_hours"] - interval["rul_ci_low_hours"])
                gaussian_interval = gaussian.interval()
                gaussian_coverage.append(gaussian_interval[0] <= true_remaining <= gaussian_interval[2])
                gaussian_error.append(abs(gaussian_interval[1] - true_remaining) / true_remaining)
                gaussian_widths.append(gaussian_interval[2] - gaussian_interval[0])
            report["subsystems"][subsystem] = {
                "heldout_runs": len(flights),
                "first_hour_90pct_interval_coverage": sum(coverage) / len(coverage) if coverage else None,
                "first_hour_median_relative_error": float(np.median(median_relative_error)) if median_relative_error else None,
                "first_window_median_interval_width_hours": float(np.median(first_intervals)) if first_intervals else None,
                "mean_desktop_seconds_per_one_hour_flight": float(np.mean(runtimes)) if runtimes else None,
                "first_hour_median_interval_width_hours": float(np.median(particle_widths)) if particle_widths else None,
                "gaussian_baseline_first_hour_90pct_coverage": sum(gaussian_coverage) / len(gaussian_coverage) if gaussian_coverage else None,
                "gaussian_baseline_first_hour_median_relative_error": float(np.median(gaussian_error)) if gaussian_error else None,
                "gaussian_baseline_mean_desktop_seconds_per_one_hour_flight": float(np.mean(gaussian_runtimes)) if gaussian_runtimes else None,
                "gaussian_baseline_first_hour_median_interval_width_hours": float(np.median(gaussian_widths)) if gaussian_widths else None,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--runs-per-subsystem", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.source, runs_per_subsystem=args.runs_per_subsystem)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
