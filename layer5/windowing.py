"""Fixed 5-second, 50%-overlap FeatureWindow assembly for Layer 5."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import WindowingConfig, load_config
from .registry import FeatureRegistry


@dataclass(frozen=True)
class FeatureWindow:
    window_start_ns: int
    window_end_ns: int
    window_length_s: float
    regime: str
    features: dict[str, float]
    feature_valid: dict[str, bool]
    cold_start: bool
    truncated: bool
    data_quality_score: float
    n_samples: dict[str, int]
    feature_schema_version: str
    feature_schema_hash: str
    ground_truth_label: str | None = None
    context: dict[str, Any] | None = None
    telemetry: dict[str, float] | None = None
    indicators: dict[str, float] | None = None
    nis_by_subsystem: dict[str, float] | None = None
    model_versions: dict[str, str] | None = None
    telemetry_timestamp_ns: int | None = None
    layer2_frame: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdaptiveWindowAssembler:
    """Compatibility name for a now strictly fixed-window assembler.

    Missing ticks are inserted on the 100 Hz grid. Per-channel residual/NIS/bias
    state is zero-order held at each invalid position while validity is retained
    separately for availability calculation.
    """

    def __init__(self, config: WindowingConfig | None = None, registry: FeatureRegistry | None = None) -> None:
        self.config = config or load_config()
        self.registry = registry or FeatureRegistry()
        if (self.config.master_rate_hz != 100 or self.config.window_seconds != 5
                or self.config.overlap != 0.5
                or self.config.window_samples != self.registry.window_samples
                or self.config.downsample_stride != self.registry.downsample_stride
                or self.config.schema_version != self.registry.version):
            raise ValueError("Windowing config and feature registry disagree")
        self._period_ns = int(round(1e9 / self.config.master_rate_hz))
        self._samples: deque[dict[str, Any]] = deque(
            maxlen=self.registry.long_window_seconds * int(self.config.master_rate_hz))
        self._last_tick_ns: int | None = None
        self._last_values: dict[str, tuple[float, float, float]] = {
            channel: (0.0, 0.0, 0.0) for channel in self.registry.channels
        }
        self._origin_ns: int | None = None
        self.session_id = None
        self._latest_source: dict[str, Any] = {}
        self._next_emit_count = self.config.window_samples
        self._total_ticks = 0

    @staticmethod
    def _finite(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

    def _grid_tick(self, timestamp_ns: int, item: dict[str, Any] | None) -> None:
        residual = (item or {}).get("residual", {})
        residual_z = (item or {}).get("residual_z", {})
        nis_by_channel = (item or {}).get("nis_by_channel", {})
        bias = (item or {}).get("bias_estimate", {})
        channel_valid = (item or {}).get("channel_valid", {})
        measurement_updated = (item or {}).get("measurement_updated", {})
        health = (item or {}).get("sensor_health", {})
        values: dict[str, tuple[float, float, float]] = {}
        valid: dict[str, bool] = {}
        for channel in self.registry.channels:
            actual = (item is not None and bool(channel_valid.get(channel, False))
                      and bool(measurement_updated.get(channel, False))
                      and health.get(channel, "ok") != "substituted"
                      and self._finite(residual.get(channel))
                      and (self._finite(nis_by_channel.get(channel)) or self._finite(residual_z.get(channel))))
            if actual:
                nis = float(nis_by_channel[channel]) if self._finite(nis_by_channel.get(channel)) else float(residual_z[channel]) ** 2
                current = (float(residual[channel]), nis,
                           float(bias[channel]) if self._finite(bias.get(channel)) else self._last_values[channel][2])
                self._last_values[channel] = current
            else:
                current = self._last_values[channel]
            values[channel] = current
            valid[channel] = bool(actual)
        if item is not None:
            self._latest_source = item
            self._latest_source["layer2_frame"] = {
                **{key: item.get(key) for key in (
                    "master_timestamp_ns", "channel_valid", "channel_source",
                    "channel_age_ms", "channel_rate_hz", "sync_quality", "context")},
                "values": dict(item.get("telemetry") or {}),
            }
        self._samples.append({
            "timestamp_ns": timestamp_ns, "values": values, "valid": valid,
            "source": item or {},
        })
        self._last_tick_ns = timestamp_ns
        self._total_ticks += 1

    def _append_to(self, timestamp_ns: int, item: dict[str, Any]) -> list[FeatureWindow]:
        if self._origin_ns is None:
            self._origin_ns = timestamp_ns
        target = (timestamp_ns - self._origin_ns + self._period_ns // 2) // self._period_ns
        if target < self._total_ticks:
            return []  # Duplicate or late sample cannot rewrite an emitted window.
        outputs: list[FeatureWindow] = []
        while self._total_ticks <= target:
            grid_ns = self._origin_ns + self._total_ticks * self._period_ns
            self._grid_tick(grid_ns, item if self._total_ticks == target else None)
            # Emit at the boundary itself, including boundaries crossed by a gap.
            if self._total_ticks == self._next_emit_count:
                outputs.append(self._emit())
                self._next_emit_count += self.config.hop_samples
        return outputs

    def _emit(self) -> FeatureWindow:
        points = list(self._samples)[-self.config.window_samples:]
        if len(points) != self.config.window_samples:
            raise RuntimeError("Attempted to emit an incomplete fixed window")
        features: dict[str, float] = {}
        valid_features: dict[str, bool] = {}
        counts: dict[str, int] = {}
        times = np.arange(self.config.window_samples, dtype=float) / self.config.master_rate_hz
        for channel in self.registry.channels:
            seq = np.asarray([point["values"][channel][0] for point in points], dtype=float)
            nis = np.asarray([point["values"][channel][1] for point in points], dtype=float)
            bias = np.asarray([point["values"][channel][2] for point in points], dtype=float)
            availability = np.asarray([point["valid"][channel] for point in points], dtype=float)
            counts[channel] = int(availability.sum())
            slope = float(np.polyfit(times, seq, 1)[0])
            raw = seq[::self.config.downsample_stride]
            if len(raw) != self.registry.RAW_SAMPLES:
                raise RuntimeError(f"Expected 50 raw samples for {channel}, received {len(raw)}")
            for index, value in enumerate(raw):
                name = self.registry.raw_name(channel, index)
                features[name] = float(value)
                valid_features[name] = True  # Missingness is carried by the availability field.
            summaries = {
                "mean": float(np.mean(seq)), "max_abs": float(np.max(np.abs(seq))), "slope": slope,
                "nis_mean": float(np.mean(nis)), "nis_max": float(np.max(nis)),
                "bias_mean": float(np.mean(bias)), "availability": float(np.mean(availability)),
            }
            for statistic, value in summaries.items():
                name = self.registry.summary_name(channel, statistic)
                features[name] = value
                valid_features[name] = True
        long_points = list(self._samples)
        long_times = np.arange(len(long_points), dtype=float) / self.config.master_rate_hz
        history_name = "history_seconds_available_60s"
        features[history_name] = len(long_points) / self.config.master_rate_hz
        valid_features[history_name] = True
        for channel in self.registry.long_channels:
            residual = np.asarray([point["values"][channel][0] for point in long_points], dtype=float)
            nis = np.asarray([point["values"][channel][1] for point in long_points], dtype=float)
            bias = np.asarray([point["values"][channel][2] for point in long_points], dtype=float)
            availability = np.asarray([point["valid"][channel] for point in long_points], dtype=float)
            long_values = {
                "mean": float(np.mean(residual)),
                "slope": float(np.polyfit(long_times, residual, 1)[0]),
                "nis_mean": float(np.mean(nis)),
                "bias_mean": float(np.mean(bias)),
                "availability": float(np.mean(availability)),
            }
            for statistic, value in long_values.items():
                name = self.registry.long_name(channel, statistic)
                features[name] = value
                valid_features[name] = True
        regime_sources = {"rpm": "rpm", "throttle": "throttle_pct", "altitude": "altitude_m"}
        for base, source_name in regime_sources.items():
            def regime_value(point: dict[str, Any]) -> Any:
                source = point["source"]
                return source.get("telemetry", {}).get(
                    source_name, source.get("context", {}).get("operating_context", {}).get(source_name))
            values = [float(regime_value(point)) for point in points if self._finite(regime_value(point))]
            for suffix in (("mean",) if base != "rpm" else ("mean", "slope")):
                name = f"{base}_{suffix}_5s"
                if name not in self.registry.regime_features:
                    continue
                if suffix == "slope":
                    ordered = [(i / self.config.master_rate_hz, float(regime_value(point)))
                               for i, point in enumerate(points)
                               if self._finite(regime_value(point))]
                    value = float(np.polyfit(*zip(*ordered), 1)[0]) if len(ordered) > 1 else 0.0
                else:
                    value = float(np.mean(values)) if values else 0.0
                features[name] = value
                valid_features[name] = bool(values)
                availability_name = name.replace("_5s", "_availability_5s")
                features[availability_name] = len(values) / len(points)
                valid_features[availability_name] = True
        for channel in getattr(self.registry, "telemetry_features", {}):
            observed = [(index / self.config.master_rate_hz, point["source"].get("telemetry", {}).get(channel))
                        for index, point in enumerate(points)]
            observed = [(t, float(v)) for t, v in observed if self._finite(v)]
            values = np.asarray([v for _, v in observed])
            summaries = {
                "mean": float(values.mean()) if len(values) else 0.0,
                "min": float(values.min()) if len(values) else 0.0,
                "max": float(values.max()) if len(values) else 0.0,
                "std": float(values.std()) if len(values) else 0.0,
                "slope": float(np.polyfit(*zip(*observed), 1)[0]) if len(observed) > 1 else 0.0,
                "availability": len(observed) / len(points),
            }
            for statistic, value in summaries.items():
                name = f"telemetry_{channel}_{statistic}_5s"
                features[name] = value
                valid_features[name] = bool(observed) or statistic == "availability"
        self.registry.require_schema(list(features), self.registry.schema_hash(list(features)))
        first, last = points[0], points[-1]
        latest = self._latest_source
        def mean_metadata(field: str) -> dict[str, float]:
            collected: dict[str, list[float]] = {}
            for point in points:
                for name, value in point['source'].get(field, {}).items():
                    if self._finite(value):
                        collected.setdefault(name, []).append(float(value))
            return {name: float(np.mean(values)) for name, values in collected.items()}

        channel_availability = [features[self.registry.summary_name(channel, "availability")]
                                for channel in self.registry.channels]
        return FeatureWindow(
            window_start_ns=int(first["timestamp_ns"]), window_end_ns=int(last["timestamp_ns"] + self._period_ns),
            window_length_s=5.0, regime="fixed", features=features, feature_valid=valid_features,
            cold_start=False, truncated=False, data_quality_score=float(np.mean(channel_availability)),
            n_samples=counts, feature_schema_version=self.registry.version,
            feature_schema_hash=self.registry.schema_hash(list(features)),
            ground_truth_label=latest.get("ground_truth_label"), context=dict(latest.get("context", {})),
            telemetry=dict(latest.get("telemetry", {})), indicators=mean_metadata("indicators"),
            layer2_frame=dict(latest.get("layer2_frame", {})),
            nis_by_subsystem=mean_metadata("nis_by_subsystem"),
            model_versions=dict(latest.get("model_versions", {})),
            telemetry_timestamp_ns=int(latest.get("master_timestamp_ns", last["timestamp_ns"])),
        )

    def process(self, item: dict[str, Any]) -> list[FeatureWindow]:
        """Consume one Layer 4 frame; return all newly completed 5-second windows."""
        session = (item.get("context") or {}).get("session_id")
        if session and session != self.session_id:
            self.__init__(self.config, self.registry)
            self.session_id = session
        timestamp = int(item["master_timestamp_ns"])
        return self._append_to(timestamp, item)


FixedWindowAssembler = AdaptiveWindowAssembler
