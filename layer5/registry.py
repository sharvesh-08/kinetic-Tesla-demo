"""Single source of truth for named fixed-window Layer 5 feature columns."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class FeatureSchemaMismatch(ValueError):
    """Raised when a Layer 5 vector does not match the trained Layer 6 schema."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    subsystem: str
    statistic: str
    units: str
    plain_language: str


class FeatureRegistry:
    """Builds stable, named columns from the versioned YAML contract."""

    RAW_SAMPLES = 50

    def __init__(self, source: str | Path | None = None) -> None:
        source = source or os.environ.get("FEATURE_REGISTRY_PATH")
        path = Path(source) if source else Path(__file__).resolve().parents[1] / "configs" / "feature_registry.yaml"
        with path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
        if raw.get("schema_version") not in (3, 4):
            raise ValueError("Unsupported feature_registry schema_version")
        self.version = str(raw["feature_schema_version"])
        self.sample_rate_hz = int(raw["sample_rate_hz"])
        self.window_seconds = int(raw["window_seconds"])
        self.hop_fraction = float(raw["hop_fraction"])
        self.downsample_stride = int(raw["downsample_stride"])
        self.long_window_seconds = int(raw["long_window_seconds"])
        self.long_channels = tuple(raw["long_channels"])
        self.long_summaries = tuple(raw["long_summaries"])
        self.regime_features = tuple(raw["regime_features"])
        self.telemetry_features = dict(raw.get("telemetry_features") or {})
        self.channels: dict[str, dict[str, Any]] = raw["channels"]
        self.summaries: dict[str, dict[str, Any]] = raw["summaries"]
        if (self.sample_rate_hz, self.window_seconds, self.hop_fraction, self.downsample_stride) != (100, 5, 0.5, 10):
            raise ValueError("Feature contract requires 100 Hz, 5 seconds, 50% overlap and stride 10")
        if set(self.summaries) != {"mean", "max_abs", "slope", "nis_mean", "nis_max", "bias_mean", "availability"}:
            raise ValueError("Feature contract requires six summaries and availability per channel")
        if self.sample_rate_hz * self.window_seconds // self.downsample_stride != self.RAW_SAMPLES:
            raise ValueError("Feature contract must emit exactly 50 raw samples per channel")
        if self.long_window_seconds != 60 or not set(self.long_channels) <= set(self.channels):
            raise ValueError("Feature contract requires a 60-second long block with registered channels")
        if set(self.long_summaries) != {"mean", "slope", "nis_mean", "bias_mean", "availability"}:
            raise ValueError("Unsupported long-window summaries")
        self._specs = self._build_specs()
        self.feature_names = tuple(sorted(self._specs))

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate_hz * self.window_seconds * self.hop_fraction)

    @property
    def window_samples(self) -> int:
        return self.sample_rate_hz * self.window_seconds

    @staticmethod
    def raw_name(channel: str, index: int) -> str:
        return f"resid_{channel}_raw_{index:02d}"

    @staticmethod
    def summary_name(channel: str, statistic: str) -> str:
        return f"resid_{channel}_{statistic}"

    @staticmethod
    def long_name(channel: str, statistic: str) -> str:
        return f"resid_{channel}_{statistic}_60s"

    def _build_specs(self) -> dict[str, FeatureSpec]:
        specs: dict[str, FeatureSpec] = {}
        for channel, channel_meta in self.channels.items():
            for index in range(self.RAW_SAMPLES):
                name = self.raw_name(channel, index)
                specs[name] = FeatureSpec(name, str(channel_meta["subsystem"]), f"raw_sample_{index:02d}",
                                          str(channel_meta["units"]), f"imputed residual at 10 Hz sample {index + 1}")
            for statistic, meta in self.summaries.items():
                name = self.summary_name(channel, statistic)
                specs[name] = FeatureSpec(name, str(channel_meta["subsystem"]), statistic,
                                          str(meta["units"]), str(meta["plain_language"]))
        for channel in self.long_channels:
            channel_meta = self.channels[channel]
            for statistic in self.long_summaries:
                name = self.long_name(channel, statistic)
                units = self.summaries[statistic]["units"]
                specs[name] = FeatureSpec(name, str(channel_meta["subsystem"]), f"{statistic}_60s",
                                          str(units), f"rolling 60-second {statistic}; partial before 60 seconds")
        specs["history_seconds_available_60s"] = FeatureSpec(
            "history_seconds_available_60s", "context", "history_seconds", "s",
            "duration of observed history in the rolling 60-second block")
        for name in self.regime_features:
            specs[name] = FeatureSpec(name, "context", name, "context",
                                      "five-second operating-regime context; zero when unavailable")
            availability_name = name.replace("_5s", "_availability_5s")
            specs[availability_name] = FeatureSpec(availability_name, "context", "availability", "fraction",
                                                   f"fraction of samples with {name} available")
        for channel, meta in self.telemetry_features.items():
            for statistic in ("mean", "min", "max", "std", "slope", "availability"):
                name = f"telemetry_{channel}_{statistic}_5s"
                specs[name] = FeatureSpec(name, meta["subsystem"], statistic, meta["unit"],
                                          f"Observed {channel} {statistic}; availability distinguishes missing data")
        return specs

    def names_for_channel(self, channel: str) -> tuple[str, ...]:
        if channel not in self.channels:
            raise ValueError(f"Unknown feature channel {channel!r}")
        return tuple(self.raw_name(channel, i) for i in range(self.RAW_SAMPLES)) + tuple(
            self.summary_name(channel, name) for name in self.summaries
        )

    def spec(self, name: str, *_: Any) -> FeatureSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise ValueError(f"Feature {name!r} is not registered") from error

    def validate(self, names: list[str], window_s: float | None = None) -> None:
        if window_s is not None and float(window_s) != self.window_seconds:
            raise FeatureSchemaMismatch(f"Expected fixed {self.window_seconds}-second features, got {window_s}")
        unknown = set(names) - self._specs.keys()
        if unknown:
            raise FeatureSchemaMismatch(f"Unregistered Layer 5 feature names: {sorted(unknown)[:5]}")
        if len(names) != len(set(names)):
            raise FeatureSchemaMismatch("Feature vector contains duplicate names")

    def schema_hash(self, names: list[str] | tuple[str, ...]) -> str:
        payload = {"version": self.version, "names": sorted(names)}
        return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()

    def require_schema(self, names: list[str] | tuple[str, ...], expected_hash: str) -> None:
        self.validate(list(names), self.window_seconds)
        actual = self.schema_hash(names)
        if actual != expected_hash:
            raise FeatureSchemaMismatch(f"Feature schema hash mismatch: trained={expected_hash}, live={actual}")
        if tuple(sorted(names)) != self.feature_names:
            raise FeatureSchemaMismatch("Feature columns differ from the complete fixed-window contract")
