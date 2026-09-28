"""Configuration for fixed-rate, fixed-length Layer 5 framing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import os


@dataclass(frozen=True)
class WindowingConfig:
    master_rate_hz: float
    window_seconds: float
    overlap: float
    downsample_stride: int
    schema_version: str

    @property
    def window_samples(self) -> int:
        return int(round(self.master_rate_hz * self.window_seconds))

    @property
    def hop_samples(self) -> int:
        return int(round(self.window_samples * (1.0 - self.overlap)))


def load_config(path: str | Path | None = None) -> WindowingConfig:
    source = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "windowing.yaml"
    with source.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if raw.get("schema_version") != 3:
        raise ValueError("Unsupported fixed windowing schema_version")
    config = WindowingConfig(
        master_rate_hz=float(raw["master_rate_hz"]),
        window_seconds=float(raw["window_seconds"]), overlap=float(raw["overlap"]),
        downsample_stride=int(raw["downsample_stride"]),
        schema_version=str(raw["feature_schema_version"]),
    )
    if path is None and os.environ.get("FEATURE_REGISTRY_PATH"):
        from dataclasses import replace
        from .registry import FeatureRegistry
        config = replace(config, schema_version=FeatureRegistry().version)
    if (config.master_rate_hz != 100 or config.window_seconds != 5 or config.overlap != 0.5
            or config.downsample_stride != 10):
        raise ValueError("Layer 5 requires fixed 100 Hz, 5-second, 50%-overlap windows and stride 10")
    return config
