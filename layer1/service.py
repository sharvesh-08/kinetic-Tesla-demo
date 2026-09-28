"""Validated Layer 1 telemetry publisher.

Hardware adapters call :meth:`publish_ecu` or :meth:`publish_vibration`; this
module is deliberately independent of any particular ECU or serial protocol.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as redis
from redis_limits import xadd_bounded, stream_maxlen

from .schema import TELEMETRY_CHANNELS, VIBRATION_SAMPLE_RATE_HZ


class TelemetryValidationError(ValueError):
    """An input cannot be represented by the canonical Layer 1 contract."""


def _timestamp(value: Any) -> str:
    if isinstance(value, bool):
        raise TelemetryValidationError("sensor_timestamp_ns must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TelemetryValidationError("sensor_timestamp_ns must be an integer") from error
    if result < 0:
        raise TelemetryValidationError("sensor_timestamp_ns must be non-negative")
    return str(result)


def validate_ecu_event(event: dict[str, Any]) -> dict[str, Any]:
    sensor = event.get("sensor")
    spec = TELEMETRY_CHANNELS.get(sensor)
    value = event.get("value")
    if spec is None:
        raise TelemetryValidationError(f"unknown telemetry channel: {sensor!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TelemetryValidationError("value must be a finite number")
    if not spec.minimum <= float(value) <= spec.maximum:
        raise TelemetryValidationError(
            f"{sensor} value {value} is outside [{spec.minimum}, {spec.maximum}] {spec.unit}"
        )
    result: dict[str, Any] = {
        "sensor": sensor,
        "value": float(value),
        "valid": event.get("valid", True) is True,
        "sensor_timestamp_ns": _timestamp(event.get("sensor_timestamp_ns")),
    }
    context = event.get("context")
    if "sample_rate_hz" in event:
        rate = event["sample_rate_hz"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= 2000:
            raise TelemetryValidationError("sample_rate_hz must be finite and in (0, 2000]")
        result["sample_rate_hz"] = float(rate)
    if context is not None:
        if not isinstance(context, dict):
            raise TelemetryValidationError("context must be an object")
        result["context"] = context
    return result


def validate_vibration_event(event: dict[str, Any]) -> dict[str, Any]:
    try:
        sample_rate = int(event["sample_rate_hz"])
        samples_per_axis = int(event["samples_per_axis"])
    except (KeyError, TypeError, ValueError) as error:
        raise TelemetryValidationError("vibration sample counts must be integers") from error
    axes = event.get("axes")
    reference = event.get("data_reference")
    if sample_rate != VIBRATION_SAMPLE_RATE_HZ or samples_per_axis <= 0:
        raise TelemetryValidationError("unsupported vibration sample rate or empty chunk")
    if not isinstance(axes, list) or not axes or any(axis not in {"x", "y", "z"} for axis in axes):
        raise TelemetryValidationError("axes must be a non-empty subset of x/y/z")
    if len(set(axes)) != len(axes) or not isinstance(reference, str) or not reference:
        raise TelemetryValidationError("axes must be unique and data_reference must be non-empty")
    return {
        "sensor_timestamp_ns": _timestamp(event.get("sensor_timestamp_ns")),
        "sample_rate_hz": sample_rate,
        "samples_per_axis": samples_per_axis,
        "axes": axes,
        "data_reference": reference,
    }


@dataclass(frozen=True)
class Layer1Config:
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"))
    ecu_stream: str = "engine:telemetry:ecu"
    vibration_stream: str = "engine:telemetry:vib"


class Layer1Service:
    def __init__(self, config: Layer1Config | None = None) -> None:
        self.config = config or Layer1Config()
        self.redis = redis.from_url(self.config.redis_url, decode_responses=True)

    async def publish_ecu(self, event: dict[str, Any]) -> str:
        payload = validate_ecu_event(event)
        return await xadd_bounded(self.redis,
            self.config.ecu_stream, {"payload": json.dumps(payload, separators=(",", ":"))}
        )

    async def publish_snapshot(self, events: list[dict[str, Any]]) -> list[str]:
        """Commit one simulator tick atomically while keeping per-sensor events."""
        payloads = [validate_ecu_event(event) for event in events]
        async with self.redis.pipeline(transaction=True) as pipe:
            for payload in payloads:
                pipe.xadd(self.config.ecu_stream,
                          {"payload": json.dumps(payload, separators=(",", ":"), allow_nan=False)},
                          maxlen=stream_maxlen(self.config.ecu_stream), approximate=True)
            return await pipe.execute()

    async def publish_vibration(self, event: dict[str, Any]) -> str:
        payload = validate_vibration_event(event)
        return await xadd_bounded(self.redis,
            self.config.vibration_stream, {"payload": json.dumps(payload, separators=(",", ":"))}
        )

    async def close(self) -> None:
        await self.redis.aclose()
