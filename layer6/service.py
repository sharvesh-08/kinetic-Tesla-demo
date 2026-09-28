"""Layer 5 FeatureWindow to Layer 6 assessment envelope."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import os
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ResponseError

from .artifact import load_artifact
from .model import Layer6Models
from layer5.registry import FeatureRegistry
from redis_limits import xadd_bounded
from redis_streams import claim_pending

LOGGER = logging.getLogger("layer6.service")


@dataclass(frozen=True)
class ServiceConfig:
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"))
    input_stream: str = "engine:features:windows"
    output_stream: str = "engine:layer6:assessments"
    consumer_group: str = "layer6-classifier"
    consumer_name: str = "edge-layer6"
    claim_idle_ms: int = 5000
    read_count: int = 100
    block_ms: int = 250


class Layer6Processor:
    def __init__(self, model: Layer6Models | None = None) -> None:
        self.model = model or load_artifact().model

    def process(self, window: dict[str, Any]) -> dict[str, Any]:
        context = dict(window.get("context") or {})
        supported = (
            float(window.get("window_length_s", 0)) == 5.0
            and int(window["window_end_ns"]) - int(window["window_start_ns"]) == 5_000_000_000
            and not bool(window.get("truncated"))
            and len(window.get("features", {})) == len(self.model.feature_names)
        )
        schema_matches = (
            window.get("feature_schema_hash") == self.model.feature_schema_hash
            and tuple(sorted(window.get("features", {}))) == self.model.feature_names
        )
        missing_context = [name for name in ("engine_serial", "sortie_id", "operating_hours")
                           if context.get(name) is None]
        black_box: dict[str, Any] = {}
        reason: str | None = None
        eligibility_reason: str | None = None
        training_scope = getattr(self.model, "training_scope", "unknown")
        data_quality = float(window.get("data_quality_score", 0.0))
        registry = getattr(self.model, "registry", None) or FeatureRegistry()
        features = window.get("features", {})
        unavailable_channels = [
            channel for channel in registry.channels
            if float(features.get(registry.summary_name(channel, "availability"), 0.0)) < 0.05
        ]
        structurally_available = True
        if not supported:
            reason = "Only complete, non-truncated fixed 5-second FeatureWindows are supported."
            structurally_available = False
        elif not schema_matches:
            reason = "FeatureWindow schema is incompatible with the loaded Layer 6 artifact."
            structurally_available = False
        if structurally_available:
            black_box = self.model.predict(window).to_dict()
        eligibility_reasons = []
        if training_scope != "verified_healthy_nominal_only":
            eligibility_reasons.append("training data is not independently verified healthy data")
        if data_quality < 0.05:
            eligibility_reasons.append(f"measurement availability is {data_quality:.3f}, below 0.050")
        if unavailable_channels:
            eligibility_reasons.append("residual availability is below 0.050 for: " + ", ".join(unavailable_channels))
        if eligibility_reasons:
            eligibility_reason = ("Inference is shown for inspection, but is not eligible for Layer 7 risk fusion because "
                                  + "; ".join(eligibility_reasons) + ".")
        return {
            "timestamp_ns": int(window["window_end_ns"]),
            "session_id": context.get("session_id"),
            "control_revision": context.get("control_revision"),
            "parameter_provenance": context.get("parameter_provenance", {}),
            "window_start_ns": int(window["window_start_ns"]),
            "window_end_ns": int(window["window_end_ns"]),
            "engine_serial": context.get("engine_serial"), "sortie_id": context.get("sortie_id"),
            "data_quality_score": float(window.get("data_quality_score", 0.0)),
            "model_training_scope": training_scope,
            "unavailable_residual_channels": unavailable_channels,
            "feature_schema_version": window.get("feature_schema_version"),
            "feature_schema_hash": window.get("feature_schema_hash"),
            "classification_available": bool(black_box), "inference_available": bool(black_box),
            "decision_eligible": (bool(black_box) and training_scope == "verified_healthy_nominal_only"
                                  and data_quality >= 0.05 and not unavailable_channels),
            "classification_unavailable_reason": reason,
            "decision_eligibility_reason": eligibility_reason,
            "missing_context": missing_context, "black_box": black_box,
            "operating_hours": context.get("operating_hours"),
            "indicators": dict(window.get("indicators") or {}),
            "telemetry": dict(window.get("telemetry") or {}),
            "layer2_frame": dict(window.get("layer2_frame") or {}),
            "telemetry_timestamp_ns": window.get("telemetry_timestamp_ns", window.get("window_end_ns")),
            "nis_by_subsystem": dict(window.get("nis_by_subsystem") or {}),
            "rul_observations": {name: float(value) for name, value in window.get("features", {}).items()
                                 if "_raw_" not in name and (name.startswith("resid_") or name == "history_seconds_available_60s")},
            "operating_context": dict(context.get("operating_context") or {}),
            "upstream_model_versions": dict(window.get("model_versions") or {}),
        }


class Layer6Service:
    def __init__(self, config: ServiceConfig | None = None, processor: Layer6Processor | None = None) -> None:
        self.config = config or ServiceConfig(); self.processor = processor or Layer6Processor()
        self.redis = redis.from_url(self.config.redis_url, decode_responses=True); self.running = False

    async def start(self) -> None:
        try:
            await self.redis.xgroup_create(self.config.input_stream, self.config.consumer_group, id="0", mkstream=True)
        except ResponseError as error:
            if "BUSYGROUP" not in str(error): raise
        await self._recover_pending(); self.running = True

    async def _recover_pending(self) -> None:
        cursor = "0-0"
        while True:
            cursor, entries, _ = await claim_pending(
                self.redis,
                self.config.input_stream, self.config.consumer_group, self.config.consumer_name,
                self.config.claim_idle_ms, cursor, count=self.config.read_count)
            for entry_id, fields in entries: await self.process_entry(entry_id, fields)
            if cursor == "0-0": return

    async def process_entry(self, entry_id: str, fields: dict[str, str]) -> None:
        try:
            result = self.processor.process(json.loads(fields["payload"]))
            output_id = f"{result['window_end_ns'] // 1_000_000}-0"
            try:
                published_id = await xadd_bounded(self.redis, self.config.output_stream,
                                                   {"payload": json.dumps(result, separators=(",", ":"), allow_nan=False)},
                                                   id=output_id)
                if not published_id:
                    LOGGER.error(json.dumps({"timestamp_ns": time.time_ns(), "module": "layer6",
                                             "event": "publish_failed", "window_end_ns": result["window_end_ns"]}))
                    return
            except ResponseError as error:
                if "equal or smaller" not in str(error): raise
            try:
                await self.redis.xack(self.config.input_stream, self.config.consumer_group, entry_id)
            except Exception:
                pass
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            LOGGER.error(json.dumps({"timestamp_ns": time.time_ns(), "module": "layer6", "event": "invalid_input", "error": str(error)}))

    async def run(self) -> None:
        await self.start()
        while self.running:
            try:
                records = await self.redis.xreadgroup(self.config.consumer_group, self.config.consumer_name,
                    {self.config.input_stream: ">"}, count=self.config.read_count, block=self.config.block_ms)
            except Exception:
                await asyncio.sleep(0.05)
                continue
            for _, entries in records:
                for entry_id, fields in entries: await self.process_entry(entry_id, fields)

    async def stop(self) -> None:
        self.running = False; await self.redis.aclose()
