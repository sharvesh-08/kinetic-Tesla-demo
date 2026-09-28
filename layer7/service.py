"""Reliable Redis Streams adapter for Layer 7 advisory outputs."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import os
from dataclasses import dataclass, field

import redis.asyncio as redis
from redis.exceptions import ResponseError
from redis_limits import xadd_bounded
from redis_streams import claim_pending

from .dashboard_publisher import build_dashboard_payload, publish_dashboard_payload
from .engine import Layer7Engine

LOGGER = logging.getLogger("layer7.service")


@dataclass(frozen=True)
class ServiceConfig:
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"))
    input_stream: str = "engine:layer6:assessments"
    output_stream: str = "engine:risk:rul"
    dashboard_output_stream: str = "engine:dashboard:telemetry"
    consumer_group: str = "layer7-risk-rul"
    consumer_name: str = "edge-layer7"
    claim_idle_ms: int = 5000
    read_count: int = 100
    block_ms: int = 250


class Layer7Service:
    def __init__(self, config: ServiceConfig | None = None, processor: Layer7Engine | None = None) -> None:
        self.config = config or ServiceConfig(); self.processor = processor or Layer7Engine()
        self.input_stream, self.output_stream = self.config.input_stream, self.config.output_stream
        self.redis = redis.from_url(self.config.redis_url, decode_responses=True); self.running = False
        self.session_id = None

    async def start(self) -> None:
        try:
            await self.redis.xgroup_create(self.input_stream, self.config.consumer_group, id="0", mkstream=True)
        except ResponseError as error:
            if "BUSYGROUP" not in str(error): raise
        await self._recover_pending(); self.running = True

    async def _recover_pending(self) -> None:
        cursor = "0-0"
        while True:
            cursor, entries, _ = await claim_pending(
                self.redis,
                self.input_stream, self.config.consumer_group, self.config.consumer_name,
                self.config.claim_idle_ms, cursor, count=self.config.read_count)
            for entry_id, fields in entries: await self.process_entry(entry_id, fields)
            if cursor == "0-0": return

    async def process_entry(self, entry_id: str, fields: dict[str, str]) -> None:
        try:
            event = json.loads(fields["payload"])
            session = event.get("session_id")
            if session and session != self.session_id:
                self.processor = Layer7Engine()
                self.session_id = session
            output_id = f"{int(event['timestamp_ns']) // 1_000_000}-0"
            existing = await self.redis.xrange(self.output_stream, min=output_id, max=output_id, count=1)
            if existing:
                risk_output = json.loads(existing[0][1]["payload"])
            else:
                risk_output = self.processor.process(event).to_dict()
                published_id = await xadd_bounded(
                    self.redis,
                    self.output_stream,
                    {"payload": json.dumps(risk_output, separators=(",", ":"), allow_nan=False)},
                    id=output_id,
                )
                if not published_id:
                    raise ResponseError("Risk assessment write failed")
            dashboard_payload = build_dashboard_payload(event, risk_output, self.processor.config)
            await publish_dashboard_payload(self.redis, self.config.dashboard_output_stream, dashboard_payload)
            try:
                await self.redis.xack(self.input_stream, self.config.consumer_group, entry_id)
            except Exception:
                pass
        except ResponseError as error:
            LOGGER.error(json.dumps({"timestamp_ns": time.time_ns(), "module": "layer7", "event": "publish_failed", "error": str(error)}))
            # Leave the source pending so recovery can retry the dashboard write.
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            LOGGER.error(json.dumps({"timestamp_ns": time.time_ns(), "module": "layer7", "event": "invalid_input", "error": str(error)}))
            try:
                await self.redis.xack(self.input_stream, self.config.consumer_group, entry_id)
            except Exception:
                pass

    async def run(self) -> None:
        await self.start()
        while self.running:
            try:
                records = await self.redis.xreadgroup(self.config.consumer_group, self.config.consumer_name,
                    {self.input_stream: ">"}, count=self.config.read_count, block=self.config.block_ms)
            except Exception:
                await asyncio.sleep(0.05)
                continue
            for _, entries in records:
                for entry_id, fields in entries: await self.process_entry(entry_id, fields)

    async def stop(self) -> None:
        self.running = False; await self.redis.aclose()
