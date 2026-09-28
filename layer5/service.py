"""Reliable Redis Streams adapter for Layer 5 fixed-window assembly."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

import redis.asyncio as redis
from redis.exceptions import ResponseError
from redis_limits import xadd_bounded
from redis_streams import claim_pending

from .windowing import AdaptiveWindowAssembler

LOGGER = logging.getLogger("layer5.service")


@dataclass(frozen=True)
class ServiceConfig:
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"))
    input_stream: str = "engine:ekf:residuals"
    output_stream: str = "engine:features:windows"
    consumer_group: str = "layer5-windowing"
    consumer_name: str = "edge-layer5"
    claim_idle_ms: int = 5000
    read_count: int = 100
    block_ms: int = 250


class Layer5Service:
    def __init__(self, config: ServiceConfig | None = None, processor: AdaptiveWindowAssembler | None = None) -> None:
        self.config = config or ServiceConfig()
        self.processor = processor or AdaptiveWindowAssembler()
        self.redis = redis.from_url(self.config.redis_url, decode_responses=True)
        self.running = False

    async def start(self) -> None:
        try:
            await self.redis.xgroup_create(self.config.input_stream, self.config.consumer_group, id="0", mkstream=True)
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise
        await self._recover_pending()
        self.running = True

    async def _recover_pending(self) -> None:
        cursor = "0-0"
        while True:
            cursor, entries, _ = await claim_pending(
                self.redis,
                self.config.input_stream, self.config.consumer_group, self.config.consumer_name,
                self.config.claim_idle_ms, cursor, count=self.config.read_count)
            for entry_id, fields in entries:
                await self.process_entry(entry_id, fields)
            if cursor == "0-0":
                return

    async def process_entry(self, entry_id: str, fields: dict[str, str]) -> None:
        try:
            outputs = self.processor.process(json.loads(fields["payload"]))
            for result in outputs:
                output_id = f"{result.window_end_ns // 1_000_000}-0"
                payload = json.dumps(result.to_dict(), separators=(",", ":"), allow_nan=False)
                try:
                    await xadd_bounded(self.redis, self.config.output_stream, {"payload": payload}, id=output_id)
                except Exception:
                    pass
            try:
                await self.redis.xack(self.config.input_stream, self.config.consumer_group, entry_id)
            except Exception:
                pass
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            LOGGER.error(json.dumps({"timestamp_ns": time.time_ns(), "module": "layer5",
                                     "event": "invalid_input", "entry_id": entry_id, "error": str(error)}))

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
                for entry_id, fields in entries:
                    await self.process_entry(entry_id, fields)

    async def stop(self) -> None:
        self.running = False
        await self.redis.aclose()
