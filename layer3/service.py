"""Reliable Redis Streams boundary for Layer 3."""

from __future__ import annotations

import asyncio
import json
import logging
import time

import redis.asyncio as redis
from redis.exceptions import ResponseError
from redis_limits import xadd_bounded
from redis_streams import claim_pending

from .config import Layer3Config
from .inference import PhysicsInference

LOGGER = logging.getLogger("layer3.service")


class Layer3Service:
    def __init__(self, config: Layer3Config | None = None, processor: PhysicsInference | None = None) -> None:
        self.config = config or Layer3Config()
        self.processor = processor or PhysicsInference(self.config)
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
                self.config.claim_idle_ms, cursor, count=self.config.read_count,
            )
            for entry_id, fields in entries:
                await self.process_entry(entry_id, fields)
            if cursor == "0-0":
                return

    async def process_entry(self, entry_id: str, fields: dict[str, str]) -> None:
        try:
            result = self.processor.process(json.loads(fields["payload"]))
            output_id = f"{int(result['master_timestamp_ns']) // 1_000_000}-0"
            payload = json.dumps(result, separators=(",", ":"), allow_nan=False)
            try:
                await xadd_bounded(self.redis, self.config.output_stream, {"payload": payload}, id=output_id)
            except Exception:
                pass
            try:
                await self.redis.xack(self.config.input_stream, self.config.consumer_group, entry_id)
            except Exception:
                pass
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            LOGGER.error(json.dumps({"timestamp_ns": time.time_ns(), "module": "layer3", "event": "invalid_input", "error": str(error)}))

    async def run(self) -> None:
        await self.start()
        while self.running:
            try:
                records = await self.redis.xreadgroup(
                    self.config.consumer_group, self.config.consumer_name, {self.config.input_stream: ">"},
                    count=self.config.read_count, block=self.config.block_ms,
                )
            except Exception:
                await asyncio.sleep(0.05)
                continue
            for _, entries in records:
                for entry_id, fields in entries:
                    await self.process_entry(entry_id, fields)

    async def stop(self) -> None:
        self.running = False
        await self.redis.aclose()
