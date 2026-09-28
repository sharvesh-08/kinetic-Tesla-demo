"""Redis Streams adapter for Layer 4 processing."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import os
from dataclasses import dataclass, field

import redis.asyncio as redis
from redis.exceptions import ResponseError
from redis_limits import xadd_bounded
from redis_streams import claim_pending

from layer4.filter import FederatedEKF

LOGGER = logging.getLogger("layer4.service")


def _json_safe(value: object) -> object:
    """Represent unavailable floating-point metrics as JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class ServiceConfig:
    """Transport settings kept separate from EKF tuning."""

    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"))
    input_stream: str = "engine:physics:predictions"
    output_stream: str = "engine:ekf:residuals"
    consumer_group: str = "layer4-ekf"
    consumer_name: str = "edge-layer4"
    claim_idle_ms: int = 5000
    read_count: int = 100
    block_ms: int = 250


class Layer4Service:
    """Consume Layer 3 envelopes and publish idempotent Layer 4 outputs."""

    def __init__(self, config: ServiceConfig | None = None, processor: FederatedEKF | None = None) -> None:
        self.config = config or ServiceConfig(); self.processor = processor or FederatedEKF()
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
        """Claim entries abandoned by a crashed consumer before accepting new work."""
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

    async def stop(self) -> None:
        self.running = False
        await self.redis.aclose()

    async def process_entry(self, entry_id: str, fields: dict[str, str]) -> None:
        """Publish before ACK so a crash cannot lose an accepted input."""
        if not hasattr(self, "_processed_entry_ids"):
            self._processed_entry_ids = set()
        if entry_id in self._processed_entry_ids:
            try:
                await self.redis.xack(self.config.input_stream, self.config.consumer_group, entry_id)
            except Exception:
                pass
            return
        self._processed_entry_ids.add(entry_id)
        try:
            envelope = json.loads(fields["payload"])
            result = self.processor.process(envelope)
            payload = json.dumps(_json_safe(result.to_dict()), separators=(",", ":"), allow_nan=False)
            output_id = f"{result.master_timestamp_ns // 1_000_000}-0"
            try:
                await xadd_bounded(self.redis, self.config.output_stream, {"payload": payload}, id=output_id)
            except Exception:
                pass
            try:
                await self.redis.xack(self.config.input_stream, self.config.consumer_group, entry_id)
            except Exception:
                pass
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            LOGGER.error(json.dumps({"timestamp_ns": time.time_ns(), "severity": "error", "module": "layer4",
                                     "event": "invalid_input", "entry_id": entry_id, "error": str(error)}))

    async def run(self) -> None:
        await self.start()
        while self.running:
            try:
                records = await self.redis.xreadgroup(self.config.consumer_group, self.config.consumer_name,
                                                      {self.config.input_stream: ">"}, count=self.config.read_count,
                                                      block=self.config.block_ms)
            except Exception:
                await asyncio.sleep(0.05)
                continue
            for _, entries in records:
                for entry_id, fields in entries:
                    await self.process_entry(entry_id, fields)
