"""Async edge synchronizer: raw Redis Streams -> retained SyncedFrames."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ResponseError
from redis_limits import xadd_bounded
from redis_streams import claim_pending

from layer1.schema import TELEMETRY_CHANNELS
from layer2.config import Layer2Config

LOGGER = logging.getLogger("layer2")


@dataclass
class Sample:
    value: float
    valid: bool
    aligned_timestamp_ms: float
    fresh: bool = True
    rate_hz: float | None = None


class Layer2Service:
    """Runs locally on the edge and does not own a durable database."""

    def __init__(self, config: Layer2Config) -> None:
        self.config = config
        self.redis = redis.from_url(config.redis_url, decode_responses=True)
        self.samples: dict[str, Sample] = {}
        self.clock_offset_ms: dict[str, float] = {}
        self.vibration_chunks: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}
        self.pending_ack_ids: dict[str, set[str]] = {}
        # Cache every asynchronous source entry until a synchronized frame
        # containing it has been durably written. Redis PEL entries are the
        # restart-safe backing store for this in-memory staging cache.
        self.pending_source_samples: dict[str, list[dict[str, Any]]] = {}
        self.running = False

    async def start(self) -> None:
        for stream in (self.config.ecu_stream, self.config.vibration_stream):
            try:
                await self.redis.xgroup_create(stream, self.config.consumer_group, id="0", mkstream=True)
            except Exception as error:
                if "BUSYGROUP" not in str(error) and "already exists" not in str(error).lower():
                    pass
        self.running = True

    async def stop(self) -> None:
        self.running = False
        await self.redis.aclose()

    async def run(self) -> None:
        await self.start()
        await asyncio.gather(
            self._recover_pending(),
            self._consume(self.config.ecu_stream, "ecu"),
            self._consume(self.config.vibration_stream, "vibration"),
            self._sync_loop(),
        )

    async def _recover_pending(self) -> None:
        for stream, kind in ((self.config.ecu_stream, "ecu"), (self.config.vibration_stream, "vibration")):
            cursor = "0-0"
            while True:
                cursor, entries, _ = await claim_pending(self.redis, stream, self.config.consumer_group, self.config.consumer_name, self.config.claim_idle_ms, cursor, count=100)
                for entry_id, fields in entries:
                    await self._process_entry(stream, entry_id, fields, kind)
                if cursor == "0-0":
                    break

    async def _consume(self, stream: str, kind: str) -> None:
        while self.running:
            try:
                records = await self.redis.xreadgroup(self.config.consumer_group, self.config.consumer_name, {stream: ">"}, count=100, block=250)
            except Exception:
                await asyncio.sleep(0.05)
                continue
            for _, entries in records:
                for entry_id, fields in entries:
                    await self._process_entry(stream, entry_id, fields, kind)

    async def _process_entry(self, stream: str, entry_id: str, fields: dict[str, str], kind: str) -> None:
        ingest_ms = time.time_ns() // 1_000_000
        payload_json = fields.get("payload", json.dumps(fields, separators=(",", ":")))
        try:
            payload = json.loads(payload_json)
            if not isinstance(payload, dict):
                raise ValueError("event payload must be a JSON object")
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            payload = {"_invalid_payload": payload_json, "_error": str(error)}
            aligned_ms = float(ingest_ms)
            accepted = False
        else:
            if kind == "ecu":
                aligned_ms = self._sensor_alignment(payload, ingest_ms)
            else:
                aligned_ms = self._vibration_alignment(payload, ingest_ms)
            accepted = aligned_ms is not None
            if aligned_ms is None:
                aligned_ms = float(ingest_ms)

        # Keep the complete source event and timing in the frame cache. Even
        # rejected measurements are forwarded with accepted=False for audit.
        cached = {
            "stream": stream,
            "stream_id": entry_id,
            "kind": kind,
            "sensor_timestamp_ns": payload.get("sensor_timestamp_ns"),
            "aligned_timestamp_ms": aligned_ms,
            "received_timestamp_ns": str(ingest_ms * 1_000_000),
            "accepted": accepted,
            "data": _json_safe(payload),
        }
        self.pending_source_samples.setdefault(stream, []).append(cached)
        # Keep the source entry in the Redis consumer group's PEL until a
        # synchronized frame carrying this cached copy has been written.
        self.pending_ack_ids.setdefault(stream, set()).add(entry_id)

    def _sensor_alignment(self, packet: dict[str, Any], ingest_ms: float) -> float | None:
        channel = packet.get("sensor")
        spec = TELEMETRY_CHANNELS.get(channel)
        sensor_ns = packet.get("sensor_timestamp_ns", packet.get("sensorTimestampNs"))
        value = packet.get("value")
        if (spec is None or sensor_ns is None or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not spec.minimum <= value <= spec.maximum):
            return None
        if isinstance(sensor_ns, bool):
            return None
        try:
            sensor_ms = int(sensor_ns) / 1_000_000
        except (TypeError, ValueError):
            return None
        return sensor_ms + self._estimate_offset("ecu", sensor_ms, ingest_ms)

    def _vibration_alignment(self, chunk: dict[str, Any], ingest_ms: float) -> float | None:
        try:
            sensor_ms = int(chunk["sensor_timestamp_ns"]) / 1_000_000
            sample_rate_hz = float(chunk["sample_rate_hz"])
            samples_per_axis = int(chunk["samples_per_axis"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0 or samples_per_axis <= 0:
            return None
        return sensor_ms + self._estimate_offset("vibration", sensor_ms, ingest_ms)

    def _estimate_offset(self, stream: str, sensor_ms: float, ingest_ms: float) -> float:
        observed = ingest_ms - sensor_ms
        previous = self.clock_offset_ms.get(stream)
        estimate = observed if previous is None else 0.98 * previous + 0.02 * observed
        self.clock_offset_ms[stream] = estimate
        return estimate

    def _accept_sensor(
        self, packet: dict[str, Any], ingest_ms: float, aligned_timestamp_ms: float | None = None,
    ) -> float | None:
        channel = packet.get("sensor")
        spec = TELEMETRY_CHANNELS.get(channel)
        sensor_ns = packet.get("sensor_timestamp_ns", packet.get("sensorTimestampNs"))
        value = packet.get("value")
        if (spec is None or sensor_ns is None or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not spec.minimum <= value <= spec.maximum):
            return None
        try:
            sensor_ms = int(sensor_ns) / 1_000_000
        except (TypeError, ValueError):
            return None
        aligned_ms = aligned_timestamp_ms
        if aligned_ms is None:
            aligned_ms = sensor_ms + self._estimate_offset("ecu", sensor_ms, ingest_ms)
        context = packet.get("context") or {}
        if context.get("session_id") and context.get("session_id") != self.context.get("session_id"):
            self.samples.clear()
            self.context.clear()
        # Every event carries a coherent control snapshot from the simulator.
        # Replace stale revision/provenance fields even when this channel's
        # sensor value is held from an earlier asynchronous event.
        if isinstance(packet.get("context"), dict):
            self.context.update(packet["context"])
        self.samples[channel] = Sample(float(value), packet.get("valid", True) is not False, aligned_ms,
                                       rate_hz=packet.get("sample_rate_hz"))
        if isinstance(packet.get("context"), dict):
            # Context is metadata, not a measurement. Preserve it without using it
            # to refresh or validate any sensor value.
            self.context.update(packet["context"])
        return aligned_ms

    def _accept_vibration(
        self, chunk: dict[str, Any], ingest_ms: float, entry_id: str,
        aligned_start_ms: float | None = None,
    ) -> float | None:
        try:
            sensor_ms = int(chunk["sensor_timestamp_ns"]) / 1_000_000
            sample_rate_hz = float(chunk["sample_rate_hz"])
            samples_per_axis = int(chunk["samples_per_axis"])
        except (KeyError, TypeError, ValueError):
            return None
        if sample_rate_hz <= 0 or samples_per_axis <= 0:
            return None
        start_ms = aligned_start_ms
        if start_ms is None:
            start_ms = sensor_ms + self._estimate_offset("vibration", sensor_ms, ingest_ms)
        self.vibration_chunks.append({"stream_id": entry_id, "sensor_timestamp_ns": str(chunk["sensor_timestamp_ns"]), "sample_rate_hz": sample_rate_hz, "aligned_start_ms": start_ms, "aligned_end_ms": start_ms + samples_per_axis / sample_rate_hz * 1000})
        return start_ms

    async def _sync_loop(self) -> None:
        period_ms = max(1, round(1000 / self.config.ekf_rate_hz))
        while self.running:
            now_ms = time.time_ns() // 1_000_000
            next_boundary_ms = (now_ms // period_ms + 1) * period_ms
            await asyncio.sleep(max(0.0, (next_boundary_ms - now_ms) / 1000.0))
            # Timestamp frames on a shared wall-clock cadence, with the
            # reorder window reserved for late asynchronous sensor events.
            master_ms = next_boundary_ms - self.config.reorder_window_ms
            dashboard_period = max(1, round(1000 / self.config.master_rate_hz))
            await self._sync_tick(master_ms, publish_ekf=True,
                                  publish_dashboard=(master_ms % dashboard_period == (-self.config.reorder_window_ms) % dashboard_period))

    async def _sync_tick(self, master_ms: int, *, publish_ekf: bool = False, publish_dashboard: bool = True) -> None:
        if hasattr(self, "_last_synced_master_ms") and master_ms <= self._last_synced_master_ms:
            return
        self._last_synced_master_ms = master_ms
        cached_inputs = [
            sample
            for samples in self.pending_source_samples.values()
            for sample in samples
            if sample["aligned_timestamp_ms"] <= master_ms + self.config.reorder_window_ms
        ]
        # Apply queued measurements in event-time order, so a late/fast sensor
        # cannot overwrite the value belonging to this master timestamp.
        cached_inputs.sort(key=lambda sample: (sample["aligned_timestamp_ms"], int(sample["received_timestamp_ns"])))
        for cached in cached_inputs:
            if not cached["accepted"]:
                continue
            received_ms = int(cached["received_timestamp_ns"]) / 1_000_000
            if cached["kind"] == "ecu":
                self._accept_sensor(cached["data"], received_ms, cached["aligned_timestamp_ms"])
            else:
                self._accept_vibration(
                    cached["data"], received_ms, cached["stream_id"], cached["aligned_timestamp_ms"]
                )

        values: dict[str, float] = {}; valid: dict[str, bool] = {}; age_ms: dict[str, float | None] = {}
        source: dict[str, str] = {}; new_sample: dict[str, bool] = {}; rates: dict[str, float] = {}
        quality = 0.0; total_weight = 0
        for channel, spec in TELEMETRY_CHANNELS.items():
            sample = self.samples.get(channel); weight = self.config.sync_weights.get(channel, 1)
            rate = sample.rate_hz if sample and sample.rate_hz else spec.native_rate_hz
            total_weight += weight; rates[channel] = rate
            if sample is None or not sample.valid:
                valid[channel] = False; age_ms[channel] = None; source[channel] = "invalid"; new_sample[channel] = False; continue
            age = max(0, master_ms - sample.aligned_timestamp_ms)
            values[channel] = sample.value; age_ms[channel] = age; new_sample[channel] = sample.fresh
            if age > self.config.stale_periods * 1000 / rate:
                valid[channel] = False; source[channel] = "invalid"
            elif sample.fresh:
                valid[channel] = True; source[channel] = "direct"; quality += weight
            else:
                valid[channel] = True; source[channel] = "zoh"; quality += weight * 0.5
            sample.fresh = False
        refs = [chunk for chunk in self.vibration_chunks if chunk["aligned_start_ms"] <= master_ms <= chunk["aligned_end_ms"]]
        self.vibration_chunks = [chunk for chunk in self.vibration_chunks if chunk["aligned_end_ms"] >= master_ms]
        input_ids: dict[str, set[str]] = {}
        if cached_inputs:
            cached_ids = {(sample["stream"], sample["stream_id"]) for sample in cached_inputs}
            for stream, entry_ids in self.pending_ack_ids.items():
                ids = {entry_id for source_stream, entry_id in cached_ids if source_stream == stream}
                if ids:
                    input_ids[stream] = ids
        frame = {"master_timestamp_ns": str(master_ms * 1_000_000), "values": values, "channel_valid": valid, "channel_age_ms": age_ms, "channel_source": source, "channel_new_sample": new_sample, "channel_rate_hz": rates, "vib_chunk_refs": refs, "input_samples": cached_inputs, "clock_offset_est_ns": {name: str(round(value * 1_000_000)) for name, value in self.clock_offset_ms.items()}, "sync_quality": quality / total_weight if total_weight else 0.0, "timing_estimated": True, "context": dict(self.context)}
        if publish_ekf:
            await self._publish_frame(frame, master_ms, input_ids, self.config.ekf_stream, acknowledge=True)
        if publish_dashboard:
            await self._publish_frame(frame, master_ms, input_ids if not publish_ekf else {},
                                      self.config.synced_stream, acknowledge=not publish_ekf)

    async def _publish_frame(self, frame: dict[str, Any], master_ms: int,
                             input_ids: dict[str, set[str]] | None = None,
                             stream: str | None = None, *, acknowledge: bool = True) -> None:
        payload = json.dumps(frame, separators=(",", ":")); entry_id = f"{master_ms}-0"
        target_stream = stream or self.config.synced_stream
        frame_persisted = True
        try:
            await xadd_bounded(self.redis, target_stream, {"payload": payload}, id=entry_id)
        except Exception:
            frame_persisted = False
        if frame_persisted and acknowledge:
            await self._ack_synchronized_inputs(input_ids or {})

    async def _ack_synchronized_inputs(self, input_ids: dict[str, set[str]]) -> None:
        """Release raw entries only after synchronization was durably completed."""
        for stream, entry_ids in input_ids.items():
            if not entry_ids:
                continue
            try:
                await self.redis.xack(stream, self.config.consumer_group, *entry_ids)
            except Exception:
                pass
            self.pending_ack_ids.get(stream, set()).difference_update(entry_ids)
            acknowledged = self.pending_source_samples.get(stream, [])
            self.pending_source_samples[stream] = [sample for sample in acknowledged if sample["stream_id"] not in entry_ids]


def _json_safe(value: Any) -> Any:
    """Keep malformed/non-finite source values serializable in cache frames."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
