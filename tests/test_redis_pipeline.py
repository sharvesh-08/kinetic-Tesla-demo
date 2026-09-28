"""Redis-boundary communication tests for Layer 1 -> Layer 2 -> Layer 3."""

from __future__ import annotations

import json
import time
import unittest

from redis.exceptions import ResponseError

from layer1.service import validate_ecu_event
from layer2.config import Layer2Config
from layer2.service import Layer2Service
from layer3.config import Layer3Config
from layer3.service import Layer3Service


class MemoryRedis:
    """Small async Redis Streams double; records write/ack order."""

    def __init__(self) -> None:
        self.entries: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.acks: list[tuple[str, str, tuple[str, ...]]] = []
        self.operations: list[tuple[str, str]] = []
        self.fail_stream: str | None = None

    async def xadd(self, stream: str, fields: dict[str, str], id: str = "*", **_options: object) -> str:
        if stream == self.fail_stream:
            raise ResponseError("simulated Redis write failure")
        self.entries.setdefault(stream, []).append((id, dict(fields)))
        self.operations.append(("xadd", stream))
        return id

    async def xack(self, stream: str, group: str, *entry_ids: str) -> int:
        self.acks.append((stream, group, tuple(entry_ids)))
        self.operations.append(("xack", stream))
        return len(entry_ids)


class PassThroughProcessor:
    def __init__(self) -> None:
        self.received: list[dict] = []

    def process(self, frame: dict) -> dict:
        self.received.append(frame)
        return {**frame, "layer3_test_processed": True}


class RedisPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_rate_path_is_layer3_default_and_dashboard_is_independent(self) -> None:
        self.assertEqual(Layer3Config().input_stream, self.l2_config.ekf_stream)
        self.assertEqual(self.l2_config.ekf_rate_hz, 100)
        await self.ingest('9000-0', self.publisher_event('rpm', 2400))
        staged = self.layer2.pending_source_samples[self.l2_config.ecu_stream]
        tick = int(staged[0]['aligned_timestamp_ms']) + 1
        await self.layer2._sync_tick(tick, publish_ekf=True, publish_dashboard=False)
        await self.layer2._sync_tick(tick + 10, publish_ekf=True, publish_dashboard=True)
        self.assertEqual(len(self.redis.entries[self.l2_config.ekf_stream]), 2)
        self.assertEqual(len(self.redis.entries[self.l2_config.synced_stream]), 1)
        first = json.loads(self.redis.entries[self.l2_config.ekf_stream][0][1]['payload'])
        second = json.loads(self.redis.entries[self.l2_config.ekf_stream][1][1]['payload'])
        self.assertTrue(first['channel_new_sample']['rpm'])
        self.assertFalse(second['channel_new_sample']['rpm'])
        self.assertEqual(self.redis.operations[:2], [('xadd', self.l2_config.ekf_stream), ('xack', self.l2_config.ecu_stream)])

    def setUp(self) -> None:
        self.l2_config = Layer2Config()
        self.redis = MemoryRedis()
        self.layer2 = Layer2Service(self.l2_config)
        self.layer2.redis = self.redis

    def publisher_event(self, sensor: str, value: float) -> dict:
        # Match the Layer 1 event contract emitted by the telemetry publisher.
        return validate_ecu_event({
            "sensor": sensor,
            "value": value,
            "sensor_timestamp_ns": time.time_ns(),
            "valid": True,
        })

    async def ingest(self, entry_id: str, event: dict) -> None:
        await self.layer2._process_entry(
            self.l2_config.ecu_stream,
            entry_id,
            {"payload": json.dumps(event)},
            "ecu",
        )

    async def test_async_events_are_buffered_then_sent_as_one_common_time_frame(self) -> None:
        self.assertEqual(self.l2_config.master_rate_hz, 1)
        await self.ingest("1000-0", self.publisher_event("rpm", 2400))
        await self.ingest("1001-0", self.publisher_event("map_kPa", 72.5))

        staged = self.layer2.pending_source_samples[self.l2_config.ecu_stream]
        self.assertEqual(len(staged), 2)
        master_ms = int(max(item["aligned_timestamp_ms"] for item in staged)) + 1
        await self.layer2._sync_tick(master_ms)

        frame_id, fields = self.redis.entries[self.l2_config.synced_stream][-1]
        frame = json.loads(fields["payload"])
        self.assertEqual(frame_id, f"{master_ms}-0")
        self.assertEqual(frame["master_timestamp_ns"], str(master_ms * 1_000_000))
        self.assertEqual({sample["stream_id"] for sample in frame["input_samples"]}, {"1000-0", "1001-0"})
        self.assertEqual({sample["data"]["sensor"] for sample in frame["input_samples"]}, {"rpm", "map_kPa"})
        self.assertEqual(frame["values"]["rpm"], 2400)
        self.assertEqual(frame["values"]["map_kPa"], 72.5)
        self.assertEqual(len(self.redis.acks), 1)
        ack_stream, ack_group, ack_ids = self.redis.acks[0]
        self.assertEqual((ack_stream, ack_group), (self.l2_config.ecu_stream, self.l2_config.consumer_group))
        self.assertEqual(set(ack_ids), {"1000-0", "1001-0"})
        self.assertEqual(self.redis.operations, [("xadd", self.l2_config.synced_stream), ("xack", self.l2_config.ecu_stream)])
        self.assertEqual(self.layer2.pending_source_samples[self.l2_config.ecu_stream], [])

    async def test_source_entries_remain_unacked_if_frame_write_fails(self) -> None:
        await self.ingest("2000-0", self.publisher_event("rpm", 2400))
        staged = self.layer2.pending_source_samples[self.l2_config.ecu_stream]
        master_ms = int(staged[0]["aligned_timestamp_ms"]) + 1
        self.redis.fail_stream = self.l2_config.synced_stream

        with self.assertRaises(ResponseError):
            await self.layer2._sync_tick(master_ms)

        self.assertEqual(self.redis.acks, [])
        self.assertEqual(len(self.layer2.pending_source_samples[self.l2_config.ecu_stream]), 1)

    async def test_layer3_receives_the_synced_frame_and_acks_after_output_write(self) -> None:
        await self.ingest("3000-0", self.publisher_event("rpm", 2400))
        staged = self.layer2.pending_source_samples[self.l2_config.ecu_stream]
        master_ms = int(staged[0]["aligned_timestamp_ms"]) + 1
        await self.layer2._sync_tick(master_ms)
        synced_id, synced_fields = self.redis.entries[self.l2_config.synced_stream][-1]

        processor = PassThroughProcessor()
        layer3_config = Layer3Config(
            input_stream=self.l2_config.synced_stream,
            output_stream="engine:test:physics",
        )
        layer3 = Layer3Service(layer3_config, processor=processor)
        layer3.redis = self.redis
        await layer3.process_entry(synced_id, synced_fields)

        self.assertEqual(len(processor.received), 1)
        self.assertEqual(processor.received[0]["input_samples"][0]["stream_id"], "3000-0")
        output_id, output_fields = self.redis.entries[layer3_config.output_stream][-1]
        output = json.loads(output_fields["payload"])
        self.assertEqual(output_id, f"{master_ms}-0")
        self.assertTrue(output["layer3_test_processed"])
        self.assertEqual(self.redis.acks[-1], (layer3_config.input_stream, layer3_config.consumer_group, (synced_id,)))
        self.assertEqual(self.redis.operations[-2:], [("xadd", layer3_config.output_stream), ("xack", layer3_config.input_stream)])


if __name__ == "__main__":
    unittest.main()
