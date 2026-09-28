"""Adversarial tests for Layer 2's edge synchronization contract."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
import uuid
from dataclasses import replace

import redis.asyncio as redis

from layer2.config import CONFIG
from layer2.service import Layer2Service


class Layer2RedTeamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        token = uuid.uuid4().hex
        self.config = replace(
            CONFIG, consumer_name=f"test-{token}",
            ecu_stream=f"redteam:{token}:ecu",
            vibration_stream=f"redteam:{token}:vib", synced_stream=f"redteam:{token}:synced",
            claim_idle_ms=0,
        )
        self.service = Layer2Service(self.config)
        self.producer = redis.from_url(self.config.redis_url, decode_responses=True)
        await self.service.start()

    async def asyncTearDown(self) -> None:
        await self.producer.aclose()
        await self.service.stop()

    async def frame(self) -> dict:
        entries = await self.producer.xrange(self.config.synced_stream)
        return json.loads(entries[-1][1]["payload"])

    async def test_clock_skew_is_estimated_and_aligned(self) -> None:
        ingest_ms = time.time_ns() // 1_000_000
        self.service._accept_sensor({"sensor": "rpm", "value": 2400.0, "sensor_timestamp_ns": str((ingest_ms - 50) * 1_000_000)}, ingest_ms)
        await self.service._sync_tick(ingest_ms - self.config.reorder_window_ms)
        frame = await self.frame()
        self.assertTrue(frame["channel_valid"]["rpm"])
        self.assertEqual(frame["channel_source"]["rpm"], "direct")
        self.assertAlmostEqual(int(frame["clock_offset_est_ns"]["ecu"]), 50_000_000, delta=2_000_000)

    async def test_zoh_then_stale_prevents_false_measurements(self) -> None:
        base_ms = time.time_ns() // 1_000_000
        self.service._accept_sensor({"sensor": "rpm", "value": 2300.0, "sensor_timestamp_ns": str(base_ms * 1_000_000)}, base_ms)
        await self.service._sync_tick(base_ms)
        await self.service._sync_tick(base_ms + 20)
        zoh = await self.frame()
        self.assertEqual(zoh["channel_source"]["rpm"], "zoh")
        self.assertFalse(zoh["channel_new_sample"]["rpm"])
        await self.service._sync_tick(base_ms + 200)
        stale = await self.frame()
        self.assertFalse(stale["channel_valid"]["rpm"])
        self.assertEqual(stale["channel_source"]["rpm"], "invalid")

    async def test_rejects_out_of_range_and_nan_values(self) -> None:
        now_ms = time.time_ns() // 1_000_000
        self.service._accept_sensor({"sensor": "rpm", "value": 9999.0, "sensor_timestamp_ns": str(now_ms * 1_000_000)}, now_ms)
        self.service._accept_sensor({"sensor": "oil_press_kPa", "value": float("nan"), "sensor_timestamp_ns": str(now_ms * 1_000_000)}, now_ms)
        await self.service._sync_tick(now_ms)
        frame = await self.frame()
        self.assertFalse(frame["channel_valid"]["rpm"])
        self.assertFalse(frame["channel_valid"]["oil_press_kPa"])

    async def test_mvem_inputs_are_synchronized_as_async_channels(self) -> None:
        now_ms = time.time_ns() // 1_000_000
        for sensor, value in (("Tm_K_k", 315.0), ("mixture_afr", 14.2), ("cowl_flap_pct", 35.0)):
            self.service._accept_sensor({"sensor": sensor, "value": value, "sensor_timestamp_ns": str(now_ms * 1_000_000)}, now_ms)
        await self.service._sync_tick(now_ms)
        frame = await self.frame()
        self.assertEqual(frame["values"]["Tm_K_k"], 315.0)
        self.assertEqual(frame["values"]["mixture_afr"], 14.2)
        self.assertEqual(frame["values"]["cowl_flap_pct"], 35.0)

    async def test_duplicate_master_tick_is_idempotent(self) -> None:
        now_ms = time.time_ns() // 1_000_000
        await self.service._sync_tick(now_ms)
        await self.service._sync_tick(now_ms)
        self.assertEqual(len(await self.producer.xrange(self.config.synced_stream)), 1)

    async def test_layer2_does_not_trim_synced_frames(self) -> None:
        now_ms = time.time_ns() // 1_000_000
        await self.service._sync_tick(now_ms)
        await self.service._sync_tick(now_ms + 700_000)
        self.assertEqual(len(await self.producer.xrange(self.config.synced_stream)), 2)

    async def test_redis_consumer_group_ingests_a_real_event(self) -> None:
        consumer = asyncio.create_task(self.service._consume(self.config.ecu_stream, "ecu"))
        now_ns = time.time_ns()
        await self.producer.xadd(self.config.ecu_stream, {"payload": json.dumps({"sensor": "rpm", "value": 2500.0, "valid": True, "sensor_timestamp_ns": str(now_ns)})})
        await asyncio.sleep(0.35)
        await self.service._sync_tick(time.time_ns() // 1_000_000)
        frame = await self.frame()
        self.assertEqual(frame["values"]["rpm"], 2500.0)
        consumer.cancel()

    async def test_pending_entry_is_recovered_after_consumer_crash(self) -> None:
        now_ns = time.time_ns()
        await self.producer.xadd(self.config.ecu_stream, {"payload": json.dumps({"sensor": "rpm", "value": 2600.0, "valid": True, "sensor_timestamp_ns": str(now_ns)})})
        # Simulate a consumer that receives the entry and then dies before XACK.
        claimed = await self.producer.xreadgroup(self.config.consumer_group, "crashed-consumer", {self.config.ecu_stream: ">"}, count=1)
        self.assertTrue(claimed)
        await self.service._recover_pending()
        await self.service._sync_tick(time.time_ns() // 1_000_000)
        frame = await self.frame()
        self.assertEqual(frame["values"]["rpm"], 2600.0)

    async def test_raw_entry_is_acked_only_after_synced_frame_exists(self) -> None:
        consumer = asyncio.create_task(self.service._consume(self.config.ecu_stream, "ecu"))
        await self.producer.xadd(self.config.ecu_stream, {"payload": json.dumps({"sensor": "rpm", "value": 2200.0, "valid": True, "sensor_timestamp_ns": str(time.time_ns())})})
        await asyncio.sleep(0.35)
        pending_before = await self.producer.xpending(self.config.ecu_stream, self.config.consumer_group)
        self.assertEqual(pending_before["pending"], 1)
        await self.service._sync_tick(time.time_ns() // 1_000_000)
        pending_after = await self.producer.xpending(self.config.ecu_stream, self.config.consumer_group)
        self.assertEqual(pending_after["pending"], 0)
        self.assertEqual(len(await self.producer.xrange(self.config.ecu_stream)), 1)
        consumer.cancel()

if __name__ == "__main__":
    unittest.main()
