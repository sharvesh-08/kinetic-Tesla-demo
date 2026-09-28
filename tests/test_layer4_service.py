"""Redis integration tests for the Layer 4 transport boundary."""

from __future__ import annotations

import json
import unittest
import uuid

import redis.asyncio as redis

from layer4.service import Layer4Service, ServiceConfig
from tests.test_layer4 import envelope


class Layer4ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        token = uuid.uuid4().hex
        self.config = ServiceConfig(
            input_stream=f"layer4-test:{token}:input",
            output_stream=f"layer4-test:{token}:output",
            consumer_group=f"layer4-test:{token}:group",
            consumer_name=f"layer4-test:{token}:consumer",
            claim_idle_ms=0,
        )
        self.service = Layer4Service(self.config)
        self.client = redis.from_url(self.config.redis_url, decode_responses=True)
        await self.service.start()

    async def asyncTearDown(self) -> None:
        await self.client.delete(self.config.input_stream, self.config.output_stream)
        await self.client.aclose()
        await self.service.stop()

    async def test_publish_precedes_ack_and_output_is_idempotent(self) -> None:
        entry_id = await self.client.xadd(self.config.input_stream, {"payload": json.dumps(envelope(0))})
        records = await self.client.xreadgroup(
            self.config.consumer_group, self.config.consumer_name,
            {self.config.input_stream: ">"}, count=1,
        )
        self.assertTrue(records)
        pending = await self.client.xpending(self.config.input_stream, self.config.consumer_group)
        self.assertEqual(pending["pending"], 1)

        await self.service.process_entry(entry_id, records[0][1][0][1])
        await self.service.process_entry(entry_id, records[0][1][0][1])

        outputs = await self.client.xrange(self.config.output_stream)
        self.assertEqual(len(outputs), 1)
        result = json.loads(outputs[0][1]["payload"])
        self.assertEqual(result["master_timestamp_ns"], envelope(0)["master_timestamp_ns"])
        self.assertIn("residual_z", result)
        pending = await self.client.xpending(self.config.input_stream, self.config.consumer_group)
        self.assertEqual(pending["pending"], 0)


if __name__ == "__main__":
    unittest.main()
