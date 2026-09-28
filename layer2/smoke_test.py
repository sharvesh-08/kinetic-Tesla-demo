"""Local Redis smoke test for the Layer 2 service."""

import asyncio
import json
import time
from dataclasses import replace

import redis.asyncio as redis

from layer2.config import CONFIG
from layer2.service import Layer2Service


async def main() -> None:
    config = replace(CONFIG, ecu_stream="smoke:ecu", vibration_stream="smoke:vib", synced_stream="smoke:synced")
    service = Layer2Service(config)
    producer = redis.from_url(config.redis_url, decode_responses=True)
    try:
        await service.start()
        await producer.xadd(config.ecu_stream, {"payload": json.dumps({"sensor": "rpm", "value": 2400.0, "valid": True, "sensor_timestamp_ns": str(time.time_ns())})})
        consumer = asyncio.create_task(service._consume(config.ecu_stream, "ecu"))
        tick = asyncio.create_task(service._sync_loop())
        await asyncio.sleep(0.4)
        frames = await producer.xrange(config.synced_stream)
        decoded = [json.loads(fields["payload"]) for _, fields in frames]
        assert any(frame["values"].get("rpm") == 2400.0 and frame["channel_valid"]["rpm"] for frame in decoded)
        print("Layer 2 smoke test passed")
        consumer.cancel(); tick.cancel()
    finally:
        await producer.aclose()
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
