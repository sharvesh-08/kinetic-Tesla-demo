"""Run the Redis consumers that receive virtual-engine telemetry.

Layer 2 consumes canonical per-sensor events from ``engine:telemetry:ecu``,
synchronizes them, and writes frames to ``engine:synced:frames``. Layer 3
consumes those frames and writes model results to ``engine:physics:predictions``.

Run from the architecture project root with ``python3 redis_subscriber.py``.
Redis must be running and the Layer 3 ONNX model artifacts must be installed.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace

from layer2.config import CONFIG as LAYER2_CONFIG
from layer2.service import Layer2Service
from layer3.config import Layer3Config
from layer3.service import Layer3Service


async def main() -> None:
    redis_url = os.environ.get("REDIS_URL", LAYER2_CONFIG.redis_url)
    layer2_config = replace(
        LAYER2_CONFIG,
        redis_url=redis_url,
        ecu_stream=os.environ.get("TELEMETRY_STREAM", LAYER2_CONFIG.ecu_stream),
        consumer_group=os.environ.get("LAYER2_CONSUMER_GROUP", LAYER2_CONFIG.consumer_group),
    )
    layer2 = Layer2Service(layer2_config)
    layer3 = Layer3Service(Layer3Config(redis_url=redis_url))
    try:
        async with asyncio.TaskGroup() as consumers:
            consumers.create_task(layer2.run(), name="layer2-redis-subscriber")
            consumers.create_task(layer3.run(), name="layer3-model-consumer")
    finally:
        await asyncio.gather(layer2.stop(), layer3.stop(), return_exceptions=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
