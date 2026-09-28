"""JSON-lines adapter for the canonical Layer 1 publisher.

By default, each stdin line is an object with ``kind`` equal to ``ecu`` or ``vibration``.
Set ``LAYER1_SOURCE_STREAM`` to bridge canonical ECU events from another Redis stream.
Real hardware adapters can import Layer1Service directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import redis.asyncio as redis

from .service import Layer1Service


async def main() -> None:
    service = Layer1Service()
    source_stream = os.environ.get("LAYER1_SOURCE_STREAM")
    if source_stream:
        source = redis.from_url(service.config.redis_url, decode_responses=True)
        cursor = os.environ.get("LAYER1_SOURCE_ID", "0-0")
        try:
            while True:
                records = await source.xread({source_stream: cursor}, count=500, block=1000)
                for _, entries in records:
                    for entry_id, fields in entries:
                        event = json.loads(fields["payload"])
                        await service.publish_ecu(event)
                        cursor = entry_id
        finally:
            await source.aclose()
            await service.close()
        return
    try:
        while line := await asyncio.to_thread(sys.stdin.readline):
            event = json.loads(line)
            kind = event.pop("kind", "ecu")
            if kind == "ecu":
                await service.publish_ecu(event)
            elif kind == "vibration":
                await service.publish_vibration(event)
            else:
                raise ValueError(f"unsupported Layer 1 event kind: {kind}")
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
