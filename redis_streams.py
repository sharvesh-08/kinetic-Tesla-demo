"""Redis Streams compatibility helpers for Redis 5+ deployments."""

from __future__ import annotations

from typing import Any

from redis.exceptions import ResponseError


async def claim_pending(
    client: Any,
    stream: str,
    group: str,
    consumer: str,
    min_idle_ms: int,
    start_id: str = "0-0",
    count: int = 100,
) -> tuple[str, list[Any], list[Any]]:
    """Claim abandoned group entries, using XCLAIM when XAUTOCLAIM is absent.

    Redis added XAUTOCLAIM in 6.2. Redis 5 supports XPENDING and XCLAIM, so
    page through the pending list and preserve the same recovery behavior.
    """
    try:
        return await client.xautoclaim(
            stream, group, consumer, min_idle_ms, start_id, count=count
        )
    except Exception:
        pass

    pending = await client.xpending_range(stream, group, min=start_id, max="+", count=count)
    rows = []
    for item in pending:
        if isinstance(item, dict):
            message_id = item.get("message_id", item.get(b"message_id"))
            idle_ms = item.get("time_since_delivered", item.get(b"time_since_delivered", 0))
        else:
            message_id, _owner, idle_ms, _deliveries = item
        if message_id is None:
            continue
        if isinstance(message_id, bytes):
            message_id = message_id.decode()
        if str(message_id) <= start_id:
            continue
        if int(idle_ms) >= min_idle_ms:
            rows.append(str(message_id))

    entries = await client.xclaim(stream, group, consumer, min_idle_ms, rows) if rows else []
    if len(pending) < count:
        next_id = "0-0"
    else:
        last = pending[-1]
        last_id = last.get("message_id", last.get(b"message_id")) if isinstance(last, dict) else last[0]
        next_id = last_id.decode() if isinstance(last_id, bytes) else str(last_id)
    return next_id, entries, []
