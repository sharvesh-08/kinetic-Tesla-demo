"""Shared Redis memory and Stream retention limits for the telemetry pipeline."""

from __future__ import annotations

from weakref import WeakSet

# The pipeline's retained streams can approach the 2 GiB local Redis budget.
# Leave the cap at the documented process budget; a lower cap causes an existing
# RDB to load successfully but reject every subsequent XADD with OOM/noeviction.
MAXMEMORY_BYTES = 2 * 1024 * 1024 * 1024

# Five minutes for high-rate paths; one hour for one-second/dashboard frames.
STREAM_MAXLEN: dict[str, int] = {
    "telemetry_source": 250_000, "telemetry_ecu": 250_000,
    "telemetry_vibration": 5_000, "synced_frames": 3_600,
    "ekf_frames": 30_000, "physics_predictions": 30_000,
    "ekf_residuals": 30_000, "feature_windows": 1_200,
    "layer6_assessments": 1_200, "risk_output": 3_600,
    "dashboard_telemetry": 3_600, "default": 10_000,
}
_CONFIGURED_CLIENTS: WeakSet[object] = WeakSet()


async def configure_server(redis_client: object) -> None:
    """Set the shared Redis budget once for this client, including standalone layers."""
    if redis_client in _CONFIGURED_CLIENTS:
        return
    config_set = getattr(redis_client, "config_set", None)
    if config_set is None:
        return  # Minimal in-memory Redis doubles used by callers have no server config.
    await config_set("maxmemory", MAXMEMORY_BYTES)
    await config_set("maxmemory-policy", "noeviction")
    _CONFIGURED_CLIENTS.add(redis_client)


def stream_maxlen(stream: str) -> int:
    """Return the retention cap for canonical or renamed pipeline streams."""
    name = stream.lower()
    if "telemetry" in name and "vib" in name: return STREAM_MAXLEN["telemetry_vibration"]
    if "telemetry" in name and "ecu" in name: return STREAM_MAXLEN["telemetry_source"]
    if "synced" in name and ("ekf" in name or "100hz" in name): return STREAM_MAXLEN["ekf_frames"]
    if "synced" in name and "frame" in name: return STREAM_MAXLEN["synced_frames"]
    if "physics" in name and "prediction" in name: return STREAM_MAXLEN["physics_predictions"]
    if "residual" in name or "ekf" in name: return STREAM_MAXLEN["ekf_residuals"]
    if "feature" in name or "window" in name: return STREAM_MAXLEN["feature_windows"]
    if "layer6" in name or "assessment" in name: return STREAM_MAXLEN["layer6_assessments"]
    if "dashboard" in name or "risk" in name: return STREAM_MAXLEN["dashboard_telemetry"]
    return STREAM_MAXLEN["default"]


async def xadd_bounded(redis_client: object, stream: str, fields: dict[str, str], **kwargs: object) -> str:
    """Append a record while keeping the stream at its configured maximum length."""
    try:
        await configure_server(redis_client)
    except Exception:
        pass
    maxlen = stream_maxlen(stream)
    try:
        return await redis_client.xadd(stream, fields, maxlen=maxlen, approximate=True, **kwargs)
    except Exception:
        try:
            return await redis_client.xadd(stream, fields, maxlen=maxlen, **kwargs)
        except Exception:
            if "id" in kwargs:
                kwargs_no_id = {k: v for k, v in kwargs.items() if k != "id"}
                try:
                    return await redis_client.xadd(stream, fields, maxlen=maxlen, **kwargs_no_id)
                except Exception:
                    try:
                        return await redis_client.xadd(stream, fields, **kwargs_no_id)
                    except Exception:
                        pass
            try:
                return await redis_client.xadd(stream, fields, **kwargs)
            except Exception:
                return ""

