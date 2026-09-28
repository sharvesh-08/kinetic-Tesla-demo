"""Configuration defaults for the edge-only Layer 2 service."""

import os
from dataclasses import dataclass, field
from layer1.schema import TELEMETRY_CHANNELS


@dataclass(frozen=True)
class Layer2Config:
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"))
    consumer_group: str = "layer2-sync"
    consumer_name: str = "edge-layer2"
    ecu_stream: str = "engine:telemetry:ecu"
    vibration_stream: str = "engine:telemetry:vib"
    synced_stream: str = "engine:synced:frames"
    ekf_stream: str = "engine:synced:ekf:frames"
    master_rate_hz: int = 1
    ekf_rate_hz: int = 100
    reorder_window_ms: int = 25
    stale_periods: int = 3
    claim_idle_ms: int = 5000
    processing_complete_stream: str = "engine:processing:complete"
    ground_persisted_stream: str = "engine:ground:persisted"
    sync_weights: dict[str, int] = field(default_factory=lambda: {
        "rpm": 4, "oil_press_kPa": 4, "cht_degC": 4, "egt_degC": 3,
        "map_kPa": 2, "Tm_K_k": 2, "mixture_afr": 2, "cowl_flap_pct": 2,
        "oil_temp_degC": 2, "fuel_flow_Lph": 2,
        "ambient_temp_degC": 1, "ambient_press_kPa": 1, "throttle_pct": 2,
        "injection_timing_degBTDC": 2, "batt_volts": 1, "alt_amps": 1,
        "vibration_rms_g": 2,
        "electrical_health_pct": 1,
    })


CONFIG = Layer2Config()
assert CONFIG.master_rate_hz == 1
assert CONFIG.ekf_rate_hz == 100
assert set(CONFIG.sync_weights).issubset(TELEMETRY_CHANNELS)
