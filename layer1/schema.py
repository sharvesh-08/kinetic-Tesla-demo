"""Canonical, unit-explicit Layer 1 telemetry contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelSpec:
    unit: str
    native_rate_hz: float
    minimum: float
    maximum: float


# Native rates follow the supplied Layer 1 contract; channels absent from that
# reference use 20 Hz. Actual source rates can be supplied with telemetry events.
TELEMETRY_CHANNELS: dict[str, ChannelSpec] = {
    "rpm": ChannelSpec("RPM", 20, 0, 5800),
    "map_kPa": ChannelSpec("kPa", 10, 15, 160),
    # MVEM runtime inputs. Ranges/rates are representative synthetic defaults
    # until the selected engine's IAT, ECU AFR, and cowl-actuator sensors are known.
    "Tm_K_k": ChannelSpec("K", 10, 200, 450),
    "mixture_afr": ChannelSpec("air/fuel mass ratio", 10, 8, 22),
    "cowl_flap_pct": ChannelSpec("percent", 5, 0, 100),
    # Limits extended to accept the connected virtual-engine source. Model
    # training ranges remain separate; an accepted sensor value can still be
    # outside a model's validated training domain.
    "cht_degC": ChannelSpec("degC", 5, -40, 600),
    "egt_degC": ChannelSpec("degC", 10, -40, 1200),
    "oil_press_kPa": ChannelSpec("kPa", 10, 0, 830),
    "oil_temp_degC": ChannelSpec("degC", 2, -40, 280),
    "fuel_flow_Lph": ChannelSpec("L/h", 5, 0, 80),
    "ambient_temp_degC": ChannelSpec("degC", 2, -60, 70),
    "ambient_press_kPa": ChannelSpec("kPa", 2, 15, 110),
    "throttle_pct": ChannelSpec("percent", 20, 0, 100),
    "injection_timing_degBTDC": ChannelSpec("degBTDC", 10, 0, 50),
    "batt_volts": ChannelSpec("V", 5, 0, 32),
    "alt_amps": ChannelSpec("A (positive output)", 5, 0, 80),
    "electrical_health_pct": ChannelSpec("percent", 20, 0, 100),
    "vibration_rms_g": ChannelSpec("g RMS", 50, 0, 30),
    # Updated engine: measured/context channels, separate from trained ONNX inputs.
    "injection_duration_ms": ChannelSpec("ms", 20, 0, 20),
    "fuel_pressure_kPa": ChannelSpec("kPa", 20, 0, 830),
    "altitude_m": ChannelSpec("m", 20, 0, 16000),
    "air_density_ratio": ChannelSpec("ratio", 20, 0, 2),
}

VIBRATION_SAMPLE_RATE_HZ = 2000
VIBRATION_CHUNK_DURATION_MS = 100
