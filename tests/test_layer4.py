"""Acceptance and degraded-input tests for Layer 4."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

from layer4.config import load_config
from layer4.filter import FederatedEKF
from layer4.service import _json_safe


BASE_VALUES = {
    "rpm": 2200.0, "throttle_pct": 55.0, "egt_degC": 690.0, "cht_degC": 175.0,
    "map_kPa": 72.0, "fuel_flow_Lph": 31.0, "oil_press_kPa": 350.0,
    "oil_temp_degC": 95.0, "vibration_rms_g": 0.25, "batt_volts": 14.2, "alt_amps": 18.0,
}
FILTER_CHANNELS = tuple(key for key in BASE_VALUES if key not in ("rpm", "throttle_pct"))


def envelope(tick: int, offsets: dict[str, float] | None = None, new: dict[str, bool] | None = None) -> dict:
    offsets = offsets or {}; measured = dict(BASE_VALUES)
    for channel, offset in offsets.items():
        measured[channel] += offset
    return {
        "master_timestamp_ns": 1_700_000_000_000_000_000 + tick * 10_000_000,
        "values": measured, "predicted": {channel: BASE_VALUES[channel] for channel in FILTER_CHANNELS},
        "channel_valid": {channel: True for channel in FILTER_CHANNELS},
        "channel_new_sample": {channel: (new or {}).get(channel, True) for channel in FILTER_CHANNELS},
        "dt_s": 0.01,
    }


class Layer4Tests(unittest.TestCase):
    def test_configuration_has_a_positive_innovation_gate(self) -> None:
        config = load_config()
        self.assertGreater(config.innovation_gate_sigma, 0.0)

    def test_nominal_residuals_are_finite_and_small(self) -> None:
        output = FederatedEKF().process(envelope(0))
        self.assertEqual(output.module_health["status"], "ok")
        self.assertTrue(all(math.isfinite(value) and abs(value) < 1e-12 for value in output.residual.values()))

    def test_held_multirate_sample_skips_update_and_inflates_covariance(self) -> None:
        ekf = FederatedEKF(); ekf.process(envelope(0))
        before = ekf.states["oil_temp_degC"].p
        output = ekf.process(envelope(1, new={"oil_temp_degC": False}))
        self.assertFalse(output.measurement_updated["oil_temp_degC"])
        self.assertNotIn("oil_temp_degC", output.residual)
        self.assertGreater(ekf.states["oil_temp_degC"].p, before)

    def test_invalid_and_missing_prediction_degrade_without_crash(self) -> None:
        item = envelope(0); item["channel_valid"]["egt_degC"] = False
        item["predicted"].pop("map_kPa")
        output = FederatedEKF().process(item)
        self.assertEqual(output.module_health["status"], "degraded")
        self.assertFalse(output.measurement_updated["egt_degC"])
        self.assertFalse(output.measurement_updated["map_kPa"])
        self.assertEqual(output.sensor_health["egt_degC"], "substituted")
        self.assertEqual(output.virtual_value["egt_degC"], BASE_VALUES["egt_degC"])

    def test_transient_inflates_q_for_one_tick(self) -> None:
        ekf = FederatedEKF(); ekf.process(envelope(0))
        item = envelope(1); item["values"]["rpm"] += 1000.0
        output = ekf.process(item)
        self.assertTrue(output.transient_active)
        self.assertEqual(output.q_inflation_factor, ekf.config.q_inflation_factor)
        settled = envelope(2); settled["values"]["rpm"] += 1000.0
        output = ekf.process(settled)
        self.assertFalse(output.transient_active)
        self.assertEqual(output.q_inflation_factor, 1.0)

    def test_isolated_sensor_drift_moves_bias_and_flattens_corrected_residual(self) -> None:
        config = replace(load_config(), minimum_bias_age_samples=3, drifting_bias_sigma=0.05, failed_bias_sigma=100.0)
        ekf = FederatedEKF(config); raw_values=[]; corrected=[]
        for tick in range(80):
            output = ekf.process(envelope(tick, {"batt_volts": tick * 0.01}))
            raw_values.append(output.residual_raw["batt_volts"]); corrected.append(output.residual["batt_volts"])
        self.assertGreater(output.bias_estimate["batt_volts"], 0.3)
        self.assertLess(abs(corrected[-1]), abs(raw_values[-1]))
        self.assertEqual(output.sensor_health["batt_volts"], "drifting")

    def test_cross_domain_degradation_is_not_absorbed_as_sensor_bias(self) -> None:
        ekf = FederatedEKF()
        for tick in range(60):
            output = ekf.process(envelope(tick, {"cht_degC": tick * 0.2, "oil_temp_degC": tick * 0.1}))
        self.assertGreater(output.residual["cht_degC"], 5.0)
        self.assertLess(abs(output.bias_estimate["cht_degC"]), 0.5)

    def test_failed_sensor_publishes_virtual_value(self) -> None:
        config = replace(load_config(), minimum_bias_age_samples=2, failed_bias_sigma=0.01)
        ekf = FederatedEKF(config)
        for tick in range(8):
            output = ekf.process(envelope(tick, {"alt_amps": 20.0}))
        self.assertEqual(output.sensor_health["alt_amps"], "substituted")
        self.assertEqual(output.virtual_value["alt_amps"], BASE_VALUES["alt_amps"])

    def test_held_channels_are_serializable_without_nis(self) -> None:
        item = envelope(0, new={channel: False for channel in FILTER_CHANNELS})
        safe = _json_safe(FederatedEKF().process(item).to_dict())
        self.assertNotIn("nis", safe)


if __name__ == "__main__":
    unittest.main()
