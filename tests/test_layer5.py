"""Acceptance tests for fixed Layer 5 feature windows."""
from __future__ import annotations

import unittest

from layer5.registry import FeatureRegistry, FeatureSchemaMismatch
from layer5.windowing import AdaptiveWindowAssembler

CHANNELS = tuple(FeatureRegistry().channels)


def item(tick: int, **extra: object) -> dict:
    values = {channel: float(tick) for channel in CHANNELS}
    result = {"master_timestamp_ns": tick * 10_000_000, "residual": dict(values),
              "residual_z": {channel: float(tick) / 10 for channel in CHANNELS},
              "bias_estimate": {channel: float(tick) / 100 for channel in CHANNELS},
              "channel_valid": {channel: True for channel in CHANNELS},
              "sensor_health": {channel: "ok" for channel in CHANNELS},
              "measurement_updated": {channel: True for channel in CHANNELS},
              "sync_quality": 1.0, "context": {"sortie_id": "test"}}
    result.update(extra)
    return result


class Layer5Tests(unittest.TestCase):
    def test_gap_crossing_multiple_hops_emits_each_exact_boundary(self) -> None:
        assembler = AdaptiveWindowAssembler()
        assembler.process(item(0))
        windows = assembler.process(item(1000))
        self.assertEqual([w.window_end_ns for w in windows], [5_000_000_000, 7_500_000_000, 10_000_000_000])
        self.assertEqual([w.window_end_ns - w.window_start_ns for w in windows], [5_000_000_000] * 3)
        self.assertEqual(windows[0].n_samples['egt_degC'], 1)
        self.assertEqual(windows[-1].n_samples['egt_degC'], 0)
        self.assertEqual(windows[-1].features['resid_egt_degC_raw_49'], 0.0)

    def test_hold_at_exact_downsample_position_and_summary_values(self) -> None:
        assembler = AdaptiveWindowAssembler()
        output = []
        for tick in range(500):
            payload = item(tick, transient_active=True, sync_quality=0.1)
            payload['nis_by_channel'] = {ch: 4.0 for ch in CHANNELS}
            if tick == 100:
                payload['measurement_updated']['egt_degC'] = False
            output.extend(assembler.process(payload))
        window = output[0]
        self.assertEqual(window.features['resid_egt_degC_raw_10'], 99.0)
        self.assertEqual(window.features['resid_cht_degC_raw_10'], 100.0)
        self.assertAlmostEqual(window.features['resid_egt_degC_mean'], 249.5 - 1 / 500)
        self.assertEqual(window.features['resid_egt_degC_max_abs'], 499.0)
        self.assertAlmostEqual(window.features['resid_cht_degC_slope'], 100.0)
        self.assertEqual(window.features['resid_egt_degC_nis_mean'], 4.0)
        self.assertEqual(window.features['resid_egt_degC_nis_max'], 4.0)
        self.assertAlmostEqual(window.features['resid_egt_degC_availability'], 499 / 500)

    def test_rul_metadata_survives_a_last_tick_without_measurements(self) -> None:
        assembler = AdaptiveWindowAssembler()
        output = []
        for tick in range(500):
            payload = item(tick)
            payload['indicators'] = {'egt_residual_degC': 12.0} if tick % 2 == 0 else {}
            output.extend(assembler.process(payload))
        self.assertEqual(output[0].indicators['egt_residual_degC'], 12.0)
        self.assertNotIn('egt_residual_degC', output[0].features)

    def test_emits_exact_named_5s_feature_contract_and_50pct_hop(self) -> None:
        assembler = AdaptiveWindowAssembler()
        output = []
        for tick in range(751):
            output.extend(assembler.process(item(tick)))
        registry = FeatureRegistry()
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].window_length_s, 5.0)
        self.assertEqual(output[0].window_end_ns - output[0].window_start_ns, 5_000_000_000)
        self.assertEqual(output[1].window_start_ns - output[0].window_start_ns, 2_500_000_000)
        self.assertEqual(tuple(sorted(output[0].features)), registry.feature_names)
        self.assertEqual(len(output[0].features), len(registry.feature_names))
        self.assertEqual(output[0].features["history_seconds_available_60s"], 5.0)
        self.assertFalse(output[0].cold_start)
        self.assertFalse(output[0].truncated)

    def test_missing_slots_are_held_and_availability_tracks_real_samples(self) -> None:
        assembler = AdaptiveWindowAssembler()
        outputs = []
        for tick in range(500):
            if tick % 2 == 0:
                outputs.extend(assembler.process(item(tick)))
            else:
                # A missing 100 Hz source tick is represented when the next timestamp arrives.
                continue
        outputs.extend(assembler.process(item(500)))
        self.assertEqual(len(outputs), 1)
        window = outputs[0]
        availability = window.features["resid_egt_degC_availability"]
        self.assertAlmostEqual(availability, 0.5, places=2)
        raw = [window.features[f"resid_egt_degC_raw_{i:02d}"] for i in range(50)]
        self.assertEqual(len(raw), 50)
        self.assertTrue(all(window.feature_valid[f"resid_egt_degC_raw_{i:02d}"] for i in range(50)))

    def test_channel_missingness_is_local_and_true_fraction_survives_hold(self) -> None:
        assembler = AdaptiveWindowAssembler()
        outputs = []
        for tick in range(501):
            payload = item(tick)
            if tick % 2:
                payload["measurement_updated"]["egt_degC"] = False
            outputs.extend(assembler.process(payload))
        self.assertEqual(len(outputs), 1)
        self.assertAlmostEqual(outputs[0].features["resid_egt_degC_availability"], 0.5, places=2)
        self.assertEqual(outputs[0].features["resid_cht_degC_availability"], 1.0)

    def test_fixed_window_contract_rejects_wrong_duration_or_schema(self) -> None:
        registry = FeatureRegistry()
        with self.assertRaises(FeatureSchemaMismatch):
            registry.validate(list(registry.feature_names), 2.0)
        with self.assertRaises(FeatureSchemaMismatch):
            registry.require_schema(["window_length_s"], "not-a-live-schema")


if __name__ == "__main__":
    unittest.main()
