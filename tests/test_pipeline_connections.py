"""Contract tests for the non-transport Layer 1 -> Layer 7 path."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np
from app.kinetic_bridge import (KineticTwin, _assessment_layer2_frame, _layer7_estimates,
                                layer2_frame_to_dashboard_values, legacy_payload)

from layer1.service import TelemetryValidationError, validate_ecu_event
from layer3.config import Layer3Config
from layer3.inference import PhysicsInference
from layer4.filter import FederatedEKF
from layer5.registry import FeatureRegistry
from layer5.windowing import AdaptiveWindowAssembler
from layer6.service import Layer6Processor
from layer7.engine import Layer7Engine


class FakeSession:
    def __init__(self, outputs: list[float]) -> None:
        self.outputs = outputs

    def run(self, _output_names: object, _inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.asarray([self.outputs], dtype=np.float32)]


class FakeRedis:
    def __init__(self, frame: dict) -> None:
        self.frame = frame

    async def xrevrange(self, _stream: str, count: int = 1):
        return [("1-0", {"payload": __import__("json").dumps(self.frame)})][:count]


class EmptyRedis:
    async def xrevrange(self, _stream: str, count: int = 1):
        return []


class FakeLayer1:
    async def publish_snapshot(self, _events: list[dict]) -> list[str]:
        return []


def sessions() -> dict[str, FakeSession]:
    return {
        "mvem": FakeSession([70_000, 0.08, 0.07, 0.003]),
        "wiebe": FakeSession([210, 453.15, 993.15, 5, 2, 40, 18]),
        "battery_alternator": FakeSession([13.8, 15, 2, 80, 1]),
        "oil": FakeSession([363.15, 315, 0.05]),
        "vibration": FakeSession([0.1, 0.1, 0.1, 0.1, 0.25, 3, 90]),
    }


def frame(timestamp_ns: int) -> dict:
    values = {
        "rpm": 2400.0, "map_kPa": 70.0, "Tm_K_k": 320.0, "throttle_pct": 55.0,
        "ambient_temp_degC": 20.0, "ambient_press_kPa": 101.3,
        "injection_timing_degBTDC": 20.0, "mixture_afr": 14.7, "cowl_flap_pct": 30.0,
        "cht_degC": 180.0, "egt_degC": 720.0, "oil_temp_degC": 90.0,
        "oil_press_kPa": 310.0, "fuel_flow_Lph": 14.59, "battery_soc_pct": 80.0,
        "friction_torque_Nm": 30.0, "batt_volts": 13.8, "alt_amps": 15.0,
        "vibration_rms_g": 0.25,
    }
    return {
        "master_timestamp_ns": str(timestamp_ns), "values": values,
        "channel_valid": {name: True for name in values},
        "channel_new_sample": {name: True for name in values},
        "sync_quality": 1.0, "timing_estimated": True,
        "context": {"engine_serial": "ENG-1", "sortie_id": "SORTIE-1", "operating_hours": 100.0},
    }


@dataclass
class FakeBlackBox:
    def to_dict(self) -> dict:
        return {
            "predicted_class": "nominal", "class_probabilities": {"nominal": 0.98},
            "confidence": 0.98, "anomaly_score": 0.01, "novel_degradation": False,
            "shap_subsystem": {}, "shap_top_features": [], "model_version": "test",
        }


class FakeLayer6Model:
    def __init__(self, window: dict) -> None:
        self.feature_names = tuple(sorted(window["features"]))
        self.feature_schema_hash = window["feature_schema_hash"]
        self.training_scope = "verified_healthy_nominal_only"
        self.registry = FeatureRegistry()

    def predict(self, _window: dict) -> FakeBlackBox:
        return FakeBlackBox()


class PipelineConnectionTests(unittest.TestCase):
    def test_live_state_keeps_simulator_clock_when_pipeline_has_no_assessment_yet(self) -> None:
        twin = KineticTwin.__new__(KineticTwin)
        twin.session_id = "session-a"
        twin.redis = EmptyRedis()
        twin.layer1 = FakeLayer1()
        twin.sample_rate_hz = 2.0
        twin.history = __import__("collections").deque(maxlen=10)
        twin.step_count = 0

        state = __import__("asyncio").run(twin.update({"timestamp": 0.5, "rpm": 2400.0}))

        self.assertEqual(state.timestamp, 0.5)
        self.assertEqual(state.payload["pipeline_status"], "control_revision_pending")

    def test_live_adapter_reads_layer2_frame_with_session_and_staleness_guard(self) -> None:
        twin = KineticTwin.__new__(KineticTwin)
        twin.session_id = "session-a"
        twin.redis = FakeRedis({"master_timestamp_ns": str(__import__("time").time_ns()),
                                "values": {"rpm": 3450.0}, "channel_valid": {"rpm": True},
                                "context": {"session_id": "session-a"}})
        result = __import__("asyncio").run(twin._latest_layer2_frame())
        self.assertEqual(result["values"]["rpm"], 3450.0)
        twin.redis.frame["context"]["session_id"] = "old-session"
        self.assertEqual(__import__("asyncio").run(twin._latest_layer2_frame()), {})

    def test_physics_and_ekf_preserve_layer2_control_context(self) -> None:
        control_context = {"session_id": "session-a", "control_revision": 17,
                           "parameter_provenance": {"rpm": {"source": "manual_simulator_override",
                                                               "measured": False}}}
        source = frame(1_000_000_000)
        source["context"].update(control_context)
        layer3 = PhysicsInference(Layer3Config(fuel_density_kg_L=0.74), sessions())
        physics = layer3.process(source)
        self.assertEqual(physics["context"], source["context"])
        ekf = FederatedEKF().process(physics).to_dict()
        self.assertEqual(ekf["context"]["control_revision"], 17)
        self.assertEqual(ekf["layer2_frame"]["context"]["parameter_provenance"],
                         control_context["parameter_provenance"])

    def test_dashboard_adapter_keeps_layer7_calculations_and_layer2_raw_frame_separate(self) -> None:
        assessment = {
            "risk_fusion": {"engine_risk_score": 42.5},
            "rul": {"hours": 123.0, "interval_hours": [100.0, 150.0]},
            "engine_state": "abnormal",
        }
        layer2_frame = {
            "master_timestamp_ns": "2000000000", "values": {"rpm": 3450.0, "cht_degC": 210.0},
            "channel_valid": {"rpm": True, "cht_degC": False},
            "channel_source": {"rpm": "direct", "cht_degC": "invalid"},
            "channel_age_ms": {"rpm": 0, "cht_degC": None},
            "channel_rate_hz": {"rpm": 20.0, "cht_degC": 5.0}, "sync_quality": 0.8,
        }
        payload = legacy_payload({"rpm": 3200.0}, assessment, {}, {})
        payload.update(layer2_frame_to_dashboard_values(layer2_frame))
        payload["layer2_sensor_frame"] = layer2_frame
        self.assertEqual(payload["rpm"], 3450.0)
        self.assertEqual(payload["pipeline"]["risk_fusion"]["engine_risk_score"], 42.5)
        self.assertEqual(payload["pipeline"]["rul"]["hours"], 123.0)
        self.assertFalse(payload["layer2_sensor_frame"]["channel_valid"]["cht_degC"])

    def test_layer2_frame_adapter_preserves_raw_channels_and_converts_for_legacy_ui_units(self) -> None:
        frame = {"values": {"rpm": 3450.0, "throttle_pct": 72.0, "map_kPa": 101.59167,
                            "oil_press_kPa": 344.73785, "Tm_K_k": 303.15,
                            "ambient_press_kPa": 101.325, "altitude_m": 914.4,
                            "air_density_ratio": 0.91}}
        values = layer2_frame_to_dashboard_values(frame)
        self.assertEqual(values["rpm"], 3450.0)
        self.assertEqual(values["throttle"], 0.72)
        self.assertAlmostEqual(values["map"], 30.0)
        self.assertAlmostEqual(values["oil_press"], 50.0)
        self.assertAlmostEqual(values["iat"], 30.0)
        self.assertAlmostEqual(values["baro_pressure"], 29.92, places=1)
        self.assertAlmostEqual(values["altitude"], 3000.0, places=1)

    def test_invalid_layer2_values_and_layer3_helper_defaults_do_not_leak_into_dashboard(self) -> None:
        payload = legacy_payload(
            {"rpm": 3200.0, "cht": 190.0}, {},
            {"predicted": {"cht_degC": 25.0}, "prediction_valid": {"cht_degC": False}}, {})
        self.assertIsNone(payload["cht_expected"])
        frame_values = layer2_frame_to_dashboard_values({"values": {"rpm": 3300.0}})
        payload.update(frame_values)
        payload["cht"] = None
        self.assertEqual(payload["rpm"], 3300.0)
        self.assertIsNone(payload["cht"])

    def test_layer7_estimates_are_exposed_without_fabricated_defaults(self) -> None:
        estimates = _layer7_estimates({"risk_fusion": {"engine_risk_score": 28.0,
                                                        "subsystem_risk": {"thermal": 41.0}},
                                       "rul": {"hours": None, "reason_unknown": "insufficient history"}})
        self.assertEqual(estimates["health"], 72.0)
        self.assertIsNone(estimates["rul_hours"])
        self.assertEqual(estimates["layer7_risk_thermal"], 41.0)

    def test_layer7_frame_passthrough_rejects_stale_or_invalid_channels(self) -> None:
        now_ns = __import__("time").time_ns()
        assessment = {"synchronized_sensor_frame": {
            "master_timestamp_ns": str(now_ns), "context": {"session_id": "active"},
            "values": {"rpm": 2300.0, "cht_degC": 190.0},
            "channel_valid": {"rpm": True, "cht_degC": False}}}
        result = _assessment_layer2_frame(assessment, "active")
        self.assertEqual(result["values"], {"rpm": 2300.0})
        self.assertEqual(_assessment_layer2_frame(assessment, "other"), {})
        assessment["synchronized_sensor_frame"]["master_timestamp_ns"] = str(now_ns - 20_000_000_000)
        self.assertEqual(_assessment_layer2_frame(assessment, "active"), {})

    def test_layer1_rejects_invalid_instead_of_publishing_valid(self) -> None:
        event = validate_ecu_event({"sensor": "rpm", "value": 2400, "sensor_timestamp_ns": 12})
        self.assertEqual(event["sensor_timestamp_ns"], "12")
        with self.assertRaises(TelemetryValidationError):
            validate_ecu_event({"sensor": "rpm", "value": float("nan"), "sensor_timestamp_ns": 12})

    def test_layer3_marks_dependency_loss_explicitly(self) -> None:
        processor = PhysicsInference(Layer3Config(fuel_density_kg_L=0.74), sessions())
        first = processor.process(frame(1_000_000_000))
        self.assertFalse(first["prediction_valid"]["vibration_rms_g"])
        second = processor.process(frame(1_010_000_000))
        self.assertTrue(all(second["prediction_valid"].values()))
        self.assertEqual(second["context"]["engine_serial"], "ENG-1")

    def test_processor_chain_reaches_advisory_output(self) -> None:
        layer3 = PhysicsInference(Layer3Config(fuel_density_kg_L=0.74), sessions())
        layer3.process(frame(1_000_000_000))
        physics = layer3.process(frame(1_010_000_000))
        residual = FederatedEKF().process(physics).to_dict()
        assembler = AdaptiveWindowAssembler()
        window = None
        for tick in range(501):
            sample = dict(residual); sample["master_timestamp_ns"] = residual["master_timestamp_ns"] + tick * 10_000_000
            emitted = assembler.process(sample)
            if emitted:
                window = emitted[-1]
        self.assertIsNotNone(window)
        assert window is not None
        payload = window.to_dict()
        self.assertEqual(payload["layer2_frame"]["context"]["engine_serial"], "ENG-1")
        for channel in FeatureRegistry().channels:
            payload["features"][FeatureRegistry.summary_name(channel, "availability")] = 1.0
        payload["data_quality_score"] = 1.0
        assessment = Layer6Processor(FakeLayer6Model(payload)).process(payload)
        self.assertTrue(assessment["classification_available"])
        output = Layer7Engine().process(assessment)
        self.assertEqual(output.recommended_action, "NORMAL")
        self.assertIsNotNone(output.rul_hours)
        self.assertIsNotNone(output.rul_interval_hours)
        self.assertEqual(output.bayesian_rul["prior_provenance"], "synthetic_fleet_fit")

    def test_development_artifact_runs_inference_but_is_not_decision_eligible(self) -> None:
        registry = FeatureRegistry()
        payload = {"window_start_ns": 0, "window_end_ns": 5_000_000_000,
                   "window_length_s": 5.0, "truncated": False,
                   "feature_schema_hash": "schema", "features": {}, "context": {},
                   "data_quality_score": 0.9}
        payload["features"] = {name: 1.0 for name in registry.feature_names}
        payload["feature_schema_hash"] = registry.schema_hash(list(payload["features"]))
        for channel in registry.channels:
            payload["features"][registry.summary_name(channel, "availability")] = 1.0

        class GatedModel(FakeLayer6Model):
            def __init__(self, window: dict, scope: str) -> None:
                super().__init__(window)
                self.training_scope = scope
                self.called = False

            def predict(self, _window: dict) -> FakeBlackBox:
                self.called = True
                return FakeBlackBox()

        development = GatedModel(payload, "synthetic_nominal_unverified")
        result = Layer6Processor(development).process(payload)
        self.assertTrue(result["classification_available"])
        self.assertTrue(result["inference_available"])
        self.assertFalse(result["decision_eligible"])
        self.assertTrue(development.called)
        self.assertIn("not eligible for Layer 7 risk fusion", result["decision_eligibility_reason"])

        payload["features"][registry.summary_name("fuel_flow_Lph", "availability")] = 0.0
        unavailable = GatedModel(payload, "verified_healthy_nominal_only")
        result = Layer6Processor(unavailable).process(payload)
        self.assertTrue(result["classification_available"])
        self.assertTrue(result["inference_available"])
        self.assertFalse(result["decision_eligible"])
        self.assertIn("fuel_flow_Lph", result["unavailable_residual_channels"])
        self.assertTrue(unavailable.called)

    def test_layer7_missing_hours_is_safe_and_explicit(self) -> None:
        output = Layer7Engine().process({"timestamp_ns": 1, "operating_hours": None, "black_box": {}, "indicators": {}})
        self.assertIsNone(output.rul_hours)
        self.assertIn("Operating hours", output.rul_reason_unknown)



if __name__ == "__main__":
    unittest.main()
