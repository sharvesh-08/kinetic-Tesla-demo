"""Tests for the Layer 7 -> dashboard Redis publishing contract."""

from __future__ import annotations

import json
import unittest

from redis.exceptions import ResponseError

from layer7.dashboard_publisher import build_dashboard_payload
from layer7.engine import Layer7Engine
from layer7.service import Layer7Service, ServiceConfig


class MemoryRedis:
    def __init__(self) -> None:
        self.entries: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.acks: list[tuple[str, str, tuple[str, ...]]] = []
        self.operations: list[tuple[str, str]] = []
        self.fail_stream: str | None = None

    async def xrange(self, stream: str, min: str, max: str, count: int = 1):
        return [entry for entry in self.entries.get(stream, []) if entry[0] == min][:count]

    async def xadd(self, stream: str, fields: dict[str, str], id: str = "*", **_options: object) -> str:
        if stream == self.fail_stream:
            raise ResponseError("simulated Redis write failure")
        self.entries.setdefault(stream, []).append((id, dict(fields)))
        self.operations.append(("xadd", stream))
        return id

    async def xack(self, stream: str, group: str, *entry_ids: str) -> int:
        self.acks.append((stream, group, tuple(entry_ids)))
        self.operations.append(("xack", stream))
        return len(entry_ids)


class Layer7DashboardPublisherTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = ServiceConfig(dashboard_output_stream="engine:test:dashboard")
        self.redis = MemoryRedis()
        self.service = Layer7Service(self.config, processor=Layer7Engine())
        self.service.redis = self.redis
        self.timestamp_ns = 1_800_000_000_123_000_000
        self.event = {
            "timestamp_ns": self.timestamp_ns,
            "telemetry_timestamp_ns": self.timestamp_ns - 1_000_000_000,
            "engine_serial": "ENG-TEST",
            "sortie_id": "SORTIE-TEST",
            "classification_available": True,
            "classification_unavailable_reason": None,
            "data_quality_score": 0.98,
            "operating_hours": None,
            "indicators": {"egt_residual_degC": 12.0},
            "telemetry": {"rpm": 2400.0, "egt_degC": 1000.0, "oil_press_kPa": 310.0},
            "layer2_frame": {"master_timestamp_ns": str(self.timestamp_ns - 1_000_000_000),
                             "values": {"rpm": 2400.0, "egt_degC": 1000.0},
                             "channel_valid": {"rpm": True, "egt_degC": True},
                             "channel_source": {"rpm": "direct", "egt_degC": "zoh"},
                             "channel_age_ms": {"rpm": 10, "egt_degC": 50},
                             "channel_rate_hz": {"rpm": 20, "egt_degC": 10},
                             "sync_quality": 0.95,
                             "context": {"session_id": "session-test"}},
            "nis_by_subsystem": {"combustion": 1.4},
            "operating_context": {},
            "black_box": {
                "predicted_class": "overheating_trend",
                "class_probabilities": {"nominal": 0.02, "overheating_trend": 0.98},
                "confidence": 0.98,
                "anomaly_score": 0.91,
                "anomaly_detected": True,
                "novel_degradation": True,
                "shap_subsystem": {"thermal": 0.7, "combustion": 0.4},
                "shap_top_features": [["resid__egt_degC__mean_w5s", 0.4]],
                "model_version": "layer6-test",
            },
        }
        self.fields = {"payload": json.dumps(self.event)}

    async def test_dashboard_message_contains_joined_model_risk_and_sensor_data(self) -> None:
        await self.service.process_entry("5000-0", self.fields)

        risk_id, risk_fields = self.redis.entries[self.config.output_stream][-1]
        dashboard_id, dashboard_fields = self.redis.entries[self.config.dashboard_output_stream][-1]
        dashboard = json.loads(dashboard_fields["payload"])
        self.assertEqual(risk_id, dashboard_id)
        self.assertEqual(dashboard["timestamp_ns"], self.timestamp_ns)
        self.assertEqual(dashboard["sensor_values_timestamp_ns"], self.timestamp_ns - 1_000_000_000)
        self.assertEqual(dashboard["synchronized_sensor_values"]["rpm"], 2400.0)
        self.assertEqual(dashboard["synchronized_sensor_frame"]["context"]["session_id"], "session-test")
        self.assertEqual(dashboard["sensor_channel_source"]["egt_degC"], "zoh")
        self.assertEqual(dashboard["sensor_value_units"]["rpm"], "RPM")
        self.assertEqual(dashboard["risk_fusion"]["redline_hits"], ["egt"])
        self.assertEqual(dashboard["lightgbm"]["predicted_class"], "overheating_trend")
        self.assertEqual(dashboard["lightgbm"]["class_probabilities"]["overheating_trend"], 0.98)
        self.assertEqual(dashboard["isolation_forest"]["anomaly_score"], 0.91)
        self.assertTrue(dashboard["isolation_forest"]["anomaly_detected"])
        self.assertEqual(dashboard["fault_category"], "overheating_trend")
        self.assertEqual(dashboard["engine_health_status"], "critical")
        self.assertIsNone(dashboard["rul"]["hours"])
        self.assertIsNotNone(dashboard["rul"]["reason_unknown"])
        self.assertEqual(dashboard["degradation_indicators"]["egt_residual_degC"], 12.0)
        thermal_evidence = dashboard["risk_fusion"]["evidence_by_subsystem"]["thermal"]
        self.assertEqual(thermal_evidence["classifier_fault_probability"], 0.98)
        self.assertEqual(thermal_evidence["isolation_forest_anomaly_score"], 0.91)
        combustion_evidence = dashboard["risk_fusion"]["evidence_by_subsystem"]["combustion"]
        self.assertEqual(combustion_evidence["physical_indicator_signal"], 0.2)
        self.assertGreaterEqual(dashboard["risk_fusion"]["engine_risk_score"], 92.0)
        self.assertIn("weighted_components", thermal_evidence)
        self.assertIn("shap_weight", dashboard["risk_fusion"]["fusion_policy"])
        self.assertEqual(json.loads(risk_fields["payload"])["recommended_action"], "IMMEDIATE")
        self.assertEqual(self.redis.acks, [(self.config.input_stream, self.config.consumer_group, ("5000-0",))])
        self.assertEqual(
            self.redis.operations,
            [("xadd", self.config.output_stream), ("xadd", self.config.dashboard_output_stream),
             ("xack", self.config.input_stream)],
        )

    async def test_dashboard_write_failure_does_not_ack_and_retry_is_idempotent(self) -> None:
        self.redis.fail_stream = self.config.dashboard_output_stream
        await self.service.process_entry("5001-0", self.fields)
        self.assertEqual(len(self.redis.entries[self.config.output_stream]), 1)
        self.assertNotIn(self.config.dashboard_output_stream, self.redis.entries)
        self.assertEqual(self.redis.acks, [])

        self.redis.fail_stream = None
        await self.service.process_entry("5001-0", self.fields)
        self.assertEqual(len(self.redis.entries[self.config.output_stream]), 1)
        self.assertEqual(len(self.redis.entries[self.config.dashboard_output_stream]), 1)
        self.assertEqual(len(self.redis.acks), 1)
        self.assertEqual(
            self.redis.operations[-2:],
            [("xadd", self.config.dashboard_output_stream), ("xack", self.config.input_stream)],
        )

    def test_sensor_trend_is_exposed_and_nominal_classifier_does_not_create_a_fault(self) -> None:
        risk_output = {
            "recommended_action": "NORMAL",
            "active_faults": [{"fault_class": "misfire", "probability": 0.01}],
            "trends": [{"indicator": "egt_residual_degC", "subsystem": "combustion", "direction": "degrading", "slope_per_op_hour": 1.2}],
            "subsystem_risk": {"combustion": 20.0},
            "subsystem_risk_confidence": {"combustion": 0.8},
        }
        event = {
            "timestamp_ns": self.timestamp_ns,
            "classification_available": True,
            "black_box": {"predicted_class": "nominal", "class_probabilities": {"nominal": 0.99, "misfire": 0.01}, "anomaly_detected": False},
            "indicators": {"egt_residual_degC": 12.0},
        }

        dashboard = build_dashboard_payload(event, risk_output, self.service.processor.config)

        self.assertEqual(dashboard["fault_category"], "nominal")
        self.assertEqual(dashboard["engine_health_status"], "abnormal")
        self.assertEqual(dashboard["sensor_degradation_trends"]["egt_residual_degC"]["slope_per_op_hour"], 1.2)
        self.assertEqual(dashboard["layer7_output"], risk_output)

    def test_isolation_forest_detection_marks_nominal_classification_abnormal(self) -> None:
        event = {
            "timestamp_ns": self.timestamp_ns,
            "classification_available": True,
            "black_box": {
                "predicted_class": "nominal",
                "class_probabilities": {"nominal": 0.99, "misfire": 0.01},
                "anomaly_detected": True,
                "novel_degradation": False,
            },
        }
        risk_output = {"recommended_action": "NORMAL", "trends": []}

        dashboard = build_dashboard_payload(event, risk_output, self.service.processor.config)

        self.assertEqual(dashboard["fault_category"], "nominal")
        self.assertEqual(dashboard["engine_health_status"], "abnormal")

    def test_development_rul_status_reaches_dashboard_record(self) -> None:
        risk_output = {"engine_state": "normal", "rul_hours": 20.0,
                       "rul_interval_hours": [5.0, 60.0],
                       "bayesian_rul": {"prior_provenance": "synthetic_fleet_fit",
                                        "validated_for_decisions": False}}
        dashboard = build_dashboard_payload({"timestamp_ns": self.timestamp_ns},
                                            risk_output, self.service.processor.config)
        self.assertEqual(dashboard["rul"]["prior_provenance"], "synthetic_fleet_fit")
        self.assertFalse(dashboard["rul"]["validated_for_decisions"])
        self.assertEqual(dashboard["rul"]["hours"], 20.0)
        self.assertEqual(dashboard["rul"]["interval_hours"], [5.0, 60.0])

    def test_layer7_engine_risk_and_rul_are_calculated_values_not_sensor_echoes(self) -> None:
        risk_output = {"engine_risk_score": 64.0, "engine_risk_confidence": 0.75,
                       "rul_hours": 18.0, "rul_interval_hours": [10.0, 28.0],
                       "rul_confidence_level": 0.9, "recommended_action": "INSPECT",
                       "bayesian_rul": {"validated_for_decisions": True}}
        dashboard = build_dashboard_payload(
            {"timestamp_ns": self.timestamp_ns, "telemetry": {"rpm": 2400.0},
             "black_box": {"predicted_class": "nominal", "class_probabilities": {"nominal": 0.9}}},
            risk_output, self.service.processor.config)
        self.assertEqual(dashboard["synchronized_sensor_values"]["rpm"], 2400.0)
        self.assertEqual(dashboard["risk_fusion"]["engine_risk_score"], 64.0)
        self.assertEqual(dashboard["rul"]["hours"], 18.0)


if __name__ == "__main__":
    unittest.main()
