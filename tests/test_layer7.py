from __future__ import annotations
from pathlib import Path
import sys
import unittest

# Permit both `python3 tests/test_layer7.py` and `cd tests && python3 test_layer7.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer7.engine import Layer7Engine


class Layer7Tests(unittest.TestCase):
    def event(self, hour: float, oil: float = 10.0, quality: float = 1.0, **extra: object) -> dict:
        base = {"timestamp_ns": int(hour * 3_600_000_000_000), "operating_hours": hour, "indicators": {"oil_press_residual_kPa": oil}, "data_quality_score": quality,
                "black_box": {"confidence": 0.1, "anomaly_score": 0.0, "shap_subsystem": {"lubrication": 8.0}, "class_probabilities": {"nominal": 0.9, "lubrication_issue": 0.1}, "model_version": "test"}}
        base.update(extra); return base

    def test_insufficient_history_has_unknown_rul(self) -> None:
        output = Layer7Engine().process(self.event(0.0))
        self.assertIsNone(output.rul_hours); self.assertIsNone(output.rul_interval_hours)
        self.assertIn("6 trend samples", output.rul_reason_unknown)
        self.assertIn("0.25 engine operating hours", output.rul_reason_unknown)

    def test_sufficient_history_without_degradation_explains_missing_rul(self) -> None:
        engine = Layer7Engine()
        for hour in (0, 0.05, 0.10, 0.15, 0.20, 0.25):
            output = engine.process(self.event(hour, oil=10.0))
        self.assertIsNone(output.rul_hours)
        self.assertEqual(output.rul_reason_unknown, "No currently degrading sensor trend; RUL cannot be estimated.")

    def test_slow_fault_produces_interval_and_shrinking_rul(self) -> None:
        engine = Layer7Engine(); outputs = []
        for hour in range(12): outputs.append(engine.process(self.event(float(hour), oil=10 + hour * 3)))
        estimates = [out for out in outputs if out.rul_hours is not None]
        self.assertGreaterEqual(len(estimates), 2)
        self.assertLess(estimates[-1].rul_hours, estimates[0].rul_hours)
        self.assertIsNotNone(estimates[-1].rul_interval_hours)

    def test_critical_subsystem_is_not_averaged_down(self) -> None:
        output = Layer7Engine().process(self.event(0.0, oil=50.0))
        self.assertEqual(output.recommended_action, "IMMEDIATE")
        self.assertGreaterEqual(output.engine_risk_score, output.subsystem_risk["lubrication"] * 0.99)

    def test_low_quality_lowers_confidence_not_risk(self) -> None:
        high = Layer7Engine().process(self.event(0.0, quality=1)); low = Layer7Engine().process(self.event(0.0, quality=0.2))
        self.assertEqual(high.engine_risk_score, low.engine_risk_score)
        self.assertLess(low.engine_risk_confidence, high.engine_risk_confidence)

    def test_redline_backstop_and_maintenance_deadline(self) -> None:
        engine = Layer7Engine()
        for hour in range(8): output = engine.process(self.event(float(hour), oil=10 + hour * 3))
        output = engine.process(self.event(9, oil=37, telemetry={"oil_press_kPa": 100}))
        self.assertEqual(output.recommended_action, "IMMEDIATE")
        self.assertEqual(output.engine_state, "critical")
        if output.rul_interval_hours: self.assertLess(output.maintenance_plan[0].deadline_operating_hours, output.rul_interval_hours[0])

    def test_commanded_shutdown_does_not_raise_low_oil_pressure_alarm(self) -> None:
        event = self.event(0.0, oil=0.0, telemetry={"rpm": 0, "oil_press_kPa": 0},
                           parameter_provenance={"rpm": {"source": "manual_simulator_override"}},
                           layer2_frame={"context": {"override_values": {"rpm": 0}}},
                           black_box={})
        output = Layer7Engine().process(event)
        self.assertEqual(output.redline_hits, [])
        self.assertEqual(output.recommended_action, "NORMAL")
        self.assertNotEqual(output.engine_state, "critical")

    def test_uncommanded_stop_is_an_engine_stall(self) -> None:
        event = self.event(0.0, oil=0.0, telemetry={"rpm": 0, "oil_press_kPa": 0},
                           black_box={})
        output = Layer7Engine().process(event)
        self.assertEqual(output.redline_hits, ["engine_stall"])
        self.assertEqual(output.recommended_action, "IMMEDIATE")
        self.assertEqual(output.engine_state, "critical")

    def test_critical_requires_joint_fault_and_anomaly_evidence(self) -> None:
        engine = Layer7Engine()
        high_fault = self.event(0, black_box={"confidence": 0.95, "anomaly_score": 0.1,
                              "class_probabilities": {"nominal": 0.05, "misfire": 0.95},
                              "model_version": "test"})
        self.assertEqual(engine.process(high_fault).engine_state, "abnormal")
        joint = self.event(0.1, black_box={"confidence": 0.95, "anomaly_score": 0.95,
                            "class_probabilities": {"nominal": 0.05, "misfire": 0.95},
                            "model_version": "test"})
        self.assertEqual(engine.process(joint).engine_state, "critical")

    def test_bayesian_rul_is_available_on_first_feature_window(self) -> None:
        event = self.event(0.0, engine_serial="E-1", sortie_id="S-1",
                           rul_observations={"history_seconds_available_60s": 5.0})
        output = Layer7Engine().process(event)
        self.assertIsNotNone(output.rul_hours)
        self.assertIsNotNone(output.rul_interval_hours)
        self.assertEqual(output.bayesian_rul["prior_provenance"], "synthetic_fleet_fit")

    def test_vibration_shap_contributes_to_mechanical_risk(self) -> None:
        event = self.event(0.0, black_box={"confidence": 0.9, "anomaly_score": 0.0,
                            "class_probabilities": {"nominal": 0.1, "abnormal_vibration": 0.9},
                            "shap_subsystem": {"vibration": 10.0}, "model_version": "test"})
        output = Layer7Engine().process(event)
        self.assertEqual(output.dominant_subsystem, "mechanical")

    def test_physical_residuals_drive_risk_when_classifier_is_not_eligible(self) -> None:
        event = self.event(0.0, black_box={
            "confidence": 0.59, "anomaly_score": 0.99,
            "class_probabilities": {"nominal": 0.41, "misfire": 0.59},
            "shap_subsystem": {"thermal": 20.0},
        }, decision_eligible=False, classification_available=True,
            indicators={"egt_residual_degC": 30.0}, data_quality_score=0.9)
        output = Layer7Engine().process(event)
        self.assertGreater(output.subsystem_risk["combustion"], 0.0)
        self.assertGreater(output.engine_risk_score, 0.0)
        self.assertGreater(output.engine_risk_confidence, 0.0)
        self.assertEqual(output.recommended_action, "NORMAL")

    def test_high_confidence_unverified_fault_contributes_provisional_risk(self) -> None:
        event = self.event(0.0, indicators={}, decision_eligible=False,
                           black_box={"confidence": 0.72, "anomaly_score": 0.78,
                                      "class_probabilities": {"sensor_drift_failure": 0.72}})
        output = Layer7Engine().process(event)
        self.assertEqual(output.engine_state, "abnormal")
        self.assertEqual(output.recommended_action, "ADVISORY")
        self.assertGreater(output.engine_risk_score, 0.0)
        event["black_box"]["class_probabilities"] = {"sensor_drift_failure": 0.59}
        excluded = Layer7Engine().process(event)
        self.assertIsNone(excluded.engine_state)
        self.assertEqual(excluded.recommended_action, "INSUFFICIENT_DATA")

    def test_catastrophic_physical_residual_has_consistent_risk_floor(self) -> None:
        event = self.event(0.0, indicators={"oil_press_residual_kPa": 60.0}, black_box={})
        output = Layer7Engine().process(event)
        self.assertEqual(output.recommended_action, "IMMEDIATE")
        self.assertGreaterEqual(output.subsystem_risk["lubrication"], 92.0)
        self.assertGreaterEqual(output.engine_risk_score, 92.0)

    def test_missing_physical_and_eligible_model_evidence_is_not_reported_as_normal(self) -> None:
        output = Layer7Engine().process({"timestamp_ns": 1, "data_quality_score": 0.0,
                                         "black_box": {}, "indicators": {}, "telemetry": {}})
        self.assertEqual(output.engine_state, None)
        self.assertEqual(output.recommended_action, "INSUFFICIENT_DATA")

    def test_environmental_stress_shortens_synthetic_rul_estimate(self) -> None:
        baseline = self.event(0.0, engine_serial="E-1", sortie_id="S-1",
                              rul_observations={"history_seconds_available_60s": 5.0},
                              operating_context={"altitude_m": 0.0, "ambient_temp_degC": 25.0})
        hot_high = {**baseline, "operating_context": {"altitude_m": 5000.0, "ambient_temp_degC": 45.0}}
        standard_output = Layer7Engine().process(baseline)
        stressed_output = Layer7Engine().process(hot_high)
        self.assertGreater(stressed_output.bayesian_rul["environmental_stress_factor"], 1.0)
        self.assertLess(stressed_output.rul_hours, standard_output.rul_hours)


if __name__ == "__main__": unittest.main()
