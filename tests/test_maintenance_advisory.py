"""Verify dashboard context and Groq response propagation without network calls."""
import json
import unittest
from unittest.mock import Mock, patch

from app.maintenance_advisory import SYSTEM_PROMPT, draft_with_groq, env_value


class MaintenanceAdvisoryTests(unittest.TestCase):
    def test_groq_receives_displayed_fault_and_risk_rul(self):
        results = {"edge_ml": {"fault_class": "sensor_drift_failure", "confidence": 0.72, "anomaly_score": 0.78},
                   "risk_rul": {"health_state": "abnormal", "remaining_useful_life_hours": 12.9,
                                "subsystem_risk": {"sensor": 33.6}}}
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "Inspect the sensor and verify its calibration."}}]}
        post = Mock(return_value=response)
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key", "GROQ_MODEL": "test-model"}):
            result = draft_with_groq(results, post=post)
        self.assertEqual(result["text"], "Inspect the sensor and verify its calibration.")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["messages"][0], {"role": "system", "content": SYSTEM_PROMPT})
        sent = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        self.assertEqual(sent, results)
        response.raise_for_status.assert_called_once()

    def test_missing_key_does_not_return_a_canned_advisory(self):
        with patch("app.maintenance_advisory.env_value", return_value=""):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                draft_with_groq({})

    def test_empty_response_is_an_error(self):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": " "}}]}
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            with self.assertRaisesRegex(ValueError, "empty"):
                draft_with_groq({}, post=Mock(return_value=response))
