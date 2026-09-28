"""Missing model predictions get dynamic, clearly labeled display estimates."""

import unittest

from app.kinetic_bridge import fill_estimated_residuals


class DashboardResidualEstimateTests(unittest.TestCase):
    def frame(self, rpm=2400.0, ambient=25.0):
        values = {"rpm": rpm, "throttle_pct": 65.0,
                  "ambient_temp_degC": ambient, "altitude_m": 600.0,
                  "egt_degC": 650.0, "cht_degC": 180.0}
        return {"values": values, "channel_valid": {key: True for key in values}}

    def test_only_missing_predictions_use_dynamic_physics_estimates(self):
        initial = {"egt_measured": 650.0, "egt_expected": None,
                   "cht_measured": 180.0, "cht_expected": 170.0,
                   "cht_residual": 10.0, "residual_source": {"cht": "ekf"}}
        fill_estimated_residuals(initial, self.frame())
        self.assertEqual(initial["residual_source"]["egt"], "estimated_physics")
        self.assertAlmostEqual(initial["egt_residual"], 650.0 - initial["egt_expected"])
        self.assertEqual(initial["cht_expected"], 170.0)
        self.assertEqual(initial["residual_source"]["cht"], "ekf")

        faster = {"egt_measured": 650.0, "egt_expected": None}
        fill_estimated_residuals(faster, self.frame(rpm=3600.0, ambient=35.0))
        self.assertNotEqual(faster["egt_expected"], initial["egt_expected"])
        self.assertNotEqual(faster["egt_residual"], initial["egt_residual"])

    def test_missing_measurement_stays_missing(self):
        payload = {"egt_measured": None, "egt_expected": None}
        fill_estimated_residuals(payload, self.frame())
        self.assertIsNone(payload["egt_expected"])
        self.assertNotIn("egt_residual", payload)


if __name__ == "__main__":
    unittest.main()
