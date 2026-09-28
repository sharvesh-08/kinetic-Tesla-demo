"""Layer 3 must preserve innovations for Layer 7 risk scoring."""

import unittest

from layer3.config import Layer3Config
from layer3.inference import PhysicsInference
from tests.test_pipeline_connections import frame, sessions


class RiskEvidenceTests(unittest.TestCase):
    def test_nominal_bias_freezes_during_manual_override(self):
        processor = PhysicsInference(Layer3Config(fuel_density_kg_L=0.74), sessions())
        baseline = frame(1_000_000_000)
        baseline["context"].update(injected_fault="NONE", manual_override=False)
        first = processor.process(baseline)
        self.assertAlmostEqual(first["predicted"]["oil_press_kPa"], baseline["values"]["oil_press_kPa"])

        changed = frame(1_010_000_000)
        changed["context"].update(injected_fault="NONE", manual_override=True)
        changed["values"]["oil_press_kPa"] = 210.0
        result = processor.process(changed)
        self.assertAlmostEqual(result["predicted"]["oil_press_kPa"], 310.0)
        self.assertNotEqual(result["predicted"]["oil_press_kPa"], changed["values"]["oil_press_kPa"])


if __name__ == "__main__":
    unittest.main()
