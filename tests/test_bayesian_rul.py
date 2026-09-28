from __future__ import annotations

import unittest
import tempfile
import csv
from pathlib import Path
import yaml

from layer7.bayesian_rul import BayesianPIRUL
from layer7.fit_priors import fit


class BayesianRULTests(unittest.TestCase):
    def _event(self, seconds: float, residual: float = 0.0) -> dict:
        features = {"history_seconds_available_60s": min(seconds, 60.0)}
        for suffix in ("", "_60s"):
            features.update({f"resid_oil_press_kPa_mean{suffix}": residual,
                             f"resid_oil_press_kPa_bias_mean{suffix}": 0.0,
                             f"resid_oil_press_kPa_nis_mean{suffix}": 1.0,
                             f"resid_oil_press_kPa_availability{suffix}": 1.0})
        return {"timestamp_ns": int(seconds * 1e9), "engine_serial": "E-1", "sortie_id": "S-1",
                "rul_observations": features, "black_box": {}}

    def test_first_window_uses_wide_prior_and_long_updates_are_spaced(self) -> None:
        estimator = BayesianPIRUL()
        first = estimator.update(self._event(5.0))
        interval = first["whole_engine"]
        self.assertGreater(interval["rul_ci_high_hours"], interval["rul_median_hours"])
        self.assertGreater(interval["rul_median_hours"], interval["rul_ci_low_hours"])
        self.assertEqual(first["prior_provenance"], "synthetic_fleet_fit")
        for hop in range(1, 36):
            output = estimator.update(self._event(5.0 + 2.5 * hop))
        self.assertEqual(output["subsystems"]["lubrication"]["long_update_count"], 1)
        estimator.update(self._event(95.0))
        self.assertEqual(estimator.states[("E-1", "S-1")]["lubrication"].long_update_count, 2)

    def test_flight_state_is_separate(self) -> None:
        estimator = BayesianPIRUL()
        estimator.update(self._event(5.0))
        estimator.update(self._event(7.5, residual=25.0))
        independent = self._event(5.0)
        independent["sortie_id"] = "S-2"
        estimator.update(independent)
        self.assertEqual(len(estimator.states), 2)
        self.assertEqual(estimator.states[("E-1", "S-2")]["lubrication"].long_update_count, 0)

    def test_long_window_spacing_avoids_spurious_rate_certainty(self) -> None:
        spaced = BayesianPIRUL()
        config = dict(spaced.config)
        config["long_update_spacing_seconds"] = 2.5  # Deliberately incorrect independence assumption.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "naive.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            naive = BayesianPIRUL(path)
            for hop in range(51):
                event = self._event(5 + hop * 2.5, residual=20.0)
                spaced_output = spaced.update(event)
                naive_output = naive.update(event)
        spaced_oil = spaced_output["subsystems"]["lubrication"]
        naive_oil = naive_output["subsystems"]["lubrication"]
        self.assertLess(spaced_oil["long_update_count"], naive_oil["long_update_count"])
        self.assertGreater(spaced_oil["posterior_theta_var"], naive_oil["posterior_theta_var"])

    def test_prior_fit_requires_independent_failure_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fleet.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["subsystem", "flight_id", "operating_hours",
                                                          "degradation_state", "failure_or_removal"])
                writer.writeheader()
                for subsystem in BayesianPIRUL().config["subsystems"]:
                    for flight in range(10):
                        for hour in (0, 10, 20):
                            writer.writerow({"subsystem": subsystem, "flight_id": f"{subsystem}-{flight}",
                                             "operating_hours": hour,
                                             "degradation_state": 0.05 + (0.03 + 0.001 * flight) * hour,
                                             "failure_or_removal": int(hour == 20)})
            destination = Path(directory) / "priors.yaml"
            fitted = fit(source, destination)
            self.assertEqual(fitted["provenance"], "fleet_trajectory_fit")
            self.assertEqual(fitted["source_flights_by_subsystem"]["thermal"], 10)
            self.assertGreater(fitted["subsystems"]["thermal"]["prior_rate_median_per_hour"], 0.03)

    def test_anomaly_shift_is_visible_within_two_hops(self) -> None:
        estimator = BayesianPIRUL()
        estimator.update(self._event(5.0))
        baseline = estimator.update(self._event(7.5))["subsystems"]["lubrication"]["rul_median_hours"]
        injected = self._event(10.0, residual=35.0)
        injected["black_box"] = {"anomaly_detected": True,
                                 "fault_flags": {"lubrication_issue": {"active": True}},
                                 "shap_by_class": {"lubrication_issue": {"lubrication": 1.0}}}
        estimator.update(injected)
        after_two_hops = estimator.update(self._event(12.5, residual=35.0))
        self.assertLess(after_two_hops["subsystems"]["lubrication"]["rul_median_hours"], baseline)

    def test_nominal_interval_narrows_without_long_window_reuse(self) -> None:
        estimator = BayesianPIRUL()
        first = estimator.update(self._event(5.0))["subsystems"]["lubrication"]
        current = first
        for hop in range(1, 1439):
            current = estimator.update(self._event(5.0 + 2.5 * hop))["subsystems"]["lubrication"]
        initial_width = first["rul_ci_high_hours"] - first["rul_ci_low_hours"]
        final_width = current["rul_ci_high_hours"] - current["rul_ci_low_hours"]
        self.assertLess(final_width, initial_width)
        self.assertLessEqual(current["long_update_count"], 60)


if __name__ == "__main__":
    unittest.main()
