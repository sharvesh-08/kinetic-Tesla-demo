from __future__ import annotations

import unittest
import importlib
import joblib
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile

from layer5.registry import FeatureRegistry
from layer6.artifact import (ArtifactCompatibilityError, ArtifactMetadata, Layer6Artifact,
                             load_artifact, migrate_legacy_artifact, save_artifact)
from layer6.dataset_audit import audit
from layer6.model import CLASSES, Layer6Models


class Layer6Tests(unittest.TestCase):
    def _row(self, label: str, replica: int) -> dict:
        registry = FeatureRegistry()
        names = list(registry.feature_names)
        # Deterministic class-specific signatures are sufficient to exercise the contract.
        class_index = CLASSES.index(label)
        features = {name: float(class_index * 10 + replica) for name in names}
        return {"window_start_ns": replica, "window_end_ns": replica + 1, "features": features,
                "feature_valid": {name: True for name in names}, "feature_schema_hash": registry.schema_hash(names),
                "ground_truth_label": label, "fault_active": label != "nominal",
                "maintenance_outcome": "healthy" if label == "nominal" else "fault_injected_source"}

    def test_fit_predict_and_schema_guard(self) -> None:
        train = [self._row(label, replica) for label in CLASSES for replica in range(3)]
        calibration = [self._row(label, replica + 20) for label in CLASSES for replica in range(2)]
        model = Layer6Models.fit(train, calibration, [row for row in train if row["ground_truth_label"] == "nominal"], model_version="test")
        result = model.predict(calibration[0])
        self.assertEqual(result.feature_schema_hash, calibration[0]["feature_schema_hash"])
        self.assertIn(result.predicted_class, CLASSES)
        self.assertTrue(result.shap_top_features)
        self.assertEqual(len(result.fault_flags), 8)
        self.assertEqual(len(result.shap_by_class), 8)
        self.assertTrue(all("_raw_" not in name for name in model.if_feature_names))

    def test_source_audit_detects_missing_nominal_sorties(self) -> None:
        source = Path("layer 3/datasets for layer 3/aero_engine_fault_injected_dataset.csv")
        if not source.is_file():
            self.skipTest("optional legacy source dataset is not included in this repository")
        report = audit(source)
        self.assertEqual(report["rows"], 87500)
        self.assertFalse(report["has_explicit_nominal_sortie_label"])
        self.assertEqual(report["sorties_per_fault_class"]["misfire"], 20)

    def _artifact(self, directory: Path) -> tuple[Path, dict]:
        train = [self._row(label, replica) for label in CLASSES for replica in range(3)]
        calibration = [self._row(label, replica + 20) for label in CLASSES for replica in range(2)]
        model = Layer6Models.fit(train, calibration, [row for row in train if row["ground_truth_label"] == "nominal"], model_version="test-v1")
        metadata = ArtifactMetadata(2, model.model_version, "2026-01-01T00:00:00+00:00", "layer6.model",
            model.feature_schema_hash, model.feature_names, tuple(model.classifier), "a" * 64, "b" * 64, "c" * 64,
            len(train), len(calibration), 3, "sigmoid_per_fault_held_out_sorties", "verified_healthy_nominal_only", 42)
        path = directory / "layer6.joblib"; save_artifact(path, Layer6Artifact(metadata, model))
        return path, calibration[0]

    def test_artifact_loads_from_root_and_preserves_inference_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path, window = self._artifact(Path(raw))
            artifact = load_artifact(path)
            output = artifact.model.predict(window)
            self.assertEqual(output.model_version, "test-v1")
            self.assertEqual(set(output.class_probabilities), set(CLASSES))
            self.assertIsInstance(output.novel_degradation, bool)
            self.assertTrue(output.shap_subsystem)

    def test_artifact_loads_from_another_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path, _ = self._artifact(Path(raw))
            command = "from layer6.artifact import load_artifact; print(load_artifact(r'%s').metadata.model_version)" % path
            environment = dict(os.environ); environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            result = subprocess.run([sys.executable, "-c", command], cwd=tempfile.gettempdir(), env=environment,
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), "test-v1")

    def test_incompatible_artifact_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path, _ = self._artifact(Path(raw))
            artifact = load_artifact(path)
            broken = Layer6Artifact(replace(artifact.metadata, feature_schema_hash="invalid"), artifact.model)
            joblib.dump(broken, path)
            with self.assertRaises(ArtifactCompatibilityError):
                load_artifact(path)

    def test_legacy_artifact_is_migrated_to_canonical_module(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw); canonical_path, _ = self._artifact(directory)
            model = load_artifact(canonical_path).model
            legacy_path = directory / "legacy.joblib"; module = importlib.import_module("layer6.model")
            original_name, original_alias = Layer6Models.__module__, sys.modules.get("model")
            try:
                Layer6Models.__module__ = "model"; sys.modules["model"] = module; joblib.dump(model, legacy_path)
            finally:
                Layer6Models.__module__ = original_name
                if original_alias is None: sys.modules.pop("model", None)
                else: sys.modules["model"] = original_alias
            metadata = ArtifactMetadata(2, "placeholder", "2026-01-01T00:00:00+00:00", "layer6.model", "", (), tuple(model.classifier),
                "a" * 64, "b" * 64, "c" * 64, 1, 1, 1, "sigmoid_per_fault_held_out_sorties", "verified_healthy_nominal_only", 42)
            migrated_path = directory / "migrated.joblib"
            migrate_legacy_artifact(legacy_path, migrated_path, metadata)
            self.assertEqual(load_artifact(migrated_path).model.model_version, "test-v1")


if __name__ == "__main__":
    unittest.main()
