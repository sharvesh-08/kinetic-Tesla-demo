"""The two permitted Layer 6 models and their safe inference contract.

TreeSHAP is obtained from LightGBM's native ``pred_contrib`` implementation; SHAP's
optional Python package is intentionally not required on the edge device.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.frozen import FrozenEstimator

from layer5.registry import FeatureRegistry


CLASSES = ("nominal", "misfire", "injector_abnormality", "cooling_degradation", "lubrication_issue",
           "sensor_drift_failure", "combustion_instability", "overheating_trend", "abnormal_vibration")
FAULT_CLASSES = tuple(name for name in CLASSES if name != "nominal")


def _fault_labels(row: dict[str, Any]) -> set[str]:
    raw = row.get("fault_labels") or row["ground_truth_label"]
    return set(raw.split("|") if isinstance(raw, str) else raw) - {"nominal"}


@dataclass(frozen=True)
class BlackBoxOutput:
    window_start_ns: int
    window_end_ns: int
    predicted_class: str
    class_probabilities: dict[str, float]
    fault_flags: dict[str, dict[str, float | bool]]
    shap_by_class: dict[str, dict[str, float]]
    shap_top_features_by_class: dict[str, list[tuple[str, float]]]
    sensor_drift_bias_subsystem: str | None
    confidence: float
    anomaly_score: float
    novel_degradation: bool
    shap_subsystem: dict[str, float]
    shap_top_features: list[tuple[str, float]]
    model_version: str
    feature_schema_hash: str
    degraded_inference: bool
    anomaly_detected: bool = False
    training_scope: str = "verified_healthy_nominal_only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceConfig:
    # These are validation-calibrated deployment values, not universal constants.
    low_confidence_threshold: float = 0.55
    high_anomaly_threshold: float = 0.80
    shap_top_n: int = 8
    isolation_forest_contamination: float | str = "auto"
    default_fault_threshold: float = 0.5


class Layer6Models:
    """Frozen LightGBM + healthy-only Isolation Forest inference artifact."""

    def __init__(self, classifier: dict[str, LGBMClassifier], probability_calibrator: dict[str, CalibratedClassifierCV],
                 anomaly_detector: IsolationForest, feature_names: list[str], if_feature_names: list[str],
                 feature_schema_hash: str, healthy_score_low: float, healthy_score_high: float,
                 model_version: str, config: InferenceConfig | None = None, registry: FeatureRegistry | None = None,
                 training_scope: str = "verified_healthy_nominal_only",
                 fault_thresholds: dict[str, float] | None = None) -> None:
        self.classifier = classifier
        self.probability_calibrator = probability_calibrator
        self.anomaly_detector = anomaly_detector
        self.feature_names = tuple(feature_names)
        self.if_feature_names = tuple(if_feature_names)
        self.fault_thresholds = fault_thresholds or {name: (config or InferenceConfig()).default_fault_threshold for name in FAULT_CLASSES}
        self.feature_schema_hash = feature_schema_hash
        self.healthy_score_low = healthy_score_low
        self.healthy_score_high = healthy_score_high
        self.model_version = model_version
        self.training_scope = training_scope
        self.config = config or InferenceConfig()
        self.registry = registry or FeatureRegistry()
        self.registry.require_schema(list(self.feature_names), self.feature_schema_hash)

    @classmethod
    def fit(cls, rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]], healthy_rows: list[dict[str, Any]], *, model_version: str, seed: int = 42,
            n_estimators: int = 250,
            config: InferenceConfig | None = None, registry: FeatureRegistry | None = None,
            allow_synthetic_nominal: bool = False) -> "Layer6Models":
        """Fit only from already split, sortie-independent training data.

        ``calibration_rows`` must be a separate, sortie-independent held-out split.
        ``healthy_rows`` is a separate nominal-only population used for Isolation Forest;
        it must never contain a fault-active window. Supplied pre-fault candidates are not
        equivalent to independently verified healthy maintenance sorties.
        """
        if allow_synthetic_nominal and "synthetic-dev" not in model_version:
            raise ValueError("Synthetic nominal training requires an explicit synthetic-dev model version")
        if not rows:
            raise ValueError("No FeatureWindow rows supplied")
        reg = registry or FeatureRegistry()
        hashes = {str(row["feature_schema_hash"]) for row in rows}
        if len(hashes) != 1:
            raise ValueError("Training rows have more than one FeatureWindow schema hash")
        names = sorted(rows[0]["features"])
        reg.require_schema(names, next(iter(hashes)))
        if any(sorted(row["features"]) != names for row in rows):
            raise ValueError("Training rows do not have identical feature names")
        labels = np.asarray([row["ground_truth_label"] for row in rows])
        unknown = set(labels) - set(CLASSES)
        if unknown:
            raise ValueError(f"Unknown LightGBM labels: {sorted(unknown)}")
        multilabel_unknown = {name for row in (*rows, *calibration_rows)
                              for name in _fault_labels(row) if name not in FAULT_CLASSES}
        if multilabel_unknown:
            raise ValueError(f"Unknown multi-label fault targets: {sorted(multilabel_unknown)}")
        matrix = np.asarray([[float(row["features"][name]) for name in names] for row in rows], dtype=float)
        nominal = np.asarray([label == "nominal" for label in labels])
        if not nominal.any() or nominal.all():
            raise ValueError("Need both nominal and fault rows for classifier training")
        # LightGBM's balanced weighting protects rare fault recall without resampling windows.
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        classifiers: dict[str, LGBMClassifier] = {}
        if not calibration_rows:
            raise ValueError("A held-out calibration split is required")
        if any(str(row["feature_schema_hash"]) != next(iter(hashes)) or sorted(row["features"]) != names
               for row in calibration_rows):
            raise ValueError("Calibration rows do not match the training feature schema")
        calibration_matrix = np.asarray([[float(row["features"][name]) for name in names] for row in calibration_rows], dtype=float)
        calibration_labels = np.asarray([row["ground_truth_label"] for row in calibration_rows])
        if set(calibration_labels) - set(CLASSES):
            raise ValueError("Calibration set includes a class absent from training")
        calibrators: dict[str, CalibratedClassifierCV] = {}
        thresholds: dict[str, float] = {}
        indices = np.arange(len(calibration_rows))
        for fault in FAULT_CLASSES:
            training_target = np.asarray([int(fault in _fault_labels(row)) for row in rows])
            calibration_target = np.asarray([int(fault in _fault_labels(row)) for row in calibration_rows])
            if len(set(training_target)) != 2 or len(set(calibration_target)) != 2:
                raise ValueError(f"Fault {fault} requires positive and negative independent runs in training and calibration")
            head = LGBMClassifier(objective="binary", class_weight="balanced", random_state=seed,
                                  n_estimators=n_estimators, max_depth=6, num_leaves=31,
                                  min_child_samples=max(5, min(30, int(np.sum(training_target) // 2))),
                                  verbosity=-1, n_jobs=1)
            head.fit(matrix, training_target)
            calibrator = CalibratedClassifierCV(FrozenEstimator(head), method="sigmoid",
                                                cv=[(indices, indices)], ensemble=False)
            calibrator.fit(calibration_matrix, calibration_target)
            scores = calibrator.predict_proba(calibration_matrix)[:, 1]
            # Per-head threshold is selected on the held-out calibration sorties.
            candidates = np.linspace(0.1, 0.9, 17)
            def f1(threshold: float) -> float:
                predicted = scores >= threshold
                tp = int(np.sum(predicted & (calibration_target == 1)))
                fp = int(np.sum(predicted & (calibration_target == 0)))
                fn = int(np.sum(~predicted & (calibration_target == 1)))
                return 2 * tp / max(2 * tp + fp + fn, 1)
            thresholds[fault] = float(max(candidates, key=f1))
            classifiers[fault] = head
            calibrators[fault] = calibrator
        # This assertion must remain immediately before the fit: faults must never normalize
        # the novelty detector's idea of healthy behaviour.
        if not healthy_rows:
            raise ValueError("A verified healthy dataset is required for Isolation Forest")
        if any(str(row["feature_schema_hash"]) != next(iter(hashes)) or sorted(row["features"]) != names for row in healthy_rows):
            raise ValueError("Healthy rows do not match the training feature schema")
        if any(row["ground_truth_label"] != "nominal" or bool(row.get("fault_active", False))
               for row in healthy_rows):
            raise AssertionError("Isolation Forest input includes a fault-active window")
        if not allow_synthetic_nominal and any(row.get("maintenance_outcome") != "healthy" for row in healthy_rows):
            raise ValueError("Verified healthy maintenance outcomes are required; synthetic development training must be explicit")
        if_names = [name for name in names if "_raw_" not in name]
        healthy = np.asarray([[float(row["features"][name]) for name in if_names] for row in healthy_rows], dtype=float)
        detector = IsolationForest(n_estimators=200, contamination=(config or InferenceConfig()).isolation_forest_contamination,
                                   random_state=seed, n_jobs=1)
        detector.fit(healthy)
        scores = detector.score_samples(healthy)
        low, high = np.percentile(scores, [1, 99])
        if not allow_synthetic_nominal:
            training_scope = "verified_healthy_nominal_only"
        elif any(row.get("source_kind") == "joint_feature_window_augmentation" for row in healthy_rows):
            training_scope = "synthetic_window_augmentation_unverified"
        elif any(row.get("maintenance_outcome") == "synthetic_prefault_block_replay" for row in healthy_rows):
            training_scope = "synthetic_prefault_block_replay"
        else:
            training_scope = "synthetic_nominal_unverified"
        return cls(classifiers, calibrators, detector, names, if_names, next(iter(hashes)), float(low), float(high), model_version, config, reg,
                   training_scope, thresholds)

    def _subsystem(self, feature_name: str) -> str:
        return self.registry.spec(feature_name).subsystem

    def predict(self, window: dict[str, Any]) -> BlackBoxOutput:
        if window.get("feature_schema_hash") != self.feature_schema_hash:
            raise ValueError("Incoming FeatureWindow schema hash differs from model artifact")
        names = sorted(window["features"])
        self.registry.require_schema(names, self.feature_schema_hash)
        if tuple(names) != self.feature_names:
            raise ValueError("Feature order/name mismatch with model artifact")
        vector = np.asarray([[float(window["features"][name]) for name in self.feature_names]], dtype=float)
        fault_probabilities = {name: float(self.probability_calibrator[name].predict_proba(vector)[0, 1])
                               for name in FAULT_CLASSES}
        nominal_probability = float(np.prod([1 - value for value in fault_probabilities.values()]))
        probabilities = {"nominal": nominal_probability, **fault_probabilities}
        active = {name: probability >= self.fault_thresholds[name]
                  for name, probability in fault_probabilities.items()}
        predicted = max((name for name in FAULT_CLASSES if active[name]),
                        key=lambda name: fault_probabilities[name], default="nominal")
        confidence = probabilities[predicted]
        if_vector = np.asarray([[float(window["features"][name]) for name in self.if_feature_names]], dtype=float)
        healthy_score = float(self.anomaly_detector.score_samples(if_vector)[0])
        # Smoothly map the healthy 1st/99th percentile anchors instead of hard
        # clipping every score below the low anchor to exactly 1.0. A score of
        # 1.0 is reserved for the asymptotic extreme; normal variation remains
        # visible to the dashboard and downstream fusion.
        score_span = max(self.healthy_score_high - self.healthy_score_low, 1e-12)
        normalized_distance = (self.healthy_score_high - healthy_score) / score_span
        anomaly = float(np.clip(0.5 + 0.5 * np.tanh(1.5 * (normalized_distance - 0.5)), 0.001, 0.999))
        shap_by_class: dict[str, dict[str, float]] = {}
        shap_top_features_by_class: dict[str, list[tuple[str, float]]] = {}
        sensor_drift_bias_subsystem: str | None = None
        for fault, head in self.classifier.items():
            # Binary LightGBM pred_contrib explains the raw margin for the positive class.
            values = np.asarray(head.booster_.predict(vector, pred_contrib=True, num_threads=1))[0, :-1]
            by_subsystem: dict[str, float] = {}
            for name, value in zip(self.feature_names, values, strict=True):
                subsystem = self._subsystem(name)
                by_subsystem[subsystem] = by_subsystem.get(subsystem, 0.0) + float(value)
            shap_by_class[fault] = by_subsystem
            ordered = sorted(zip(self.feature_names, (float(value) for value in values), strict=True),
                             key=lambda item: abs(item[1]), reverse=True)
            shap_top_features_by_class[fault] = ordered[:self.config.shap_top_n]
            if fault == "sensor_drift_failure":
                bias_attributions = [(name, float(value)) for name, value in zip(self.feature_names, values, strict=True)
                                     if "bias_mean" in name and name.startswith("resid_")]
                if bias_attributions:
                    strongest = max(bias_attributions, key=lambda item: abs(item[1]))
                    if abs(strongest[1]) > 0:
                        sensor_drift_bias_subsystem = self._subsystem(strongest[0])
        explanation_class = predicted if predicted != "nominal" else max(FAULT_CLASSES, key=fault_probabilities.get)
        by_subsystem = shap_by_class[explanation_class]
        ordered = shap_top_features_by_class[explanation_class]
        valid = window.get("feature_valid", {})
        degraded = any(not bool(valid.get(name, False)) for name in self.feature_names) or any(
            float(window["features"][self.registry.summary_name(channel, "availability")]) < 1.0
            for channel in self.registry.channels)
        return BlackBoxOutput(
            window_start_ns=int(window["window_start_ns"]), window_end_ns=int(window["window_end_ns"]),
            predicted_class=predicted, class_probabilities=probabilities,
            fault_flags={name: {"active": active[name], "confidence": fault_probabilities[name]}
                         for name in FAULT_CLASSES},
            shap_by_class=shap_by_class, shap_top_features_by_class=shap_top_features_by_class,
            sensor_drift_bias_subsystem=sensor_drift_bias_subsystem,
            confidence=confidence, anomaly_score=anomaly,
            novel_degradation=max(fault_probabilities.values()) < self.config.low_confidence_threshold and anomaly >= self.config.high_anomaly_threshold,
            shap_subsystem=by_subsystem, shap_top_features=ordered[:self.config.shap_top_n],
            model_version=self.model_version, feature_schema_hash=self.feature_schema_hash, degraded_inference=degraded,
            anomaly_detected=anomaly >= self.config.high_anomaly_threshold,
            training_scope=self.training_scope,
        )
