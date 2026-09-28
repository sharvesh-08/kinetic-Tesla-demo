"""Versioned, package-safe Layer 6 artifact storage and loading."""

from __future__ import annotations

import hashlib
import importlib
import joblib
import sys
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from layer5.registry import FeatureRegistry, FeatureSchemaMismatch
from layer6.model import FAULT_CLASSES, Layer6Models

ARTIFACT_FORMAT_VERSION = 2
CANONICAL_MODEL_MODULE = "layer6.model"


class ArtifactLoadError(RuntimeError):
    """The artifact cannot be loaded safely."""


class ArtifactCompatibilityError(ArtifactLoadError):
    """The artifact loaded but is incompatible with the live Layer 6 contract."""


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_format_version: int
    model_version: str
    created_at: str
    model_module: str
    feature_schema_hash: str
    feature_names: tuple[str, ...]
    class_list: tuple[str, ...]
    training_dataset_sha256: str
    healthy_dataset_sha256: str
    calibration_split_sha256: str
    training_rows: int
    calibration_rows: int
    healthy_rows: int
    calibration_method: str
    isolation_forest_population: str
    seed: int


@dataclass(frozen=True)
class Layer6Artifact:
    metadata: ArtifactMetadata
    model: Layer6Models


def repository_root() -> Path:
    """Return the repository root independently of the calling process directory."""
    return Path(__file__).resolve().parents[1]


def default_artifact_path() -> Path:
    import os
    configured = os.environ.get("LAYER6_ARTIFACT")
    return Path(configured) if configured else repository_root() / "models" / "layer6-v3-synthetic-dev.joblib"


def resolve_artifact_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit path or the repository's canonical artifact location."""
    candidate = default_artifact_path() if path is None else Path(path).expanduser()
    return candidate.resolve(strict=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(model: Layer6Models, *, training_dataset: str | Path, healthy_dataset: str | Path,
                   calibration_split_sha256: str, training_rows: int, calibration_rows: int,
                   healthy_rows: int, seed: int) -> ArtifactMetadata:
    return ArtifactMetadata(
        artifact_format_version=ARTIFACT_FORMAT_VERSION, model_version=model.model_version,
        created_at=datetime.now(timezone.utc).isoformat(), model_module=CANONICAL_MODEL_MODULE,
        feature_schema_hash=model.feature_schema_hash, feature_names=model.feature_names,
        class_list=tuple(model.classifier),
        training_dataset_sha256=sha256_file(training_dataset), healthy_dataset_sha256=sha256_file(healthy_dataset),
        calibration_split_sha256=calibration_split_sha256, training_rows=training_rows,
        calibration_rows=calibration_rows, healthy_rows=healthy_rows, calibration_method="sigmoid_per_fault_held_out_sorties",
        isolation_forest_population=model.training_scope, seed=seed,
    )


def _validate(artifact: Layer6Artifact) -> None:
    metadata, model = artifact.metadata, artifact.model
    if metadata.artifact_format_version != ARTIFACT_FORMAT_VERSION:
        raise ArtifactCompatibilityError(f"Unsupported artifact format {metadata.artifact_format_version}")
    if metadata.model_module != CANONICAL_MODEL_MODULE or type(model).__module__ != CANONICAL_MODEL_MODULE:
        raise ArtifactCompatibilityError("Artifact was not serialized from canonical layer6.model")
    if not metadata.model_version or metadata.model_version != model.model_version:
        raise ArtifactCompatibilityError("Artifact model version is missing or mismatched")
    if tuple(metadata.feature_names) != tuple(model.feature_names):
        raise ArtifactCompatibilityError("Artifact feature names do not match the model")
    if metadata.feature_schema_hash != model.feature_schema_hash:
        raise ArtifactCompatibilityError("Artifact feature schema hash does not match the model")
    try:
        FeatureRegistry().require_schema(list(model.feature_names), model.feature_schema_hash)
    except FeatureSchemaMismatch as error:
        raise ArtifactCompatibilityError(str(error)) from error
    classifier_classes = tuple(model.classifier)
    if tuple(metadata.class_list) != classifier_classes or set(classifier_classes) != set(FAULT_CLASSES):
        raise ArtifactCompatibilityError("Artifact class list is incompatible with Layer 6")
    if metadata.calibration_method != "sigmoid_per_fault_held_out_sorties":
        raise ArtifactCompatibilityError("Artifact is missing required held-out confidence calibration")
    if set(model.fault_thresholds) != set(FAULT_CLASSES) or any(not 0 < value < 1 for value in model.fault_thresholds.values()):
        raise ArtifactCompatibilityError("Artifact is missing calibrated per-fault thresholds")
    if not model.if_feature_names or any("_raw_" in name for name in model.if_feature_names):
        raise ArtifactCompatibilityError("Isolation Forest must use summary/context features only")
    if (metadata.isolation_forest_population not in {
            "verified_healthy_nominal_only", "synthetic_prefault_block_replay",
            "synthetic_nominal_unverified", "synthetic_window_augmentation_unverified",
        }
            or metadata.isolation_forest_population != model.training_scope or metadata.healthy_rows <= 0):
        raise ArtifactCompatibilityError("Artifact is missing nominal-only Isolation Forest provenance")
    if any(not getattr(metadata, field) for field in ("training_dataset_sha256", "healthy_dataset_sha256", "calibration_split_sha256", "created_at")):
        raise ArtifactCompatibilityError("Artifact metadata is incomplete")


def save_artifact(path: str | Path, artifact: Layer6Artifact) -> Path:
    _validate(artifact)
    destination = resolve_artifact_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".joblib", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(artifact, temporary)
        load_artifact(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_artifact(path: str | Path | None = None) -> Layer6Artifact:
    source = resolve_artifact_path(path)
    if not source.is_file():
        raise ArtifactLoadError(f"Layer 6 artifact not found: {source}")
    try:
        artifact = joblib.load(source)
    except ModuleNotFoundError as error:
        if error.name == "model":
            raise ArtifactLoadError("Legacy artifact uses module 'model'. Run `python3 -m layer6.migrate_artifact <legacy> <output>` or retrain with `python3 -m layer6.train`.") from error
        raise ArtifactLoadError(f"Missing dependency while loading {source}: {error.name}") from error
    except Exception as error:
        raise ArtifactLoadError(f"Could not load Layer 6 artifact {source}: {error}") from error
    if not isinstance(artifact, Layer6Artifact):
        raise ArtifactCompatibilityError("Artifact is a legacy raw model, not a versioned Layer6Artifact")
    _validate(artifact)
    return artifact


@contextmanager
def _legacy_model_alias() -> Any:
    """One-time compatibility alias for artifacts written as module ``model``.

    This alias is never used by normal inference and does not depend on the current directory.
    """
    previous = sys.modules.get("model")
    sys.modules["model"] = importlib.import_module(CANONICAL_MODEL_MODULE)
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = previous


def migrate_legacy_artifact(legacy_path: str | Path, output_path: str | Path, metadata: ArtifactMetadata) -> Path:
    """Re-export a known legacy raw model with explicit provenance metadata."""
    source = resolve_artifact_path(legacy_path)
    if not source.is_file():
        raise ArtifactLoadError(f"Legacy artifact not found: {source}")
    with _legacy_model_alias():
        try:
            model = joblib.load(source)
        except Exception as error:
            raise ArtifactLoadError(f"Could not migrate legacy artifact {source}: {error}") from error
    if not isinstance(model, Layer6Models):
        raise ArtifactCompatibilityError("Legacy artifact does not contain a Layer6Models instance")
    if not isinstance(model.classifier, dict):
        raise ArtifactCompatibilityError("Single-label artifacts require retraining for the v3 multi-label feature contract")
    # A raw legacy pickle has no embedded provenance.  The caller supplies it explicitly;
    # model-derived fields are copied instead of trusting potentially stale CLI values.
    canonical_metadata = replace(metadata, model_version=model.model_version,
                                 feature_schema_hash=model.feature_schema_hash,
                                 feature_names=tuple(model.feature_names),
                                 class_list=tuple(model.classifier))
    return save_artifact(output_path, Layer6Artifact(metadata=canonical_metadata, model=model))


def metadata_json(artifact: Layer6Artifact) -> dict[str, Any]:
    return asdict(artifact.metadata)
