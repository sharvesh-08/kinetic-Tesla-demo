"""One-time canonical re-export of a legacy Layer 6 raw joblib artifact."""
from __future__ import annotations
import argparse
from pathlib import Path
from layer6.artifact import ArtifactMetadata, migrate_legacy_artifact
from layer6.model import CLASSES

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_artifact", type=Path); parser.add_argument("output_artifact", type=Path)
    parser.add_argument("--training-sha256", required=True); parser.add_argument("--healthy-sha256", required=True)
    parser.add_argument("--calibration-sha256", required=True); parser.add_argument("--created-at", required=True)
    parser.add_argument("--version", default="layer6-migrated-legacy")
    args = parser.parse_args()
    # Source datasets are not recoverable from an old raw pickle; explicit hashes make that
    # limitation visible rather than inventing provenance.
    metadata = ArtifactMetadata(1, args.version, args.created_at, "layer6.model", "", (), CLASSES,
        args.training_sha256, args.healthy_sha256, args.calibration_sha256, 0, 0, 1,
        "isotonic_held_out_sorties", "verified_healthy_nominal_only", 42)
    # Populate schema/version directly from the legacy model inside migration would be less
    # transparent; retraining is recommended for this repository's current artifact.
    migrate_legacy_artifact(args.legacy_artifact, args.output_artifact, metadata)

if __name__ == "__main__": main()
