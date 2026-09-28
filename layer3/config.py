"""Transport and model configuration for Layer 3."""

from __future__ import annotations

from dataclasses import dataclass
import os
from dataclasses import field
from pathlib import Path


@dataclass(frozen=True)
class Layer3Config:
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"))
    input_stream: str = "engine:synced:ekf:frames"
    output_stream: str = "engine:physics:predictions"
    consumer_group: str = "layer3-physics"
    consumer_name: str = "edge-layer3"
    claim_idle_ms: int = 5000
    read_count: int = 100
    block_ms: int = 250
    artifact_root: Path = Path(__file__).resolve().parents[1] / "layer 3"
    # Required only to convert MVEM fuel mass flow to the Layer 1 L/h channel.
    # It intentionally has no default because fuel density is installation-specific.
    fuel_density_kg_L: float | None = field(default_factory=lambda:
        float(os.environ["FUEL_DENSITY_KG_L"]) if os.environ.get("FUEL_DENSITY_KG_L") else 0.72)

    def __post_init__(self):
        if self.fuel_density_kg_L is not None:
            import math
            if not math.isfinite(self.fuel_density_kg_L) or not .5 <= self.fuel_density_kg_L <= 1.2:
                raise ValueError("FUEL_DENSITY_KG_L must be finite and between 0.5 and 1.2 kg/L")
