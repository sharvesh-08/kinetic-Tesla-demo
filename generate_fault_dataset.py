"""
Synthetic Fault Classification Dataset Generator.

Generates data/fault_classification.csv containing 4,000 synthetic samples
across four fault classes:
  - NORMAL (1000 samples)
  - OVERHEATING (1000 samples)
  - LUBRICATION_ISSUE (1000 samples)
  - EXHAUST_LEAK (1000 samples)

Physics-consistent features:
  [rpm, throttle, egt, cht, oil_temp, egt_residual, cht_residual, label]
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.physics_model import EnginePhysicsModel


def generate_fault_dataset(
    n_samples_per_class: int = 1000,
    seed: int = 42,
    output_csv: Path | str = PROJECT_ROOT / "data" / "fault_classification.csv",
) -> pd.DataFrame:
    """Generate synthetic telemetry dataset for multi-class fault classification across 10 channels."""
    np.random.seed(seed)
    physics = EnginePhysicsModel(ambient_temp=25.0)

    records = []

    classes = ["NORMAL", "OVERHEATING", "LUBRICATION_ISSUE", "EXHAUST_LEAK", "FUEL_RESTRICTION", "CYLINDER_MISFIRE"]

    for fault_type in classes:
        for _ in range(n_samples_per_class):
            throttle = float(np.random.uniform(0.3, 0.95))
            rpm_drag = 0.0
            egt_bias = 0.0
            cht_bias = 0.0
            oil_temp_bias = 0.0
            map_bias = 0.0
            oil_press_bias = 0.0
            fuel_flow_bias = 0.0
            iat_bias = 0.0
            vib_bias = 0.0

            if fault_type == "OVERHEATING":
                egt_bias = float(np.random.uniform(25.0, 50.0))
                cht_bias = float(np.random.uniform(15.0, 30.0))
                iat_bias = float(np.random.uniform(8.0, 18.0))
            elif fault_type == "LUBRICATION_ISSUE":
                oil_temp_bias = float(np.random.uniform(18.0, 35.0))
                oil_press_bias = float(np.random.uniform(-35.0, -18.0))
                vib_bias = float(np.random.uniform(0.8, 1.8))
                rpm_drag = 150.0
            elif fault_type == "EXHAUST_LEAK":
                egt_bias = float(np.random.uniform(-80.0, -40.0))
                map_bias = float(np.random.uniform(-4.0, -1.5))
            elif fault_type == "FUEL_RESTRICTION":
                fuel_flow_bias = float(np.random.uniform(-6.0, -3.0))
                egt_bias = float(np.random.uniform(40.0, 80.0)) # lean burn spike
                rpm_drag = 180.0
            elif fault_type == "CYLINDER_MISFIRE":
                vib_bias = float(np.random.uniform(2.5, 4.5))
                egt_bias = float(np.random.uniform(-50.0, -25.0))
                rpm_drag = 250.0

            rpm_true = max(1000.0, 2000.0 + throttle * 2500.0 - rpm_drag + float(np.random.normal(0, 30.0)))
            expected = physics.predict(rpm=rpm_true, throttle=throttle, ambient_temp=25.0)

            egt_meas = max(0.0, expected.egt_expected + egt_bias + float(np.random.normal(0.0, 3.0)))
            cht_meas = max(0.0, expected.cht_expected + cht_bias + float(np.random.normal(0.0, 2.0)))
            oil_meas = max(0.0, expected.oil_temp_expected + oil_temp_bias + float(np.random.normal(0.0, 1.5)))
            map_meas = max(5.0, expected.map_expected + map_bias + float(np.random.normal(0.0, 0.2)))
            oil_press_meas = max(0.0, expected.oil_press_expected + oil_press_bias + float(np.random.normal(0.0, 1.2)))
            fuel_flow_meas = max(0.0, expected.fuel_flow_expected + fuel_flow_bias + float(np.random.normal(0.0, 0.15)))
            iat_meas = expected.iat_expected + iat_bias + float(np.random.normal(0.0, 0.8))
            vib_meas = max(0.0, expected.vibration_expected + vib_bias + float(np.random.normal(0.0, 0.04)))

            egt_res = egt_meas - expected.egt_expected
            cht_res = cht_meas - expected.cht_expected
            oil_press_res = oil_press_meas - expected.oil_press_expected
            vib_res = vib_meas - expected.vibration_expected

            records.append({
                "rpm": round(rpm_true, 2),
                "throttle": round(throttle, 4),
                "egt": round(egt_meas, 2),
                "cht": round(cht_meas, 2),
                "oil_temp": round(oil_meas, 2),
                "map": round(map_meas, 2),
                "oil_press": round(oil_press_meas, 2),
                "fuel_flow": round(fuel_flow_meas, 2),
                "iat": round(iat_meas, 2),
                "vibration": round(vib_meas, 3),
                "egt_residual": round(egt_res, 2),
                "cht_residual": round(cht_res, 2),
                "oil_press_residual": round(oil_press_res, 2),
                "vibration_residual": round(vib_res, 3),
                "label": fault_type,
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[OK] Generated {len(df)} samples into {out_path}")
    print("Class distribution:\n", df["label"].value_counts())
    return df


if __name__ == "__main__":
    generate_fault_dataset()
