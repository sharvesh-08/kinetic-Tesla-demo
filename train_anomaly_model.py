"""
Training Script for Isolation Forest Anomaly Detection Model.

This script performs the complete one-time offline ML preparation pipeline:
1. Ensures data/ and models/ directories exist.
2. Generates a clean normal engine telemetry dataset (fault_active=False).
3. Computes physics-based expected values and residuals for all samples.
4. Validates the dataset (checks for NaNs, infs, correct feature types, and 0% fault samples).
5. Saves the normal dataset to data/normal_telemetry.csv.
6. Trains an IsolationForest model on the 7 core feature channels.
7. Computes decision-function normalization parameters on normal data.
8. Saves the trained model bundle to models/isolation_forest_anomaly.pkl.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

from app.simulator import EngineSimulator
from app.physics_model import predict_expected

# ---------------------------------------------------------------------------
# Paths and Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

CSV_PATH = DATA_DIR / "normal_telemetry.csv"
MODEL_PATH = MODELS_DIR / "isolation_forest_anomaly.pkl"

FEATURE_COLS = [
    "rpm",
    "throttle",
    "egt",
    "cht",
    "oil_temp",
    "map",
    "oil_press",
    "fuel_flow",
    "iat",
    "vibration",
    "res_egt",
    "res_cht",
    "res_oil_press",
    "res_vibration",
]


def generate_normal_data(n_samples: int = 3000, dt: float = 0.5) -> pd.DataFrame:
    """
    Generate normal telemetry dataset using EngineSimulator with enable_fault=False.
    Covers the full throttle range [0.2, 0.95] under healthy operation.
    """
    records = []

    # Part 1: Dynamic mission sweep (1500 samples)
    sim = EngineSimulator(dt=dt, seed=42, enable_fault=False)
    for _ in range(1500):
        t_data = sim.step()
        exp_state = predict_expected(rpm=t_data["rpm"], throttle=t_data["throttle"], ambient_temp=25.0)
        res_egt = t_data["egt"] - exp_state.egt_expected
        res_cht = t_data["cht"] - exp_state.cht_expected
        res_oil_press = t_data["oil_press"] - exp_state.oil_press_expected
        res_vib = t_data["vibration"] - exp_state.vibration_expected
        records.append({
            "timestamp": t_data["timestamp"],
            "rpm": t_data["rpm"],
            "throttle": t_data["throttle"],
            "egt": t_data["egt"],
            "cht": t_data["cht"],
            "oil_temp": t_data["oil_temp"],
            "map": t_data["map"],
            "oil_press": t_data["oil_press"],
            "fuel_flow": t_data["fuel_flow"],
            "iat": t_data["iat"],
            "vibration": t_data["vibration"],
            "fault_active": False,
            "egt_expected": round(exp_state.egt_expected, 2),
            "cht_expected": round(exp_state.cht_expected, 2),
            "res_egt": round(res_egt, 2),
            "res_cht": round(res_cht, 2),
            "res_oil_press": round(res_oil_press, 2),
            "res_vibration": round(res_vib, 3),
        })

    # Part 2: Manual throttle grid sweep across [0.20, 0.95] (1500 samples)
    np.random.seed(123)
    for thr in np.linspace(0.20, 0.95, 25):
        sim_man = EngineSimulator(dt=dt, seed=int(thr * 1000), enable_fault=False)
        sim_man.set_manual_control(throttle=float(thr), injected_fault="NONE")
        for _ in range(60):
            t_data = sim_man.step()
            exp_state = predict_expected(rpm=t_data["rpm"], throttle=t_data["throttle"], ambient_temp=25.0)
            res_egt = t_data["egt"] - exp_state.egt_expected
            res_cht = t_data["cht"] - exp_state.cht_expected
            res_oil_press = t_data["oil_press"] - exp_state.oil_press_expected
            res_vib = t_data["vibration"] - exp_state.vibration_expected
            records.append({
                "timestamp": t_data["timestamp"],
                "rpm": t_data["rpm"],
                "throttle": t_data["throttle"],
                "egt": t_data["egt"],
                "cht": t_data["cht"],
                "oil_temp": t_data["oil_temp"],
                "map": t_data["map"],
                "oil_press": t_data["oil_press"],
                "fuel_flow": t_data["fuel_flow"],
                "iat": t_data["iat"],
                "vibration": t_data["vibration"],
                "fault_active": False,
                "egt_expected": round(exp_state.egt_expected, 2),
                "cht_expected": round(exp_state.cht_expected, 2),
                "res_egt": round(res_egt, 2),
                "res_cht": round(res_cht, 2),
                "res_oil_press": round(res_oil_press, 2),
                "res_vibration": round(res_vib, 3),
            })

    return pd.DataFrame(records)


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate quality and integrity of normal dataset before training.
    """
    print("--- Validating Dataset ---")
    assert not df.empty, "ERROR: Dataset is empty!"
    assert df["fault_active"].sum() == 0, "ERROR: Dataset contains fault-injected samples!"
    
    for col in FEATURE_COLS:
        assert col in df.columns, f"ERROR: Missing column {col}"
        assert not df[col].isnull().any(), f"ERROR: NaN values found in column {col}"
        assert not np.isinf(df[col]).any(), f"ERROR: Infinite values found in column {col}"
        assert pd.api.types.is_numeric_dtype(df[col]), f"ERROR: Column {col} is not numeric"

    print("[OK] Dataset validation passed!")
    print(f"  Total normal samples: {len(df)}")
    print(f"  Fault samples included: {df['fault_active'].sum()}")
    print(f"  Features verified: {FEATURE_COLS}")


def main():
    print("==================================================")
    print("      UAV ENGINE DIGITAL TWIN ML TRAINING         ")
    print("==================================================")

    # 1. Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Generate clean normal dataset
    print("\n[1/4] Generating normal telemetry dataset...")
    df = generate_normal_data(n_samples=2400, dt=0.5)  # 20 minutes of simulation
    
    # 3. Validate dataset
    print("\n[2/4] Validating dataset...")
    validate_dataset(df)

    # 4. Save CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved normal dataset to: {CSV_PATH}")

    # 5. Extract feature matrix X
    X = df[FEATURE_COLS].values

    # 6. Train Isolation Forest
    print("\n[3/4] Training Isolation Forest model...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    print("[OK] Model training complete.")

    # Compute raw anomaly decision scores for normalization baseline
    # decision_function: positive for inliers (normal), negative for outliers (anomalies)
    raw_decisions = model.decision_function(X)
    # Raw inverted score: higher value = more anomalous
    raw_scores = -raw_decisions
    score_mean = float(np.mean(raw_scores))
    score_std = float(np.std(raw_scores))
    score_min = float(np.min(raw_scores))
    score_99 = float(np.percentile(raw_scores, 99))

    print(f"  Normal baseline score range: min={score_min:.4f}, mean={score_mean:.4f}, std={score_std:.4f}, p99={score_99:.4f}")

    # 7. Save model artifact bundle
    print("\n[4/4] Saving model bundle...")
    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "score_mean": score_mean,
        "score_std": score_std,
        "score_min": score_min,
        "score_99": score_99,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"[OK] Model bundle saved successfully to: {MODEL_PATH}")
    print("\n==================================================")
    print("          TRAINING COMPLETED SUCCESSFULLY          ")
    print("==================================================")


if __name__ == "__main__":
    main()
