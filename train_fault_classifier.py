"""
Random Forest Fault Classifier Training Script.

Trains a Random Forest classifier on data/fault_classification.csv to predict:
  - NORMAL
  - OVERHEATING
  - LUBRICATION_ISSUE
  - EXHAUST_LEAK

Saves the model bundle to models/fault_classifier.pkl.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "fault_classification.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fault_classifier.pkl"

FEATURE_NAMES = [
    "rpm", "throttle", "egt", "cht", "oil_temp", "map", "oil_press", "fuel_flow", "iat", "vibration",
    "egt_residual", "cht_residual", "oil_press_residual", "vibration_residual"
]
TARGET_NAME = "label"


def train_fault_classifier() -> None:
    """Load dataset, train Random Forest, evaluate metrics, and save model bundle."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found at {DATA_PATH}. Run generate_fault_dataset.py first.")

    print(f"[INFO] Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Validate columns
    for col in FEATURE_NAMES + [TARGET_NAME]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from dataset!")

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    # Stratified Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[INFO] Training RandomForestClassifier on {len(X_train)} samples...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"FAULT CLASSIFIER EVALUATION RESULTS (Accuracy: {acc:.4f})")
    print("=" * 50)

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=clf.classes_))
    print(f"Classes: {list(clf.classes_)}")

    # Save model bundle
    bundle = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "classes": list(clf.classes_),
        "accuracy": float(acc),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n[OK] Model successfully saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_fault_classifier()
