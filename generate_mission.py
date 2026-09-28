"""
Mission Telemetry Recording Generator.

Generates recorded mission telemetry CSV files for replay testing and analysis.

Usage
-----
python generate_mission.py --mission endurance --duration 180 --output mission_001.csv
python generate_mission.py --mission high_altitude --duration 180 --output mission_002.csv
"""

import argparse
from pathlib import Path
import pandas as pd

from app.simulator import EngineSimulator
from app.mission_profiles import get_mission_profile

PROJECT_ROOT = Path(__file__).parent.resolve()
MISSIONS_DIR = PROJECT_ROOT / "data" / "missions"


def generate_mission_csv(
    mission_name: str = "endurance",
    duration_sec: float = 180.0,
    output_filename: str = "mission_001.csv",
    dt: float = 0.5,
) -> Path:
    """
    Simulate a UAV flight mission and record telemetry to CSV.
    """
    MISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MISSIONS_DIR / output_filename

    sim = EngineSimulator(
        dt=dt,
        seed=42,
        enable_fault=True,
        mission_profile=mission_name,
    )

    n_steps = int(duration_sec / dt)
    records = []

    print(f"Generating mission '{mission_name}' for {duration_sec}s ({n_steps} steps)...")

    for _ in range(n_steps):
        telem = sim.step()
        records.append({
            "timestamp": round(telem["timestamp"], 4),
            "rpm": round(telem["rpm"], 2),
            "throttle": round(telem["throttle"], 4),
            "egt": round(telem["egt"], 2),
            "cht": round(telem["cht"], 2),
            "oil_temp": round(telem["oil_temp"], 2),
            "map": round(telem["map"], 2),
            "oil_press": round(telem["oil_press"], 2),
            "fuel_flow": round(telem["fuel_flow"], 2),
            "iat": round(telem["iat"], 2),
            "vibration": round(telem["vibration"], 3),
            "fault_active": telem["fault_active"],
        })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved mission recording ({len(df)} rows) to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate UAV Mission Telemetry Recording")
    parser.add_argument("--mission", type=str, default="nominal_cruise", help="Mission profile name")
    parser.add_argument("--duration", type=float, default=120.0, help="Duration in seconds (default: 120s)")
    parser.add_argument("--output", type=str, default="mission_001_nominal_cruise.csv", help="Output filename")
    
    args = parser.parse_args()

    # If called without arguments, generate all 4 standard mission files
    profiles = [
        ("nominal_cruise", "mission_001_nominal_cruise.csv"),
        ("thermal_overload", "mission_002_thermal_overload.csv"),
        ("lubrication_issue", "mission_003_lubrication_issue.csv"),
        ("exhaust_leak", "mission_004_exhaust_leak.csv"),
    ]

    for p_name, p_out in profiles:
        generate_mission_csv(
            mission_name=p_name,
            duration_sec=120.0,
            output_filename=p_out,
        )


if __name__ == "__main__":
    main()
