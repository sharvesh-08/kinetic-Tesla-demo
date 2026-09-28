"""
Mission Replay Engine for UAV Aero-Piston Engine Digital Twin.

Loads recorded mission CSV files, validates schema and data integrity,
and sequentially feeds telemetry rows through the Digital Twin Core.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MISSIONS_DIR = PROJECT_ROOT / "data" / "missions"

REQUIRED_COLUMNS = ["timestamp", "rpm", "throttle", "egt", "cht", "oil_temp"]


class MissionReplayEngine:
    """
    Replay engine that reads recorded telemetry CSV files row-by-row.
    """

    def __init__(self, mission_name_or_filename: str) -> None:
        self.mission_name = mission_name_or_filename
        self.csv_path = self._resolve_path(mission_name_or_filename)
        self.df: pd.DataFrame = self._load_and_validate(self.csv_path)
        self.total_rows: int = len(self.df)
        self.current_index: int = 0

    @staticmethod
    def _resolve_path(name: str) -> Path:
        """Resolve a filename or mission identifier to a Path object."""
        if not name.endswith(".csv"):
            filename = f"{name}.csv"
        else:
            filename = name
        
        path = MISSIONS_DIR / filename
        if not path.exists():
            # Fallback check if full path passed directly
            direct_path = Path(name)
            if direct_path.exists():
                return direct_path
            raise FileNotFoundError(f"Mission recording file not found: '{filename}' in {MISSIONS_DIR}")
        return path

    @staticmethod
    def _load_and_validate(path: Path) -> pd.DataFrame:
        """Load CSV and validate columns, types, and values."""
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Mission file {path} is empty.")
        
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required telemetry column '{col}' in {path}")
            if df[col].isnull().any():
                raise ValueError(f"NaN values found in column '{col}' of {path}")

        # Ensure fault_active column exists
        if "fault_active" not in df.columns:
            df["fault_active"] = False

        return df

    def get_step(self) -> tuple[dict[str, Any], bool]:
        """
        Fetch the next telemetry record from the mission CSV.

        Returns
        -------
        tuple[dict, bool]
            (telemetry_row_dict, is_completed_flag)
        """
        if self.current_index >= self.total_rows:
            # Reached end of mission recording
            last_row = self.df.iloc[-1].to_dict()
            return last_row, True

        row_dict = self.df.iloc[self.current_index].to_dict()
        self.current_index += 1
        is_finished = self.current_index >= self.total_rows

        return row_dict, is_finished

    def reset(self) -> None:
        """Reset playback index to start of recording."""
        self.current_index = 0

    @property
    def progress_pct(self) -> float:
        """Return playback progress percentage (0.0 to 100.0)."""
        if self.total_rows == 0:
            return 100.0
        return round(100.0 * self.current_index / self.total_rows, 1)


def list_available_missions() -> list[str]:
    """Discover all available mission CSV files in data/missions/."""
    MISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(MISSIONS_DIR.glob("*.csv"))
    return [f.stem for f in csv_files]
