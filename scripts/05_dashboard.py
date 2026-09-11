"""Launch the Streamlit dashboard.

Usage:
    uv run python scripts/05_dashboard.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "src" / "fraud_detection" / "dashboard.py"


def main() -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP)], check=False)


if __name__ == "__main__":
    main()
