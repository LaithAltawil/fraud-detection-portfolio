"""Export the small, shareable dashboard data bundle.

Usage:
    uv run python scripts/06_export_dashboard.py
"""

from __future__ import annotations

from fraud_detection.config import load_config
from fraud_detection.export_dashboard import export_dashboard_data


def main() -> None:
    cfg = load_config()
    out = export_dashboard_data(cfg)
    print(f"dashboard bundle written to {out}")


if __name__ == "__main__":
    main()
