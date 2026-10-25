"""Feature engineering pipeline script for Corporación Favorita Store Sales.

Loads data/processed/sales_daily.csv, computes calendar-day demand lags,
calendar-window rolling statistics, calendar cyclical encodings, and store-specific
holiday indicators, exporting data/processed/sales_features.csv and an audit report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.retail_forecasting.features import build_feature_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build leakage-safe features on processed Corporación Favorita sales data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to sales_daily.csv (default: data/processed/sales_daily.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save sales_features.csv (default: data/processed/sales_features.csv)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save feature_engineering_report.json (default: reports/metrics/feature_engineering_report.json)",
    )
    return parser.parse_args()


def resolve_path(cli_arg: str | None, default_path: Path) -> Path:
    """Resolve path argument portably from relative or absolute input."""
    if cli_arg:
        p = Path(cli_arg)
        return p if p.is_absolute() else (Path.cwd() / p).resolve()
    return default_path.resolve()


def main() -> int:
    """Execute feature engineering workflow."""
    args = parse_args()

    default_in = PROJECT_ROOT / "data" / "processed" / "sales_daily.csv"
    default_out = PROJECT_ROOT / "data" / "processed" / "sales_features.csv"
    default_rep = PROJECT_ROOT / "reports" / "metrics" / "feature_engineering_report.json"

    in_path = resolve_path(args.input, default_in)
    out_path = resolve_path(args.output, default_out)
    rep_path = resolve_path(args.report, default_rep)

    print("=" * 75)
    print("Corporación Favorita — Feature Engineering Pipeline (Phase 4)")
    print("=" * 75)
    print(f"Target Processed Input: {in_path}")
    print(f"Target Feature Output:  {out_path}")
    print(f"Audit Report Output:    {rep_path}")
    print("-" * 75)

    if not in_path.exists():
        print(f"[ERROR] Processed sales file not found at: '{in_path}'", file=sys.stderr)
        print("Please run 'scripts/prepare_data.py' first to generate clean sales records.", file=sys.stderr)
        return 1

    try:
        t0 = time.time()
        print(f"[INFO] Ingesting processed sales data from '{in_path}'...")
        sales_df = pd.read_csv(in_path, parse_dates=["date"])
        print(f"[INFO] Loaded {len(sales_df):,} rows in {time.time() - t0:.2f}s.")

        # Load supporting metadata if available
        holidays_path = PROJECT_ROOT / "data" / "raw" / "holidays_events.csv"
        stores_path = PROJECT_ROOT / "data" / "raw" / "stores.csv"

        holidays_df = pd.read_csv(holidays_path) if holidays_path.exists() else None
        stores_df = pd.read_csv(stores_path) if stores_path.exists() else None

        if holidays_df is not None:
            print(f"[INFO] Loaded holidays data ({len(holidays_df)} events).")
        if stores_df is not None:
            print(f"[INFO] Loaded store hierarchy ({len(stores_df)} stores).")

        print("[INFO] Constructing leakage-safe features...")
        t_feat = time.time()
        feat_df, report = build_feature_pipeline(
            sales_df,
            holidays_df=holidays_df,
            stores_df=stores_df,
            lags=[1, 7, 14, 28],
            windows=[7, 14, 30],
            include_cyclical=True,
        )
        print(f"[INFO] Engineered {report['feature_count']} features in {time.time() - t_feat:.2f}s.")

        # Ensure output directories exist
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Saving feature matrix to '{out_path}'...")
        t_save = time.time()
        feat_df.to_csv(out_path, index=False)
        print(f"[INFO] Exported feature matrix in {time.time() - t_save:.2f}s.")

        print(f"[INFO] Persisting audit report to '{rep_path}'...")
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("-" * 75)
        print("[SUCCESS] Feature engineering pipeline completed successfully.")
        print(f"  Input Rows:          {report['input_row_count']:,}")
        print(f"  Output Rows:         {report['output_row_count']:,}")
        print(f"  Engineered Features: {report['feature_count']}")
        print(f"  Total Columns:       {report['total_column_count']}")
        print(f"  Date Range:          {report['date_range']['min_date']} to {report['date_range']['max_date']}")
        print(f"  Elapsed Time:        {time.time() - t0:.2f}s")
        print("=" * 75)
        return 0

    except Exception as exc:
        print(f"\n[FATAL ERROR] Feature engineering aborted:\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

