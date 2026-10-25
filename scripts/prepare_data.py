"""Data preparation pipeline script for Corporación Favorita Store Sales.

Executes schema validation, date normalization, sorting, and exports clean
daily sales data along with a data-quality audit report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retail_forecasting.data import get_default_raw_sales_path, load_raw_sales
from src.retail_forecasting.preprocessing import clean_sales_records


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare and validate Corporación Favorita Store Sales dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw train.csv (default: configs/project_config.json raw_sales_file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save processed sales_daily.csv (default: data/processed/sales_daily.csv)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save data_quality_report.json (default: reports/metrics/data_quality_report.json)",
    )
    parser.add_argument(
        "--impute-promotions-zero",
        action="store_true",
        default=False,
        help="Override policy: force zero-imputation for missing onpromotion values (default: preserve missing)",
    )
    return parser.parse_args()


def resolve_path(cli_arg: str | None, default_path: Path) -> Path:
    """Resolve path argument portably from relative or absolute input."""
    if cli_arg:
        p = Path(cli_arg)
        return p if p.is_absolute() else (Path.cwd() / p).resolve()
    return default_path.resolve()


def main() -> int:
    """Execute data preparation workflow."""
    args = parse_args()

    default_raw = get_default_raw_sales_path()
    default_out = PROJECT_ROOT / "data" / "processed" / "sales_daily.csv"
    default_rep = PROJECT_ROOT / "reports" / "metrics" / "data_quality_report.json"

    raw_path = resolve_path(args.input, default_raw)
    out_path = resolve_path(args.output, default_out)
    rep_path = resolve_path(args.report, default_rep)

    print("=" * 75)
    print("Corporación Favorita — Data Preparation & Validation Pipeline")
    print("=" * 75)
    print(f"Target Raw Input:       {raw_path}")
    print(f"Target Processed Output: {out_path}")
    print(f"Data Quality Report:    {rep_path}")
    print(f"Preserve Null Promos:   {not args.impute_promotions_zero}")
    print("-" * 75)

    if not raw_path.exists():
        print(f"[ERROR] Raw dataset file not found at: '{raw_path}'", file=sys.stderr)
        print(
            "\n[ACTION REQUIRED] Please obtain 'train.csv' from the Kaggle Corporación Favorita competition:\n"
            "  https://www.kaggle.com/competitions/store-sales-time-series-forecasting\n"
            f"and place it at '{raw_path}' before running this pipeline.\n",
            file=sys.stderr,
        )
        return 1

    try:
        print(f"[INFO] Ingesting raw sales data from '{raw_path}'...")
        raw_df = load_raw_sales(file_path=raw_path)
        print(f"[INFO] Loaded {len(raw_df):,} rows.")

        print("[INFO] Running schema validation and safe preprocessing...")
        clean_df, report = clean_sales_records(
            raw_df,
            input_filename=raw_path.name,
            preserve_missing_promotions=not args.impute_promotions_zero,
        )

        # Output writes only occur after clean_sales_records completely succeeds.
        # This guarantees any existing files are preserved if validation fails.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Saving processed records to '{out_path}'...")
        clean_df.to_csv(out_path, index=False)

        print(f"[INFO] Persisting data quality report to '{rep_path}'...")
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("-" * 75)
        print("[SUCCESS] Data preparation pipeline completed successfully.")
        print(f"  Rows Processed: {report['output_row_count']:,}")
        print(f"  Date Range:     {report['date_range']['min_date']} to {report['date_range']['max_date']}")
        print(f"  Unique Stores:  {report['store_count']}")
        print(f"  Unique Families:{report['family_count']}")
        print(f"  Duplicates:     {report['duplicate_key_count']}")
        print("=" * 75)
        return 0

    except Exception as exc:
        print(f"\n[FATAL ERROR] Pipeline aborted due to validation/processing failure:\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
