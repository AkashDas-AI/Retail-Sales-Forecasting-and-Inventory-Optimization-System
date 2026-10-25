"""Safe data preprocessing and normalization for Corporación Favorita sales data.

Validates raw inputs, normalizes dates and numeric data types, sorts by modeling grain,
and emits both processed dataset and a comprehensive data quality report without
premature zero-filling, Cartesian expansion, or silent promotion imputation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from src.retail_forecasting.validation import MODELING_KEY, validate_sales_schema


def clean_sales_records(
    df: pd.DataFrame,
    input_filename: str = "train.csv",
    preserve_missing_promotions: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and normalize sales records preserving observed grain without imputation.

    Steps:
    1. Validate input against schema requirements (fatal errors stop execution).
    2. Normalize `date` to datetime64[ns] format without time component.
    3. Cast `store_nbr` to integer, `sales` to float64, `family` to trimmed string.
    4. Normalize `onpromotion` to nullable integer (Int64) preserving missing values by default.
    5. Sort deterministically by `(date, store_nbr, family)`.
    6. Retain original observed rows without Cartesian expansion or zero-filling.

    Args:
        df: Input DataFrame.
        input_filename: Name or path of source file for report provenance.
        preserve_missing_promotions: If True, preserves nulls in onpromotion as pd.NA.
            If False, explicitly imputes 0 (requires documented policy override).

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (clean_df, quality_report)

    Raises:
        ValueError: If input DataFrame fails fatal schema validation.
    """
    # 1. Pre-transformation validation
    val_report = validate_sales_schema(df, raise_on_error=True)

    # 2. Work on an explicit copy to preserve input immutability
    clean_df = df.copy()

    transformations = []

    # 3. Date normalization
    clean_df["date"] = pd.to_datetime(clean_df["date"]).dt.normalize()
    transformations.append("Normalized 'date' column to datetime (YYYY-MM-DD format).")

    # 4. Strict type casting
    clean_df["store_nbr"] = clean_df["store_nbr"].astype(int)
    clean_df["family"] = clean_df["family"].astype(str).str.strip()
    clean_df["sales"] = clean_df["sales"].astype(float)

    # 5. Promotion handling (Preserve missing values by default)
    if "onpromotion" in clean_df.columns:
        if preserve_missing_promotions:
            clean_df["onpromotion"] = pd.to_numeric(clean_df["onpromotion"], errors="coerce").astype("Int64")
            transformations.append(
                "Normalized 'onpromotion' to nullable integer (Int64), strictly preserving unobserved/missing values."
            )
        else:
            clean_df["onpromotion"] = (
                pd.to_numeric(clean_df["onpromotion"], errors="coerce").fillna(0).astype("Int64")
            )
            transformations.append(
                "Normalized 'onpromotion' to integer with explicit zero-imputation policy override."
            )

    # 6. Deterministic sorting by modeling key
    clean_df = clean_df.sort_values(by=MODELING_KEY, ascending=True).reset_index(drop=True)
    transformations.append(f"Sorted deterministically by modeling grain: {MODELING_KEY}.")

    # 7. Generate Data Quality Report
    warnings_limitations = [
        "Missing dates between first and last observed dates are NOT zero-filled.",
        "Cartesian expansion (Store x Family x Date) intentionally omitted to prevent artificial zero-sales bias.",
        "Store openings, closures, and assortment eligibility require secondary verification before grid construction.",
    ]
    if preserve_missing_promotions:
        warnings_limitations.append(
            "Missing 'onpromotion' values are preserved as unobserved (<NA>); no automatic zero-imputation applied."
        )

    quality_report: Dict[str, Any] = {
        "input_filename": input_filename,
        "input_row_count": len(df),
        "output_row_count": len(clean_df),
        "date_range": {
            "min_date": val_report.min_date,
            "max_date": val_report.max_date,
        },
        "store_count": val_report.store_count,
        "family_count": val_report.family_count,
        "duplicate_key_count": val_report.duplicate_key_count,
        "missing_values_by_column": val_report.missing_values_by_column,
        "invalid_value_counts": val_report.invalid_value_counts,
        "transformations_performed": transformations,
        "warnings_and_limitations": warnings_limitations + val_report.warnings,
    }

    return clean_df, quality_report


def preprocess_sales_pipeline(
    input_df: Optional[pd.DataFrame] = None,
    input_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    report_path: Optional[Union[str, Path]] = None,
    preserve_missing_promotions: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete ingestion, validation, and safe preprocessing pipeline.

    Args:
        input_df: Optional in-memory DataFrame (e.g. for testing). If None, loaded from input_path.
        input_path: Path to raw train.csv.
        output_path: Destination path for processed CSV (default: data/processed/sales_daily.csv).
        report_path: Destination path for JSON report (default: reports/metrics/data_quality_report.json).
        preserve_missing_promotions: Whether to retain null promotions as pd.NA (default: True).

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (processed_df, quality_report)

    Raises:
        ValueError: If fatal validation checks fail. Pre-existing output files are left untouched.
    """
    from src.retail_forecasting.data import get_default_raw_sales_path, load_raw_sales, resolve_project_root

    root = resolve_project_root()

    if input_df is None:
        raw_path = Path(input_path) if input_path else get_default_raw_sales_path()
        df = load_raw_sales(file_path=raw_path)
        source_name = str(raw_path.name)
    else:
        df = input_df
        source_name = "in_memory_dataframe"

    # Pre-validation & cleaning: if validation fails, clean_sales_records raises ValueError
    # BEFORE any file operations, ensuring pre-existing output files are completely preserved.
    clean_df, report = clean_sales_records(
        df,
        input_filename=source_name,
        preserve_missing_promotions=preserve_missing_promotions,
    )

    # Resolve output destination safely
    out_file = Path(output_path) if output_path else root / "data" / "processed" / "sales_daily.csv"
    if not out_file.is_absolute():
        out_file = (Path.cwd() / out_file).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_file, index=False)

    # Resolve report destination safely
    rep_file = Path(report_path) if report_path else root / "reports" / "metrics" / "data_quality_report.json"
    if not rep_file.is_absolute():
        rep_file = (Path.cwd() / rep_file).resolve()
    rep_file.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return clean_df, report
