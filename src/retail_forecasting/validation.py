"""Data integrity and schema validation checks for Corporación Favorita sales data.

Validates column presence, datatypes, finite value ranges, key uniqueness,
and missing value statistics, separating fatal errors from quality warnings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.retail_forecasting.data import REQUIRED_TRAIN_COLUMNS

MODELING_KEY = ["date", "store_nbr", "family"]


@dataclass
class ValidationReport:
    """Structured report detailing validation findings, errors, and metadata."""

    is_valid: bool
    row_count: int
    fatal_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    store_count: int = 0
    family_count: int = 0
    duplicate_key_count: int = 0
    missing_values_by_column: Dict[str, int] = field(default_factory=dict)
    invalid_value_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Generate human-readable diagnostic summary without dumping full dataset."""
        lines = [
            f"Validation Status: {'PASSED' if self.is_valid else 'FAILED'}",
            f"Total Rows: {self.row_count:,}",
            f"Date Range: {self.min_date} to {self.max_date}",
            f"Active Stores: {self.store_count} | Active Families: {self.family_count}",
            f"Duplicate Modeling Keys: {self.duplicate_key_count:,}",
        ]
        if self.fatal_errors:
            lines.append("Fatal Errors:")
            for err in self.fatal_errors:
                lines.append(f"  - [FATAL] {err}")
        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  - [WARN] {warn}")
        return "\n".join(lines)


def validate_sales_schema(
    df: pd.DataFrame,
    raise_on_error: bool = False,
) -> ValidationReport:
    """Validate DataFrame against Corporación Favorita schema and integrity constraints.

    Checks:
    1. Required columns exist (`id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion`).
    2. `date` can be parsed/coerced to valid non-null datetime.
    3. `store_nbr` is valid integer identifier without nulls.
    4. `family` is non-empty string category without nulls.
    5. `sales` is numeric, finite, non-null, and non-negative.
    6. `onpromotion` is numeric and non-negative (missing values preserved and warned; malformed values are fatal).
    7. Modeling key `(date, store_nbr, family)` has zero duplicates.
    8. Missing values counted by column.

    Args:
        df: Input DataFrame to validate.
        raise_on_error: If True, raises ValueError upon any fatal validation errors.

    Returns:
        ValidationReport: Detailed validation summary and status.

    Raises:
        ValueError: If raise_on_error is True and fatal errors are detected.
    """
    fatal_errors: List[str] = []
    warnings: List[str] = []
    invalid_counts: Dict[str, int] = {}

    if not isinstance(df, pd.DataFrame):
        fatal_errors.append(f"Expected pandas DataFrame, received {type(df).__name__}.")
        report = ValidationReport(
            is_valid=False,
            row_count=0,
            fatal_errors=fatal_errors,
        )
        if raise_on_error:
            raise ValueError("\n".join(fatal_errors))
        return report

    row_count = len(df)
    if row_count == 0:
        fatal_errors.append("DataFrame is empty (0 rows).")
        report = ValidationReport(
            is_valid=False,
            row_count=0,
            fatal_errors=fatal_errors,
        )
        if raise_on_error:
            raise ValueError("\n".join(fatal_errors))
        return report

    # 1. Required Columns Check
    missing_cols = [col for col in REQUIRED_TRAIN_COLUMNS if col not in df.columns]
    if missing_cols:
        fatal_errors.append(f"Missing required columns: {missing_cols}")

    # 2. Missing Values per Column
    missing_by_col = {col: int(df[col].isna().sum()) for col in df.columns}

    # If critical columns are missing, abort further deep column inspections
    if missing_cols:
        report = ValidationReport(
            is_valid=False,
            row_count=row_count,
            fatal_errors=fatal_errors,
            missing_values_by_column=missing_by_col,
        )
        if raise_on_error:
            raise ValueError("\n".join(fatal_errors))
        return report

    # 3. Date Validation
    parsed_dates = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    nat_dates_count = int(parsed_dates.isna().sum())
    if nat_dates_count > 0:
        fatal_errors.append(f"Found {nat_dates_count} invalid or unparseable dates in 'date' column.")
        invalid_counts["invalid_dates"] = nat_dates_count
        min_date_str = None
        max_date_str = None
    else:
        min_date_str = parsed_dates.min().strftime("%Y-%m-%d")
        max_date_str = parsed_dates.max().strftime("%Y-%m-%d")

    # 4. store_nbr Validation
    if missing_by_col["store_nbr"] > 0:
        fatal_errors.append(f"Column 'store_nbr' contains {missing_by_col['store_nbr']} missing values.")

    try:
        numeric_stores = pd.to_numeric(df["store_nbr"], errors="coerce")
        invalid_store_count = int((numeric_stores.isna() | (numeric_stores <= 0)).sum())
        if invalid_store_count > 0:
            fatal_errors.append(f"Column 'store_nbr' contains {invalid_store_count} non-positive or non-numeric values.")
            invalid_counts["invalid_store_nbr"] = invalid_store_count
        store_count = int(df["store_nbr"].nunique())
    except Exception as exc:
        fatal_errors.append(f"Error validating 'store_nbr': {exc}")
        store_count = 0

    # 5. family Validation
    if missing_by_col["family"] > 0:
        fatal_errors.append(f"Column 'family' contains {missing_by_col['family']} missing values.")

    empty_family_count = int((df["family"].astype(str).str.strip() == "").sum())
    if empty_family_count > 0:
        fatal_errors.append(f"Column 'family' contains {empty_family_count} blank/empty string entries.")
        invalid_counts["empty_family"] = empty_family_count

    family_count = int(df["family"].nunique())

    # 6. sales Validation (Numeric, Finite, Non-negative)
    sales_series = pd.to_numeric(df["sales"], errors="coerce")
    non_numeric_sales = int(sales_series.isna().sum())
    if non_numeric_sales > 0:
        fatal_errors.append(f"Column 'sales' contains {non_numeric_sales} non-numeric or NaN values.")
        invalid_counts["non_numeric_sales"] = non_numeric_sales

    non_finite_sales = int((~np.isfinite(sales_series) & ~sales_series.isna()).sum())
    if non_finite_sales > 0:
        fatal_errors.append(f"Column 'sales' contains {non_finite_sales} non-finite (infinite) values.")
        invalid_counts["infinite_sales"] = non_finite_sales

    negative_sales = int((sales_series < 0).sum())
    if negative_sales > 0:
        fatal_errors.append(
            f"Column 'sales' contains {negative_sales} negative sales values. "
            "Sales demand must be non-negative in retail volume modeling."
        )
        invalid_counts["negative_sales"] = negative_sales

    # 7. onpromotion Validation (Preserve missing, flag malformed non-numeric, reject negative)
    orig_null_promos = missing_by_col["onpromotion"]
    if orig_null_promos > 0:
        warnings.append(
            f"Column 'onpromotion' contains {orig_null_promos:,} null values. "
            "Missing promotion flags are preserved as unobserved (<NA>) by default and must not be "
            "silently imputed as 0 without an explicit, documented modeling decision."
        )

    promo_series = pd.to_numeric(df["onpromotion"], errors="coerce")
    coerced_null_promos = int(promo_series.isna().sum())
    malformed_promos = coerced_null_promos - orig_null_promos
    if malformed_promos > 0:
        fatal_errors.append(
            f"Column 'onpromotion' contains {malformed_promos} malformed non-numeric values. "
            "Promotion counts must be numeric integers or null."
        )
        invalid_counts["malformed_onpromotion"] = malformed_promos

    negative_promos = int((promo_series < 0).sum())
    if negative_promos > 0:
        fatal_errors.append(f"Column 'onpromotion' contains {negative_promos} negative values.")
        invalid_counts["negative_onpromotion"] = negative_promos

    # 8. Modeling Key Duplicate Check: (date, store_nbr, family)
    duplicate_count = 0
    if nat_dates_count == 0:
        key_df = pd.DataFrame({
            "date": parsed_dates,
            "store_nbr": df["store_nbr"],
            "family": df["family"],
        })
        duplicate_count = int(key_df.duplicated(subset=MODELING_KEY).sum())
        if duplicate_count > 0:
            fatal_errors.append(
                f"Found {duplicate_count:,} duplicate records for modeling key (date, store_nbr, family). "
                "The modeling grain must be unique per daily store-family observation and will not be "
                "silently dropped or aggregated."
            )
            invalid_counts["duplicate_keys"] = duplicate_count

    is_valid = len(fatal_errors) == 0

    report = ValidationReport(
        is_valid=is_valid,
        row_count=row_count,
        fatal_errors=fatal_errors,
        warnings=warnings,
        min_date=min_date_str,
        max_date=max_date_str,
        store_count=store_count,
        family_count=family_count,
        duplicate_key_count=duplicate_count,
        missing_values_by_column=missing_by_col,
        invalid_value_counts=invalid_counts,
    )

    if raise_on_error and not is_valid:
        raise ValueError(f"Schema validation failed:\n{report.summary()}")

    return report


def check_missing_temporal_gaps(
    df: pd.DataFrame,
    date_col: str = "date",
    group_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Audit series for intermittent unobserved dates across active Store-Family series.

    Does not modify data or impute values. Reports observational gaps for review.

    Args:
        df: Input DataFrame.
        date_col: Date column name.
        group_cols: Series grouping keys (default: ['store_nbr', 'family']).

    Returns:
        dict: Summary of observed temporal span and series coverage.
    """
    if group_cols is None:
        group_cols = ["store_nbr", "family"]

    dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    min_date = dates.min()
    max_date = dates.max()
    expected_days = (max_date - min_date).days + 1 if pd.notna(min_date) else 0

    series_counts = df.groupby(group_cols).size()

    return {
        "global_min_date": str(min_date.date()) if pd.notna(min_date) else None,
        "global_max_date": str(max_date.date()) if pd.notna(max_date) else None,
        "total_calendar_days": expected_days,
        "total_active_series": len(series_counts),
        "mean_observations_per_series": float(series_counts.mean()) if not series_counts.empty else 0.0,
        "min_observations_per_series": int(series_counts.min()) if not series_counts.empty else 0,
        "max_observations_per_series": int(series_counts.max()) if not series_counts.empty else 0,
    }
