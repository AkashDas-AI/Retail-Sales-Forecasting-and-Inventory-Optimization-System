"""Leakage-safe feature engineering engine for Corporación Favorita sales forecasting.

Computes exact calendar-day aligned demand lags, calendar-window rolling statistics
strictly excluding the current target (closed='left'), deterministic calendar cyclical
signals, and store-specific holiday/event indicators without lookahead bias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.retail_forecasting.validation import MODELING_KEY


def validate_feature_input_schema(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
) -> None:
    """Validate input DataFrame schema and key uniqueness before feature transformations.

    Args:
        df: Input DataFrame.
        required_cols: List of required columns (default: ['date', 'store_nbr', 'family', 'sales']).

    Raises:
        ValueError: If input is empty, missing required columns, or contains duplicate modeling keys.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, received {type(df).__name__}.")

    if df.empty:
        raise ValueError("Cannot engineer features on an empty DataFrame.")

    if required_cols is None:
        required_cols = ["date", "store_nbr", "family", "sales"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns for feature engineering: {missing}")

    # Enforce uniqueness of modeling grain (date, store_nbr, family)
    dup_count = int(df.duplicated(subset=MODELING_KEY).sum())
    if dup_count > 0:
        raise ValueError(
            f"Input DataFrame contains {dup_count:,} duplicate (date, store_nbr, family) keys. "
            "Feature engineering requires unique observational grain."
        )


def generate_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    lags: Optional[List[int]] = None,
    group_cols: Optional[List[str]] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Generate exact calendar-day aligned demand lags using date-keyed joins.

    For each observation at date t, store s, and family f, lag k retrieves
    sales strictly from (t - k days, s, f). If (t - k days, s, f) was not
    observed, the lag is preserved as missing (NaN).

    Args:
        df: Input DataFrame. Must not be mutated.
        target_col: Column name of target to lag (default: 'sales').
        lags: List of calendar-day lags in days (default: [1, 7, 14, 28]).
        group_cols: Series grouping keys (default: ['store_nbr', 'family']).
        date_col: Date column name (default: 'date').

    Returns:
        pd.DataFrame: New DataFrame containing original columns plus lag features.
    """
    if lags is None:
        lags = [1, 7, 14, 28]
    if group_cols is None:
        group_cols = ["store_nbr", "family"]

    validate_feature_input_schema(df, required_cols=[date_col, target_col] + group_cols)

    out_df = df.copy()
    out_df[date_col] = pd.to_datetime(out_df[date_col]).dt.normalize()

    lookup_base = out_df[[date_col] + group_cols + [target_col]].copy()

    for k in lags:
        col_name = f"{target_col}_lag_{k}d"
        lookup = lookup_base.copy()
        # Shift date forward by k days so joining on 'date' retrieves observation from t - k days
        lookup[date_col] = lookup[date_col] + pd.to_timedelta(k, unit="D")
        lookup.rename(columns={target_col: col_name}, inplace=True)
        out_df = out_df.merge(lookup, on=[date_col] + group_cols, how="left")

    return out_df


def generate_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    windows: Optional[List[int]] = None,
    group_cols: Optional[List[str]] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Generate leakage-safe calendar-window rolling statistics strictly excluding current target.

    For each observation at date t, aggregates history strictly over [t - W days, t - 1 day].
    Current day sales at date t is excluded (closed='left'). Missing dates within the
    window are not filled with zeros; only observed values are aggregated.
    If 0 observations exist in window, mean and std are NaN.
    If < 2 observations exist, sample standard deviation is NaN.

    Args:
        df: Input DataFrame. Must not be mutated.
        target_col: Target column name (default: 'sales').
        windows: Calendar-day rolling window durations in days (default: [7, 14, 30]).
        group_cols: Series grouping keys (default: ['store_nbr', 'family']).
        date_col: Date column name (default: 'date').

    Returns:
        pd.DataFrame: New DataFrame containing original columns plus rolling statistics.
    """
    if windows is None:
        windows = [7, 14, 30]
    if group_cols is None:
        group_cols = ["store_nbr", "family"]

    validate_feature_input_schema(df, required_cols=[date_col, target_col] + group_cols)

    # Work on a copy and ensure strict date typing
    work_df = df.copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col]).dt.normalize()

    # Sort strictly by group and date for continuous time-series rolling
    sort_keys = group_cols + [date_col]
    work_df = work_df.sort_values(sort_keys).reset_index(drop=True)

    # Set DateTimeIndex for calendar-offset rolling
    indexed_df = work_df.set_index(date_col)
    grouped = indexed_df.groupby(group_cols, sort=False)[target_col]

    for w in windows:
        mean_col = f"{target_col}_rolling_mean_{w}d"
        std_col = f"{target_col}_rolling_std_{w}d"

        # closed='left' strictly includes [t - w, t) and excludes t (zero leakage)
        roller = grouped.rolling(f"{w}D", closed="left")
        stats = roller.agg(["mean", "std"])

        work_df[mean_col] = stats["mean"].values
        work_df[std_col] = stats["std"].values

    return work_df


def generate_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
    include_cyclical: bool = True,
) -> pd.DataFrame:
    """Generate deterministic calendar signals and cyclical encodings.

    Features generated:
    - day_of_week (0=Mon, 6=Sun)
    - day_of_month (1-31)
    - month (1-12)
    - quarter (1-4)
    - year
    - is_weekend (1 if Sat/Sun, else 0)
    - day_of_week_sin / day_of_week_cos (optional cyclical representation)
    - month_sin / month_cos (optional cyclical representation)

    Args:
        df: Input DataFrame.
        date_col: Column name containing dates.
        include_cyclical: Whether to include sine/cosine cyclical transformations.

    Returns:
        pd.DataFrame: DataFrame with calendar features attached.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

    out_df = df.copy()
    dates = pd.to_datetime(out_df[date_col])

    dow = dates.dt.dayofweek
    dom = dates.dt.day
    m = dates.dt.month

    out_df["day_of_week"] = dow.astype(int)
    out_df["day_of_month"] = dom.astype(int)
    out_df["month"] = m.astype(int)
    out_df["quarter"] = dates.dt.quarter.astype(int)
    out_df["year"] = dates.dt.year.astype(int)
    out_df["is_weekend"] = (dow >= 5).astype(int)

    if include_cyclical:
        out_df["day_of_week_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        out_df["day_of_week_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        out_df["month_sin"] = np.sin(2.0 * np.pi * (m - 1.0) / 12.0)
        out_df["month_cos"] = np.cos(2.0 * np.pi * (m - 1.0) / 12.0)

    return out_df


def generate_holiday_features(
    df: pd.DataFrame,
    holidays_df: Optional[pd.DataFrame] = None,
    stores_df: Optional[pd.DataFrame] = None,
    date_col: str = "date",
    store_col: str = "store_nbr",
) -> pd.DataFrame:
    """Generate store-specific holiday and business event indicators.

    Rules applied:
    - Days with transferred == True are regular working days (is_holiday = 0).
    - Days with type == 'Work Day' are regular working days (is_holiday = 0).
    - Celebrated non-working days (Holiday, Transfer, Bridge, Additional) set is_holiday = 1.
    - Type == 'Event' (e.g. World Cup, Earthquake disaster relief) sets is_event = 1.
    - National events apply to all stores.
    - Regional events apply to stores in the matching state.
    - Local events apply to stores in the matching city.
    - Multiple events on the same date/store are aggregated with max() to strictly prevent row duplication.

    Args:
        df: Input DataFrame.
        holidays_df: Optional holidays DataFrame. If None, loaded from data/raw/holidays_events.csv if present.
        stores_df: Optional stores DataFrame. If None, loaded from data/raw/stores.csv if present.
        date_col: Date column name.
        store_col: Store column name.

    Returns:
        pd.DataFrame: DataFrame with 'is_holiday' and 'is_event' binary indicator features.
    """
    out_df = df.copy()

    # If DataFrames not supplied, attempt to resolve from standard raw paths
    if holidays_df is None:
        h_path = Path("data/raw/holidays_events.csv")
        if h_path.is_file():
            holidays_df = pd.read_csv(h_path)

    if stores_df is None:
        s_path = Path("data/raw/stores.csv")
        if s_path.is_file():
            stores_df = pd.read_csv(s_path)

    if holidays_df is None:
        # Default fallback if holiday tables are unavailable: zero indicators
        out_df["is_holiday"] = 0
        out_df["is_event"] = 0
        return out_df

    h_clean = holidays_df.copy()
    h_clean[date_col] = pd.to_datetime(h_clean[date_col]).dt.normalize()

    # Filter out transferred holidays and make-up work days
    valid_events = h_clean[(h_clean["transferred"] == False) & (h_clean["type"] != "Work Day")].copy()

    records = []
    # If store hierarchy is available, match locales exactly
    if stores_df is not None and store_col in stores_df.columns:
        for _, store in stores_df.iterrows():
            s_nbr = store[store_col]
            city = str(store.get("city", "")).strip()
            state = str(store.get("state", "")).strip()

            matching = valid_events[
                (valid_events["locale"] == "National")
                | ((valid_events["locale"] == "Regional") & (valid_events["locale_name"].str.strip() == state))
                | ((valid_events["locale"] == "Local") & (valid_events["locale_name"].str.strip() == city))
            ]
            for _, ev in matching.iterrows():
                is_ev = 1 if ev["type"] == "Event" else 0
                is_hol = 1 if ev["type"] != "Event" else 0
                records.append({
                    date_col: ev[date_col],
                    store_col: s_nbr,
                    "is_holiday": is_hol,
                    "is_event": is_ev,
                })
    else:
        # Fallback to national events across all stores
        national_events = valid_events[valid_events["locale"] == "National"]
        for _, ev in national_events.iterrows():
            is_ev = 1 if ev["type"] == "Event" else 0
            is_hol = 1 if ev["type"] != "Event" else 0
            records.append({
                date_col: ev[date_col],
                "is_holiday": is_hol,
                "is_event": is_ev,
            })

    if records:
        event_df = pd.DataFrame(records)
        if store_col in event_df.columns:
            event_map = event_df.groupby([date_col, store_col]).max().reset_index()
            out_df[date_col] = pd.to_datetime(out_df[date_col]).dt.normalize()
            out_df = out_df.merge(event_map, on=[date_col, store_col], how="left")
        else:
            event_map = event_df.groupby(date_col).max().reset_index()
            out_df[date_col] = pd.to_datetime(out_df[date_col]).dt.normalize()
            out_df = out_df.merge(event_map, on=date_col, how="left")

        out_df["is_holiday"] = out_df["is_holiday"].fillna(0).astype(int)
        out_df["is_event"] = out_df["is_event"].fillna(0).astype(int)
    else:
        out_df["is_holiday"] = 0
        out_df["is_event"] = 0

    return out_df


def build_feature_pipeline(
    df: pd.DataFrame,
    holidays_df: Optional[pd.DataFrame] = None,
    stores_df: Optional[pd.DataFrame] = None,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
    include_cyclical: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute end-to-end leakage-safe feature engineering pipeline.

    Transforms:
    1. Validates input schema and key uniqueness (date, store_nbr, family).
    2. Exact calendar-day demand lags (1, 7, 14, 28 days).
    3. Calendar-window rolling statistics (7, 14, 30 days) excluding current target.
    4. Deterministic calendar and cyclical features.
    5. Store-matched holiday and event indicators.
    6. Deterministically sorts final output by (date, store_nbr, family).
    7. Emits structured data quality audit report.

    Args:
        df: Input DataFrame.
        holidays_df: Optional holidays DataFrame.
        stores_df: Optional stores metadata DataFrame.
        lags: List of calendar lag days (default: [1, 7, 14, 28]).
        windows: List of calendar rolling windows (default: [7, 14, 30]).
        include_cyclical: Whether to include sine/cosine cyclical features.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (features_df, feature_report)
    """
    if lags is None:
        lags = [1, 7, 14, 28]
    if windows is None:
        windows = [7, 14, 30]

    validate_feature_input_schema(df)

    input_count = len(df)
    transformations = []

    # Step 1: Calendar Day Lags
    feat_df = generate_lag_features(df, target_col="sales", lags=lags)
    transformations.append(f"Generated exact calendar-day demand lags for {lags} days.")

    # Step 2: Leakage-Safe Rolling Statistics
    feat_df = generate_rolling_features(feat_df, target_col="sales", windows=windows)
    transformations.append(
        f"Generated calendar-window rolling means and standard deviations for {windows} days (closed='left')."
    )

    # Step 3: Calendar Features
    feat_df = generate_calendar_features(feat_df, date_col="date", include_cyclical=include_cyclical)
    transformations.append("Generated deterministic calendar signals and cyclical sine/cosine encodings.")

    # Step 4: Holiday & Business Event Features
    feat_df = generate_holiday_features(feat_df, holidays_df=holidays_df, stores_df=stores_df)
    transformations.append("Generated store-specific holiday and business event indicators.")

    # Step 5: Deterministic Sorting by Modeling Grain
    feat_df = feat_df.sort_values(MODELING_KEY).reset_index(drop=True)
    transformations.append(f"Sorted deterministically by modeling grain: {MODELING_KEY}.")

    output_count = len(feat_df)
    if output_count != input_count:
        raise RuntimeError(
            f"Row count mismatch! Input has {input_count:,} rows, output has {output_count:,} rows. "
            "Feature engineering must never drop or expand rows."
        )

    # Identify engineered feature columns
    base_cols = {"id", "date", "store_nbr", "family", "sales", "onpromotion"}
    engineered_cols = [c for c in feat_df.columns if c not in base_cols]

    missing_by_feature = {c: int(feat_df[c].isna().sum()) for c in engineered_cols}

    dates = pd.to_datetime(feat_df["date"])
    report: Dict[str, Any] = {
        "input_row_count": input_count,
        "output_row_count": output_count,
        "feature_count": len(engineered_cols),
        "total_column_count": len(feat_df.columns),
        "engineered_feature_names": engineered_cols,
        "date_range": {
            "min_date": dates.min().strftime("%Y-%m-%d"),
            "max_date": dates.max().strftime("%Y-%m-%d"),
        },
        "missing_values_by_feature": missing_by_feature,
        "transformations_performed": transformations,
        "warnings_and_limitations": [
            "Lags strictly require exact calendar date match; missing historical dates result in NaN.",
            "Rolling statistics strictly exclude current target (closed='left') across previous W calendar days.",
            "Sample standard deviation is NaN when fewer than 2 observations are present in rolling window.",
            "No future target values or lookahead observations were used.",
            "Unobserved dates were not filled with synthetic zero-sales records.",
        ],
    }

    return feat_df, report
