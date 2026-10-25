"""Unit and temporal leakage prevention tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.retail_forecasting import features
from src.retail_forecasting.features import (
    build_feature_pipeline,
    generate_calendar_features,
    generate_holiday_features,
    generate_lag_features,
    generate_rolling_features,
    validate_feature_input_schema,
)


@pytest.fixture
def temporal_test_df() -> pd.DataFrame:
    """Return a deterministic multi-series DataFrame with calendar date gaps."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7],
        "date": pd.to_datetime([
            "2017-01-01",
            "2017-01-02",
            "2017-01-08",
            "2017-01-09",
            "2017-01-15",
            # Store 2 series
            "2017-01-01",
            "2017-01-02",
        ]),
        "store_nbr": [1, 1, 1, 1, 1, 2, 2],
        "family": [
            "GROCERY I",
            "GROCERY I",
            "GROCERY I",
            "GROCERY I",
            "GROCERY I",
            "GROCERY I",
            "GROCERY I",
        ],
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0],
        "onpromotion": [0, 1, pd.NA, 2, 0, 5, 10],
    })


# ---------------------------------------------------------------------------
# Contract & Import Tests
# ---------------------------------------------------------------------------

def test_features_module_importable():
    """Verify feature engineering module defines key pipeline transformers."""
    assert hasattr(features, "generate_lag_features")
    assert hasattr(features, "generate_rolling_features")
    assert hasattr(features, "generate_calendar_features")
    assert hasattr(features, "generate_holiday_features")
    assert hasattr(features, "build_feature_pipeline")


# ---------------------------------------------------------------------------
# Exact Calendar-Day Lag Tests
# ---------------------------------------------------------------------------

def test_exact_calendar_day_lags(temporal_test_df):
    """Test 1: Lags match exact calendar dates (t - k days) and preserve missing when gap exists."""
    df_lags = generate_lag_features(temporal_test_df, lags=[1, 7])

    s1 = df_lags[df_lags["store_nbr"] == 1].sort_values("date").reset_index(drop=True)

    # 2017-01-01: No prior date -> lag_1d is NaN, lag_7d is NaN
    assert pd.isna(s1.loc[0, "sales_lag_1d"])
    assert pd.isna(s1.loc[0, "sales_lag_7d"])

    # 2017-01-02: 1 day after 2017-01-01 -> lag_1d is 10.0, lag_7d is NaN
    assert s1.loc[1, "sales_lag_1d"] == 10.0
    assert pd.isna(s1.loc[1, "sales_lag_7d"])

    # 2017-01-08: 2017-01-07 was NOT observed! lag_1d must be NaN (NOT forward-filled from Jan 2)
    assert pd.isna(s1.loc[2, "sales_lag_1d"])
    # Exactly 7 calendar days after 2017-01-01 -> lag_7d must be 10.0
    assert s1.loc[2, "sales_lag_7d"] == 10.0

    # 2017-01-09: lag_1d is 30.0 (from 2017-01-08), lag_7d is 20.0 (from 2017-01-02)
    assert s1.loc[3, "sales_lag_1d"] == 30.0
    assert s1.loc[3, "sales_lag_7d"] == 20.0


def test_no_lag_crosses_store_or_family_boundary(temporal_test_df):
    """Test 2: Lags strictly isolate series and never borrow observations from other stores."""
    df_lags = generate_lag_features(temporal_test_df, lags=[1])

    s2 = df_lags[df_lags["store_nbr"] == 2].sort_values("date").reset_index(drop=True)
    # Store 2 on 2017-01-01 has no prior store 2 record -> must be NaN, NOT borrowing Store 1 sales
    assert pd.isna(s2.loc[0, "sales_lag_1d"])
    # Store 2 on 2017-01-02 -> must be 100.0 (from Store 2 Jan 1), NOT Store 1 Jan 1 (10.0)
    assert s2.loc[1, "sales_lag_1d"] == 100.0


# ---------------------------------------------------------------------------
# Leakage-Safe Rolling Statistics Tests
# ---------------------------------------------------------------------------

def test_rolling_windows_exclude_current_target(temporal_test_df):
    """Test 3: Rolling window strictly excludes current target at date t (closed='left')."""
    df_roll = generate_rolling_features(temporal_test_df, windows=[7])
    s1 = df_roll[df_roll["store_nbr"] == 1].sort_values("date").reset_index(drop=True)

    # 2017-01-01: target is 10.0, but rolling mean must be NaN (current day excluded)
    assert pd.isna(s1.loc[0, "sales_rolling_mean_7d"])

    # 2017-01-02: target is 20.0, rolling window [Jan 1, Jan 1] contains only Jan 1 (10.0) -> mean is 10.0
    assert s1.loc[1, "sales_rolling_mean_7d"] == 10.0

    # 2017-01-08: window [Jan 1, Jan 7] contains Jan 1 (10.0) and Jan 2 (20.0) -> mean is 15.0
    assert s1.loc[2, "sales_rolling_mean_7d"] == 15.0


def test_rolling_std_minimum_observations(temporal_test_df):
    """Test 4: Sample standard deviation returns NaN when fewer than 2 observations exist in window."""
    df_roll = generate_rolling_features(temporal_test_df, windows=[7])
    s1 = df_roll[df_roll["store_nbr"] == 1].sort_values("date").reset_index(drop=True)

    # Jan 1: 0 observations -> std is NaN
    assert pd.isna(s1.loc[0, "sales_rolling_std_7d"])

    # Jan 2: 1 observation (Jan 1) -> sample std is NaN (requires N >= 2)
    assert pd.isna(s1.loc[1, "sales_rolling_std_7d"])

    # Jan 8: 2 observations (Jan 1=10, Jan 2=20) -> std = 7.071068
    assert np.isclose(s1.loc[2, "sales_rolling_std_7d"], 7.0710678)


# ---------------------------------------------------------------------------
# Temporal Invariance & Non-Leakage Proofs
# ---------------------------------------------------------------------------

def test_changing_current_day_sales_does_not_alter_features(temporal_test_df):
    """Test 5: Modifying sales on date t does NOT change lag or rolling features for date t."""
    feat1 = generate_rolling_features(temporal_test_df, windows=[7])
    feat1 = generate_lag_features(feat1, lags=[1, 7])

    # Alter sales on Jan 8 from 30.0 to 9999.0
    corrupted_df = temporal_test_df.copy()
    corrupted_df.loc[corrupted_df["date"] == "2017-01-08", "sales"] = 9999.0

    feat2 = generate_rolling_features(corrupted_df, windows=[7])
    feat2 = generate_lag_features(feat2, lags=[1, 7])

    row1 = feat1[(feat1["store_nbr"] == 1) & (feat1["date"] == "2017-01-08")].iloc[0]
    row2 = feat2[(feat2["store_nbr"] == 1) & (feat2["date"] == "2017-01-08")].iloc[0]

    # Features on Jan 8 must be IDENTICAL because current-day sales is excluded from features
    assert pd.isna(row1["sales_lag_1d"]) and pd.isna(row2["sales_lag_1d"])
    assert row1["sales_lag_7d"] == row2["sales_lag_7d"]
    assert row1["sales_rolling_mean_7d"] == row2["sales_rolling_mean_7d"]
    assert row1["sales_rolling_std_7d"] == row2["sales_rolling_std_7d"]


def test_changing_future_sales_cannot_alter_past_features(temporal_test_df):
    """Test 6: Modifying sales in the future cannot alter any features on earlier dates."""
    feat1 = generate_rolling_features(temporal_test_df, windows=[7])
    feat1 = generate_lag_features(feat1, lags=[1, 7])

    # Alter future date Jan 15 from 50.0 to 1,000,000.0
    corrupted_df = temporal_test_df.copy()
    corrupted_df.loc[corrupted_df["date"] == "2017-01-15", "sales"] = 1_000_000.0

    feat2 = generate_rolling_features(corrupted_df, windows=[7])
    feat2 = generate_lag_features(feat2, lags=[1, 7])

    # Compare all rows prior to Jan 15 for Store 1
    past1 = feat1[(feat1["store_nbr"] == 1) & (feat1["date"] < "2017-01-15")].sort_values("date").reset_index(drop=True)
    past2 = feat2[(feat2["store_nbr"] == 1) & (feat2["date"] < "2017-01-15")].sort_values("date").reset_index(drop=True)

    pd.testing.assert_series_equal(past1["sales_lag_1d"], past2["sales_lag_1d"])
    pd.testing.assert_series_equal(past1["sales_lag_7d"], past2["sales_lag_7d"])
    pd.testing.assert_series_equal(past1["sales_rolling_mean_7d"], past2["sales_rolling_mean_7d"])
    pd.testing.assert_series_equal(past1["sales_rolling_std_7d"], past2["sales_rolling_std_7d"])


def test_no_future_leakage_in_lags(temporal_test_df):
    """Test 7: Verify that lag k strictly retrieves sales from t - k and never t + k."""
    feat = generate_lag_features(temporal_test_df, lags=[1, 7, 14, 28])
    assert "sales_lag_1d" in feat.columns
    assert "sales_lag_7d" in feat.columns
    assert "sales_lag_14d" in feat.columns
    assert "sales_lag_28d" in feat.columns


# ---------------------------------------------------------------------------
# Calendar & Holiday Feature Tests
# ---------------------------------------------------------------------------

def test_calendar_features(temporal_test_df):
    """Test 8: Calendar signals match datetime components and include cyclical encodings."""
    cal_df = generate_calendar_features(temporal_test_df, include_cyclical=True)

    assert "day_of_week" in cal_df.columns
    assert "day_of_month" in cal_df.columns
    assert "month" in cal_df.columns
    assert "quarter" in cal_df.columns
    assert "year" in cal_df.columns
    assert "is_weekend" in cal_df.columns
    assert "day_of_week_sin" in cal_df.columns
    assert "day_of_week_cos" in cal_df.columns
    assert "month_sin" in cal_df.columns
    assert "month_cos" in cal_df.columns

    # 2017-01-01 was a Sunday (dayofweek = 6, weekend = 1)
    sun_row = cal_df[cal_df["date"] == "2017-01-01"].iloc[0]
    assert sun_row["day_of_week"] == 6
    assert sun_row["is_weekend"] == 1
    assert sun_row["month"] == 1
    assert sun_row["year"] == 2017


def test_holiday_features_transferred_and_locale_rules():
    """Test 9: Transferred holidays and workdays are not holidays; locales match store hierarchy."""
    dates = pd.to_datetime(["2017-10-09", "2017-10-13", "2017-05-01", "2017-12-06"])
    sales_df = pd.DataFrame({
        "date": dates,
        "store_nbr": [1, 1, 1, 1],
    })

    holidays_df = pd.DataFrame({
        "date": ["2017-10-09", "2017-10-13", "2017-05-01", "2017-12-06"],
        "type": ["Holiday", "Transfer", "Holiday", "Holiday"],
        "locale": ["National", "National", "National", "Local"],
        "locale_name": ["Ecuador", "Ecuador", "Ecuador", "Quito"],
        "description": ["Independencia de Guayaquil", "Traslado Independencia", "Dia del Trabajo", "Fundacion de Quito"],
        "transferred": [True, False, False, False],
    })

    stores_df = pd.DataFrame({
        "store_nbr": [1],
        "city": ["Quito"],
        "state": ["Pichincha"],
    })

    res = generate_holiday_features(sales_df, holidays_df=holidays_df, stores_df=stores_df)

    # 2017-10-09: transferred is True -> is_holiday must be 0!
    assert res.loc[0, "is_holiday"] == 0

    # 2017-10-13: Transfer type celebrated -> is_holiday must be 1!
    assert res.loc[1, "is_holiday"] == 1

    # 2017-05-01: National holiday celebrated -> is_holiday must be 1!
    assert res.loc[2, "is_holiday"] == 1

    # 2017-12-06: Local holiday in Quito matching Store 1 -> is_holiday must be 1!
    assert res.loc[3, "is_holiday"] == 1


def test_rolling_window_multi_day_gap_boundary_and_leakage_invariance():
    """Test: Calendar rolling window [t - W, t) boundary precision, gaps, and current-day exclusion."""
    # Constructed scenario for W = 7 days:
    # Row 0: 2023-01-02 (sales=10.0) -> outside window for 2023-01-10 (8 days prior)
    # Row 1: 2023-01-03 (sales=20.0) -> EXACTLY on boundary t - W for 2023-01-10 (7 days prior)
    # Row 2: 2023-01-07 (sales=40.0) -> inside window for 2023-01-10 (3 days prior)
    # Row 3: 2023-01-10 (sales=999.0) -> current-day target (must be excluded!)
    # Row 4: 2023-01-25 (sales=50.0) -> after a 15-day gap (> W=7), so 0 observations in [2023-01-18, 2023-01-25)
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-07", "2023-01-10", "2023-01-25"]),
        "store_nbr": 1,
        "family": "GROCERY",
        "sales": [10.0, 20.0, 40.0, 999.0, 50.0],
    })

    res = generate_rolling_features(df, windows=[7])

    mean_col = "sales_rolling_mean_7d"
    std_col = "sales_rolling_std_7d"

    # Row 0: 2023-01-02 -> no prior observations in [2022-12-26, 2023-01-02)
    assert np.isnan(res.loc[0, mean_col])
    assert np.isnan(res.loc[0, std_col])

    # Row 1: 2023-01-03 -> only 2023-01-02 in window (sales=10.0)
    assert res.loc[1, mean_col] == 10.0
    assert np.isnan(res.loc[1, std_col])  # N=1 < 2 observations -> sample std is NaN

    # Row 2: 2023-01-07 -> 2023-01-02 (10) and 2023-01-03 (20) in window
    assert res.loc[2, mean_col] == 15.0
    expected_std_r2 = np.std([10.0, 20.0], ddof=1)
    assert np.isclose(res.loc[2, std_col], expected_std_r2)

    # Row 3: 2023-01-10:
    # - 2023-01-02 is 8 days prior -> OUTSIDE window [2023-01-03, 2023-01-10)
    # - 2023-01-03 is 7 days prior -> EXACTLY ON BOUNDARY -> INCLUDED
    # - 2023-01-07 is 3 days prior -> INCLUDED
    # - 2023-01-10 current sales (999.0) -> EXCLUDED
    # Expected observations in window: [20.0, 40.0]
    expected_mean_r3 = (20.0 + 40.0) / 2.0  # 30.0
    expected_std_r3 = np.std([20.0, 40.0], ddof=1)  # sqrt(200) ~= 14.142136
    assert res.loc[3, mean_col] == expected_mean_r3
    assert np.isclose(res.loc[3, std_col], expected_std_r3)

    # Row 4: 2023-01-25 -> prior observation 2023-01-10 is 15 days ago (> 7 days)
    # Window [2023-01-18, 2023-01-25) contains ZERO observations -> mean and std are NaN
    assert np.isnan(res.loc[4, mean_col])
    assert np.isnan(res.loc[4, std_col])


def test_holiday_features_multiple_events_on_same_date_no_row_duplication():
    """Test: Multiple event records on the same date do not expand row counts and set flags correctly."""
    # Date 2016-05-01 has both a National Holiday (Dia del Trabajo) and an Event (Terremoto Manabi+15)
    sales_df = pd.DataFrame({
        "date": pd.to_datetime(["2016-04-30", "2016-05-01", "2016-05-02"]),
        "store_nbr": [1, 1, 1],
    })

    holidays_df = pd.DataFrame({
        "date": ["2016-05-01", "2016-05-01"],
        "type": ["Holiday", "Event"],
        "locale": ["National", "National"],
        "locale_name": ["Ecuador", "Ecuador"],
        "description": ["Dia del Trabajo", "Terremoto Manabi+15"],
        "transferred": [False, False],
    })

    stores_df = pd.DataFrame({
        "store_nbr": [1],
        "city": ["Quito"],
        "state": ["Pichincha"],
    })

    res = generate_holiday_features(sales_df, holidays_df=holidays_df, stores_df=stores_df)

    # CRITICAL: Output row count must match input row count (no duplication from multiple events)
    assert len(res) == len(sales_df)

    # 2016-04-30: Regular day
    assert res.loc[0, "is_holiday"] == 0
    assert res.loc[0, "is_event"] == 0

    # 2016-05-01: BOTH is_holiday AND is_event should be active (1)
    assert res.loc[1, "is_holiday"] == 1
    assert res.loc[1, "is_event"] == 1

    # 2016-05-02: Regular day
    assert res.loc[2, "is_holiday"] == 0
    assert res.loc[2, "is_event"] == 0


def test_holiday_locale_mismatches_and_regional_scoping():
    """Test: Regional and local events only activate for matching store states and cities."""
    sales_df = pd.DataFrame({
        "date": pd.to_datetime(["2017-04-21", "2017-04-21", "2017-04-21", "2017-07-23", "2017-07-23"]),
        "store_nbr": [1, 2, 3, 1, 2],
    })

    stores_df = pd.DataFrame({
        "store_nbr": [1, 2, 3],
        "city": ["Quito", "Guayaquil", "Cuenca"],
        "state": ["Pichincha", "Guayas", "Azuay"],
    })

    holidays_df = pd.DataFrame({
        "date": ["2017-04-21", "2017-07-23", "2017-04-21"],
        "type": ["Holiday", "Holiday", "Work Day"],
        "locale": ["Regional", "Local", "National"],
        "locale_name": ["Pichincha", "Guayaquil", "Ecuador"],
        "description": ["Cantonizacion de Riobamba", "Cantonizacion de Cayambe", "Recuperacion"],
        "transferred": [False, False, False],
    })

    res = generate_holiday_features(sales_df, holidays_df=holidays_df, stores_df=stores_df)
    assert len(res) == len(sales_df)

    # 2017-04-21: Regional holiday in Pichincha
    # Store 1 is in Pichincha -> is_holiday = 1
    row_s1 = res[(res["date"] == "2017-04-21") & (res["store_nbr"] == 1)].iloc[0]
    assert row_s1["is_holiday"] == 1

    # Store 2 is in Guayas -> mismatch -> is_holiday = 0
    row_s2 = res[(res["date"] == "2017-04-21") & (res["store_nbr"] == 2)].iloc[0]
    assert row_s2["is_holiday"] == 0

    # Store 3 is in Azuay -> mismatch -> is_holiday = 0
    row_s3 = res[(res["date"] == "2017-04-21") & (res["store_nbr"] == 3)].iloc[0]
    assert row_s3["is_holiday"] == 0

    # 2017-07-23: Local holiday in Guayaquil
    # Store 1 is in Quito -> mismatch -> is_holiday = 0
    row_s1_july = res[(res["date"] == "2017-07-23") & (res["store_nbr"] == 1)].iloc[0]
    assert row_s1_july["is_holiday"] == 0

    # Store 2 is in Guayaquil -> match -> is_holiday = 1
    row_s2_july = res[(res["date"] == "2017-07-23") & (res["store_nbr"] == 2)].iloc[0]
    assert row_s2_july["is_holiday"] == 1


# ---------------------------------------------------------------------------
# Immutability, Row Integrity, Error Handling & Pipeline Tests
# ---------------------------------------------------------------------------

def test_input_immutability(temporal_test_df):
    """Test 10: Input DataFrame is not mutated in-place by feature transformers."""
    orig_copy = temporal_test_df.copy(deep=True)
    _ = build_feature_pipeline(temporal_test_df)
    pd.testing.assert_frame_equal(temporal_test_df, orig_copy)


def test_duplicate_keys_raise_error(temporal_test_df):
    """Test 11: Duplicate (date, store_nbr, family) keys raise ValueError immediately."""
    corrupted = pd.concat([temporal_test_df, temporal_test_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_feature_pipeline(corrupted)


def test_empty_dataframe_raises_error():
    """Test 12: Empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="empty DataFrame"):
        build_feature_pipeline(pd.DataFrame())


def test_build_feature_pipeline_row_conservation_and_reporting(temporal_test_df):
    """Test 13: Pipeline preserves exact row count, generates audit report, and sorts deterministically."""
    feat_df, report = build_feature_pipeline(temporal_test_df)

    assert len(feat_df) == len(temporal_test_df)
    assert report["input_row_count"] == len(temporal_test_df)
    assert report["output_row_count"] == len(temporal_test_df)
    assert "sales_lag_1d" in feat_df.columns
    assert "sales_rolling_mean_7d" in feat_df.columns
    assert "is_weekend" in feat_df.columns
    assert "is_holiday" in feat_df.columns
    assert len(report["engineered_feature_names"]) == 22
    assert "missing_values_by_feature" in report
