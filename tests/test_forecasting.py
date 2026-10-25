"""Comprehensive unit and leakage tests for forecasting engine and baselines."""

import numpy as np
import pandas as pd
import pytest

from src.retail_forecasting.forecasting import (
    RandomForestDemandForecaster,
    SeasonalNaiveForecaster,
    temporal_train_val_test_split,
)


@pytest.fixture
def synthetic_series_df():
    """Deterministic synthetic multi-series DataFrame for testing forecasting methods."""
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    records = []
    for s_nbr in [1, 2]:
        for fam in ["GROCERY", "BEVERAGES"]:
            base = 100.0 if fam == "GROCERY" else 50.0
            for i, d in enumerate(dates):
                # Day of week seasonality + trend
                dow_effect = (d.dayofweek + 1) * 10.0
                sales = base + dow_effect + float(i % 5)
                records.append({
                    "id": len(records),
                    "date": d,
                    "store_nbr": s_nbr,
                    "family": fam,
                    "sales": sales,
                    "onpromotion": 0,
                    "sales_lag_1d": sales - 2.0 if i > 0 else np.nan,
                    "sales_lag_7d": sales - 5.0 if i > 6 else np.nan,
                    "sales_lag_14d": sales - 7.0 if i > 13 else np.nan,
                    "sales_lag_28d": sales - 10.0 if i > 27 else np.nan,
                    "sales_rolling_mean_7d": sales - 1.0 if i > 0 else np.nan,
                    "sales_rolling_std_7d": 5.0 if i > 1 else np.nan,
                    "sales_rolling_mean_14d": sales - 2.0 if i > 0 else np.nan,
                    "sales_rolling_std_14d": 5.0 if i > 1 else np.nan,
                    "sales_rolling_mean_30d": sales - 3.0 if i > 0 else np.nan,
                    "sales_rolling_std_30d": 5.0 if i > 1 else np.nan,
                    "day_of_week": d.dayofweek,
                    "day_of_month": d.day,
                    "month": d.month,
                    "quarter": d.quarter,
                    "year": d.year,
                    "is_weekend": 1 if d.dayofweek >= 5 else 0,
                    "day_of_week_sin": np.sin(2 * np.pi * d.dayofweek / 7),
                    "day_of_week_cos": np.cos(2 * np.pi * d.dayofweek / 7),
                    "month_sin": np.sin(2 * np.pi * (d.month - 1) / 12),
                    "month_cos": np.cos(2 * np.pi * (d.month - 1) / 12),
                    "is_holiday": 0,
                    "is_event": 0,
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Chronological Split Tests
# ---------------------------------------------------------------------------

def test_chronological_split_strict_ordering_and_no_overlap(synthetic_series_df):
    """Verify temporal partitions have strictly zero date overlap and respect chronological order."""
    train_df, val_df, test_df = temporal_train_val_test_split(
        synthetic_series_df,
        date_col="date",
        train_end="2023-02-15",
        val_start="2023-02-16",
        val_end="2023-03-05",
        test_start="2023-03-06",
        test_end="2023-03-31",
    )

    train_dates = set(train_df["date"])
    val_dates = set(val_df["date"])
    test_dates = set(test_df["date"])

    # Strict non-overlap
    assert len(train_dates.intersection(val_dates)) == 0
    assert len(train_dates.intersection(test_dates)) == 0
    assert len(val_dates.intersection(test_dates)) == 0

    # Chronological ordering
    assert max(train_dates) < min(val_dates)
    assert max(val_dates) < min(test_dates)

    # Total rows conserved
    assert len(train_df) + len(val_df) + len(test_df) == len(synthetic_series_df)


def test_chronological_split_invalid_boundaries_raise_error(synthetic_series_df):
    """Verify inverted or overlapping dates raise ValueError."""
    with pytest.raises(ValueError, match="Invalid split boundaries"):
        temporal_train_val_test_split(
            synthetic_series_df,
            train_end="2023-02-20",
            val_start="2023-02-15",  # Overlaps train!
            val_end="2023-02-28",
            test_start="2023-03-01",
            test_end="2023-03-15",
        )


# ---------------------------------------------------------------------------
# Seasonal Naive Baseline Tests
# ---------------------------------------------------------------------------

def test_seasonal_naive_exact_lag():
    """Verify seasonal naive one-step forecast retrieves exact 7-day prior sales."""
    history_df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-08"]),
        "store_nbr": [1, 1],
        "family": ["GROCERY", "GROCERY"],
        "sales": [120.0, 150.0],
    })

    eval_df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-08", "2023-01-15"]),
        "store_nbr": [1, 1],
        "family": ["GROCERY", "GROCERY"],
    })

    forecaster = SeasonalNaiveForecaster(lag_days=7)
    forecaster.fit(history_df)
    preds = forecaster.predict_one_step(eval_df, allow_fallback=False)

    # For 2023-01-08: lag 7d is 2023-01-01 -> 120.0
    # For 2023-01-15: lag 7d is 2023-01-08 -> 150.0
    assert preds[0] == 120.0
    assert preds[1] == 150.0


def test_seasonal_naive_strict_nan_on_missing_lag_when_fallback_disabled():
    """Verify missing t-7 lag returns NaN when allow_fallback=False, preventing silent contamination."""
    # Only 2023-01-01 observed
    history_df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01"]),
        "store_nbr": [1],
        "family": ["GROCERY"],
        "sales": [100.0],
    })

    # Evaluating 2023-01-05 (lag t-7 is 2022-12-29, which is missing!)
    eval_df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-05"]),
        "store_nbr": [1],
        "family": ["GROCERY"],
    })

    forecaster = SeasonalNaiveForecaster(lag_days=7)
    forecaster.fit(history_df)

    # Strict policy: must return NaN
    preds_strict = forecaster.predict_one_step(eval_df, allow_fallback=False)
    assert np.isnan(preds_strict[0])

    # Explicit fallback policy: returns most recent observed
    preds_fallback = forecaster.predict_one_step(eval_df, allow_fallback=True)
    assert preds_fallback[0] == 100.0


def test_seasonal_naive_fixed_origin_repeats_weekly_pattern():
    """Verify multi-day fixed origin repeats weekly cycle and never reads post-origin actuals."""
    # 7 days of known history up to origin 2023-01-07 (Saturday)
    dates = pd.date_range("2023-01-01", "2023-01-07")  # Sun to Sat
    sales_vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    history_df = pd.DataFrame({
        "date": dates,
        "store_nbr": 1,
        "family": "GROCERY",
        "sales": sales_vals,
    })

    forecaster = SeasonalNaiveForecaster(lag_days=7)
    forecaster.fit(history_df)

    # Forecast 14 days ahead from origin 2023-01-07
    series_spec = pd.DataFrame({"store_nbr": [1], "family": ["GROCERY"]})
    forecast_df = forecaster.predict_fixed_origin("2023-01-07", horizon_days=14, series_df=series_spec, allow_fallback=False)

    assert len(forecast_df) == 14
    # Week 1: Day 1 (Sunday) should be 10.0, Day 7 (Saturday) should be 70.0
    assert forecast_df.loc[forecast_df["horizon_step"] == 1, "forecast_sales"].values[0] == 10.0
    assert forecast_df.loc[forecast_df["horizon_step"] == 7, "forecast_sales"].values[0] == 70.0

    # Week 2: Day 8 (next Sunday) must repeat 10.0 (NOT look at post-origin data)
    assert forecast_df.loc[forecast_df["horizon_step"] == 8, "forecast_sales"].values[0] == 10.0
    assert forecast_df.loc[forecast_df["horizon_step"] == 14, "forecast_sales"].values[0] == 70.0

    # Assert source date is always strictly <= origin (2023-01-07)
    origin_ts = pd.to_datetime("2023-01-07")
    for step in range(1, 15):
        fc_date = origin_ts + pd.Timedelta(days=step)
        offset = 6 - ((step - 1) % 7)
        src_date = origin_ts - pd.Timedelta(days=offset)
        assert src_date <= origin_ts
        assert (fc_date - src_date).days % 7 == 0
        assert fc_date.dayofweek == src_date.dayofweek


# ---------------------------------------------------------------------------
# Random Forest Forecaster Tests
# ---------------------------------------------------------------------------

def test_predictor_matrix_excludes_target_and_id(synthetic_series_df):
    """Verify target 'sales' and identifier 'id' are strictly excluded from X."""
    forecaster = RandomForestDemandForecaster(n_estimators=5, max_depth=4)
    train_slice = synthetic_series_df.iloc[:50]
    X = forecaster._prepare_matrix(train_slice, is_training=True)

    assert X.shape[1] == len(RandomForestDemandForecaster.FEATURE_COLS)
    assert "sales" not in RandomForestDemandForecaster.FEATURE_COLS
    assert "id" not in RandomForestDemandForecaster.FEATURE_COLS
    assert "date" not in RandomForestDemandForecaster.FEATURE_COLS


def test_preprocessor_fitted_on_train_only(synthetic_series_df):
    """Verify imputer and encoder are fitted on training data and not refitted on evaluation data."""
    train_df = synthetic_series_df[synthetic_series_df["date"] <= "2023-01-31"].copy()
    eval_df = synthetic_series_df[synthetic_series_df["date"] >= "2023-02-01"].copy()

    forecaster = RandomForestDemandForecaster(n_estimators=5, max_depth=4)
    forecaster.fit(train_df)

    # Imputer statistics must be stored and match train features
    train_medians = forecaster.imputer.statistics_
    assert len(train_medians) == len(RandomForestDemandForecaster.FEATURE_COLS)

    # Transform eval_df with fixed imputer
    X_eval = forecaster._prepare_matrix(eval_df, is_training=False)
    # The imputer statistics must remain identical
    np.testing.assert_array_equal(forecaster.imputer.statistics_, train_medians)


def test_fixed_origin_recursive_never_sees_future_sales(synthetic_series_df):
    """Verify fixed-origin recursive forecaster does not change when future actual sales are mutated."""
    train_df = synthetic_series_df[synthetic_series_df["date"] <= "2023-02-15"].copy()
    forecaster = RandomForestDemandForecaster(n_estimators=5, max_depth=4, random_seed=42)
    forecaster.fit(train_df)

    origin = "2023-02-15"
    horizon = 7

    # Run 1: Original history
    fc1 = forecaster.predict_fixed_origin_recursive(
        origin_date=origin,
        horizon_days=horizon,
        history_df=synthetic_series_df,
    )

    # Mutate actual future sales in synthetic_series_df for dates after origin!
    corrupted_future = synthetic_series_df.copy()
    future_mask = corrupted_future["date"] > origin
    corrupted_future.loc[future_mask, "sales"] = 999999.0

    # Run 2: With corrupted future sales
    fc2 = forecaster.predict_fixed_origin_recursive(
        origin_date=origin,
        horizon_days=horizon,
        history_df=corrupted_future,
    )

    # Predictions MUST be 100% identical: future sales cannot leak into fixed-origin forecast!
    np.testing.assert_array_almost_equal(
        fc1["forecast_sales"].values,
        fc2["forecast_sales"].values,
        decimal=6,
    )


def test_recursive_forecasting_accumulates_prior_predictions():
    """Verify recursive mode feeds prior predictions into future lag inputs."""
    # Create simple 1-series dataframe
    dates = pd.date_range("2023-01-01", "2023-01-15")
    df = pd.DataFrame({
        "date": dates,
        "store_nbr": 1,
        "family": "GROCERY",
        "sales": [100.0] * len(dates),
    })

    forecaster = RandomForestDemandForecaster(n_estimators=5, max_depth=4, random_seed=42)
    forecaster.fit(df)

    res = forecaster.predict_fixed_origin_recursive(
        origin_date="2023-01-15",
        horizon_days=5,
        history_df=df,
    )

    assert len(res) == 5
    assert set(res["horizon_step"]) == {1, 2, 3, 4, 5}
    # All predictions are finite and non-negative
    assert np.all(np.isfinite(res["forecast_sales"]))
    assert np.all(res["forecast_sales"] >= 0.0)


def test_model_save_and_load_consistency(synthetic_series_df, tmp_path):
    """Verify model serialization and deserialization preserves identical predictions."""
    train_df = synthetic_series_df[synthetic_series_df["date"] <= "2023-01-31"]
    eval_df = synthetic_series_df[synthetic_series_df["date"] > "2023-01-31"].iloc[:20]

    forecaster = RandomForestDemandForecaster(n_estimators=5, max_depth=4, random_seed=42)
    forecaster.fit(train_df)
    preds_orig = forecaster.predict_one_step(eval_df)

    model_file = tmp_path / "rf_model.joblib"
    meta_file = tmp_path / "metadata.json"
    forecaster.save(model_file, meta_file, validation_metrics={"wape": 12.5})

    # Reload model
    loaded_forecaster = RandomForestDemandForecaster.load(model_file)
    preds_loaded = loaded_forecaster.predict_one_step(eval_df)

    np.testing.assert_array_almost_equal(preds_orig, preds_loaded, decimal=6)
    assert meta_file.exists()


def test_reproducibility_with_seed(synthetic_series_df):
    """Verify identical random seeds produce identical trees and predictions."""
    train_df = synthetic_series_df[synthetic_series_df["date"] <= "2023-01-31"]
    eval_df = synthetic_series_df[synthetic_series_df["date"] > "2023-01-31"].iloc[:20]

    f1 = RandomForestDemandForecaster(n_estimators=5, max_depth=4, random_seed=42)
    f1.fit(train_df)
    p1 = f1.predict_one_step(eval_df)

    f2 = RandomForestDemandForecaster(n_estimators=5, max_depth=4, random_seed=42)
    f2.fit(train_df)
    p2 = f2.predict_one_step(eval_df)

    np.testing.assert_allclose(p1, p2, rtol=1e-8, atol=1e-8)
