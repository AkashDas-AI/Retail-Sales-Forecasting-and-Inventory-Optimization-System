"""Unit tests for forecasting evaluation metrics."""

import math
import numpy as np
import pandas as pd
import pytest

from src.retail_forecasting import metrics


def test_metrics_module_importable():
    """Verify metrics module defines standard evaluation functions."""
    assert hasattr(metrics, "calculate_mae")
    assert hasattr(metrics, "calculate_rmse")
    assert hasattr(metrics, "calculate_wape")
    assert hasattr(metrics, "calculate_bias")
    assert hasattr(metrics, "calculate_service_level")
    assert hasattr(metrics, "evaluate_forecast_metrics")


def test_mae_calculation_correctness():
    """Verify MAE calculation matches exact hand-calculated values."""
    actual = [10.0, 20.0, 30.0]
    forecast = [12.0, 18.0, 35.0]
    # Errors: |10-12|=2, |20-18|=2, |30-35|=5. Mean: (2+2+5)/3 = 3.0
    assert metrics.calculate_mae(actual, forecast) == 3.0


def test_rmse_calculation_correctness():
    """Verify RMSE calculation matches exact hand-calculated values."""
    actual = [10.0, 20.0, 30.0]
    forecast = [13.0, 16.0, 30.0]
    # Errors: 3^2=9, (-4)^2=16, 0^2=0. Mean: 25/3. Sqrt: sqrt(25/3) = 2.886751...
    expected = math.sqrt(25.0 / 3.0)
    assert np.isclose(metrics.calculate_rmse(actual, forecast), expected)


def test_wape_calculation_correctness():
    """Verify WAPE calculation matches mathematical definition."""
    actual = [100.0, 200.0, 300.0]  # sum = 600.0
    forecast = [110.0, 180.0, 330.0]  # abs diffs = 10, 20, 30 -> sum = 60.0
    # WAPE = (60.0 / 600.0) * 100% = 10.0%
    assert metrics.calculate_wape(actual, forecast) == 10.0


def test_wape_zero_actual_handling():
    """Verify explicit handling of zero actual sales denominator."""
    # Case 1: Both actual and forecast are zero -> 0.0% error
    assert metrics.calculate_wape([0.0, 0.0], [0.0, 0.0]) == 0.0

    # Case 2: Actual is zero, forecast is non-zero -> float('inf')
    assert metrics.calculate_wape([0.0, 0.0], [10.0, 5.0]) == float("inf")


def test_bias_calculation_correctness():
    """Verify normalized signed forecast bias and direction semantics."""
    actual = [100.0, 100.0]  # sum = 200.0

    # Over-forecasting: forecast > actual -> positive bias
    over_forecast = [110.0, 120.0]  # sum error = +30.0 -> +15.0%
    assert metrics.calculate_bias(actual, over_forecast) == 15.0

    # Under-forecasting: forecast < actual -> negative bias
    under_forecast = [90.0, 80.0]  # sum error = -30.0 -> -15.0%
    assert metrics.calculate_bias(actual, under_forecast) == -15.0

    # Zero actuals handling
    assert metrics.calculate_bias([0.0], [0.0]) == 0.0
    assert metrics.calculate_bias([0.0], [10.0]) == float("inf")
    assert metrics.calculate_bias([0.0], [-10.0]) == float("-inf")


def test_invalid_input_shapes_raise_error():
    """Verify empty inputs or mismatched lengths raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        metrics.calculate_mae([], [])

    with pytest.raises(ValueError, match="Length mismatch"):
        metrics.calculate_wape([1.0, 2.0], [1.0])


def test_evaluate_forecast_metrics_grouped_weighting():
    """Verify multi-group WAPE is correctly weighted by sales volume."""
    # Group A: actual=1000, error=100 -> WAPE = 10%
    # Group B: actual=10, error=5 -> WAPE = 50%
    # Unweighted average of WAPEs: (10 + 50) / 2 = 30%
    # Correct global WAPE: (100 + 5) / (1000 + 10) = 105 / 1010 = 10.396%
    df = pd.DataFrame({
        "family": ["A", "B"],
        "actual": [1000.0, 10.0],
        "forecast": [1100.0, 15.0],
    })

    report = metrics.evaluate_forecast_metrics(df, group_cols=["family"])

    assert report["overall"]["wape_pct"] == round((105.0 / 1010.0) * 100.0, 4)
    assert report["by_family"]["A"]["wape_pct"] == 10.0
    assert report["by_family"]["B"]["wape_pct"] == 50.0
