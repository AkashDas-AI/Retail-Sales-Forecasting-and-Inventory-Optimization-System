"""Evaluation metrics for demand forecasting and inventory performance.

Provides mathematical implementations of retail metrics including WAPE, MAE,
RMSE, Signed Forecast Bias, and horizon/group-level aggregation utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def calculate_mae(actual: Union[np.ndarray, pd.Series, List[float]], forecast: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """Calculate Mean Absolute Error (MAE).

    MAE = (1 / N) * sum(|actual - forecast|)

    Args:
        actual: Ground-truth target values.
        forecast: Predicted forecast values.

    Returns:
        float: Mean Absolute Error.
    """
    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(forecast, dtype=np.float64)

    if len(y_true) == 0:
        raise ValueError("Cannot calculate MAE on empty arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: actual ({len(y_true)}) vs forecast ({len(y_pred)}).")

    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(actual: Union[np.ndarray, pd.Series, List[float]], forecast: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    RMSE = sqrt((1 / N) * sum((actual - forecast)^2))

    Args:
        actual: Ground-truth target values.
        forecast: Predicted forecast values.

    Returns:
        float: Root Mean Squared Error.
    """
    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(forecast, dtype=np.float64)

    if len(y_true) == 0:
        raise ValueError("Cannot calculate RMSE on empty arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: actual ({len(y_true)}) vs forecast ({len(y_pred)}).")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_wape(actual: Union[np.ndarray, pd.Series, List[float]], forecast: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE).

    WAPE = (sum(|actual - forecast|) / sum(|actual|)) * 100%

    Zero-denominator handling:
    - If sum(|actual|) == 0 and sum(|actual - forecast|) == 0: returns 0.0%.
    - If sum(|actual|) == 0 and sum(|actual - forecast|) > 0: returns float('inf').

    Args:
        actual: Ground-truth target values.
        forecast: Predicted forecast values.

    Returns:
        float: WAPE expressed as a percentage (e.g. 15.2 means 15.2%).
    """
    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(forecast, dtype=np.float64)

    if len(y_true) == 0:
        raise ValueError("Cannot calculate WAPE on empty arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: actual ({len(y_true)}) vs forecast ({len(y_pred)}).")

    sum_abs_err = float(np.sum(np.abs(y_true - y_pred)))
    sum_actual = float(np.sum(np.abs(y_true)))

    if sum_actual == 0.0:
        return 0.0 if sum_abs_err == 0.0 else float("inf")

    return float((sum_abs_err / sum_actual) * 100.0)


def calculate_bias(actual: Union[np.ndarray, pd.Series, List[float]], forecast: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """Calculate Normalized Signed Forecast Bias percentage.

    Bias = (sum(forecast - actual) / sum(actual)) * 100%

    Direction convention:
    - Positive Bias (> 0%): Over-forecasting (forecasts systematically exceed actual sales).
    - Negative Bias (< 0%): Under-forecasting (forecasts systematically fall below actual sales).

    Zero-denominator handling:
    - If sum(actual) == 0 and sum(forecast - actual) == 0: returns 0.0%.
    - If sum(actual) == 0 and sum(forecast - actual) != 0: returns float('inf') or float('-inf').

    Args:
        actual: Ground-truth target values.
        forecast: Predicted forecast values.

    Returns:
        float: Percentage bias.
    """
    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(forecast, dtype=np.float64)

    if len(y_true) == 0:
        raise ValueError("Cannot calculate bias on empty arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: actual ({len(y_true)}) vs forecast ({len(y_pred)}).")

    sum_error = float(np.sum(y_pred - y_true))
    sum_actual = float(np.sum(y_true))

    if sum_actual == 0.0:
        if sum_error == 0.0:
            return 0.0
        return float("inf") if sum_error > 0 else float("-inf")

    return float((sum_error / sum_actual) * 100.0)


def calculate_service_level(fulfilled_demand: Any, total_demand: Any) -> float:
    """Calculate Item Fill Rate (IFR).

    IFR = sum(fulfilled_demand) / sum(total_demand)

    TODO (Phase 6): Implement service level metric during inventory evaluation phase.
    """
    raise NotImplementedError("Service level calculation pending inventory evaluation phase.")


def evaluate_forecast_metrics(
    df: pd.DataFrame,
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    group_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate overall and grouped retail forecast performance metrics.

    Ensures correct aggregation weighting: WAPE across multiple series is computed
    by aggregating absolute errors and actual sales globally (sum of absolute errors
    divided by sum of actuals), NOT by unweighted averaging of group percentages.

    Args:
        df: DataFrame containing actual and forecasted values.
        actual_col: Name of actual target column.
        forecast_col: Name of forecast column.
        group_cols: Optional list of columns to group by (e.g. ['family'] or ['horizon_step']).

    Returns:
        Dict[str, Any]: Comprehensive metrics dictionary.
    """
    if len(df) == 0:
        raise ValueError("Cannot evaluate metrics on empty DataFrame.")
    for col in [actual_col, forecast_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    actual = df[actual_col].values
    forecast = df[forecast_col].values

    overall = {
        "sample_count": int(len(df)),
        "total_actual_sales": float(np.sum(actual)),
        "total_forecast_sales": float(np.sum(forecast)),
        "mae": round(calculate_mae(actual, forecast), 4),
        "rmse": round(calculate_rmse(actual, forecast), 4),
        "wape_pct": round(calculate_wape(actual, forecast), 4),
        "bias_pct": round(calculate_bias(actual, forecast), 4),
    }

    report: Dict[str, Any] = {"overall": overall}

    if group_cols:
        for grp_col in group_cols:
            if grp_col not in df.columns:
                continue
            grouped_metrics: Dict[str, Dict[str, Any]] = {}
            for grp_val, grp_df in df.groupby(grp_col):
                y_t = grp_df[actual_col].values
                y_p = grp_df[forecast_col].values
                key = str(grp_val)
                grouped_metrics[key] = {
                    "sample_count": int(len(grp_df)),
                    "total_actual_sales": float(np.sum(y_t)),
                    "total_forecast_sales": float(np.sum(y_p)),
                    "mae": round(calculate_mae(y_t, y_p), 4),
                    "rmse": round(calculate_rmse(y_t, y_p), 4),
                    "wape_pct": round(calculate_wape(y_t, y_p), 4),
                    "bias_pct": round(calculate_bias(y_t, y_p), 4),
                }
            report[f"by_{grp_col}"] = grouped_metrics

    return report
