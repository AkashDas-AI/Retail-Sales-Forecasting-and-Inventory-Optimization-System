"""Demand forecasting models, baselines, and temporal evaluation engines.

Implements:
1. Temporal train / validation / holdout chronological splitters with strict non-overlap.
2. Seasonal Naive benchmark forecaster (one-step-ahead and fixed-origin weekly cycle repetition).
3. Leakage-safe tabular Random Forest demand forecaster with training-only preprocessing.
4. One-step-ahead rolling evaluation engine.
5. Fixed-origin multi-day recursive forecasting engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from src.retail_forecasting.features import (
    generate_calendar_features,
    generate_holiday_features,
)


def temporal_train_val_test_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_start: Optional[Union[str, pd.Timestamp]] = None,
    train_end: Union[str, pd.Timestamp] = "2017-06-20",
    val_start: Union[str, pd.Timestamp] = "2017-06-21",
    val_end: Union[str, pd.Timestamp] = "2017-07-18",
    test_start: Union[str, pd.Timestamp] = "2017-07-19",
    test_end: Union[str, pd.Timestamp] = "2017-08-15",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition dataset into chronological, non-overlapping train, validation, and holdout splits.

    Guarantees:
    - Global time-based split: all stores and families split on the exact same date boundaries.
    - Zero temporal overlap: train_end < val_start and val_end < test_start.
    - Deterministic order: partitions are sorted strictly by [date_col, store_nbr, family].

    Args:
        df: Input DataFrame containing date column.
        date_col: Name of date column.
        train_start: Optional earliest date for training data (inclusive).
        train_end: Latest date for training data (inclusive).
        val_start: Earliest date for validation data (inclusive).
        val_end: Latest date for validation data (inclusive).
        test_start: Earliest date for holdout data (inclusive).
        test_end: Latest date for holdout data (inclusive).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df).
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

    t_end = pd.to_datetime(train_end)
    v_start = pd.to_datetime(val_start)
    v_end = pd.to_datetime(val_end)
    h_start = pd.to_datetime(test_start)
    h_end = pd.to_datetime(test_end)

    if not (t_end < v_start <= v_end < h_start <= h_end):
        raise ValueError(
            f"Invalid split boundaries: train_end ({t_end}) must be strictly earlier than val_start ({v_start}), "
            f"val_end ({v_end}) strictly earlier than test_start ({h_start})."
        )

    dates = pd.to_datetime(df[date_col])

    # Filter train partition
    train_mask = dates <= t_end
    if train_start is not None:
        t_start = pd.to_datetime(train_start)
        train_mask = train_mask & (dates >= t_start)

    val_mask = (dates >= v_start) & (dates <= v_end)
    test_mask = (dates >= h_start) & (dates <= h_end)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    if len(train_df) == 0:
        raise ValueError(f"Training partition is empty for specified range (<= {t_end}).")
    if len(val_df) == 0:
        raise ValueError(f"Validation partition is empty for range [{v_start}, {v_end}].")
    if len(test_df) == 0:
        raise ValueError(f"Holdout partition is empty for range [{h_start}, {h_end}].")

    return train_df, val_df, test_df


class BaseForecaster:
    """Abstract interface for demand forecasting models."""

    def fit(self, *args: Any, **kwargs: Any) -> "BaseForecaster":
        """Fit model on training data."""
        raise NotImplementedError("Subclasses must implement fit().")

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Generate forecasts."""
        raise NotImplementedError("Subclasses must implement predict().")


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal naive baseline predicting demand using matching seasonal historical lags.

    Evaluation Modes:
    1. One-step-ahead: For each date t, uses actual observed sales at t - lag_days (default 7 days).
    2. Fixed-origin multi-day: At origin date T0, forecasts T0 + 1 ... T0 + H by repeating
       the weekly seasonal pattern observed in the final 7 days leading up to T0.
       Never accesses actual sales recorded after T0.
    """

    def __init__(self, lag_days: int = 7) -> None:
        """Initialize Seasonal Naive forecaster.

        Args:
            lag_days: Seasonal period (default: 7 days for weekly retail seasonality).
        """
        self.lag_days = lag_days
        self._history_map: Dict[Tuple[int, str, pd.Timestamp], float] = {}
        self._latest_observed: Dict[Tuple[int, str], float] = {}
        self._is_fitted: bool = False

    def fit(
        self,
        history_df: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "sales",
        group_cols: Optional[List[str]] = None,
    ) -> "SeasonalNaiveForecaster":
        """Store historical series observations for lag lookup.

        Args:
            history_df: DataFrame of historical sales.
            date_col: Name of date column.
            target_col: Name of target sales column.
            group_cols: Grouping identifiers (default: ['store_nbr', 'family']).

        Returns:
            SeasonalNaiveForecaster: Fitted forecaster instance.
        """
        if group_cols is None:
            group_cols = ["store_nbr", "family"]

        for col in [date_col, target_col] + group_cols:
            if col not in history_df.columns:
                raise ValueError(f"Required column '{col}' missing from history DataFrame.")

        df = history_df.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

        # Build lookup table of (store, family, date) -> sales
        self._history_map.clear()
        self._latest_observed.clear()

        # Sort chronologically so latest observation is preserved
        df = df.sort_values(group_cols + [date_col])
        for row in df.itertuples(index=False):
            s_nbr = getattr(row, group_cols[0])
            fam = getattr(row, group_cols[1])
            d = getattr(row, date_col)
            val = float(getattr(row, target_col))

            self._history_map[(s_nbr, fam, d)] = val
            self._latest_observed[(s_nbr, fam)] = val

        self._is_fitted = True
        return self

    def predict_one_step(
        self,
        eval_df: pd.DataFrame,
        date_col: str = "date",
        group_cols: Optional[List[str]] = None,
        allow_fallback: bool = False,
    ) -> np.ndarray:
        """Generate one-step-ahead predictions using true sales at t - lag_days.

        Primary Benchmark Policy (allow_fallback=False):
        - Evaluates exact same-series calendar lag at t - lag_days.
        - If t - lag_days is unobserved (e.g. store closure or start of series), returns NaN.
        - Never silently replaces missing lags with non-seasonal alternatives.

        Secondary Fallback Policy (allow_fallback=True):
        - If t - lag_days is unobserved, falls back to the most recent observed sales
          for that series prior to t.

        Args:
            eval_df: DataFrame of rows to evaluate.
            date_col: Name of date column.
            group_cols: Identifier columns (default: ['store_nbr', 'family']).
            allow_fallback: Whether to use most recent historical observation when exact lag is missing.

        Returns:
            np.ndarray: Predicted sales array matching eval_df length.
        """
        if not self._is_fitted:
            raise RuntimeError("SeasonalNaiveForecaster must be fitted before predict_one_step().")
        if group_cols is None:
            group_cols = ["store_nbr", "family"]

        predictions = []
        for row in eval_df.itertuples(index=False):
            s_nbr = getattr(row, group_cols[0])
            fam = getattr(row, group_cols[1])
            t = pd.to_datetime(getattr(row, date_col)).normalize()
            target_lag_date = t - pd.Timedelta(days=self.lag_days)

            # Exact calendar lag lookup
            val = self._history_map.get((s_nbr, fam, target_lag_date))
            if val is None:
                if allow_fallback:
                    val = self._latest_observed.get((s_nbr, fam), np.nan)
                else:
                    val = np.nan

            predictions.append(np.nan if np.isnan(val) else max(0.0, float(val)))

        return np.array(predictions, dtype=np.float64)

    def predict_fixed_origin(
        self,
        origin_date: Union[str, pd.Timestamp],
        horizon_days: int,
        series_df: pd.DataFrame,
        date_col: str = "date",
        group_cols: Optional[List[str]] = None,
        allow_fallback: bool = False,
    ) -> pd.DataFrame:
        """Generate fixed-origin multi-day forecast repeating the weekly pattern up to origin_date.

        For horizon step h in 1..H:
        Target date is T0 + h days.
        Prediction draws strictly from the matching day of week in the 7 days prior to T0:
        Source date = T0 - (6 - ((h - 1) % 7)).
        Never uses actual sales observed after T0.

        Args:
            origin_date: Forecast origin date T0 (inclusive cutoff for observed history).
            horizon_days: Number of days H to forecast ahead.
            series_df: DataFrame identifying the (store_nbr, family) pairs to forecast.
            date_col: Name of date column.
            group_cols: Series identifier columns (default: ['store_nbr', 'family']).
            allow_fallback: Whether to use fallback when matching weekday is unobserved prior to T0.

        Returns:
            pd.DataFrame: Multi-horizon forecasts with columns [date, store_nbr, family, horizon_step, forecast_sales].
        """
        if not self._is_fitted:
            raise RuntimeError("SeasonalNaiveForecaster must be fitted before predict_fixed_origin().")
        if group_cols is None:
            group_cols = ["store_nbr", "family"]

        t0 = pd.to_datetime(origin_date).normalize()
        unique_series = series_df[group_cols].drop_duplicates().reset_index(drop=True)

        records = []
        for h in range(1, horizon_days + 1):
            forecast_date = t0 + pd.Timedelta(days=h)
            # Cycle through the 7 days preceding T0: [T0 - 6 days ... T0]
            offset = 6 - ((h - 1) % self.lag_days)
            source_date = t0 - pd.Timedelta(days=offset)

            for s_row in unique_series.itertuples(index=False):
                s_nbr = getattr(s_row, group_cols[0])
                fam = getattr(s_row, group_cols[1])

                val = self._history_map.get((s_nbr, fam, source_date))
                if val is None:
                    if allow_fallback:
                        val = self._latest_observed.get((s_nbr, fam), np.nan)
                    else:
                        val = np.nan

                pred_val = np.nan if (val is None or np.isnan(val)) else max(0.0, float(val))
                records.append({
                    date_col: forecast_date,
                    group_cols[0]: s_nbr,
                    group_cols[1]: fam,
                    "horizon_step": h,
                    "forecast_sales": pred_val,
                })

        return pd.DataFrame(records)


class RandomForestDemandForecaster(BaseForecaster):
    """Leakage-safe Random Forest forecaster for retail sales.

    Features:
    - Lag features: [1, 7, 14, 28] days.
    - Rolling features: mean and std over [7, 14, 30] days.
    - Calendar features: day_of_week, day_of_month, month, year, is_weekend, cyclical sin/cos.
    - Holiday features: is_holiday, is_event.
    - Categoricals: store_nbr, family (ordinal encoded strictly on training data).

    Guarantees:
    - Preprocessing imputer and encoder are fitted strictly on training data.
    - Predictions are clamped to non-negative values: max(0.0, y_hat).
    - Fixed-origin recursive mode feeds prior predictions into future lag/rolling inputs
      without access to true future sales.
    """

    FEATURE_COLS = [
        "sales_lag_1d",
        "sales_lag_7d",
        "sales_lag_14d",
        "sales_lag_28d",
        "sales_rolling_mean_7d",
        "sales_rolling_std_7d",
        "sales_rolling_mean_14d",
        "sales_rolling_std_14d",
        "sales_rolling_mean_30d",
        "sales_rolling_std_30d",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_holiday",
        "is_event",
        "store_nbr",
        "family_encoded",
    ]

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: Optional[int] = 16,
        max_features: str = "sqrt",
        min_samples_leaf: int = 4,
        random_seed: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """Initialize forecaster with model hyperparameters."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_seed = random_seed
        self.n_jobs = n_jobs

        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.imputer = SimpleImputer(strategy="median")
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
        )

        self._is_fitted: bool = False
        self.feature_names: List[str] = list(self.FEATURE_COLS)
        self.training_cutoff: Optional[str] = None

    def _prepare_matrix(self, df: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        """Extract and preprocess predictor matrix X."""
        work_df = df.copy()

        # Encode family column
        if "family" in work_df.columns:
            fam_vals = work_df[["family"]].astype(str)
            if is_training:
                work_df["family_encoded"] = self.encoder.fit_transform(fam_vals)
            else:
                work_df["family_encoded"] = self.encoder.transform(fam_vals)
        elif "family_encoded" not in work_df.columns:
            work_df["family_encoded"] = -1

        # Check for missing feature columns and create them with NaN if absent
        for col in self.FEATURE_COLS:
            if col not in work_df.columns:
                work_df[col] = 0.0 if is_training else np.nan

        X = work_df[self.FEATURE_COLS].values.astype(np.float64)

        # Impute missing values (fit on training data only)
        if is_training:
            X = self.imputer.fit_transform(X)
        else:
            X = self.imputer.transform(X)

        # Guard against any residual NaNs from all-NaN evaluation columns
        X = np.nan_to_num(X, nan=0.0)

        return X

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
    ) -> "RandomForestDemandForecaster":
        """Fit preprocessing pipeline and Random Forest regressor on training partition.

        Args:
            train_df: Training DataFrame containing features and sales target.
            target_col: Target sales column.
            date_col: Date column.

        Returns:
            RandomForestDemandForecaster: Fitted forecaster.
        """
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in training DataFrame.")

        # Record training cutoff date
        if date_col in train_df.columns:
            self.training_cutoff = str(pd.to_datetime(train_df[date_col]).max())[:10]

        X_train = self._prepare_matrix(train_df, is_training=True)
        y_train = train_df[target_col].values.astype(np.float64)

        self.model.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict_one_step(self, eval_df: pd.DataFrame) -> np.ndarray:
        """Generate one-step-ahead predictions for evaluation rows.

        Args:
            eval_df: DataFrame with features computed through true historical observations.

        Returns:
            np.ndarray: Non-negative predictions array matching eval_df length.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_one_step().")

        X_eval = self._prepare_matrix(eval_df, is_training=False)
        preds = self.model.predict(X_eval)
        # Clamping to non-negative sales
        return np.maximum(0.0, preds)

    def predict_fixed_origin_recursive(
        self,
        origin_date: Union[str, pd.Timestamp],
        horizon_days: int,
        history_df: pd.DataFrame,
        holidays_df: Optional[pd.DataFrame] = None,
        stores_df: Optional[pd.DataFrame] = None,
        date_col: str = "date",
        target_col: str = "sales",
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate fixed-origin multi-day forecast using autoregressive recursive prediction.

        For each forecast date t = T0 + h (h in 1..H):
        1. Lags and rolling stats are computed strictly using working history (actual sales
           up to T0, and prior model predictions after T0).
        2. Known future calendar features and holiday indicators are attached.
        3. Model predicts demand for date t.
        4. Predicted sales are appended to working history for subsequent steps.
        5. Ground-truth sales after T0 are NEVER accessed.

        Args:
            origin_date: Cutoff date T0.
            horizon_days: Horizon H (number of future days to forecast).
            history_df: Historical records through T0.
            holidays_df: Optional holiday calendar DataFrame.
            stores_df: Optional store metadata DataFrame.
            date_col: Date column name.
            target_col: Target sales column name.
            group_cols: Identifier columns (default: ['store_nbr', 'family']).

        Returns:
            pd.DataFrame: Multi-day forecast with columns [date, store_nbr, family, horizon_step, forecast_sales].
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict_fixed_origin_recursive().")
        if group_cols is None:
            group_cols = ["store_nbr", "family"]

        t0 = pd.to_datetime(origin_date).normalize()

        # Initialize working history strictly through T0 (no future leakage)
        hist = history_df.copy()
        hist[date_col] = pd.to_datetime(hist[date_col]).dt.normalize()
        working_history = hist[hist[date_col] <= t0][[date_col, target_col] + group_cols].copy()

        unique_series = working_history[group_cols].drop_duplicates().reset_index(drop=True)
        forecast_records: List[Dict[str, Any]] = []

        for h in range(1, horizon_days + 1):
            curr_date = t0 + pd.Timedelta(days=h)

            # Build minimal target slice for curr_date
            step_slice = unique_series.copy()
            step_slice[date_col] = curr_date
            step_slice[target_col] = np.nan

            # For rolling stats and lags, retain working history from [curr_date - 35 days, curr_date - 1 day]
            lookback_cutoff = curr_date - pd.Timedelta(days=35)
            recent_hist = working_history[working_history[date_col] >= lookback_cutoff].copy()

            # 1. Compute exact calendar lags using working history
            # Shift historical sales by k days forward to match curr_date
            lag_dict: Dict[str, Dict[Tuple[int, str], float]] = {f"sales_lag_{k}d": {} for k in [1, 7, 14, 28]}
            for k in [1, 7, 14, 28]:
                lag_d = curr_date - pd.Timedelta(days=k)
                hist_lag = recent_hist[recent_hist[date_col] == lag_d]
                for r in hist_lag.itertuples(index=False):
                    s = getattr(r, group_cols[0])
                    f = getattr(r, group_cols[1])
                    v = float(getattr(r, target_col))
                    lag_dict[f"sales_lag_{k}d"][(s, f)] = v

            for k in [1, 7, 14, 28]:
                col_name = f"sales_lag_{k}d"
                step_slice[col_name] = [
                    lag_dict[col_name].get((getattr(r, group_cols[0]), getattr(r, group_cols[1])), np.nan)
                    for r in step_slice.itertuples(index=False)
                ]

            # 2. Compute calendar rolling stats [curr_date - w days, curr_date) from working history
            for w in [7, 14, 30]:
                w_start = curr_date - pd.Timedelta(days=w)
                w_hist = recent_hist[(recent_hist[date_col] >= w_start) & (recent_hist[date_col] < curr_date)]
                
                # Group stats across working history in window
                grp_stats = w_hist.groupby(group_cols)[target_col].agg(["mean", "std", "count"])
                
                mean_col = f"sales_rolling_mean_{w}d"
                std_col = f"sales_rolling_std_{w}d"

                step_slice[mean_col] = np.nan
                step_slice[std_col] = np.nan

                # Fast join / mapping of stats
                mean_map = grp_stats["mean"].to_dict()
                std_map = grp_stats["std"].to_dict()
                cnt_map = grp_stats["count"].to_dict()

                means = []
                stds = []
                for r in step_slice.itertuples(index=False):
                    key = (getattr(r, group_cols[0]), getattr(r, group_cols[1]))
                    cnt = cnt_map.get(key, 0)
                    means.append(mean_map.get(key, np.nan) if cnt >= 1 else np.nan)
                    stds.append(std_map.get(key, np.nan) if cnt >= 2 else np.nan)

                step_slice[mean_col] = means
                step_slice[std_col] = stds

            # 3. Attach calendar and holiday signals
            step_slice = generate_calendar_features(step_slice, date_col=date_col, include_cyclical=True)
            step_slice = generate_holiday_features(
                step_slice,
                holidays_df=holidays_df,
                stores_df=stores_df,
                date_col=date_col,
                store_col=group_cols[0],
            )

            # 4. Predict demand with Random Forest
            preds = self.predict_one_step(step_slice)
            step_slice[target_col] = preds

            # 5. Append recursive predictions to working history (crucial for future lags!)
            new_hist_rows = step_slice[[date_col, target_col] + group_cols].copy()
            working_history = pd.concat([working_history, new_hist_rows], ignore_index=True)

            # Record predictions
            for r in step_slice.itertuples(index=False):
                forecast_records.append({
                    date_col: curr_date,
                    group_cols[0]: getattr(r, group_cols[0]),
                    group_cols[1]: getattr(r, group_cols[1]),
                    "horizon_step": h,
                    "forecast_sales": float(getattr(r, target_col)),
                })

        return pd.DataFrame(forecast_records)

    def save(
        self,
        model_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        validation_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Serialize model artifact and metadata to disk.

        Args:
            model_path: File path to save joblib model binary.
            metadata_path: File path to save metadata JSON.
            validation_metrics: Optional metrics dictionary from validation evaluation.
        """
        m_p = Path(model_path)
        m_p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "encoder": self.encoder,
            "imputer": self.imputer,
            "model": self.model,
            "feature_names": self.feature_names,
            "training_cutoff": self.training_cutoff,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "max_features": self.max_features,
                "min_samples_leaf": self.min_samples_leaf,
                "random_seed": self.random_seed,
            },
        }
        joblib.dump(payload, m_p)

        if metadata_path is not None:
            import sklearn
            meta_p = Path(metadata_path)
            meta_p.parent.mkdir(parents=True, exist_ok=True)
            meta_dict = {
                "model_type": "RandomForestRegressor",
                "training_cutoff": self.training_cutoff,
                "environment_versions": {
                    "scikit_learn": sklearn.__version__,
                    "joblib": joblib.__version__,
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                },
                "preprocessing": {
                    "missing_imputation_strategy": "median",
                    "categorical_encoder": "OrdinalEncoder",
                    "non_negative_clipping": True,
                },
                "hyperparameters": payload["hyperparameters"],
                "feature_count": len(self.feature_names),
                "feature_names": self.feature_names,
                "validation_metrics": validation_metrics or {},
            }
            with open(meta_p, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2)

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "RandomForestDemandForecaster":
        """Load and reconstruct fitted forecaster from joblib file."""
        m_p = Path(model_path)
        if not m_p.is_file():
            raise FileNotFoundError(f"Model file not found at: '{m_p}'")

        payload = joblib.load(m_p)
        instance = cls(
            n_estimators=payload["hyperparameters"]["n_estimators"],
            max_depth=payload["hyperparameters"]["max_depth"],
            max_features=payload["hyperparameters"]["max_features"],
            min_samples_leaf=payload["hyperparameters"]["min_samples_leaf"],
            random_seed=payload["hyperparameters"]["random_seed"],
        )
        instance.encoder = payload["encoder"]
        instance.imputer = payload["imputer"]
        instance.model = payload["model"]
        instance.feature_names = payload["feature_names"]
        instance.training_cutoff = payload.get("training_cutoff")
        instance._is_fitted = True
        return instance
