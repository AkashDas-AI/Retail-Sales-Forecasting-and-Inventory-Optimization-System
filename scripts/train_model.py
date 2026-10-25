"""Production Model Training and Evaluation Pipeline for Favorita Demand Forecasting.

Orchestrates:
1. Ingesting real feature matrix (data/processed/sales_features.csv).
2. Chronological data splitting into Train (2016-01-01 to 2017-06-20), Validation (2017-06-21 to 2017-07-18), and Final Holdout (2017-07-19 to 2017-08-15).
3. Training Seasonal Naive baseline and Random Forest demand forecaster.
4. Mode A Evaluation: One-step-ahead rolling predictions.
5. Mode B Evaluation: Fixed-origin multi-day recursive forecasting over 14-day horizon.
6. Exporting fitted models, predictions, and comprehensive metrics reports.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.retail_forecasting.forecasting import (
    RandomForestDemandForecaster,
    SeasonalNaiveForecaster,
    temporal_train_val_test_split,
)
from src.retail_forecasting.metrics import evaluate_forecast_metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI options for training pipeline."""
    parser = argparse.ArgumentParser(description="Train and evaluate demand forecasting engine.")
    parser.add_argument(
        "--features-file",
        type=str,
        default="data/processed/sales_features.csv",
        help="Path to feature matrix (default: data/processed/sales_features.csv)",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default="2016-01-01",
        help="Earliest training date (default: 2016-01-01 to capture recent seasonal dynamics)",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2017-06-20",
        help="Training cutoff date (default: 2017-06-20)",
    )
    parser.add_argument(
        "--val-start",
        type=str,
        default="2017-06-21",
        help="Validation start date (default: 2017-06-21)",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default="2017-07-18",
        help="Validation end date (default: 2017-07-18, 28 days)",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default="2017-07-19",
        help="Holdout test start date (default: 2017-07-19)",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default="2017-08-15",
        help="Holdout test end date (default: 2017-08-15, 28 days)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=60,
        help="Number of trees in Random Forest (default: 60)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=16,
        help="Maximum tree depth (default: 16)",
    )
    parser.add_argument(
        "--recursive-horizon",
        type=int,
        default=14,
        help="Horizon days for fixed-origin recursive forecast (default: 14)",
    )
    return parser.parse_args()


def main() -> int:
    """Execute end-to-end forecasting pipeline."""
    args = parse_args()
    t_start_total = time.time()

    feat_path = (PROJECT_ROOT / args.features_file).resolve()
    model_dir = PROJECT_ROOT / "models"
    output_dir = PROJECT_ROOT / "outputs" / "forecasts"
    report_dir = PROJECT_ROOT / "reports" / "metrics"

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Corporación Favorita — Demand Forecasting Engine (Phase 5)")
    print("=" * 80)
    print(f"Feature Source:     {feat_path}")
    print(f"Train Window:       {args.train_start} to {args.train_end}")
    print(f"Validation Window:  {args.val_start} to {args.val_end} (28 days)")
    print(f"Holdout Window:     {args.test_start} to {args.test_end} (28 days)")
    print(f"Model:              RandomForestRegressor (n_estimators={args.n_estimators}, max_depth={args.max_depth})")
    print("-" * 80)

    if not feat_path.exists():
        print(f"[FATAL ERROR] Features file not found at: '{feat_path}'", file=sys.stderr)
        return 1

    try:
        # 1. Ingest features
        t0 = time.time()
        print(f"[INFO] Ingesting feature matrix from '{feat_path}'...")
        df = pd.read_csv(feat_path, parse_dates=["date"])
        print(f"[INFO] Loaded {len(df):,} rows ({df.shape[1]} columns) in {time.time() - t0:.2f}s.")

        # 2. Chronological data splitting
        print("[INFO] Creating chronological train, validation, and holdout splits...")
        train_df, val_df, holdout_df = temporal_train_val_test_split(
            df,
            date_col="date",
            train_start=args.train_start,
            train_end=args.train_end,
            val_start=args.val_start,
            val_end=args.val_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        print(f"  Training Set:   {len(train_df):,} rows ({train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')})")
        print(f"  Validation Set: {len(val_df):,} rows ({val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')})")
        print(f"  Holdout Set:    {len(holdout_df):,} rows ({holdout_df['date'].min().strftime('%Y-%m-%d')} to {holdout_df['date'].max().strftime('%Y-%m-%d')})")

        # 3. Fit Seasonal Naive baseline
        print("\n[INFO] Fitting Seasonal Naive baseline on training history...")
        t_sn = time.time()
        naive_forecaster = SeasonalNaiveForecaster(lag_days=7)
        naive_forecaster.fit(train_df, date_col="date", target_col="sales")
        print(f"[INFO] Seasonal Naive baseline fitted in {time.time() - t_sn:.2f}s.")

        # 4. Fit Random Forest forecaster
        print("\n[INFO] Fitting Random Forest demand forecaster on training partition...")
        t_rf = time.time()
        rf_forecaster = RandomForestDemandForecaster(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_features="sqrt",
            min_samples_leaf=4,
            random_seed=42,
            n_jobs=-1,
        )
        rf_forecaster.fit(train_df, target_col="sales", date_col="date")
        print(f"[INFO] Random Forest forecaster fitted in {time.time() - t_rf:.2f}s.")

        # -------------------------------------------------------------------
        # Mode A: One-Step-Ahead Rolling Evaluation
        # -------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("MODE A: One-Step-Ahead Rolling Evaluation")
        print("=" * 80)
        print("[INFO] Evaluating on Validation and Holdout partitions...")

        # In Mode A rolling evaluation, actual observations up to t - 1 are legitimately available.
        # Primary benchmark: Exact same-series calendar lag at t - 7 days (allow_fallback=False).
        naive_val_forecaster = SeasonalNaiveForecaster(lag_days=7)
        naive_val_forecaster.fit(pd.concat([train_df, val_df], ignore_index=True), date_col="date", target_col="sales")

        val_eval = val_df.copy()
        val_eval["forecast_naive"] = naive_val_forecaster.predict_one_step(val_df, allow_fallback=False)
        val_eval["forecast_rf"] = rf_forecaster.predict_one_step(val_df)

        val_metrics_naive = evaluate_forecast_metrics(val_eval, actual_col="sales", forecast_col="forecast_naive", group_cols=["family"])
        val_metrics_rf = evaluate_forecast_metrics(val_eval, actual_col="sales", forecast_col="forecast_rf", group_cols=["family"])

        print(f"  [Validation] Seasonal Naive WAPE: {val_metrics_naive['overall']['wape_pct']:.2f}% | MAE: {val_metrics_naive['overall']['mae']:.2f} | RMSE: {val_metrics_naive['overall']['rmse']:.2f} | Bias: {val_metrics_naive['overall']['bias_pct']:.2f}%")
        print(f"  [Validation] Random Forest  WAPE: {val_metrics_rf['overall']['wape_pct']:.2f}% | MAE: {val_metrics_rf['overall']['mae']:.2f} | RMSE: {val_metrics_rf['overall']['rmse']:.2f} | Bias: {val_metrics_rf['overall']['bias_pct']:.2f}%")

        # Final Holdout evaluations (Mode A: rolling one-step using actuals through t - 1)
        naive_holdout_forecaster = SeasonalNaiveForecaster(lag_days=7)
        naive_holdout_forecaster.fit(pd.concat([train_df, val_df, holdout_df], ignore_index=True), date_col="date", target_col="sales")

        holdout_eval = holdout_df.copy()
        holdout_eval["forecast_naive"] = naive_holdout_forecaster.predict_one_step(holdout_df, allow_fallback=False)
        holdout_eval["forecast_rf"] = rf_forecaster.predict_one_step(holdout_df)

        holdout_metrics_naive = evaluate_forecast_metrics(holdout_eval, actual_col="sales", forecast_col="forecast_naive", group_cols=["family"])
        holdout_metrics_rf = evaluate_forecast_metrics(holdout_eval, actual_col="sales", forecast_col="forecast_rf", group_cols=["family"])

        print(f"  [Holdout]    Seasonal Naive WAPE: {holdout_metrics_naive['overall']['wape_pct']:.2f}% | MAE: {holdout_metrics_naive['overall']['mae']:.2f} | RMSE: {holdout_metrics_naive['overall']['rmse']:.2f} | Bias: {holdout_metrics_naive['overall']['bias_pct']:.2f}%")
        print(f"  [Holdout]    Random Forest  WAPE: {holdout_metrics_rf['overall']['wape_pct']:.2f}% | MAE: {holdout_metrics_rf['overall']['mae']:.2f} | RMSE: {holdout_metrics_rf['overall']['rmse']:.2f} | Bias: {holdout_metrics_rf['overall']['bias_pct']:.2f}%")

        # Export Mode A predictions
        one_step_preds_path = output_dir / "predictions_one_step.csv"
        export_cols = ["date", "store_nbr", "family", "sales", "forecast_naive", "forecast_rf"]
        combined_one_step = pd.concat([
            val_eval[export_cols].assign(partition="validation"),
            holdout_eval[export_cols].assign(partition="holdout"),
        ], ignore_index=True)
        combined_one_step.to_csv(one_step_preds_path, index=False)
        print(f"[INFO] Exported one-step predictions to '{one_step_preds_path}'.")

        # -------------------------------------------------------------------
        # Mode B: Fixed-Origin Multi-Day Recursive Forecasting
        # -------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("MODE B: Fixed-Origin Multi-Day Recursive Forecasting")
        print("=" * 80)
        origin_date = args.val_end  # 2017-07-18 (day before holdout starts)
        horizon = args.recursive_horizon
        print(f"Forecast Origin (T0): {origin_date}")
        print(f"Forecast Horizon:     {horizon} days ahead")

        # Load holidays and stores metadata for recursive calendar matching
        h_path = PROJECT_ROOT / "data" / "raw" / "holidays_events.csv"
        s_path = PROJECT_ROOT / "data" / "raw" / "stores.csv"
        holidays_df = pd.read_csv(h_path) if h_path.exists() else None
        stores_df = pd.read_csv(s_path) if s_path.exists() else None

        # Fixed-origin Seasonal Naive forecast (strictly using history <= origin_date)
        t_rec_naive = time.time()
        print("[INFO] Computing fixed-origin Seasonal Naive forecast...")
        pre_origin_history = pd.concat([train_df, val_df], ignore_index=True)
        naive_fixed_forecaster = SeasonalNaiveForecaster(lag_days=7)
        naive_fixed_forecaster.fit(pre_origin_history, date_col="date", target_col="sales")

        naive_fixed_df = naive_fixed_forecaster.predict_fixed_origin(
            origin_date=origin_date,
            horizon_days=horizon,
            series_df=holdout_df[["store_nbr", "family"]].drop_duplicates(),
            allow_fallback=False,
        )
        print(f"[INFO] Generated {len(naive_fixed_df):,} baseline forecasts in {time.time() - t_rec_naive:.2f}s.")

        # Fixed-origin Random Forest recursive forecast
        t_rec_rf = time.time()
        print("[INFO] Generating autoregressive recursive Random Forest forecasts...")
        # History available through origin is Train + Validation
        pre_origin_history = pd.concat([train_df, val_df], ignore_index=True)
        rf_fixed_df = rf_forecaster.predict_fixed_origin_recursive(
            origin_date=origin_date,
            horizon_days=horizon,
            history_df=pre_origin_history,
            holidays_df=holidays_df,
            stores_df=stores_df,
        )
        print(f"[INFO] Generated {len(rf_fixed_df):,} recursive forecasts in {time.time() - t_rec_rf:.2f}s.")

        # Merge actual ground truth from holdout for evaluation
        actual_slice = holdout_df[["date", "store_nbr", "family", "sales"]].copy()
        
        fixed_eval = rf_fixed_df.rename(columns={"forecast_sales": "forecast_rf"}).merge(
            naive_fixed_df.rename(columns={"forecast_sales": "forecast_naive"}),
            on=["date", "store_nbr", "family", "horizon_step"],
            how="inner",
        ).merge(
            actual_slice,
            on=["date", "store_nbr", "family"],
            how="inner",
        )

        fixed_metrics_naive = evaluate_forecast_metrics(
            fixed_eval, actual_col="sales", forecast_col="forecast_naive", group_cols=["horizon_step", "family"]
        )
        fixed_metrics_rf = evaluate_forecast_metrics(
            fixed_eval, actual_col="sales", forecast_col="forecast_rf", group_cols=["horizon_step", "family"]
        )

        print(f"  [Fixed-Origin 14d] Seasonal Naive WAPE: {fixed_metrics_naive['overall']['wape_pct']:.2f}% | MAE: {fixed_metrics_naive['overall']['mae']:.2f} | RMSE: {fixed_metrics_naive['overall']['rmse']:.2f}")
        print(f"  [Fixed-Origin 14d] Random Forest  WAPE: {fixed_metrics_rf['overall']['wape_pct']:.2f}% | MAE: {fixed_metrics_rf['overall']['mae']:.2f} | RMSE: {fixed_metrics_rf['overall']['rmse']:.2f}")

        # Export Mode B predictions
        fixed_preds_path = output_dir / "predictions_fixed_origin.csv"
        fixed_eval.to_csv(fixed_preds_path, index=False)
        print(f"[INFO] Exported fixed-origin predictions to '{fixed_preds_path}'.")

        # -------------------------------------------------------------------
        # Save Model Artifact & Metadata
        # -------------------------------------------------------------------
        model_file = model_dir / "random_forest_demand_model.joblib"
        meta_file = model_dir / "model_metadata.json"
        print(f"\n[INFO] Saving model artifact to '{model_file}'...")
        rf_forecaster.save(
            model_path=model_file,
            metadata_path=meta_file,
            validation_metrics=val_metrics_rf["overall"],
        )
        print(f"[INFO] Saved model metadata to '{meta_file}'.")

        # -------------------------------------------------------------------
        # Compile Comprehensive Metrics Report
        # -------------------------------------------------------------------
        rep_file = report_dir / "forecast_evaluation_report.json"
        full_report = {
            "execution_metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_elapsed_seconds": round(time.time() - t_start_total, 2),
                "features_input_file": str(feat_path),
                "model_type": "RandomForestRegressor",
                "hyperparameters": {
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "max_features": "sqrt",
                    "min_samples_leaf": 4,
                    "random_seed": 42,
                },
                "split_boundaries": {
                    "train_start": args.train_start,
                    "train_end": args.train_end,
                    "val_start": args.val_start,
                    "val_end": args.val_end,
                    "holdout_start": args.test_start,
                    "holdout_end": args.test_end,
                },
                "feature_count": len(rf_forecaster.feature_names),
                "feature_names": rf_forecaster.feature_names,
            },
            "mode_a_one_step_ahead": {
                "validation": {
                    "seasonal_naive": val_metrics_naive["overall"],
                    "random_forest": val_metrics_rf["overall"],
                    "by_family_rf": val_metrics_rf.get("by_family", {}),
                },
                "holdout": {
                    "seasonal_naive": holdout_metrics_naive["overall"],
                    "random_forest": holdout_metrics_rf["overall"],
                    "by_family_rf": holdout_metrics_rf.get("by_family", {}),
                },
            },
            "mode_b_fixed_origin_recursive": {
                "forecast_origin": origin_date,
                "horizon_days": horizon,
                "seasonal_naive": fixed_metrics_naive["overall"],
                "random_forest": fixed_metrics_rf["overall"],
                "by_horizon_rf": fixed_metrics_rf.get("by_horizon_step", {}),
                "by_horizon_naive": fixed_metrics_naive.get("by_horizon_step", {}),
                "by_family_rf": fixed_metrics_rf.get("by_family", {}),
            },
        }

        with open(rep_file, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        print(f"[INFO] Persisted comprehensive evaluation report to '{rep_file}'.")

        # -------------------------------------------------------------------
        # Kaggle test.csv (Inference only, unobserved ground truth)
        # -------------------------------------------------------------------
        raw_test_path = PROJECT_ROOT / "data" / "raw" / "test.csv"
        if raw_test_path.exists():
            print("\n[INFO] Generating multi-step recursive forecasts for 'data/raw/test.csv' (2017-08-16 to 2017-08-31)...")
            test_raw = pd.read_csv(raw_test_path)
            full_history = df[["date", "sales", "store_nbr", "family"]].copy()
            test_preds_df = rf_forecaster.predict_fixed_origin_recursive(
                origin_date=args.test_end,  # 2017-08-15
                horizon_days=16,
                history_df=full_history,
                holidays_df=holidays_df,
                stores_df=stores_df,
            )
            # Merge with test_raw to preserve test.csv id order
            test_preds_df["date"] = pd.to_datetime(test_preds_df["date"]).dt.strftime("%Y-%m-%d")
            test_raw["date"] = pd.to_datetime(test_raw["date"]).dt.strftime("%Y-%m-%d")
            submission_df = test_raw.merge(
                test_preds_df[["date", "store_nbr", "family", "forecast_sales"]],
                on=["date", "store_nbr", "family"],
                how="left",
            ).rename(columns={"forecast_sales": "sales"})

            test_out_path = output_dir / "test_predictions.csv"
            submission_df[["id", "sales"]].to_csv(test_out_path, index=False)
            print(f"[INFO] Exported test inference predictions to '{test_out_path}' (Note: unobserved ground truth; not scored).")

        print("\n" + "=" * 80)
        print("[SUCCESS] Phase 5 Model Training & Evaluation completed successfully.")
        print(f"Total Pipeline Runtime: {time.time() - t_start_total:.2f}s")
        print("=" * 80)
        return 0

    except Exception as exc:
        print(f"\n[FATAL ERROR] Training workflow aborted:\n{exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
