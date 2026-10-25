"""Recommendation generation pipeline script.

Combines out-of-sample forecast validation errors and demand predictions
to estimate series-level demand uncertainty, compute dynamic Safety Stocks (SS),
Reorder Points (ROP), and Economic Order Quantities (EOQ), and output prioritized
replenishment purchase orders for all 1,782 store-family planning units.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retail_forecasting.inventory import (
    calculate_inventory_policy,
    estimate_series_demand_uncertainty,
)
from src.retail_forecasting.recommendations import generate_purchase_orders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_recommendations")


def load_json_config(path: Path) -> Dict[str, Any]:
    """Load JSON configuration file safely."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_recommendation_pipeline(
    project_config_path: Path = PROJECT_ROOT / "configs" / "project_config.json",
    inventory_config_path: Path = PROJECT_ROOT / "configs" / "inventory_config.json",
    predictions_path: Path = PROJECT_ROOT / "outputs" / "forecasts" / "predictions_one_step.csv",
    output_dir: Path = PROJECT_ROOT / "outputs" / "recommendations",
    reports_dir: Path = PROJECT_ROOT / "reports" / "metrics",
) -> Dict[str, Any]:
    """Execute end-to-end inventory policy calculation and purchase order recommendation."""
    logger.info("Initializing inventory recommendation pipeline...")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    proj_cfg = load_json_config(project_config_path)
    inv_cfg = load_json_config(inventory_config_path)

    logger.info(f"Loading out-of-sample forecast predictions from: {predictions_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Forecast predictions artifact missing: {predictions_path}")

    all_preds_df = pd.read_csv(predictions_path)

    # Strictly extract validation partition for uncertainty estimation
    # to avoid contaminating policy parameters with final holdout observations!
    val_df = all_preds_df[all_preds_df["partition"] == "validation"].copy()
    if val_df.empty:
        raise ValueError("No validation partition records found in predictions_one_step.csv")

    logger.info(
        f"Calibrating demand uncertainty across {val_df.shape[0]} validation records "
        f"({val_df['date'].min()} to {val_df['date'].max()})..."
    )

    unc_cfg = inv_cfg.get("uncertainty_policy", {})
    shrinkage_k = int(unc_cfg.get("shrinkage_k", 7))
    floor_std = float(unc_cfg.get("residual_std_floor", 0.05))
    compounding_mode = str(unc_cfg.get("lead_time_compounding_mode", "ar1_analytical"))
    ar1_rho = float(unc_cfg.get("ar1_rho_fallback", 0.25))

    lead_time_days = int(inv_cfg.get("default_lead_time_days", 7))
    service_level = float(inv_cfg.get("target_service_level", 0.95))
    holding_cost_rate = float(inv_cfg.get("default_holding_cost_annual_rate", 0.20))
    order_cost = float(inv_cfg.get("default_fixed_order_cost_usd", 50.0))
    unit_cost = float(inv_cfg.get("default_unit_cost_usd", 10.0))
    min_order_qty = float(inv_cfg.get("minimum_order_quantity", 10.0))
    batch_multiple = float(inv_cfg.get("batch_order_multiples", 5.0))

    # 1. Estimate series-level demand uncertainty
    uncertainty_df = estimate_series_demand_uncertainty(
        forecast_df=val_df,
        pred_col="forecast_rf",
        actual_col="sales",
        shrinkage_k=shrinkage_k,
        floor_std=floor_std,
        lead_time_days=lead_time_days,
        compounding_mode=compounding_mode,
        ar1_rho=ar1_rho,
    )
    logger.info(f"Uncertainty calibrated for {len(uncertainty_df)} series.")

    # 2. Calculate inventory policy parameters (SS, ROP, EOQ)
    logger.info("Computing Safety Stock, Reorder Point, and EOQ for all series...")
    policy_df = calculate_inventory_policy(
        forecast_demand=val_df,
        series_uncertainty=uncertainty_df,
        lead_time_days=lead_time_days,
        service_level=service_level,
        holding_cost_rate=holding_cost_rate,
        order_cost=order_cost,
        unit_cost=unit_cost,
        min_order_qty=min_order_qty,
        batch_multiple=batch_multiple,
        demand_col="forecast_rf",
    )

    # 3. Simulate current inventory position at origin date
    # In absence of observed ERP stock, assume initial stock = 7 days of supply
    logger.info("Simulating initial inventory states (assumption: 7 days of supply)...")
    sim_cfg = inv_cfg.get("simulation", {})
    init_dos = float(sim_cfg.get("initial_inventory_days_of_supply", 7.0))

    current_inv_records = []
    for _, row in policy_df.iterrows():
        current_inv_records.append({
            "store_nbr": int(row["store_nbr"]),
            "family": str(row["family"]),
            # Simulated stock position with minor variation for demonstration
            "on_hand": max(0.0, float(row["daily_demand_mean"]) * init_dos),
            "on_order": 0.0,
            "backorders": 0.0,
        })
    current_inv_df = pd.DataFrame(current_inv_records)

    # 4. Generate purchase order recommendations
    planning_date = str(val_df["date"].max())
    logger.info(f"Generating purchase orders for origin date: {planning_date}...")
    po_df = generate_purchase_orders(
        current_inventory_df=current_inv_df,
        policy_df=policy_df,
        lead_time_days=lead_time_days,
        current_date=planning_date,
        default_unit_cost=unit_cost,
    )

    # 5. Export artifacts
    policies_csv = output_dir / "inventory_policies.csv"
    orders_csv = output_dir / "purchase_orders.csv"
    policy_df.to_csv(policies_csv, index=False)
    po_df.to_csv(orders_csv, index=False)
    logger.info(f"Exported inventory policies ({len(policy_df)} rows) to: {policies_csv}")
    logger.info(f"Exported purchase order recommendations ({len(po_df)} rows) to: {orders_csv}")

    # 6. Generate Machine-Readable Planning Report
    orders_triggered = int(po_df["order_recommended"].sum())
    total_spend = float(po_df["estimated_purchase_cost"].sum())
    total_units = float(po_df["recommended_order_qty"].sum())

    flag_counts = policy_df["quality_flag"].value_counts().to_dict()

    report = {
        "report_type": "inventory_planning_and_recommendation_report",
        "timestamp_generated": pd.Timestamp.now().isoformat(),
        "planning_grain": "(store_nbr, family)",
        "series_count": len(policy_df),
        "calibration_window": {
            "partition": "validation",
            "start_date": str(val_df["date"].min()),
            "end_date": str(val_df["date"].max()),
            "total_observations": int(len(val_df)),
        },
        "configured_parameters": {
            "target_service_level_csl": service_level,
            "service_level_z": float(policy_df["service_level_z"].iloc[0]),
            "lead_time_days": lead_time_days,
            "annual_holding_cost_rate": holding_cost_rate,
            "default_unit_cost_usd": unit_cost,
            "annual_holding_cost_per_unit_usd": round(holding_cost_rate * unit_cost, 4),
            "daily_holding_cost_per_unit_usd": round((holding_cost_rate * unit_cost) / 365.0, 8),
            "order_cost_usd": order_cost,
            "minimum_order_quantity": min_order_qty,
            "batch_order_multiple": batch_multiple,
            "uncertainty_shrinkage_k": shrinkage_k,
            "residual_std_floor": floor_std,
            "lead_time_compounding_mode": compounding_mode,
            "ar1_rho_fallback": ar1_rho,
        },
        "data_provenance_and_assumptions": {
            "demand": "Model out-of-sample forecast from validation set (observed/derived)",
            "forecast_residuals": "Actual sales minus model prediction on validation set (observed/derived)",
            "on_hand_inventory": f"Simulated baseline of {init_dos} days of average demand (synthetic assumption)",
            "lead_time": "Configured baseline assumption (7 calendar days)",
            "costs": "Configured baseline assumptions (holding rate 20%, order cost $50, unit cost $10)",
            "shortage_penalty": "Synthetic simulation parameter (not observed)",
        },
        "uncertainty_quality_distribution": flag_counts,
        "recommendation_summary": {
            "total_planning_units": len(po_df),
            "orders_triggered": orders_triggered,
            "percent_units_reordered": round(float(orders_triggered / len(po_df) * 100.0), 2),
            "total_recommended_order_units": round(total_units, 2),
            "total_estimated_purchase_spend_usd": round(total_spend, 2),
            "priority_breakdown": po_df["urgency_priority"].value_counts().to_dict(),
        },
        "policy_metrics_summary": {
            "mean_daily_demand": round(float(policy_df["daily_demand_mean"].mean()), 2),
            "mean_safety_stock": round(float(policy_df["safety_stock"].mean()), 2),
            "mean_reorder_point": round(float(policy_df["reorder_point"].mean()), 2),
            "mean_eoq_recommended": round(float(policy_df["eoq_recommended"].mean()), 2),
            "total_safety_stock_buffer_units": round(float(policy_df["safety_stock"].sum()), 2),
        },
        "output_artifacts": {
            "inventory_policies_file": str(policies_csv),
            "purchase_orders_file": str(orders_csv),
        },
    }

    report_json_path = reports_dir / "inventory_planning_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Inventory planning report generated at: {report_json_path}")

    return report


def main() -> None:
    """CLI entry point for recommendation generation."""
    parser = argparse.ArgumentParser(description="Generate inventory policies and replenishment recommendations.")
    parser.add_argument("--project-config", type=Path, default=PROJECT_ROOT / "configs" / "project_config.json")
    parser.add_argument("--inventory-config", type=Path, default=PROJECT_ROOT / "configs" / "inventory_config.json")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "outputs" / "forecasts" / "predictions_one_step.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "recommendations")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "metrics")

    args = parser.parse_args()
    report = run_recommendation_pipeline(
        project_config_path=args.project_config,
        inventory_config_path=args.inventory_config,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
    )
    print("\n=======================================================")
    print(" INVENTORY PLANNING & RECOMMENDATIONS SUMMARY")
    print("=======================================================")
    print(f"Planning Units Evaluated : {report['series_count']}")
    print(f"Orders Triggered         : {report['recommendation_summary']['orders_triggered']} ({report['recommendation_summary']['percent_units_reordered']}%)")
    print(f"Total Recommended Units  : {report['recommendation_summary']['total_recommended_order_units']:,.2f}")
    print(f"Total Estimated Spend    : ${report['recommendation_summary']['total_estimated_purchase_spend_usd']:,.2f}")
    print(f"Mean Safety Stock        : {report['policy_metrics_summary']['mean_safety_stock']:.2f} units")
    print(f"Mean Reorder Point (ROP) : {report['policy_metrics_summary']['mean_reorder_point']:.2f} units")
    print(f"Mean Recommended EOQ     : {report['policy_metrics_summary']['mean_eoq_recommended']:.2f} units")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
