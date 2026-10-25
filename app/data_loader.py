"""Data loading and caching utilities for the Streamlit decision-support interface.

All functions use robust paths relative to the project root and implement Streamlit caching
to ensure responsive page interactions without repeated disk I/O.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Project root resolution relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Return the resolved project root directory."""
    return PROJECT_ROOT


@st.cache_data(show_spinner=False)
def load_json_artifact(relative_path: str) -> Dict[str, Any]:
    """Load a JSON artifact from a relative path from the project root."""
    target_path = PROJECT_ROOT / relative_path
    if not target_path.exists():
        raise FileNotFoundError(
            f"Required pipeline artifact '{relative_path}' is missing. "
            "Please ensure earlier pipeline stages have executed (e.g., scripts/train_model.py, "
            "scripts/generate_recommendations.py, or scripts/run_simulation.py)."
        )
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv_artifact(relative_path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a CSV artifact from a relative path with optional column projection."""
    target_path = PROJECT_ROOT / relative_path
    if not target_path.exists():
        raise FileNotFoundError(
            f"Required pipeline CSV artifact '{relative_path}' is missing. "
            "Please ensure earlier pipeline stages have executed (e.g., scripts/train_model.py, "
            "scripts/generate_recommendations.py, or scripts/run_simulation.py)."
        )
    return pd.read_csv(target_path, usecols=usecols)


# ---------------------------------------------------------------------------
# Specific Artifact Loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_project_config() -> Dict[str, Any]:
    """Load core project configuration."""
    return load_json_artifact("configs/project_config.json")


@st.cache_data(show_spinner=False)
def load_inventory_config() -> Dict[str, Any]:
    """Load inventory and simulation configuration."""
    return load_json_artifact("configs/inventory_config.json")


@st.cache_data(show_spinner=False)
def load_forecast_evaluation_report() -> Dict[str, Any]:
    """Load forecasting performance metrics report."""
    return load_json_artifact("reports/metrics/forecast_evaluation_report.json")


@st.cache_data(show_spinner=False)
def load_inventory_planning_report() -> Dict[str, Any]:
    """Load inventory planning and recommendations summary report."""
    return load_json_artifact("reports/metrics/inventory_planning_report.json")


@st.cache_data(show_spinner=False)
def load_simulation_evaluation_report() -> Dict[str, Any]:
    """Load simulation and policy comparison evaluation report."""
    return load_json_artifact("reports/metrics/simulation_evaluation_report.json")


@st.cache_data(show_spinner=False)
def load_inventory_policies() -> pd.DataFrame:
    """Load precomputed inventory policies across all 1,782 series."""
    return load_csv_artifact("outputs/recommendations/inventory_policies.csv")


@st.cache_data(show_spinner=False)
def load_purchase_orders() -> pd.DataFrame:
    """Load prioritized purchase order recommendations across all 1,782 series."""
    return load_csv_artifact("outputs/recommendations/purchase_orders.csv")


@st.cache_data(show_spinner=False)
def load_simulation_results() -> pd.DataFrame:
    """Load 50-series holdout simulation comparison results."""
    return load_csv_artifact("outputs/simulation/simulation_results.csv")


@st.cache_data(show_spinner=False)
def load_ablation_results() -> pd.DataFrame:
    """Load incremental ablation study results across the 4 stages."""
    return load_csv_artifact("outputs/simulation/ablation_study_results.csv")


@st.cache_data(show_spinner=False)
def load_sensitivity_results() -> pd.DataFrame:
    """Load 12-scenario sensitivity analysis results across 50 representative series."""
    return load_csv_artifact("outputs/simulation/sensitivity_analysis_results.csv")


@st.cache_data(show_spinner=False)
def load_predictions_one_step() -> pd.DataFrame:
    """Load Mode A rolling one-step predictions for validation and holdout partitions."""
    return load_csv_artifact("outputs/forecasts/predictions_one_step.csv")


@st.cache_data(show_spinner=False)
def load_predictions_fixed_origin() -> pd.DataFrame:
    """Load Mode B recursive fixed-origin predictions for holdout horizon."""
    return load_csv_artifact("outputs/forecasts/predictions_fixed_origin.csv")


@st.cache_data(show_spinner=True)
def load_sales_history_sample(
    store_nbr: Optional[int] = None,
    family: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load and filter daily sales history on demand without loading entire 3M rows repeatedly.

    If store and family are specified, extracts only the relevant series trajectory.
    """
    sales_path = PROJECT_ROOT / "data" / "processed" / "sales_daily.csv"
    if not sales_path.exists():
        raise FileNotFoundError(f"Sales history file not found at: {sales_path}")

    # Usecols projection minimizes memory usage
    cols = ["date", "store_nbr", "family", "sales", "onpromotion"]
    df = pd.read_csv(sales_path, usecols=cols)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    if store_nbr is not None:
        df = df[df["store_nbr"] == int(store_nbr)]
    if family is not None:
        df = df[df["family"] == str(family)]
    if start_date is not None:
        df = df[df["date"] >= str(start_date)]
    if end_date is not None:
        df = df[df["date"] <= str(end_date)]

    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pure Helper Utilities (Easily Testable)
# ---------------------------------------------------------------------------

def compute_policy_derivation(
    policy_row: pd.Series,
    po_row: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Compute step-by-step mathematical derivation breakdown for a single planning unit.

    Parameters
    ----------
    policy_row : pd.Series
        Row from inventory_policies.csv.
    po_row : Optional[pd.Series]
        Matching row from purchase_orders.csv, if available.

    Returns
    -------
    Dict[str, Any]
        Structured dictionary with exact derivation intermediate values.
    """
    d_mean = float(policy_row["daily_demand_mean"])
    annual_d = float(policy_row["annual_demand"])
    lt_days = int(policy_row["lead_time_days"])
    lt_demand = float(policy_row["lead_time_demand"])
    csl = float(policy_row["service_level"])
    z_score = float(policy_row["service_level_z"])
    lt_std = float(policy_row["lead_time_std"])
    ss = float(policy_row["safety_stock"])
    rop = float(policy_row["reorder_point"])
    raw_eoq = float(policy_row["eoq_unconstrained"])
    rec_eoq = float(policy_row["eoq_recommended"])
    unit_cost = float(policy_row["unit_cost"])
    order_cost = float(policy_row["order_cost"])
    annual_h_rate = float(policy_row["holding_cost_annual_rate"])
    annual_h_unit = float(policy_row["holding_cost_unit_year"])
    daily_h_unit = (annual_h_rate * unit_cost) / 365.0
    quality_flag = str(policy_row["quality_flag"])

    on_hand = float(po_row["on_hand"]) if po_row is not None else 0.0
    on_order = float(po_row["on_order"]) if po_row is not None else 0.0
    backorders = float(po_row["backorders"]) if po_row is not None else 0.0
    ip = on_hand + on_order - backorders
    order_recommended = bool(ip <= rop)
    rec_qty = rec_eoq if order_recommended else 0.0
    purchase_spend = rec_qty * unit_cost
    priority = str(po_row["urgency_priority"]) if po_row is not None else ("NORMAL" if order_recommended else "NONE")

    return {
        "store_nbr": int(policy_row["store_nbr"]),
        "family": str(policy_row["family"]),
        "daily_demand_mean": d_mean,
        "annual_demand": annual_d,
        "lead_time_days": lt_days,
        "lead_time_demand": lt_demand,
        "service_level": csl,
        "service_level_z": z_score,
        "lead_time_std": lt_std,
        "safety_stock": ss,
        "reorder_point": rop,
        "eoq_unconstrained": raw_eoq,
        "eoq_recommended": rec_eoq,
        "unit_cost": unit_cost,
        "order_cost": order_cost,
        "annual_holding_rate": annual_h_rate,
        "annual_holding_cost_unit": annual_h_unit,
        "daily_holding_cost_unit": daily_h_unit,
        "quality_flag": quality_flag,
        "on_hand": on_hand,
        "on_order": on_order,
        "backorders": backorders,
        "inventory_position": ip,
        "order_recommended": order_recommended,
        "recommended_order_qty": rec_qty,
        "estimated_purchase_spend": purchase_spend,
        "urgency_priority": priority,
    }

