"""Tests for the Streamlit decision-support dashboard and data loaders.

Validates cached data loaders, policy derivation math, chart builder functions,
and view callable status.
"""

from __future__ import annotations

import os
from typing import Any, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.components.charts import (
    build_ablation_ladder_chart,
    build_cost_breakdown_bar_chart,
    build_day_of_week_seasonality_chart,
    build_forecast_vs_actual_chart,
    build_inventory_trajectory_chart,
    build_sales_trend_chart,
)
from app.data_loader import (
    compute_policy_derivation,
    load_ablation_results,
    load_forecast_evaluation_report,
    load_inventory_config,
    load_inventory_planning_report,
    load_inventory_policies,
    load_predictions_fixed_origin,
    load_predictions_one_step,
    load_project_config,
    load_purchase_orders,
    load_sensitivity_results,
    load_simulation_evaluation_report,
    load_simulation_results,
)
from app.streamlit_app import main as streamlit_main
from app.views.executive_overview import render_executive_overview
from app.views.forecast_performance import render_forecast_performance
from app.views.inventory_planning import render_inventory_planning
from app.views.purchase_orders import render_purchase_orders
from app.views.sales_exploration import render_sales_exploration
from app.views.simulation_analysis import render_simulation_analysis


# =====================================================================
# 1. Data Loader & Artifact Integrity Tests
# =====================================================================

def test_load_project_and_inventory_configs() -> None:
    """Verify that project and inventory configs load correctly with expected structure."""
    proj_cfg = load_project_config()
    assert isinstance(proj_cfg, dict)
    assert "data_paths" in proj_cfg
    assert "modeling" in proj_cfg

    inv_cfg = load_inventory_config()
    assert isinstance(inv_cfg, dict)
    assert inv_cfg["default_lead_time_days"] == 7
    assert inv_cfg["default_holding_cost_annual_rate"] == 0.20
    assert inv_cfg["default_fixed_order_cost_usd"] == 50.0
    assert inv_cfg["stockout_penalty_multiplier"] == 1.5


def test_load_reports() -> None:
    """Verify that JSON evaluation reports load and contain expected benchmark keys."""
    fc_rep = load_forecast_evaluation_report()
    assert isinstance(fc_rep, dict)
    assert "mode_a_one_step_ahead" in fc_rep
    assert "mode_b_fixed_origin_recursive" in fc_rep

    inv_rep = load_inventory_planning_report()
    assert isinstance(inv_rep, dict)
    assert "recommendation_summary" in inv_rep
    assert inv_rep["series_count"] == 1782

    sim_rep = load_simulation_evaluation_report()
    assert isinstance(sim_rep, dict)
    assert "policy_comparison_summary" in sim_rep
    assert "sensitivity_analysis_summary" in sim_rep


def test_load_inventory_policies() -> None:
    """Verify that inventory policies DataFrame has exactly 1,782 series and valid columns."""
    policies_df = load_inventory_policies()
    assert isinstance(policies_df, pd.DataFrame)
    assert len(policies_df) == 1782
    expected_cols = [
        "store_nbr", "family", "daily_demand_mean", "annual_demand",
        "lead_time_days", "lead_time_demand", "service_level", "service_level_z",
        "lead_time_std", "safety_stock", "reorder_point", "eoq_unconstrained",
        "eoq_recommended", "unit_cost", "order_cost", "holding_cost_annual_rate",
        "holding_cost_unit_year", "quality_flag",
    ]
    for col in expected_cols:
        assert col in policies_df.columns
    assert policies_df["reorder_point"].isna().sum() == 0
    assert (policies_df["safety_stock"] >= 0).all()
    assert (policies_df["reorder_point"] >= policies_df["safety_stock"]).all()


def test_load_purchase_orders() -> None:
    """Verify purchase orders schema, unit counts, and purchase spend separation."""
    orders_df = load_purchase_orders()
    assert isinstance(orders_df, pd.DataFrame)
    assert len(orders_df) == 1782
    assert "order_recommended" in orders_df.columns
    assert "recommended_order_qty" in orders_df.columns
    assert "estimated_purchase_cost" in orders_df.columns
    assert "urgency_priority" in orders_df.columns

    # Verify order count and spend match Phase 6 contract
    triggered = orders_df[orders_df["order_recommended"] == True]
    assert len(triggered) == 299
    assert triggered["recommended_order_qty"].sum() == 46445.0
    assert triggered["estimated_purchase_cost"].sum() == 464450.0

    # Verify capital math: spend = qty * unit_cost ($10)
    calculated_spend = (triggered["recommended_order_qty"] * triggered["unit_cost"]).sum()
    assert pytest.approx(calculated_spend, rel=1e-5) == 464450.0


def test_load_simulation_results() -> None:
    """Verify simulation holdout results contain expected metrics and columns."""
    sim_df = load_simulation_results()
    assert isinstance(sim_df, pd.DataFrame)
    assert len(sim_df) == 50
    assert "baseline_fill_rate" in sim_df.columns
    assert "optimized_fill_rate" in sim_df.columns
    assert "baseline_operating_cost" in sim_df.columns
    assert "optimized_operating_cost" in sim_df.columns
    assert "cost_delta_usd" in sim_df.columns
    assert sim_df["balance_verified"].all()


def test_load_predictions_and_sensitivities() -> None:
    """Verify prediction artifacts and sensitivity tables load cleanly."""
    p_one = load_predictions_one_step()
    assert isinstance(p_one, pd.DataFrame)
    assert not p_one.empty
    assert "forecast_rf" in p_one.columns
    assert "forecast_naive" in p_one.columns

    p_fixed = load_predictions_fixed_origin()
    assert isinstance(p_fixed, pd.DataFrame)
    assert not p_fixed.empty

    ablation = load_ablation_results()
    assert isinstance(ablation, pd.DataFrame)
    assert not ablation.empty

    sens = load_sensitivity_results()
    assert isinstance(sens, pd.DataFrame)
    assert not sens.empty


# =====================================================================
# 2. Mathematical Policy Derivation Test
# =====================================================================

def test_compute_policy_derivation() -> None:
    """Verify step-by-step mathematical breakdown for a known series."""
    policies_df = load_inventory_policies()
    orders_df = load_purchase_orders()

    policy_row = policies_df[
        (policies_df["store_nbr"] == 1) & (policies_df["family"] == "GROCERY I")
    ].iloc[0]
    po_row = orders_df[
        (orders_df["store_nbr"] == 1) & (orders_df["family"] == "GROCERY I")
    ].iloc[0]

    deriv = compute_policy_derivation(
        policy_row=policy_row,
        po_row=po_row,
    )

    assert isinstance(deriv, dict)
    assert deriv["store_nbr"] == 1
    assert deriv["family"] == "GROCERY I"
    assert deriv["lead_time_days"] == 7
    assert deriv["annual_holding_rate"] == 0.20
    assert pytest.approx(deriv["daily_holding_cost_unit"], rel=1e-4) == (0.20 * 10.0 / 365.0)
    assert deriv["reorder_point"] >= deriv["safety_stock"]

    # Test inventory position equation: IP = OH + OO - BO
    expected_ip = deriv["on_hand"] + deriv["on_order"] - deriv["backorders"]
    assert pytest.approx(deriv["inventory_position"], rel=1e-4) == expected_ip

    # Test trigger logic
    expected_triggered = deriv["inventory_position"] <= deriv["reorder_point"]
    assert deriv["order_recommended"] == expected_triggered


# =====================================================================
# 3. Chart Builder Component Tests
# =====================================================================

def test_build_sales_trend_chart() -> None:
    """Verify sales trend Plotly figure generation."""
    sample_data = pd.DataFrame({
        "date": pd.date_range("2017-01-01", periods=10),
        "sales": np.random.uniform(50, 150, 10),
        "onpromotion": [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    })
    fig = build_sales_trend_chart(sample_data, "Test Sales Trend")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # sales line + promo markers


def test_build_day_of_week_seasonality_chart() -> None:
    """Verify DOW seasonality Plotly figure generation from daily dates and sales."""
    sample_data = pd.DataFrame({
        "date": pd.date_range("2017-01-01", periods=14),
        "sales": [100, 110, 105, 120, 180, 250, 230, 95, 115, 100, 125, 190, 260, 240],
    })
    fig = build_day_of_week_seasonality_chart(sample_data, "Test DOW Seasonality")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_build_forecast_vs_actual_chart() -> None:
    """Verify forecast comparison Plotly figure generation."""
    sample_data = pd.DataFrame({
        "date": pd.date_range("2017-07-19", periods=5),
        "sales": [100, 120, 110, 130, 115],
        "forecast_rf": [102, 118, 112, 128, 116],
        "forecast_naive": [95, 125, 105, 135, 110],
    })
    fig = build_forecast_vs_actual_chart(sample_data, "Test Forecast Comparison")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # Actual, RF Forecast, Baseline Naive


def test_build_inventory_trajectory_chart() -> None:
    """Verify inventory trajectory Plotly figure generation."""
    sample_data = pd.DataFrame({
        "date": pd.date_range("2017-07-19", periods=5),
        "ending_on_hand": [50, 40, 30, 75, 65],
        "demand": [10, 15, 12, 8, 14],
        "receipts": [0, 0, 50, 0, 0],
        "order_placed": [0, 50, 0, 0, 0],
        "lost_sales": [0, 0, 0, 0, 0],
    })
    fig = build_inventory_trajectory_chart(sample_data, rop=45.0, safety_stock=25.0, title="Test Trajectory")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_build_cost_breakdown_bar_chart() -> None:
    """Verify cost breakdown comparison Plotly figure generation."""
    baseline_costs = {
        "holding_cost_usd": 500.0,
        "ordering_cost_usd": 150.0,
        "shortage_penalty_usd": 50.0,
        "total_operating_cost_usd": 700.0,
    }
    optimized_costs = {
        "holding_cost_usd": 750.0,
        "ordering_cost_usd": 100.0,
        "shortage_penalty_usd": 0.0,
        "total_operating_cost_usd": 850.0,
    }
    fig = build_cost_breakdown_bar_chart(baseline_costs, optimized_costs, "Test Cost Breakdown")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Baseline bar and Optimized bar


def test_build_ablation_ladder_chart() -> None:
    """Verify ablation progression ladder Plotly figure generation."""
    ablation_summary = [
        {"stage": "Stage 0", "stockout_days": 86, "cost_operating_total": 750739.0},
        {"stage": "Stage 1", "stockout_days": 42, "cost_operating_total": 380000.0},
        {"stage": "Stage 2", "stockout_days": 1, "cost_operating_total": 95856.0},
    ]
    fig = build_ablation_ladder_chart(ablation_summary)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Stockouts bar + Cost line


# =====================================================================
# 4. View and Entry Point Callables Test
# =====================================================================

def test_view_functions_are_callable() -> None:
    """Verify that all dashboard view renderers and main app function are valid callables."""
    assert callable(render_executive_overview)
    assert callable(render_sales_exploration)
    assert callable(render_forecast_performance)
    assert callable(render_inventory_planning)
    assert callable(render_simulation_analysis)
    assert callable(render_purchase_orders)
    assert callable(streamlit_main)


# =====================================================================
# 5. Interactive Execution & Filter Validation Tests
# =====================================================================

def test_all_view_renderers_execute_without_exceptions() -> None:
    """Verify that all six page views and main app function execute without uncaught exceptions."""
    import unittest.mock as mock

    def mock_selectbox(label, options, index=0, **kwargs):
        if isinstance(options, list) and len(options) > 0:
            return options[index] if index < len(options) else options[0]
        return "ALL"

    def mock_radio(label, options, index=0, **kwargs):
        if isinstance(options, list) and len(options) > 0:
            return options[index] if index < len(options) else options[0]
        return "Triggered Orders Only"

    with mock.patch("streamlit.set_page_config"), \
         mock.patch("streamlit.sidebar.radio", return_value="📊 Executive Overview"), \
         mock.patch("streamlit.radio", side_effect=mock_radio), \
         mock.patch("streamlit.selectbox", side_effect=mock_selectbox), \
         mock.patch("streamlit.multiselect", return_value=["RELIABLE", "SPARSE_SHRUNK"]), \
         mock.patch("streamlit.text_input", return_value=""), \
         mock.patch("streamlit.date_input", return_value=[pd.to_datetime("2017-01-01").date(), pd.to_datetime("2017-08-15").date()]), \
         mock.patch("streamlit.button", autospec=True, return_value=False), \
         mock.patch("streamlit.toggle", autospec=True, return_value=False), \
         mock.patch("streamlit.slider", autospec=True, return_value=7), \
         mock.patch("streamlit.select_slider", autospec=True, return_value=0.95), \
         mock.patch("streamlit.plotly_chart"), \
         mock.patch("streamlit.dataframe"), \
         mock.patch("streamlit.download_button"):

        # Render all six pages
        render_executive_overview()
        render_sales_exploration()
        render_forecast_performance()
        render_inventory_planning()
        render_simulation_analysis()
        render_purchase_orders()
        streamlit_main()


def test_simulation_scenario_parameter_sensitivities() -> None:
    """Verify that scenario controls (CSL, lead time, DOS, shortage multiplier) impact simulations."""
    from scipy import stats
    from src.retail_forecasting.inventory import calculate_reorder_point, calculate_safety_stock
    from src.retail_forecasting.simulation import InventoryPolicy, InventorySimulator

    policies_df = load_inventory_policies()
    preds_df = load_predictions_one_step()

    pol_row = policies_df[(policies_df["store_nbr"] == 1) & (policies_df["family"] == "GROCERY I")].iloc[0]
    holdout_df = preds_df[
        (preds_df["partition"] == "holdout")
        & (preds_df["store_nbr"] == 1)
        & (preds_df["family"] == "GROCERY I")
    ].sort_values("date")

    demand = holdout_df["sales"].values
    dates = holdout_df["date"].values
    d_mean = float(pol_row["daily_demand_mean"])
    base_lt_std = float(pol_row["lead_time_std"])
    eoq = float(pol_row["eoq_recommended"])
    unit_c = float(pol_row["unit_cost"])

    # 1. Config A: Baseline CSL=0.95, LT=7, DOS=14, Penalty=1.5x
    z_a = float(stats.norm.ppf(0.95))
    ss_a = calculate_safety_stock(base_lt_std, z_a)
    rop_a = calculate_reorder_point(d_mean * 7, ss_a)
    sim_a = InventorySimulator(
        initial_stock=d_mean * 14,
        lead_time_days=7,
        unit_cost=unit_c,
        order_cost=50.0,
        holding_cost_annual_rate=0.20,
        shortage_penalty_per_unit=1.5 * unit_c,
        backlog_allowed=False,
    )
    pol_a = InventoryPolicy("p_a", "rop_eoq", rop=rop_a, order_qty=eoq, review_period_days=1, min_order_qty=10, batch_multiple=5)
    res_a = sim_a.run(demand, pol_a, dates)

    # 2. Config B: Stressed CSL=0.99, LT=14, DOS=7, Penalty=2.5x
    z_b = float(stats.norm.ppf(0.99))
    lt_std_b = base_lt_std * (np.sqrt(14.0) / np.sqrt(7.0))
    ss_b = calculate_safety_stock(lt_std_b, z_b)
    rop_b = calculate_reorder_point(d_mean * 14, ss_b)
    sim_b = InventorySimulator(
        initial_stock=d_mean * 7,
        lead_time_days=14,
        unit_cost=unit_c,
        order_cost=50.0,
        holding_cost_annual_rate=0.20,
        shortage_penalty_per_unit=2.5 * unit_c,
        backlog_allowed=False,
    )
    pol_b = InventoryPolicy("p_b", "rop_eoq", rop=rop_b, order_qty=eoq, review_period_days=1, min_order_qty=10, batch_multiple=5)
    res_b = sim_b.run(demand, pol_b, dates)

    # Verify initial inventory change
    assert pytest.approx(res_b.summary_metrics["initial_stock"] * 2.0, rel=1e-4) == res_a.summary_metrics["initial_stock"]

    # Verify CSL updates safety stock and ROP
    assert ss_b > ss_a
    assert rop_b > rop_a

    # Verify lead time delay induces stockouts in stressed Config B
    assert res_a.summary_metrics["stockout_days"] == 0
    assert res_b.summary_metrics["stockout_days"] > 0
    assert res_b.summary_metrics["cost_shortage_penalty"] > 0.0

    # 3. Invariance Test: When policy and physical parameters are fixed, changing shortage multiplier
    # ONLY affects shortage cost, leaving physical trajectories bitwise identical
    sim_pen1 = InventorySimulator(initial_stock=d_mean * 7, lead_time_days=14, unit_cost=unit_c, order_cost=50.0, holding_cost_annual_rate=0.20, shortage_penalty_per_unit=1.0 * unit_c, backlog_allowed=False)
    sim_pen3 = InventorySimulator(initial_stock=d_mean * 7, lead_time_days=14, unit_cost=unit_c, order_cost=50.0, holding_cost_annual_rate=0.20, shortage_penalty_per_unit=3.0 * unit_c, backlog_allowed=False)
    res_pen1 = sim_pen1.run(demand, pol_b, dates)
    res_pen3 = sim_pen3.run(demand, pol_b, dates)

    assert np.allclose(res_pen1.daily_history["ending_on_hand"], res_pen3.daily_history["ending_on_hand"])
    assert np.allclose(res_pen1.daily_history["sales"], res_pen3.daily_history["sales"])
    assert np.allclose(res_pen1.daily_history["lost_sales"], res_pen3.daily_history["lost_sales"])
    assert pytest.approx(res_pen3.summary_metrics["cost_shortage_penalty"], rel=1e-4) == 3.0 * res_pen1.summary_metrics["cost_shortage_penalty"]


def test_purchase_orders_filtering_and_csv_export_invariants() -> None:
    """Verify purchase order filtering and export match UI presentation."""
    orders_df = load_purchase_orders()
    display_cols = [
        "store_nbr", "family", "urgency_priority", "order_recommended",
        "recommended_order_qty", "estimated_purchase_cost", "on_hand",
        "on_order", "backorders", "inventory_position", "safety_stock",
        "reorder_point", "order_date", "expected_delivery_date",
    ]
    rename_map = {
        "store_nbr": "Store",
        "family": "Product Family",
        "urgency_priority": "Priority",
        "order_recommended": "Order Triggered?",
        "recommended_order_qty": "Order Qty (units)",
        "estimated_purchase_cost": "Purchase Spend ($)",
        "on_hand": "On-Hand",
        "on_order": "On-Order",
        "backorders": "Backorders",
        "inventory_position": "IP",
        "safety_stock": "Safety Stock",
        "reorder_point": "ROP",
        "order_date": "Order Date",
        "expected_delivery_date": "Expected Delivery",
    }

    # Filter 1: Triggered Orders Only
    triggered_df = orders_df[orders_df["order_recommended"] == True]
    assert len(triggered_df) == 299
    assert triggered_df["recommended_order_qty"].sum() == 46445.0
    assert triggered_df["estimated_purchase_cost"].sum() == 464450.0

    # Filter 2: High priority
    high_df = triggered_df[triggered_df["urgency_priority"] == "HIGH"]
    assert len(high_df) == 162

    # CSV Export format and row count parity
    export_df = high_df[display_cols].rename(columns=rename_map)
    csv_str = export_df.to_csv(index=False)
    csv_lines = [line for line in csv_str.strip().split("\n") if line]
    assert len(csv_lines) == 163  # 1 header + 162 data rows
    assert "Purchase Spend ($)" in csv_lines[0]
    assert "Store" in csv_lines[0]


def test_missing_artifact_helpful_error_message() -> None:
    """Verify that querying a missing artifact raises a descriptive error explaining pipeline steps."""
    from app.data_loader import load_json_artifact, load_csv_artifact

    with pytest.raises(FileNotFoundError, match="Required pipeline artifact 'configs/non_existent.json' is missing"):
        load_json_artifact("configs/non_existent.json")

    with pytest.raises(FileNotFoundError, match="Required pipeline CSV artifact 'outputs/non_existent.csv' is missing"):
        load_csv_artifact("outputs/non_existent.csv")
