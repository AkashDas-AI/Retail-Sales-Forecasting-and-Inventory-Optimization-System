"""Executive Overview dashboard page.

Presents portfolio-level KPIs spanning demand forecasting accuracy, inventory planning scope,
and retrospective discrete-event simulation performance.
"""

from typing import Any, Dict

import streamlit as st

from app.components.charts import (
    build_ablation_ladder_chart,
    build_cost_breakdown_bar_chart,
)
from app.components.disclaimers import (
    render_banner_disclaimer,
    render_retrospective_evaluation_notice,
)
from app.data_loader import (
    load_forecast_evaluation_report,
    load_inventory_planning_report,
    load_simulation_evaluation_report,
)


def render_executive_overview() -> None:
    """Render the Executive Overview dashboard view."""
    st.header("📊 Executive Portfolio Overview")
    render_banner_disclaimer()

    # Load machine-readable reports
    forecast_rep = load_forecast_evaluation_report()
    plan_rep = load_inventory_planning_report()
    sim_rep = load_simulation_evaluation_report()

    # Row 1: High-Level Forecasting & Catalog KPIs
    st.subheader("1. Catalog Scale & Demand Forecasting Baseline Beat")
    c1, c2, c3, c4 = st.columns(4)

    mode_a_holdout = forecast_rep.get("mode_a_one_step_ahead", {}).get("holdout", {})
    rf_holdout_wape = mode_a_holdout.get("random_forest", {}).get("wape_pct", 12.87)
    naive_holdout_wape = mode_a_holdout.get("seasonal_naive", {}).get("wape_pct", 18.51)
    wape_delta = rf_holdout_wape - naive_holdout_wape

    with c1:
        st.metric(
            label="Total Planning Units",
            value=f"{plan_rep.get('series_count', 1782):,}",
            help="1,782 distinct (store, family) planning units across 54 stores and 33 merchandise categories.",
        )
    with c2:
        st.metric(
            label="Evaluation Horizon",
            value="28 Days",
            delta="2017-07-19 to 08-15",
            delta_color="off",
            help="Final 4-week Kaggle competition holdout period.",
        )
    with c3:
        st.metric(
            label="Forecasting Model WAPE",
            value=f"{rf_holdout_wape:.2f}%",
            delta=f"{wape_delta:.2f}% vs Baseline",
            delta_color="inverse",
            help="Random Forest Demand Forecaster aggregate WAPE across holdout partition.",
        )
    with c4:
        st.metric(
            label="Seasonal Naive Baseline",
            value=f"{naive_holdout_wape:.2f}%",
            help="Mode A lag-7 seasonal naive benchmark (pure t-7 sales repetition).",
        )

    st.markdown("---")

    # Row 2: Inventory Planning & Replenishment Orders
    st.subheader("2. Inventory Policy Buffers & Immediate Replenishment Advice")
    rec_sum = plan_rep.get("recommendation_summary", {})
    pol_sum = plan_rep.get("policy_metrics_summary", {})

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric(
            label="Orders Recommended",
            value=f"{rec_sum.get('orders_triggered', 299):,}",
            delta=f"{rec_sum.get('percent_units_reordered', 16.78):.1f}% of catalog",
            delta_color="off",
            help="Planning units where Inventory Position (IP) <= Reorder Point (ROP).",
        )
    with c6:
        st.metric(
            label="Total Recommended Units",
            value=f"{rec_sum.get('total_recommended_order_units', 46445.0):,.0f}",
            help="Total batch-constrained replenishment units recommended across triggered orders.",
        )
    with c7:
        st.metric(
            label="Estimated Purchase Spend",
            value=f"${rec_sum.get('total_estimated_purchase_spend_usd', 464450.0):,.2f}",
            help="Estimated inventory acquisition expenditure (Q * unit_cost). Note: Working capital investment, NOT an operating friction cost.",
        )
    with c8:
        st.metric(
            label="Catalog Safety Stock Buffer",
            value=f"{pol_sum.get('total_safety_stock_buffer_units', 726628.86):,.0f} units",
            help=f"Mean SS buffer: {pol_sum.get('mean_safety_stock', 407.76):.1f} units/series across 1,782 series.",
        )

    st.markdown("---")

    # Row 3: Discrete-Event Simulation Backtest Comparison
    st.subheader("3. Discrete-Event Simulation: Baseline vs. Optimized ROP+EOQ Policy")
    render_retrospective_evaluation_notice()

    sim_comp = sim_rep.get("policy_comparison_summary", {})
    base_pol = sim_comp.get("baseline_policy", {})
    opt_pol = sim_comp.get("optimized_policy", {})

    col_sim1, col_sim2, col_sim3, col_sim4 = st.columns(4)
    with col_sim1:
        base_fill = base_pol.get("volume_weighted_fill_rate", 97.65)
        opt_fill = opt_pol.get("volume_weighted_fill_rate", 99.89)
        st.metric(
            label="Volume-Weighted Fill Rate",
            value=f"{opt_fill:.2f}%",
            delta=f"+{opt_fill - base_fill:.2f}% vs Baseline ({base_fill:.2f}%)",
            help="Total sales issued divided by total customer demand across 50 representative holdout series.",
        )
    with col_sim2:
        base_stockouts = base_pol.get("total_stockout_days", 86)
        opt_stockouts = opt_pol.get("total_stockout_days", 1)
        st.metric(
            label="Total Stockout Days",
            value=f"{opt_stockouts}",
            delta=f"{opt_stockouts - base_stockouts} days (-98.8%)",
            delta_color="inverse",
            help="Cumulative days where customer demand exceeded available stock.",
        )
    with col_sim3:
        base_cost = base_pol.get("total_operating_cost_usd", 750731.99)
        opt_cost = opt_pol.get("total_operating_cost_usd", 95926.40)
        st.metric(
            label="Total Operating Cost",
            value=f"${opt_cost:,.2f}",
            delta=f"-${base_cost - opt_cost:,.2f} (-87.2%)",
            delta_color="inverse",
            help="Holding Cost + Ordering Setup Cost + Shortage Penalty. Strictly excludes purchase spend.",
        )
    with col_sim4:
        st.metric(
            label="Mean Daily On-Hand Inventory",
            value=f"{opt_pol.get('mean_daily_inventory', 6991.35):,.1f} units",
            delta=f"{opt_pol.get('mean_daily_inventory', 6991.35) - base_pol.get('mean_daily_inventory', 8062.43):,.1f} units (-13.3%)",
            delta_color="inverse",
            help="Average physical inventory held in store/warehouse over the 28-day horizon.",
        )

    # Visualizations
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        cost_fig = build_cost_breakdown_bar_chart(
            baseline_costs=base_pol.get("cost_breakdown", {}),
            optimized_costs=opt_pol.get("cost_breakdown", {}),
        )
        st.plotly_chart(cost_fig, use_container_width=True)

    with col_v2:
        ablation_ladder = sim_rep.get("incremental_ablation_ladder", [])
        if ablation_ladder:
            ladder_fig = build_ablation_ladder_chart(ablation_ladder)
            st.plotly_chart(ladder_fig, use_container_width=True)

    # Detailed Assumptions Expander
    with st.expander("ℹ️ Detailed Operational & Financial Configuration Assumptions"):
        cfg_params = sim_rep.get("configured_simulation_parameters", {})
        st.markdown(
            f"""
            - **Replenishment Lead Time (L)**: `{cfg_params.get('lead_time_days', 7)} calendar days` (deterministic benchmark assumption).
            - **Annual Holding Cost Rate (h)**: `{cfg_params.get('annual_holding_cost_rate', 0.20) * 100:.0f}% per year` on unit valuation.
            - **Unit Purchase Valuation (c)**: `${cfg_params.get('unit_cost_usd', 10.0):.2f} / unit` (configured benchmark).
            - **Standard Daily Holding Cost**: `h * c / 365 = ${cfg_params.get('daily_holding_cost_per_unit_usd', 0.00547945):.8f} / unit / day`.
            - **Fixed Ordering Setup Cost (K)**: `${cfg_params.get('order_cost_usd', 50.0):.2f} / order`.
            - **Shortage Stockout Penalty**: `${cfg_params.get('shortage_penalty_per_unit_usd', 15.0):.2f} / unit unmet` (1.5x unit cost illustrative penalty).
            - **Simulated Initial Inventory**: `14 days of supply` based on series average demand.
            - **Target Cycle Service Level (CSL)**: `95.0%` (z = 1.645).
            """
        )

