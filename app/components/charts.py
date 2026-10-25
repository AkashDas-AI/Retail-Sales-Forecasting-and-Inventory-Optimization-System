"""Plotly visualization builders for retail sales, forecast accuracy, and inventory trajectories."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_sales_trend_chart(
    df: pd.DataFrame,
    title: str = "Historical Daily Sales Demand",
    show_promotions: bool = True,
) -> go.Figure:
    """Create an interactive daily sales trend chart with optional promotional markers."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["sales"],
            mode="lines",
            name="Daily Sales (Units)",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="<b>Date</b>: %{x}<br><b>Sales</b>: %{y:,.1f} units<extra></extra>",
        )
    )

    if show_promotions and "onpromotion" in df.columns:
        promo_df = df[df["onpromotion"] > 0]
        if not promo_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=promo_df["date"],
                    y=promo_df["sales"],
                    mode="markers",
                    name="Promotion Active",
                    marker=dict(color="#ff7f0e", size=6, symbol="triangle-up"),
                    hovertemplate="<b>Promo Date</b>: %{x}<br><b>Promo Count</b>: %{customdata} items<extra></extra>",
                    customdata=promo_df["onpromotion"],
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Date", showgrid=True),
        yaxis=dict(title="Sales Units", showgrid=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
    )
    return fig


def build_day_of_week_seasonality_chart(
    df: pd.DataFrame,
    title: str = "Weekly Demand Pattern (Average Daily Sales)",
) -> go.Figure:
    """Create a weekly seasonality bar chart showing average sales by day of week."""
    work_df = df.copy()
    work_df["date_dt"] = pd.to_datetime(work_df["date"])
    work_df["day_name"] = work_df["date_dt"].dt.day_name()
    work_df["day_num"] = work_df["date_dt"].dt.dayofweek

    dow_summary = work_df.groupby(["day_num", "day_name"])["sales"].mean().reset_index()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_summary["day_name"] = pd.Categorical(dow_summary["day_name"], categories=day_order, ordered=True)
    dow_summary = dow_summary.sort_values("day_name")

    fig = px.bar(
        dow_summary,
        x="day_name",
        y="sales",
        labels={"day_name": "Day of Week", "sales": "Average Sales (Units)"},
        title=title,
        color="sales",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def build_forecast_vs_actual_chart(
    df: pd.DataFrame,
    title: str = "Forecast vs Actual Demand (Holdout Evaluation)",
) -> go.Figure:
    """Create a multi-line comparison of actual customer sales against model and baseline forecasts."""
    fig = go.Figure()

    # Actual sales
    if "sales" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["sales"],
                mode="lines",
                name="Actual Sales",
                line=dict(color="#2ca02c", width=2.5),
                hovertemplate="<b>Date</b>: %{x}<br><b>Actual</b>: %{y:,.1f}<extra></extra>",
            )
        )

    # Random forest model forecast
    if "forecast_rf" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["forecast_rf"],
                mode="lines",
                name="Random Forest Model",
                line=dict(color="#1f77b4", width=2, dash="dash"),
                hovertemplate="<b>Date</b>: %{x}<br><b>RF Forecast</b>: %{y:,.1f}<extra></extra>",
            )
        )

    # Seasonal naive baseline forecast
    if "forecast_naive" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["forecast_naive"],
                mode="lines",
                name="Seasonal Naive (t-7)",
                line=dict(color="#d62728", width=1.5, dash="dot"),
                hovertemplate="<b>Date</b>: %{x}<br><b>Naive Baseline</b>: %{y:,.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Date", showgrid=True),
        yaxis=dict(title="Demand Units", showgrid=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
    )
    return fig


def build_inventory_trajectory_chart(
    daily_df: pd.DataFrame,
    rop: Optional[float] = None,
    safety_stock: Optional[float] = None,
    title: str = "Daily Inventory Trajectory and Replenishment Events",
) -> go.Figure:
    """Create a detailed inventory trajectory chart showing on-hand stock, orders, and ROP."""
    fig = go.Figure()

    # Ending On-Hand stock
    fig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["ending_on_hand"],
            mode="lines+markers",
            name="Ending On-Hand Stock",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>Date</b>: %{x}<br><b>On-Hand</b>: %{y:,.1f} units<extra></extra>",
        )
    )

    # Customer Demand
    fig.add_trace(
        go.Bar(
            x=daily_df["date"],
            y=daily_df["demand"],
            name="Daily Demand",
            marker=dict(color="rgba(100, 100, 100, 0.3)"),
            hovertemplate="<b>Date</b>: %{x}<br><b>Demand</b>: %{y:,.1f} units<extra></extra>",
        )
    )

    # Receipts Arriving
    receipt_days = daily_df[daily_df["receipts"] > 0]
    if not receipt_days.empty:
        fig.add_trace(
            go.Scatter(
                x=receipt_days["date"],
                y=receipt_days["ending_on_hand"],
                mode="markers",
                name="Shipment Received",
                marker=dict(color="#2ca02c", size=12, symbol="arrow-down"),
                hovertemplate="<b>Date</b>: %{x}<br><b>Received</b>: %{customdata:,.1f} units<extra></extra>",
                customdata=receipt_days["receipts"],
            )
        )

    # Orders Placed
    order_days = daily_df[daily_df["order_placed"] > 0]
    if not order_days.empty:
        fig.add_trace(
            go.Scatter(
                x=order_days["date"],
                y=order_days["ending_on_hand"],
                mode="markers",
                name="Purchase Order Placed",
                marker=dict(color="#ff7f0e", size=12, symbol="arrow-up"),
                hovertemplate="<b>Date</b>: %{x}<br><b>Order Placed</b>: %{customdata:,.1f} units<extra></extra>",
                customdata=order_days["order_placed"],
            )
        )

    # Stockout / Lost Sales
    stockout_col = "lost_sales" if "lost_sales" in daily_df.columns else "ending_backlog"
    stockout_days = daily_df[daily_df[stockout_col] > 0]
    if not stockout_days.empty:
        fig.add_trace(
            go.Scatter(
                x=stockout_days["date"],
                y=[0] * len(stockout_days),
                mode="markers",
                name="Stockout / Unmet Demand",
                marker=dict(color="#d62728", size=10, symbol="x"),
                hovertemplate="<b>Date</b>: %{x}<br><b>Unmet Demand</b>: %{customdata:,.1f} units<extra></extra>",
                customdata=stockout_days[stockout_col],
            )
        )

    # Reference ROP line
    if rop is not None:
        fig.add_hline(
            y=rop,
            line_dash="dash",
            line_color="#9467bd",
            annotation_text=f"Reorder Point ({rop:,.0f})",
            annotation_position="top left",
        )

    # Reference Safety Stock line
    if safety_stock is not None:
        fig.add_hline(
            y=safety_stock,
            line_dash="dot",
            line_color="#e377c2",
            annotation_text=f"Safety Stock ({safety_stock:,.0f})",
            annotation_position="bottom left",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Date", showgrid=True),
        yaxis=dict(title="Units", showgrid=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
    )
    return fig


def build_cost_breakdown_bar_chart(
    baseline_costs: Dict[str, float],
    optimized_costs: Dict[str, float],
    title: str = "Operating Cost Breakdown: Baseline vs Optimized Policy",
) -> go.Figure:
    """Create a grouped bar chart comparing holding, ordering, shortage, and total operating cost."""
    categories = ["Holding Cost", "Ordering Cost", "Shortage Penalty", "Total Operating Cost"]
    base_vals = [
        baseline_costs.get("holding_cost_usd", baseline_costs.get("cost_holding", 0.0)),
        baseline_costs.get("ordering_cost_usd", baseline_costs.get("cost_ordering", 0.0)),
        baseline_costs.get("shortage_penalty_usd", baseline_costs.get("cost_shortage", 0.0)),
        baseline_costs.get("total_operating_cost_usd", baseline_costs.get("cost_operating_total", 0.0)),
    ]
    opt_vals = [
        optimized_costs.get("holding_cost_usd", optimized_costs.get("cost_holding", 0.0)),
        optimized_costs.get("ordering_cost_usd", optimized_costs.get("cost_ordering", 0.0)),
        optimized_costs.get("shortage_penalty_usd", optimized_costs.get("cost_shortage", 0.0)),
        optimized_costs.get("total_operating_cost_usd", optimized_costs.get("cost_operating_total", 0.0)),
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Baseline (Weekly Review)",
                x=categories,
                y=base_vals,
                marker_color="#d62728",
                text=[f"${v:,.2f}" for v in base_vals],
                textposition="auto",
            ),
            go.Bar(
                name="Optimized (Dynamic ROP + EOQ)",
                x=categories,
                y=opt_vals,
                marker_color="#1f77b4",
                text=[f"${v:,.2f}" for v in opt_vals],
                textposition="auto",
            ),
        ]
    )

    fig.update_layout(
        barmode="group",
        title=dict(text=title, font=dict(size=16)),
        yaxis=dict(title="Cost (USD)", showgrid=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
    )
    return fig


def build_ablation_ladder_chart(
    ablation_summary: List[Dict[str, Any]],
    title: str = "Incremental Ablation Progression: Service vs Operating Cost",
) -> go.Figure:
    """Create a combo chart tracking stockout days reduction and operating cost across ablation stages."""
    stages = [row["stage"] for row in ablation_summary]
    stockouts = [row["stockout_days"] for row in ablation_summary]
    costs = [row["cost_operating_total"] for row in ablation_summary]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=stages,
            y=stockouts,
            name="Stockout Days",
            marker_color="#e377c2",
            yaxis="y",
            text=stockouts,
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stages,
            y=costs,
            mode="lines+markers+text",
            name="Operating Cost ($)",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8),
            text=[f"${c:,.0f}" for c in costs],
            textposition="top center",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis=dict(title="Stockout Days", showgrid=True),
        yaxis2=dict(title="Operating Cost (USD)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
    )
    return fig
