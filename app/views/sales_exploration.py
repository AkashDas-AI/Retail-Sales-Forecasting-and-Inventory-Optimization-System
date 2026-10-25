"""Sales Exploration dashboard page.

Provides interactive visual analytics for historical daily sales trends, promotional lifts,
and weekly day-of-week seasonality across stores and product families.
"""

import pandas as pd
import streamlit as st

from app.components.charts import (
    build_day_of_week_seasonality_chart,
    build_sales_trend_chart,
)
from app.components.disclaimers import render_banner_disclaimer
from app.components.filters import (
    render_date_range_selector,
    render_store_family_selector,
)
from app.data_loader import (
    load_inventory_policies,
    load_sales_history_sample,
)


def render_sales_exploration() -> None:
    """Render the Sales Exploration dashboard view."""
    st.header("📈 Historical Sales Demand Exploration")
    render_banner_disclaimer()

    # Load policies to obtain available stores and product families
    policies_df = load_inventory_policies()

    st.subheader("1. Series & Horizon Filter")
    store_nbr, family = render_store_family_selector(
        df=policies_df,
        default_store=1,
        default_family="GROCERY I",
        key_prefix="sales_exp",
    )

    # Date range selection (default to 2017 for responsiveness)
    start_date, end_date = render_date_range_selector(
        min_date_str="2013-01-01",
        max_date_str="2017-08-15",
        default_days=365,
        key_prefix="sales_exp",
    )

    # Fetch data on demand (cached)
    with st.spinner(f"Loading daily sales history for Store {store_nbr} - {family}..."):
        series_sales_df = load_sales_history_sample(
            store_nbr=store_nbr,
            family=family,
            start_date=start_date,
            end_date=end_date,
        )

    if series_sales_df.empty:
        st.warning(f"No sales observations found for Store {store_nbr} - {family} within {start_date} to {end_date}.")
        return

    # Check for missing calendar dates explicitly without zero-filling
    series_sales_df["date_dt"] = pd.to_datetime(series_sales_df["date"])
    full_calendar_range = pd.date_range(start=series_sales_df["date_dt"].min(), end=series_sales_df["date_dt"].max())
    observed_dates = set(series_sales_df["date_dt"])
    missing_dates = set(full_calendar_range) - observed_dates

    # Summary metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Observed Days", f"{len(series_sales_df):,}")
    with m2:
        st.metric("Total Sales Volume", f"{series_sales_df['sales'].sum():,.0f} units")
    with m3:
        st.metric("Mean Daily Sales", f"{series_sales_df['sales'].mean():,.1f} units")
    with m4:
        st.metric("Peak Single-Day Sales", f"{series_sales_df['sales'].max():,.1f} units")
    with m5:
        promo_days = (series_sales_df["onpromotion"] > 0).sum()
        st.metric("Active Promotion Days", f"{promo_days} days ({promo_days / len(series_sales_df) * 100:.1f}%)")

    if missing_dates:
        st.caption(
            f"ℹ️ **Calendar Gap Notice**: There are {len(missing_dates)} calendar dates in this range without recorded store sales "
            f"(e.g., store closure on New Year's Day). The system displays actual observations without synthetic zero-imputation."
        )

    # Historical trend chart
    st.subheader(f"2. Daily Demand Trajectory: Store {store_nbr} — {family}")
    trend_fig = build_sales_trend_chart(
        df=series_sales_df,
        title=f"Store {store_nbr} — {family} ({start_date} to {end_date})",
        show_promotions=True,
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    # Seasonality charts
    st.subheader("3. Demand Seasonality Patterns")
    col_dow, col_dist = st.columns(2)

    with col_dow:
        dow_fig = build_day_of_week_seasonality_chart(
            df=series_sales_df,
            title=f"Day-of-Week Seasonality: Store {store_nbr} — {family}",
        )
        st.plotly_chart(dow_fig, use_container_width=True)

    with col_dist:
        # Promotion impact comparison
        promo_mean = series_sales_df[series_sales_df["onpromotion"] > 0]["sales"].mean() if promo_days > 0 else 0.0
        non_promo_mean = series_sales_df[series_sales_df["onpromotion"] == 0]["sales"].mean()

        comp_df = pd.DataFrame([
            {"Status": "Non-Promotional Days", "Average Sales": non_promo_mean},
            {"Status": "Promotional Days", "Average Sales": promo_mean},
        ])

        import plotly.express as px
        promo_bar = px.bar(
            comp_df,
            x="Status",
            y="Average Sales",
            title=f"Promotional Demand Lift: Store {store_nbr} — {family}",
            color="Status",
            color_discrete_map={"Non-Promotional Days": "#1f77b4", "Promotional Days": "#ff7f0e"},
            text_auto=".1f",
        )
        promo_bar.update_layout(
            template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(promo_bar, use_container_width=True)

