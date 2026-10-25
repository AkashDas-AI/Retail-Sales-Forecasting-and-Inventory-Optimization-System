"""Retail Sales Forecasting & Inventory Optimization Dashboard.

Main interactive application providing executive decision-support, demand forecasting
evaluation, inventory buffer calibration, scenario simulation, and purchase recommendations.
"""

import streamlit as st

from app.views.executive_overview import render_executive_overview
from app.views.forecast_performance import render_forecast_performance
from app.views.inventory_planning import render_inventory_planning
from app.views.purchase_orders import render_purchase_orders
from app.views.sales_exploration import render_sales_exploration
from app.views.simulation_analysis import render_simulation_analysis


def inject_custom_css() -> None:
    """Inject custom CSS for enhanced portfolio-grade aesthetics."""
    st.markdown(
        """
        <style>
        /* Modern metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.90rem !important;
            color: #64748b !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            color: #f8fafc;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
            color: #cbd5e1 !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }
        /* Card-like borders for containers */
        div.stBlock {
            border-radius: 8px;
        }
        /* Clean tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Streamlit web application entry point."""
    st.set_page_config(
        page_title="Retail Demand Forecasting & Inventory Optimization",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Sidebar Navigation & Branding
    st.sidebar.title("📦 Retail Demand AI")
    st.sidebar.caption("Enterprise Demand Forecasting & Inventory Optimization")

    st.sidebar.markdown("---")

    views = {
        "📊 Executive Overview": render_executive_overview,
        "📈 Sales Exploration": render_sales_exploration,
        "🎯 Forecast Performance": render_forecast_performance,
        "📦 Inventory Planning": render_inventory_planning,
        "🔬 Simulation & Scenarios": render_simulation_analysis,
        "📋 Purchase Orders": render_purchase_orders,
    }

    selected_view = st.sidebar.radio(
        "Navigation",
        options=list(views.keys()),
        index=0,
    )

    st.sidebar.markdown("---")

    # System Parameters Summary in Sidebar
    st.sidebar.subheader("System Architecture")
    st.sidebar.markdown(
        """
        - **Demand Data**: Corporación Favorita (Ecuador)
        - **Catalog Scale**: 1,782 series (54 stores × 33 families)
        - **Forecasting Engine**: Walk-Forward Random Forest
        - **Policy Model**: Continuous `(s, Q)` with Empirical Bayes buffer
        - **Review Period (R)**: 7 days
        - **Supplier Lead Time (L)**: 7 days
        - **Holding Cost (h)**: 20% annual (`$0.00548`/unit/day)
        - **Order Cost (K)**: `$50` / order
        - **Stockout Penalty (π)**: `$15` / unit
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "⚠️ **Disclaimer**: Demand history is from Corporación Favorita. "
        "Costs, lead times, and inventory balances are configured simulation assumptions. "
        "Purchase recommendations are decision-support outputs."
    )

    # Route to selected page view
    render_func = views[selected_view]
    render_func()


if __name__ == "__main__":
    main()
