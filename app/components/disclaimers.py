"""Standardized disclosure banners and disclaimers for the decision-support system.

Ensures transparent communication that historical demand is derived from Corporación Favorita,
while all cost parameters, supplier lead times, and initial inventory are configured assumptions.
"""

import streamlit as st


def render_banner_disclaimer() -> None:
    """Render the standard top-level system context banner."""
    st.caption(
        "📌 **System Architecture & Data Notice**: "
        "Demand forecasts are trained on historical retail transactions from the **Corporación Favorita** Kaggle benchmark. "
        "Financial cost parameters (`h = 20%`, `K = $50`, `shortage penalty = $15/unit`), supplier lead times (`L = 7` days), "
        "and starting stock balances are **configured illustrative assumptions** (Favorita does not provide ERP inventory data). "
        "Purchase orders are **offline decision-support recommendations**, not live transactional orders."
    )


def render_retrospective_evaluation_notice() -> None:
    """Render notice explaining the retrospective nature of holdout simulations."""
    st.info(
        "🔍 **Retrospective Evaluation Notice**: "
        "Simulation metrics represent retrospective backtesting over the 28-day final holdout (2017-07-19 to 2017-08-15). "
        "Demand uncertainty and policy parameters were strictly calibrated on earlier validation residuals without holdout leakage. "
        "Past simulation performance illustrates policy mechanics and does not constitute a guarantee of future real-world operational savings."
    )


def render_purchase_order_disclaimer() -> None:
    """Render notice for purchase order recommendation exports."""
    st.warning(
        "⚠️ **Decision-Support Notice**: "
        "The purchase recommendations displayed below are calculated from simulated starting inventory positions and statistical reorder points. "
        "They are designed for supply-chain review and are **not connected to a live retailer ERP or supplier EDI ordering system**."
    )

