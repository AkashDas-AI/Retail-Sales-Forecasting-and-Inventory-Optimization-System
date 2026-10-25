"""Inventory Planning dashboard page.

Displays safety stock, reorder points, economic order quantities, and unit-level
mathematical policy derivations for all 1,782 planning series.
"""

import pandas as pd
import streamlit as st

from app.components.disclaimers import render_banner_disclaimer
from app.components.filters import render_store_family_selector
from app.data_loader import (
    compute_policy_derivation,
    load_inventory_planning_report,
    load_inventory_policies,
    load_purchase_orders,
)


def render_inventory_planning() -> None:
    """Render the Inventory Planning dashboard view."""
    st.header("📦 Inventory Policy Planning & Buffer Calibration")
    render_banner_disclaimer()

    # Load artifacts
    policies_df = load_inventory_policies()
    orders_df = load_purchase_orders()
    plan_rep = load_inventory_planning_report()

    # Merge for complete planning table
    merged_df = pd.merge(
        policies_df,
        orders_df[[
            "store_nbr", "family", "on_hand", "on_order", "backorders",
            "inventory_position", "order_recommended", "recommended_order_qty",
            "estimated_purchase_cost", "urgency_priority",
        ]],
        on=["store_nbr", "family"],
        how="left",
    )

    # Catalog Overview Stats
    st.subheader("1. Catalog Policy Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Total Planning Series", f"{len(merged_df):,}")
    with s2:
        st.metric("Mean Daily Demand", f"{merged_df['daily_demand_mean'].mean():,.1f} units")
    with s3:
        st.metric("Mean Safety Stock", f"{merged_df['safety_stock'].mean():,.1f} units")
    with s4:
        st.metric("Mean Reorder Point", f"{merged_df['reorder_point'].mean():,.1f} units")
    with s5:
        st.metric("Mean Recommended EOQ", f"{merged_df['eoq_recommended'].mean():,.1f} units")

    st.caption(
        r"ℹ️ **Calibration Source**: Demand uncertainty ($\sigma_L$) and Safety Stock were calibrated from out-of-sample forecast "
        "residuals over the validation partition (2017-06-21 to 2017-07-18). On-hand inventory is a simulated initial baseline (7 days of supply)."
    )

    st.markdown("---")

    # Interactive Planning Table
    st.subheader("2. Searchable Inventory Policy Catalog")
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        quality_filter = st.multiselect(
            "Filter Quality Flag",
            options=["RELIABLE", "SPARSE_SHRUNK"],
            default=["RELIABLE", "SPARSE_SHRUNK"],
            help="RELIABLE: N_demand >= 4 with Empirical Bayes weight >= 0.5. SPARSE_SHRUNK: Shrunk toward family variance.",
        )
    with col_f2:
        search_family = st.text_input("Search Product Family or Store Number", placeholder="e.g., GROCERY, 1, BEVERAGES")

    filtered_table = merged_df[merged_df["quality_flag"].isin(quality_filter)]
    if search_family:
        search_term = search_family.strip().upper()
        filtered_table = filtered_table[
            filtered_table["family"].str.contains(search_term, na=False)
            | filtered_table["store_nbr"].astype(str).str.contains(search_term, na=False)
        ]

    if filtered_table.empty:
        st.warning("⚠️ No inventory policies match the current quality flags or search query. Broaden your filter selection.")

    display_cols = [
        "store_nbr", "family", "daily_demand_mean", "safety_stock", "reorder_point",
        "eoq_recommended", "on_hand", "inventory_position", "order_recommended",
        "recommended_order_qty", "urgency_priority", "estimated_purchase_cost", "quality_flag",
    ]
    st.dataframe(
        filtered_table[display_cols].rename(columns={
            "store_nbr": "Store",
            "family": "Family",
            "daily_demand_mean": "Daily Demand (units)",
            "safety_stock": "Safety Stock (units)",
            "reorder_point": "Reorder Point (units)",
            "eoq_recommended": "EOQ (units)",
            "on_hand": "On-Hand Stock",
            "inventory_position": "Inventory Position",
            "order_recommended": "Reorder Triggered",
            "recommended_order_qty": "Order Qty",
            "urgency_priority": "Priority",
            "estimated_purchase_cost": "Purchase Spend ($)",
            "quality_flag": "Uncertainty Flag",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Detailed Planning-Unit Derivation View
    st.subheader("3. Single Planning Unit Mathematical Derivation")
    st.caption("Select a specific store and merchandise family to inspect how each parameter was calculated step by step.")

    sel_store, sel_family = render_store_family_selector(
        df=policies_df,
        default_store=1,
        default_family="GROCERY I",
        key_prefix="inv_plan",
    )

    sel_pol = policies_df[(policies_df["store_nbr"] == sel_store) & (policies_df["family"] == sel_family)]
    sel_po = orders_df[(orders_df["store_nbr"] == sel_store) & (orders_df["family"] == sel_family)]

    if not sel_pol.empty:
        pol_row = sel_pol.iloc[0]
        po_row = sel_po.iloc[0] if not sel_po.empty else None
        deriv = compute_policy_derivation(pol_row, po_row)

        c_d1, c_d2, c_d3 = st.columns(3)
        with c_d1:
            st.markdown(
                f"""
                **Step 1: Expected Lead-Time Demand ($D_L$)**
                - Mean Daily Demand ($\\bar{{D}}$): `{deriv['daily_demand_mean']:,.2f} units/day`
                - Supplier Lead Time ($L$): `{deriv['lead_time_days']} calendar days`
                - $$D_L = \\bar{{D}} \\times L = {deriv['lead_time_demand']:,.2f} \\text{{ units}}$$
                - Annualized Demand ($D_{{\\text{{ann}}}}$): `{deriv['annual_demand']:,.0f} units/year`
                """
            )
        with c_d2:
            st.markdown(
                f"""
                **Step 2: Safety Stock ($SS$) & ROP**
                - Service Level Target ($CSL$): `{deriv['service_level'] * 100:.1f}\\%` ($z = {deriv['service_level_z']:.3f}$)
                - Compounded Uncertainty ($\\sigma_L$): `{deriv['lead_time_std']:,.2f} units`
                - Quality Flag: `{deriv['quality_flag']}`
                - $$SS = z \\times \\sigma_L = {deriv['safety_stock']:,.2f} \\text{{ units}}$$
                - $$ROP = D_L + SS = {deriv['reorder_point']:,.2f} \\text{{ units}}$$
                """
            )
        with c_d3:
            st.markdown(
                f"""
                **Step 3: Economic Order Quantity ($EOQ$)**
                - Setup Cost ($K$): `${deriv['order_cost']:.2f}/order`
                - Annual Unit Holding Cost ($H$): `${deriv['annual_holding_cost_unit']:.2f}/unit/year`
                - Unconstrained EOQ: `{deriv['eoq_unconstrained']:,.1f} units`
                - MOQ (10) & Batch (5) Rounding:
                - $$EOQ_{{\\text{{rec}}}} = {deriv['eoq_recommended']:,.0f} \\text{{ units}}$$
                """
            )

        st.markdown(
            f"""
            **Step 4: Current Position Review & Replenishment Trigger**
            - Physical On-Hand Stock ($I_t$): `{deriv['on_hand']:,.1f} units` (simulated 7d supply assumption)
            - In-Flight Orders ($Q_t^{{\\text{{in-transit}}}}$): `{deriv['on_order']:,.1f} units`
            - Customer Backorders ($B_t$): `{deriv['backorders']:,.1f} units`
            - Inventory Position: $$IP_t = I_t + Q_t^{{\\text{{in-transit}}}} - B_t = {deriv['inventory_position']:,.1f} \\text{{ units}}$$
            - **Decision Rule**: Is $IP_t \\le ROP$ ({deriv['inventory_position']:,.1f} $\\le$ {deriv['reorder_point']:,.1f})?
              - **Result**: `{"YES -> ORDER RECOMMENDED" if deriv['order_recommended'] else "NO -> STOCK SUFFICIENT"}`
              - **Recommended Purchase Qty**: `{deriv['recommended_order_qty']:,.0f} units`
              - **Urgency Priority**: `{deriv['urgency_priority']}`
              - **Estimated Spend**: `${deriv['estimated_purchase_spend']:,.2f} USD`
            """
        )

        st.info(
            "📌 **EOQ vs. Recommended Order Quantity Clarification**: "
            f"The Economic Order Quantity ({deriv['eoq_recommended']:,.0f} units) represents the optimal batch size "
            f"when an order is needed. It is only converted into a purchase recommendation when the Inventory Position "
            f"falls to or below the Reorder Point ({deriv['reorder_point']:,.0f} units)."
        )

