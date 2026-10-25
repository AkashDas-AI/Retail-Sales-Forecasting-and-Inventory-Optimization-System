"""Purchase Orders dashboard page.

Provides interactive filtering, urgency prioritization, and CSV export
for automated inventory replenishment purchase recommendations.
"""

import pandas as pd
import streamlit as st

from app.components.disclaimers import (
    render_banner_disclaimer,
    render_purchase_order_disclaimer,
)
from app.data_loader import (
    load_inventory_config,
    load_inventory_policies,
    load_purchase_orders,
)


def render_purchase_orders() -> None:
    """Render the Purchase Order Recommendations dashboard view."""
    st.header("📋 Purchase Order Recommendations & Replenishment")
    render_banner_disclaimer()
    st.markdown("")
    render_purchase_order_disclaimer()

    # Load artifacts
    orders_df = load_purchase_orders()
    policies_df = load_inventory_policies()
    inv_cfg = load_inventory_config()

    # Overall Summary KPIs
    st.subheader("1. Replenishment Order Summary")
    triggered_orders = orders_df[orders_df["order_recommended"] == True]

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Orders Triggered", f"{len(triggered_orders):,}", delta=f"{(len(triggered_orders)/len(orders_df))*100:.1f}% of catalog")
    with k2:
        st.metric("Total Replenishment Units", f"{triggered_orders['recommended_order_qty'].sum():,.0f} units")
    with k3:
        st.metric("Estimated Purchase Spend", f"${triggered_orders['estimated_purchase_cost'].sum():,.2f}")
    with k4:
        high_count = len(orders_df[orders_df["urgency_priority"] == "HIGH"])
        st.metric("High Urgency Orders", f"{high_count:,}", help="Inventory Position is below Safety Stock (buffer breached)")
    with k5:
        norm_count = len(orders_df[orders_df["urgency_priority"] == "NORMAL"])
        st.metric("Normal Urgency Orders", f"{norm_count:,}", help="Inventory Position is between Safety Stock and Reorder Point")

    st.info(
        "💡 **Capital Accounting Separation**: "
        "Estimated Purchase Spend represents upfront working capital for acquiring physical inventory at standard wholesale cost (`$10`/unit). "
        "This is strictly distinct from recurring operational carrying and stockout costs "
        "(`h = 20%` annual rate, `K = $50`/order, `shortage penalty = $15`/unit)."
    )

    st.markdown("---")

    # Priority Rules & Architecture Explanation
    with st.expander("ℹ️ Priority Classification Rules & Purchase Order Logic", expanded=False):
        st.markdown(
            """
            ### Reorder Trigger Logic
            An order is triggered if and only if the current **Inventory Position ($IP$)** falls at or below the **Reorder Point ($ROP$)**:
            $$IP = \\text{On-Hand} + \\text{On-Order} - \\text{Backorders} \\le ROP$$
            
            When triggered, the recommended batch size equals:
            $$Q^* = \\max\\left(EOQ, ROP - IP\\right)$$
            
            ### Urgency Priority Tiers
            - **CRITICAL** ($IP \\le 0$): Immediate stockout or backorders present. Stock is completely exhausted.
            - **HIGH** ($0 < IP \\le SS$): Safety stock buffer has been partially or fully breached. Elevated vulnerability to stockouts during lead time.
            - **NORMAL** ($SS < IP \\le ROP$): Reorder threshold reached, but safety stock buffer is intact.
            - **NONE** ($IP > ROP$): Inventory position is healthy. No replenishment purchase is required at this review.
            """
        )

    # Interactive Filters
    st.subheader("2. Recommended Purchase Order Catalog")
    col_filter_mode, col_prio, col_search = st.columns([1, 1, 2])

    with col_filter_mode:
        view_mode = st.radio(
            "Filter Scope",
            options=["Triggered Orders Only", "All Series (Full Catalog)"],
            index=0,
            horizontal=True,
        )

    with col_prio:
        priority_options = ["ALL", "HIGH", "NORMAL", "CRITICAL", "NONE"]
        selected_prio = st.selectbox("Filter Urgency Priority", priority_options, index=0)

    with col_search:
        search_query = st.text_input(
            "Search Store or Product Family",
            placeholder="e.g. 1, GROCERY, BEVERAGES, MEATS...",
        )

    # Apply filters
    filtered_df = orders_df.copy()
    if view_mode == "Triggered Orders Only":
        filtered_df = filtered_df[filtered_df["order_recommended"] == True]

    if selected_prio != "ALL":
        filtered_df = filtered_df[filtered_df["urgency_priority"] == selected_prio]

    if search_query:
        query_str = search_query.strip().upper()
        filtered_df = filtered_df[
            filtered_df["family"].str.contains(query_str, na=False)
            | filtered_df["store_nbr"].astype(str).str.contains(query_str, na=False)
        ]

    # Quick summary of filtered view
    st.caption(
        f"Displaying **{len(filtered_df):,}** matching rows | "
        f"Filtered Units: **{filtered_df['recommended_order_qty'].sum():,.0f}** | "
        f"Filtered Purchase Spend: **${filtered_df['estimated_purchase_cost'].sum():,.2f}**"
    )

    # Display columns and formatting
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

    # Empty state handling
    if filtered_df.empty:
        st.warning("⚠️ No purchase order recommendations match the selected filters. Broaden your search or priority criteria.")

    # Formatted display dataframe
    export_df = filtered_df[display_cols].rename(columns=rename_map)

    st.dataframe(
        export_df.style.format({
            "Order Qty (units)": "{:,.0f}",
            "Purchase Spend ($)": "${:,.2f}",
            "On-Hand": "{:,.1f}",
            "On-Order": "{:,.1f}",
            "Backorders": "{:,.1f}",
            "IP": "{:,.1f}",
            "Safety Stock": "{:,.1f}",
            "ROP": "{:,.1f}",
        }),
        use_container_width=True,
        height=400,
    )

    # CSV Download Button with exact column names and row matching
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Purchase Order Recommendations (CSV)",
        data=csv_bytes,
        file_name="purchase_order_recommendations.csv",
        mime="text/csv",
        help="Export current filtered recommendations as a standardized purchase requisition CSV.",
    )

    st.markdown("---")

    # Single Series Order Inspector
    st.subheader("3. Single-Series Replenishment Diagnostic")
    st.markdown(
        "Select an individual Store and Product Family to audit its exact replenishment math and supply pipeline."
    )

    stores = sorted(orders_df["store_nbr"].unique())
    col_sel_s, col_sel_f = st.columns(2)
    with col_sel_s:
        chosen_store = st.selectbox("Select Store", stores, index=0)
    
    store_families = sorted(orders_df[orders_df["store_nbr"] == chosen_store]["family"].unique())
    with col_sel_f:
        chosen_family = st.selectbox("Select Product Family", store_families, index=0)

    chosen_row = orders_df[
        (orders_df["store_nbr"] == chosen_store) & (orders_df["family"] == chosen_family)
    ].iloc[0]

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.metric("Inventory Position (IP)", f"{chosen_row['inventory_position']:,.1f}")
    with d2:
        st.metric("Reorder Point (ROP)", f"{chosen_row['reorder_point']:,.1f}")
    with d3:
        st.metric("Safety Stock (SS)", f"{chosen_row['safety_stock']:,.1f}")
    with d4:
        st.metric("Order Triggered?", "YES (Reorder)" if chosen_row["order_recommended"] else "NO")
    with d5:
        prio_color = {
            "CRITICAL": "🔴 CRITICAL",
            "HIGH": "🟠 HIGH",
            "NORMAL": "🟢 NORMAL",
            "NONE": "⚪ NONE",
        }.get(chosen_row["urgency_priority"], chosen_row["urgency_priority"])
        st.metric("Urgency Level", prio_color)

    diag_left, diag_right = st.columns(2)
    with diag_left:
        st.markdown(
            f"""
            #### Inventory Breakdown
            - **On-Hand Physical Stock**: `{chosen_row['on_hand']:,.1f}` units
            - **In-Transit Pipeline (On-Order)**: `{chosen_row['on_order']:,.1f}` units
            - **Unfulfilled Backorders**: `{chosen_row['backorders']:,.1f}` units
            - **Net Inventory Position**: `{chosen_row['inventory_position']:,.1f}` units
            """
        )
    with diag_right:
        st.markdown(
            f"""
            #### Order Recommendation
            - **Recommended Order Batch**: `{chosen_row['recommended_order_qty']:,.0f}` units
            - **Unit Wholesale Cost**: `${chosen_row['unit_cost']:,.2f}`
            - **Estimated Order Spend**: `${chosen_row['estimated_purchase_cost']:,.2f}`
            - **Review / Order Date**: `{chosen_row['order_date']}`
            - **Projected Arrival Date**: `{chosen_row['expected_delivery_date']}` (Lead Time = `{inv_cfg.get('default_lead_time_days', 7)}` days)
            """
        )

