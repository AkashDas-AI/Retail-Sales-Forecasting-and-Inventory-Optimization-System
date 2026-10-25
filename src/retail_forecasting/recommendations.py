"""Replenishment order generation and recommendation engine.

Translates current stock positions and forecast-driven Reorder Points
into prioritized, batch-constrained purchase order recommendations.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def generate_purchase_orders(
    current_inventory_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    lead_time_days: int = 7,
    current_date: Optional[str] = None,
    default_unit_cost: float = 10.0,
) -> pd.DataFrame:
    """Evaluate current inventory positions against ROPs and output replenishment recommendations.

    Parameters
    ----------
    current_inventory_df : pd.DataFrame
        DataFrame with ['store_nbr', 'family', 'on_hand'] and optional ['on_order', 'backorders'].
    policy_df : pd.DataFrame
        DataFrame with ['store_nbr', 'family', 'reorder_point', 'safety_stock', 'eoq_recommended']
        and optional ['unit_cost', 'lead_time_days'].
    lead_time_days : int
        Default supplier lead time in calendar days if not present in policy_df.
    current_date : Optional[str]
        Current planning date string (YYYY-MM-DD). If provided, computes expected_delivery_date.
    default_unit_cost : float
        Fallback unit cost ($/unit) if not present in policy_df.

    Returns
    -------
    pd.DataFrame
        Purchase orders recommendations table containing:
        ['store_nbr', 'family', 'on_hand', 'on_order', 'backorders', 'inventory_position',
         'safety_stock', 'reorder_point', 'order_recommended', 'recommended_order_qty',
         'unit_cost', 'estimated_purchase_cost', 'urgency_priority',
         'order_date', 'expected_delivery_date'].
    """
    if current_inventory_df.empty:
        raise ValueError("current_inventory_df cannot be empty.")
    if policy_df.empty:
        raise ValueError("policy_df cannot be empty.")

    inv_df = current_inventory_df.copy()
    if "on_order" not in inv_df.columns:
        inv_df["on_order"] = 0.0
    if "backorders" not in inv_df.columns:
        inv_df["backorders"] = 0.0

    merged = pd.merge(inv_df, policy_df, on=["store_nbr", "family"], how="inner")
    if merged.empty:
        raise ValueError("No matching (store_nbr, family) pairs between current inventory and policy DataFrames.")

    # Calculate Inventory Position: IP = On-Hand + On-Order - Backorders
    merged["inventory_position"] = merged["on_hand"] + merged["on_order"] - merged["backorders"]

    # Trigger rule: order if IP <= ROP
    merged["order_recommended"] = merged["inventory_position"] <= merged["reorder_point"]

    # Recommended quantity: EOQ if triggered, else 0.0
    qty_col = "eoq_recommended" if "eoq_recommended" in merged.columns else "order_qty"
    merged["recommended_order_qty"] = np.where(merged["order_recommended"], merged[qty_col], 0.0)

    # Unit cost and estimated expenditure
    cost_col = "unit_cost" if "unit_cost" in merged.columns else None
    if cost_col is None:
        merged["unit_cost"] = default_unit_cost
    merged["estimated_purchase_cost"] = merged["recommended_order_qty"] * merged["unit_cost"]

    # Priority classification:
    # - CRITICAL: on_hand == 0 or on_hand < backorders (immediate stockout)
    # - HIGH: on_hand < safety_stock (safety buffer depleted)
    # - NORMAL: order_recommended and on_hand >= safety_stock
    # - NONE: no order needed
    priorities: List[str] = []
    for _, row in merged.iterrows():
        if not row["order_recommended"]:
            priorities.append("NONE")
        elif row["on_hand"] <= 0 or row["on_hand"] < row["backorders"]:
            priorities.append("CRITICAL")
        elif row["on_hand"] < row["safety_stock"]:
            priorities.append("HIGH")
        else:
            priorities.append("NORMAL")

    merged["urgency_priority"] = priorities

    # Dates
    if current_date is not None:
        order_dt = pd.to_datetime(current_date)
        lt_col = merged["lead_time_days"] if "lead_time_days" in merged.columns else lead_time_days
        merged["order_date"] = current_date
        merged["expected_delivery_date"] = (order_dt + pd.to_timedelta(lt_col, unit="D")).dt.strftime("%Y-%m-%d")
    else:
        merged["order_date"] = "T_0"
        merged["expected_delivery_date"] = f"T_0 + {lead_time_days}d"

    # Select and order output columns
    output_cols = [
        "store_nbr",
        "family",
        "on_hand",
        "on_order",
        "backorders",
        "inventory_position",
        "safety_stock",
        "reorder_point",
        "order_recommended",
        "recommended_order_qty",
        "unit_cost",
        "estimated_purchase_cost",
        "urgency_priority",
        "order_date",
        "expected_delivery_date",
    ]

    return merged[output_cols].sort_values(
        by=["order_recommended", "urgency_priority", "store_nbr", "family"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
