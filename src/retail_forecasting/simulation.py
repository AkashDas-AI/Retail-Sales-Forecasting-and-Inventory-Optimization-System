"""Discrete-event inventory policy simulation and backtesting.

Simulates daily inventory state transitions over historical time horizons:
I_{t+1} = max(0, I_t + Receipts_t - Demand_t)
accounting for order lead times, stockout events, inventory position tracking,
and reconciled holding, ordering, and shortage cost accounting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.retail_forecasting.inventory import calculate_daily_holding_cost_per_unit


# ---------------------------------------------------------------------------
# Policy Definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InventoryPolicy:
    """Replenishment policy configuration for inventory simulation.

    Attributes
    ----------
    policy_name : str
        Descriptive label for the policy (e.g. 'Optimized_ROP_EOQ', 'Baseline_Weekly').
    policy_type : str
        Replenishment rule: 'rop_eoq' or 'order_up_to'.
    rop : float
        Reorder point threshold in units (>= 0).
    order_qty : float
        Fixed order quantity (EOQ) in units (>= 0).
    order_up_to : Optional[float]
        Target inventory level for order-up-to policies (>= rop).
    review_period_days : int
        Review interval in days (1 = daily review continuous).
    min_order_qty : float
        Minimum order quantity MOQ in units (>= 0).
    batch_multiple : float
        Batch packaging increment in units (>= 1.0).
    """

    policy_name: str
    policy_type: str = "rop_eoq"
    rop: float = 0.0
    order_qty: float = 0.0
    order_up_to: Optional[float] = None
    review_period_days: int = 1
    min_order_qty: float = 0.0
    batch_multiple: float = 1.0

    def __post_init__(self) -> None:
        """Validate policy parameters."""
        if not self.policy_name or not self.policy_name.strip():
            raise ValueError("policy_name must be a non-empty string.")
        if self.policy_type not in ("rop_eoq", "order_up_to"):
            raise ValueError(f"policy_type must be 'rop_eoq' or 'order_up_to', got {self.policy_type!r}")
        if self.rop < 0:
            raise ValueError(f"rop must be non-negative, got {self.rop}")
        if self.order_qty < 0:
            raise ValueError(f"order_qty must be non-negative, got {self.order_qty}")
        if self.review_period_days < 1:
            raise ValueError(f"review_period_days must be >= 1, got {self.review_period_days}")
        if self.min_order_qty < 0:
            raise ValueError(f"min_order_qty must be non-negative, got {self.min_order_qty}")
        if self.batch_multiple < 1.0:
            raise ValueError(f"batch_multiple must be >= 1.0, got {self.batch_multiple}")
        if self.policy_type == "order_up_to" and self.order_up_to is not None:
            if self.order_up_to < self.rop:
                raise ValueError(
                    f"order_up_to ({self.order_up_to}) must be >= rop ({self.rop}) for order_up_to policy"
                )


# ---------------------------------------------------------------------------
# Simulation Results Container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Results and metrics produced by an inventory simulation run.

    Attributes
    ----------
    policy_name : str
        Label of the simulated policy.
    daily_history : pd.DataFrame
        Detailed daily trajectory of stock, receipts, demand, sales, shortages, orders, and costs.
    summary_metrics : Dict[str, Any]
        Aggregated Key Performance Indicators (KPIs).
    balance_reconciled : bool
        Whether the fundamental inventory conservation equation strictly holds.
    """

    policy_name: str
    daily_history: pd.DataFrame
    summary_metrics: Dict[str, Any]
    balance_reconciled: bool


# ---------------------------------------------------------------------------
# Discrete-Event Simulator Engine
# ---------------------------------------------------------------------------

class InventorySimulator:
    """Discrete-event inventory simulator evaluating replenishment policies over time.

    Event Sequence on Day t:
    1. Beginning of Day: Scheduled purchase orders arriving today (t_arrival == t) are
       received and added to on-hand inventory:
       I_t^{avail} = I_{t-1}^{end} + Receipts_t
    2. Demand Realization & Fulfillment: Customer demand D_t occurs.
       Sales_t = min(I_t^{avail}, D_t + Backorders_{t-1})
       If lost sales: LostSales_t = max(0, D_t - I_t^{avail}), Backorders_t = 0
       If backorders: Backorders_t = Backorders_{t-1} + D_t - Sales_t, LostSales_t = 0
       I_t^{end} = I_t^{avail} - Sales_t >= 0
    3. Inventory Position Review:
       IP_t = I_t^{end} + OnOrder_t - Backorders_t
    4. Replenishment Decision:
       If review day and IP_t <= ROP:
         Compute Q_t, apply MOQ and batch multiples.
         Schedule arrival at t + lead_time_days.
         OnOrder_t += Q_t
    5. Daily Cost Accounting:
       Holding cost = I_t^{end} * (h * c / 365)
       Order cost = K * I(Q_t > 0)
       Purchase cost = c * Q_t
       Shortage penalty = penalty_rate * UnmetDemand_t
    6. Balance Invariant Verification:
       Delta I_t = I_t^{end} - I_{t-1}^{end} == Receipts_t - Sales_t
    """

    def __init__(
        self,
        initial_stock: float = 0.0,
        lead_time_days: int = 7,
        unit_cost: float = 10.0,
        order_cost: float = 50.0,
        holding_cost_annual_rate: float = 0.20,
        shortage_penalty_per_unit: Optional[float] = None,
        backlog_allowed: bool = False,
        backorder_penalty_mode: str = "one_time",
        initial_on_order: Optional[List[Dict[str, Any]]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the simulator with operational and cost parameters.

        Parameters
        ----------
        initial_stock : float
            Starting physical on-hand inventory at t = 0 (>= 0).
        lead_time_days : int
            Replenishment lead time in calendar days (>= 1).
        unit_cost : float
            Unit purchase cost in $/unit (> 0).
        order_cost : float
            Fixed order setup cost in $/order (>= 0).
        holding_cost_annual_rate : float
            Annual holding cost rate as a fraction of unit cost (> 0, e.g. 0.20).
        shortage_penalty_per_unit : Optional[float]
            Synthetic penalty per unit of unmet demand ($/unit).
        backlog_allowed : bool
            True for backorder system, False for lost sales system (default False).
        backorder_penalty_mode : str
            'one_time' (charges penalty once per unit of unmet demand) or
            'per_day' (charges penalty per unit-day that backorders remain unfulfilled).
        initial_on_order : Optional[List[Dict[str, Any]]]
            List of existing in-flight orders: [{'arrival_day': int, 'quantity': float}].
        random_seed : Optional[int]
            Random seed for reproducibility if stochastic elements are introduced.
        """
        if initial_stock < 0:
            raise ValueError(f"initial_stock must be non-negative, got {initial_stock}")
        if lead_time_days < 1:
            raise ValueError(f"lead_time_days must be >= 1, got {lead_time_days}")
        if unit_cost <= 0:
            raise ValueError(f"unit_cost must be strictly positive, got {unit_cost}")
        if order_cost < 0:
            raise ValueError(f"order_cost must be non-negative, got {order_cost}")
        if holding_cost_annual_rate <= 0:
            raise ValueError(
                f"holding_cost_annual_rate must be strictly positive, got {holding_cost_annual_rate}"
            )
        if backorder_penalty_mode not in ("one_time", "per_day"):
            raise ValueError(f"backorder_penalty_mode must be 'one_time' or 'per_day', got {backorder_penalty_mode!r}")

        self.initial_stock = float(initial_stock)
        self.lead_time_days = int(lead_time_days)
        self.unit_cost = float(unit_cost)
        self.order_cost = float(order_cost)
        self.holding_cost_annual_rate = float(holding_cost_annual_rate)
        self.daily_holding_cost_per_unit = calculate_daily_holding_cost_per_unit(
            annual_holding_rate=self.holding_cost_annual_rate,
            unit_cost=self.unit_cost,
            days_per_year=365.0,
        )
        self.shortage_penalty_per_unit = (
            float(shortage_penalty_per_unit) if shortage_penalty_per_unit is not None else 0.0
        )
        self.backlog_allowed = bool(backlog_allowed)
        self.backorder_penalty_mode = backorder_penalty_mode
        self.initial_on_order = initial_on_order or []
        self.random_seed = random_seed

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def run(
        self,
        actual_demand: Union[List[float], np.ndarray, pd.Series],
        replenishment_policy: InventoryPolicy,
        dates: Optional[Union[List[Any], pd.Series]] = None,
    ) -> SimulationResult:
        """Run daily inventory simulation across a demand series under the given policy.

        Variables Definition:
        - Customer demand (d_t): Units requested by consumers on day t.
        - Receipts (receipts_t): Scheduled purchase orders arriving at start of day t.
        - Available stock (avail_stock): Physical on-hand stock ready before demand fulfillment.
        - Fulfilled sales (sales_t): Actual physical units issued from inventory to customers.
        - Lost sales (lost_sales_t): Unfulfilled demand permanently lost (in lost-sales mode).
        - Backorders (current_backorders): Cumulative unfilled customer orders awaiting stock.
        - Ending on-hand (end_on_hand): Physical stock remaining at close of day (strictly >= 0).
        - Outstanding orders (on_order_qty): Units ordered but not yet received.
        - Inventory position (ip_t): end_on_hand + on_order_qty - current_backorders.
        """
        demands = np.asarray(actual_demand, dtype=float)
        if len(demands) == 0:
            raise ValueError("actual_demand series cannot be empty.")
        if np.any(np.isnan(demands)):
            raise ValueError("actual_demand contains NaN values.")
        if np.any(demands < 0):
            raise ValueError("actual_demand contains negative values.")

        T = len(demands)
        date_seq = list(dates) if dates is not None else [f"Day_{t}" for t in range(T)]
        if len(date_seq) != T:
            raise ValueError(f"Length of dates ({len(date_seq)}) must match actual_demand ({T}).")

        # Initialize pipeline orders: arrival_day -> list of quantities
        pending_arrivals: Dict[int, float] = {}
        initial_pipeline_qty = 0.0
        for order in self.initial_on_order:
            arr_day = int(order.get("arrival_day", 0))
            qty = float(order.get("quantity", 0.0))
            if qty > 0:
                pending_arrivals[arr_day] = pending_arrivals.get(arr_day, 0.0) + qty
                initial_pipeline_qty += qty

        # State tracking
        current_on_hand = self.initial_stock
        current_backorders = 0.0
        on_order_qty = initial_pipeline_qty

        history_records: List[Dict[str, Any]] = []

        total_demand = 0.0
        total_sales = 0.0
        total_immediate_sales = 0.0
        total_lost_sales = 0.0
        stockout_days_count = 0
        replenishment_cycles_count = 0
        stockout_cycles_count = 0
        total_orders_placed = 0
        total_units_ordered = 0.0
        total_units_received = 0.0

        total_holding_cost = 0.0
        total_ordering_cost = 0.0
        total_purchase_cost = 0.0
        total_shortage_cost = 0.0

        for t in range(T):
            d_t = float(demands[t])
            date_t = date_seq[t]
            start_on_hand = current_on_hand

            # Step 1: Beginning of Day Receipts
            receipts_t = pending_arrivals.pop(t, 0.0)
            avail_stock = start_on_hand + receipts_t
            on_order_qty = max(0.0, on_order_qty - receipts_t)
            total_units_received += receipts_t

            # Step 2: Demand Fulfillment & Shortages
            prior_backorders = current_backorders
            if self.backlog_allowed:
                # In backorder mode: outstanding backorders fulfilled first, then current demand
                effective_demand = d_t + prior_backorders
                sales_t = min(avail_stock, effective_demand)
                current_backorders = effective_demand - sales_t
                lost_sales_t = 0.0

                # Immediate sales on current day's demand
                # Units available for current demand after fulfilling prior backorders
                stock_for_current = max(0.0, avail_stock - prior_backorders)
                immediate_sales_t = min(stock_for_current, d_t)
                new_unmet_t = max(0.0, d_t - immediate_sales_t)

                if self.backorder_penalty_mode == "per_day":
                    penalty_base = current_backorders
                else:
                    penalty_base = new_unmet_t
            else:
                sales_t = min(avail_stock, d_t)
                lost_sales_t = max(0.0, d_t - sales_t)
                immediate_sales_t = sales_t
                new_unmet_t = lost_sales_t
                current_backorders = 0.0
                penalty_base = lost_sales_t

            # Ending physical stock (cannot be negative)
            end_on_hand = avail_stock - sales_t

            # Step 3: Service Tracking
            total_demand += d_t
            total_sales += sales_t
            total_immediate_sales += immediate_sales_t
            total_lost_sales += lost_sales_t
            is_stockout_day = (new_unmet_t > 0.0)
            if is_stockout_day:
                stockout_days_count += 1

            # Step 4: Inventory Position Review & Replenishment
            ip_t = end_on_hand + on_order_qty - current_backorders

            is_review_day = (t % replenishment_policy.review_period_days == 0)
            order_qty_t = 0.0

            if is_review_day and (ip_t <= replenishment_policy.rop):
                replenishment_cycles_count += 1
                if is_stockout_day:
                    stockout_cycles_count += 1

                # Calculate order quantity
                if replenishment_policy.policy_type == "order_up_to":
                    target_s = replenishment_policy.order_up_to if replenishment_policy.order_up_to is not None else replenishment_policy.rop
                    raw_q = max(0.0, target_s - ip_t)
                else:
                    raw_q = replenishment_policy.order_qty

                if raw_q > 0:
                    q_moq = max(raw_q, replenishment_policy.min_order_qty)
                    q_final = np.ceil(q_moq / replenishment_policy.batch_multiple) * replenishment_policy.batch_multiple

                    order_qty_t = float(q_final)
                    arrival_day = t + self.lead_time_days
                    pending_arrivals[arrival_day] = pending_arrivals.get(arrival_day, 0.0) + order_qty_t
                    on_order_qty += order_qty_t

                    total_orders_placed += 1
                    total_units_ordered += order_qty_t

            # Step 5: Cost Accounting
            holding_cost_t = end_on_hand * self.daily_holding_cost_per_unit
            ordering_cost_t = self.order_cost if order_qty_t > 0 else 0.0
            purchase_cost_t = self.unit_cost * order_qty_t
            shortage_cost_t = self.shortage_penalty_per_unit * penalty_base

            total_holding_cost += holding_cost_t
            total_ordering_cost += ordering_cost_t
            total_purchase_cost += purchase_cost_t
            total_shortage_cost += shortage_cost_t

            # Step 6: Daily Balance Invariant Check:
            # delta I = end_on_hand - start_on_hand == receipts_t - sales_t
            balance_delta = (end_on_hand - start_on_hand) - (receipts_t - sales_t)
            daily_balanced = abs(balance_delta) < 1e-7

            # Backlog conservation check:
            # ending_backlog == beginning_backlog + demand - sales
            backlog_delta = current_backorders - (prior_backorders + d_t - sales_t)
            backlog_balanced = abs(backlog_delta) < 1e-7

            backlog_cleared_t = min(avail_stock, prior_backorders) if self.backlog_allowed else 0.0

            history_records.append({
                "day_index": t,
                "date": date_t,
                "beginning_on_hand": start_on_hand,
                "receipts": receipts_t,
                "available_stock": avail_stock,
                "demand": d_t,
                "current_demand": d_t,
                "beginning_backlog": prior_backorders,
                "total_requested": effective_demand if self.backlog_allowed else d_t,
                "backlog_cleared_today": backlog_cleared_t,
                "current_demand_fulfilled_on_time": immediate_sales_t,
                "immediate_sales": immediate_sales_t,
                "sales": sales_t,
                "total_units_fulfilled": sales_t,
                "lost_sales": lost_sales_t,
                "ending_backlog": current_backorders,
                "backorders": current_backorders,
                "ending_on_hand": end_on_hand,
                "on_order": on_order_qty,
                "inventory_position": ip_t,
                "order_placed": order_qty_t,
                "holding_cost": holding_cost_t,
                "ordering_cost": ordering_cost_t,
                "purchase_cost": purchase_cost_t,
                "shortage_cost": shortage_cost_t,
                "daily_cost": holding_cost_t + ordering_cost_t + shortage_cost_t,
                "balance_reconciled": daily_balanced,
                "backlog_balance_reconciled": backlog_balanced,
            })

            current_on_hand = end_on_hand

        daily_df = pd.DataFrame(history_records)

        # Cumulative Inventory Conservation Invariant:
        # EndingStock_T - InitialStock == TotalReceipts - TotalSales
        net_stock_change = current_on_hand - self.initial_stock
        net_physical_flow = total_units_received - total_sales
        cumulative_balanced = abs(net_stock_change - net_physical_flow) < 1e-6

        # Key Performance Indicators
        on_time_fill_rate = float(total_immediate_sales / total_demand) if total_demand > 0 else 1.0
        physical_fulfillment_rate = float(total_sales / total_demand) if total_demand > 0 else 1.0
        total_backlog_cleared = float(total_sales - total_immediate_sales)
        non_stockout_day_ratio = float((T - stockout_days_count) / T) if T > 0 else 1.0
        cycle_service_level = (
            float((replenishment_cycles_count - stockout_cycles_count) / replenishment_cycles_count)
            if replenishment_cycles_count > 0
            else 1.0
        )
        avg_inventory = float(daily_df["ending_on_hand"].mean())
        total_operating_cost = float(total_holding_cost + total_ordering_cost + total_shortage_cost)

        summary = {
            "policy_name": replenishment_policy.policy_name,
            "policy_type": replenishment_policy.policy_type,
            "horizon_days": T,
            "initial_stock": self.initial_stock,
            "final_on_hand": current_on_hand,
            "final_on_order": on_order_qty,
            "final_backorders": current_backorders,
            "total_demand": total_demand,
            "total_sales": total_sales,
            "total_immediate_sales": total_immediate_sales,
            "total_backlog_cleared": total_backlog_cleared,
            "total_lost_sales": total_lost_sales,
            "on_time_fill_rate": on_time_fill_rate,
            "physical_fulfillment_rate": physical_fulfillment_rate,
            "fill_rate": on_time_fill_rate,
            "eventual_fill_rate": physical_fulfillment_rate,  # Retained as backwards-compatible alias
            "non_stockout_day_ratio": non_stockout_day_ratio,
            "cycle_service_level": cycle_service_level,
            "stockout_days": stockout_days_count,
            "total_orders_placed": total_orders_placed,
            "total_units_ordered": total_units_ordered,
            "total_units_received": total_units_received,
            "average_on_hand_inventory": avg_inventory,
            "cost_holding": total_holding_cost,
            "cost_ordering": total_ordering_cost,
            "cost_purchase": total_purchase_cost,
            "cost_shortage_penalty": total_shortage_cost,
            "cost_operating_total": total_operating_cost,
            "cost_all_inclusive": total_operating_cost + total_purchase_cost,
            "balance_reconciled": cumulative_balanced,
        }

        return SimulationResult(
            policy_name=replenishment_policy.policy_name,
            daily_history=daily_df,
            summary_metrics=summary,
            balance_reconciled=cumulative_balanced,
        )


# ---------------------------------------------------------------------------
# Policy Comparison Runner
# ---------------------------------------------------------------------------

def compare_inventory_policies(
    demand_series: Union[List[float], np.ndarray, pd.Series],
    policies: List[InventoryPolicy],
    simulator_params: Optional[Dict[str, Any]] = None,
    dates: Optional[Union[List[Any], pd.Series]] = None,
) -> Tuple[pd.DataFrame, Dict[str, SimulationResult]]:
    """Simulate and compare multiple inventory policies on an identical demand sequence.

    Parameters
    ----------
    demand_series : Union[List[float], np.ndarray, pd.Series]
        Customer demand sequence.
    policies : List[InventoryPolicy]
        List of candidate replenishment policies to evaluate.
    simulator_params : Optional[Dict[str, Any]]
        Shared simulator configuration (initial_stock, lead_time_days, costs, etc.).
    dates : Optional[Union[List[Any], pd.Series]]
        Corresponding calendar dates.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, SimulationResult]]
        - Comparison summary table with one row per policy.
        - Dictionary mapping policy name to full SimulationResult.
    """
    params = simulator_params or {}
    results_dict: Dict[str, SimulationResult] = {}
    summary_rows: List[Dict[str, Any]] = []

    for policy in policies:
        sim = InventorySimulator(**params)
        res = sim.run(demand_series, replenishment_policy=policy, dates=dates)
        results_dict[policy.policy_name] = res
        summary_rows.append(res.summary_metrics)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, results_dict
