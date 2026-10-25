"""Unit tests for discrete-event inventory simulation engine, balance invariants, and policy comparison."""

import numpy as np
import pandas as pd
import pytest

from src.retail_forecasting import simulation
from src.retail_forecasting.simulation import (
    InventoryPolicy,
    InventorySimulator,
    SimulationResult,
    compare_inventory_policies,
)


# ---------------------------------------------------------------------------
# Module Interface & Dataclass Validation
# ---------------------------------------------------------------------------

def test_simulation_module_importable():
    """Verify simulation module exposes InventorySimulator and InventoryPolicy."""
    assert hasattr(simulation, "InventorySimulator")
    assert hasattr(simulation, "InventoryPolicy")
    assert hasattr(simulation, "compare_inventory_policies")


def test_inventory_policy_validation():
    """Verify InventoryPolicy enforces parameter constraints."""
    # Valid policy
    pol = InventoryPolicy(policy_name="TestPolicy", rop=50.0, order_qty=100.0)
    assert pol.policy_name == "TestPolicy"
    assert pol.policy_type == "rop_eoq"

    # Empty name
    with pytest.raises(ValueError, match="policy_name must be a non-empty string"):
        InventoryPolicy(policy_name="", rop=50.0, order_qty=100.0)

    # Invalid policy type
    with pytest.raises(ValueError, match="policy_type must be"):
        InventoryPolicy(policy_name="Test", policy_type="invalid_type", rop=50.0, order_qty=100.0)

    # Negative ROP or Order Qty
    with pytest.raises(ValueError, match="rop must be non-negative"):
        InventoryPolicy(policy_name="Test", rop=-10.0, order_qty=100.0)
    with pytest.raises(ValueError, match="order_qty must be non-negative"):
        InventoryPolicy(policy_name="Test", rop=50.0, order_qty=-5.0)

    # Order up to smaller than ROP
    with pytest.raises(ValueError, match="order_up_to .* must be >= rop"):
        InventoryPolicy(policy_name="Test", policy_type="order_up_to", rop=50.0, order_up_to=40.0)


def test_simulator_initialization_validation():
    """Verify InventorySimulator rejects invalid physical and financial parameters."""
    with pytest.raises(ValueError, match="initial_stock must be non-negative"):
        InventorySimulator(initial_stock=-10.0)

    with pytest.raises(ValueError, match="lead_time_days must be >= 1"):
        InventorySimulator(lead_time_days=0)

    with pytest.raises(ValueError, match="unit_cost must be strictly positive"):
        InventorySimulator(unit_cost=0.0)

    with pytest.raises(ValueError, match="order_cost must be non-negative"):
        InventorySimulator(order_cost=-5.0)

    with pytest.raises(ValueError, match="holding_cost_annual_rate must be strictly positive"):
        InventorySimulator(holding_cost_annual_rate=-0.1)


# ---------------------------------------------------------------------------
# Fundamental Balance Invariant & Deterministic Step-by-Step Scenario
# ---------------------------------------------------------------------------

def test_inventory_balance_equation():
    """Verify fundamental inventory conservation equation holds for arbitrary demand trajectories.

    Daily Invariant:
    Delta I_t = I_t^{end} - I_{t-1}^{end} == Receipts_t - Sales_t

    Cumulative Invariant:
    EndingStock_T - InitialStock == TotalReceipts - TotalSales
    """
    demands = [12.0, 25.0, 0.0, 40.0, 15.0, 8.0, 30.0, 50.0, 10.0, 20.0]
    policy = InventoryPolicy(policy_name="Dynamic_ROP", rop=35.0, order_qty=60.0)

    sim = InventorySimulator(
        initial_stock=50.0,
        lead_time_days=2,
        unit_cost=10.0,
        order_cost=50.0,
        holding_cost_annual_rate=0.20,
    )

    res = sim.run(actual_demand=demands, replenishment_policy=policy)

    assert res.balance_reconciled is True
    daily = res.daily_history

    # Verify daily reconciliation for every single step
    for _, row in daily.iterrows():
        assert row["balance_reconciled"] is True
        net_stock = row["ending_on_hand"] - row["beginning_on_hand"]
        net_flow = row["receipts"] - row["sales"]
        assert pytest.approx(net_stock, abs=1e-7) == net_flow

    # Verify cumulative conservation
    summary = res.summary_metrics
    assert summary["balance_reconciled"] is True
    net_cum_stock = summary["final_on_hand"] - summary["initial_stock"]
    net_cum_flow = summary["total_units_received"] - summary["total_sales"]
    assert pytest.approx(net_cum_stock, abs=1e-6) == net_cum_flow


def test_hand_checked_deterministic_scenario():
    """Verify exact step-by-step numbers against a hand-calculated deterministic trace.

    Scenario Parameters:
    - Initial Stock: 30 units
    - Lead Time L: 2 days
    - Unit Cost: $10.00
    - Order Cost K: $50.00
    - Policy: ROP = 20 units, EOQ = 40 units (Lost Sales mode)
    - Demands: [15, 10, 20, 10] (4 days)

    Day 0:
      Start On-Hand = 30. Receipts = 0. Avail = 30.
      Demand = 15 -> Sales = 15, Lost = 0, End On-Hand = 15.
      IP = End On-Hand (15) + On-Order (0) = 15 <= ROP (20).
      Trigger Order: Q_0 = 40. Scheduled arrival at Day 0 + 2 = Day 2.
      On-Order becomes 40.

    Day 1:
      Start On-Hand = 15. Receipts = 0. Avail = 15.
      Demand = 10 -> Sales = 10, Lost = 0, End On-Hand = 5.
      IP = End On-Hand (5) + On-Order (40) = 45 > ROP (20).
      No Order Placed.

    Day 2:
      Start On-Hand = 5. Receipts = 40 (Q_0 arrives!). Avail = 45. On-Order becomes 0.
      Demand = 20 -> Sales = 20, Lost = 0, End On-Hand = 25.
      IP = End On-Hand (25) + On-Order (0) = 25 > ROP (20).
      No Order Placed.

    Day 3:
      Start On-Hand = 25. Receipts = 0. Avail = 25.
      Demand = 10 -> Sales = 10, Lost = 0, End On-Hand = 15.
      IP = End On-Hand (15) + On-Order (0) = 15 <= ROP (20).
      Trigger Order: Q_3 = 40. Scheduled arrival at Day 3 + 2 = Day 5.
      On-Order becomes 40.

    Summary:
      Total Demand = 15 + 10 + 20 + 10 = 55
      Total Sales = 55 (Fill Rate = 100%)
      Total Orders Placed = 2 (Day 0 and Day 3)
      Total Units Ordered = 80
      Total Units Received = 40
      Final On-Hand = 15
      Initial Stock = 30
      Net stock change: 15 - 30 = -15
      Net flow: 40 (received) - 55 (sales) = -15 (Reconciled!)
    """
    demands = [15.0, 10.0, 20.0, 10.0]
    policy = InventoryPolicy(policy_name="HandChecked", rop=20.0, order_qty=40.0)

    sim = InventorySimulator(
        initial_stock=30.0,
        lead_time_days=2,
        unit_cost=10.0,
        order_cost=50.0,
        holding_cost_annual_rate=0.20,
    )

    res = sim.run(actual_demand=demands, replenishment_policy=policy)
    daily = res.daily_history

    # Verify Day 0
    assert daily.loc[0, "beginning_on_hand"] == 30.0
    assert daily.loc[0, "receipts"] == 0.0
    assert daily.loc[0, "sales"] == 15.0
    assert daily.loc[0, "ending_on_hand"] == 15.0
    assert daily.loc[0, "order_placed"] == 40.0
    assert daily.loc[0, "on_order"] == 40.0

    # Verify Day 1
    assert daily.loc[1, "beginning_on_hand"] == 15.0
    assert daily.loc[1, "receipts"] == 0.0
    assert daily.loc[1, "sales"] == 10.0
    assert daily.loc[1, "ending_on_hand"] == 5.0
    assert daily.loc[1, "order_placed"] == 0.0
    assert daily.loc[1, "on_order"] == 40.0

    # Verify Day 2 (Order arrives)
    assert daily.loc[2, "beginning_on_hand"] == 5.0
    assert daily.loc[2, "receipts"] == 40.0
    assert daily.loc[2, "available_stock"] == 45.0
    assert daily.loc[2, "sales"] == 20.0
    assert daily.loc[2, "ending_on_hand"] == 25.0
    assert daily.loc[2, "order_placed"] == 0.0
    assert daily.loc[2, "on_order"] == 0.0

    # Verify Day 3
    assert daily.loc[3, "beginning_on_hand"] == 25.0
    assert daily.loc[3, "receipts"] == 0.0
    assert daily.loc[3, "sales"] == 10.0
    assert daily.loc[3, "ending_on_hand"] == 15.0
    assert daily.loc[3, "order_placed"] == 40.0
    assert daily.loc[3, "on_order"] == 40.0

    # Verify summary metrics
    s = res.summary_metrics
    assert s["total_demand"] == 55.0
    assert s["total_sales"] == 55.0
    assert s["fill_rate"] == 1.0
    assert s["total_orders_placed"] == 2
    assert s["total_units_ordered"] == 80.0
    assert s["total_units_received"] == 40.0
    assert s["final_on_hand"] == 15.0
    assert s["final_on_order"] == 40.0
    assert s["balance_reconciled"] is True


# ---------------------------------------------------------------------------
# Edge Cases & Service Level Tests
# ---------------------------------------------------------------------------

def test_zero_demand_scenario():
    """Verify behavior when customer demand is zero across the horizon."""
    demands = [0.0] * 10
    policy = InventoryPolicy(policy_name="ZeroDemand", rop=20.0, order_qty=30.0)

    sim = InventorySimulator(initial_stock=25.0, lead_time_days=3)
    res = sim.run(actual_demand=demands, replenishment_policy=policy)

    s = res.summary_metrics
    assert s["total_demand"] == 0.0
    assert s["total_sales"] == 0.0
    assert s["total_orders_placed"] == 0
    assert s["stockout_days"] == 0
    assert s["fill_rate"] == 1.0
    assert s["final_on_hand"] == 25.0
    assert res.balance_reconciled is True


def test_massive_demand_spike_and_stockout():
    """Verify stockout handling and lost sales tracking under severe demand spike."""
    # Day 0 demand = 100 on initial stock 20 -> 20 fulfilled, 80 lost sales
    demands = [100.0, 10.0, 10.0]
    policy = InventoryPolicy(policy_name="SpikeTest", rop=25.0, order_qty=50.0)

    sim = InventorySimulator(
        initial_stock=20.0,
        lead_time_days=2,
        shortage_penalty_per_unit=5.0,
        backlog_allowed=False,
    )
    res = sim.run(actual_demand=demands, replenishment_policy=policy)
    daily = res.daily_history

    # Day 0: 20 sold, 80 lost sales, ending on-hand = 0 (never negative!)
    assert daily.loc[0, "sales"] == 20.0
    assert daily.loc[0, "lost_sales"] == 80.0
    assert daily.loc[0, "ending_on_hand"] == 0.0

    # Shortage penalty = 80 * 5 = $400
    assert daily.loc[0, "shortage_cost"] == 400.0

    s = res.summary_metrics
    assert s["stockout_days"] >= 1
    assert s["total_lost_sales"] >= 80.0
    assert s["fill_rate"] < 1.0
    assert res.balance_reconciled is True


def test_backorder_mode_fulfillment():
    """Verify backorders accumulate and fulfill when incoming receipts arrive."""
    # Day 0: Demand 30, Initial stock 10 -> Sales 10, Backorders 20. Order 40 placed (L=1).
    # Day 1: Receipts 40. Available stock = 40. Effective demand = Demand (5) + Backorders (20) = 25.
    # Sales = 25, Backorders cleared to 0, Ending stock = 40 - 25 = 15.
    demands = [30.0, 5.0]
    policy = InventoryPolicy(policy_name="BackorderPolicy", rop=20.0, order_qty=40.0)

    sim = InventorySimulator(
        initial_stock=10.0,
        lead_time_days=1,
        backlog_allowed=True,
    )
    res = sim.run(actual_demand=demands, replenishment_policy=policy)
    daily = res.daily_history

    assert daily.loc[0, "sales"] == 10.0
    assert daily.loc[0, "backorders"] == 20.0
    assert daily.loc[0, "ending_on_hand"] == 0.0

    assert daily.loc[1, "receipts"] == 40.0
    assert daily.loc[1, "sales"] == 25.0  # 20 backordered + 5 current
    assert daily.loc[1, "backorders"] == 0.0
    assert daily.loc[1, "ending_on_hand"] == 15.0

    assert res.balance_reconciled is True


def test_initial_pipeline_orders_in_flight():
    """Verify existing initial on-order pipeline shipments are honored and received."""
    # Initial pipeline order of 30 units arriving on Day 1
    initial_orders = [{"arrival_day": 1, "quantity": 30.0}]
    demands = [5.0, 10.0, 10.0]
    policy = InventoryPolicy(policy_name="PipelineTest", rop=10.0, order_qty=20.0)

    sim = InventorySimulator(
        initial_stock=10.0,
        lead_time_days=2,
        initial_on_order=initial_orders,
    )
    res = sim.run(actual_demand=demands, replenishment_policy=policy)
    daily = res.daily_history

    assert daily.loc[1, "receipts"] == 30.0
    assert res.balance_reconciled is True


# ---------------------------------------------------------------------------
# Policy Comparison Runner Tests
# ---------------------------------------------------------------------------

def test_compare_inventory_policies_runner():
    """Verify compare_inventory_policies executes multiple policies on identical demand."""
    demands = [20.0, 25.0, 30.0, 15.0, 10.0, 35.0, 20.0, 40.0]

    baseline_policy = InventoryPolicy(
        policy_name="Baseline_Fixed",
        rop=15.0,       # Low ROP -> frequent stockouts
        order_qty=25.0,
    )
    optimized_policy = InventoryPolicy(
        policy_name="Optimized_ROP_EOQ",
        rop=60.0,       # Well-buffered ROP
        order_qty=70.0,
    )

    summary_df, results = compare_inventory_policies(
        demand_series=demands,
        policies=[baseline_policy, optimized_policy],
        simulator_params={
            "initial_stock": 40.0,
            "lead_time_days": 2,
            "unit_cost": 10.0,
            "order_cost": 50.0,
            "holding_cost_annual_rate": 0.20,
            "shortage_penalty_per_unit": 10.0,
        },
    )

    assert len(summary_df) == 2
    assert "Baseline_Fixed" in results
    assert "Optimized_ROP_EOQ" in results

    # Both must maintain reconciled balances
    assert results["Baseline_Fixed"].balance_reconciled is True
    assert results["Optimized_ROP_EOQ"].balance_reconciled is True

    # Optimized policy should achieve higher fill rate than under-buffered baseline
    opt_fill = summary_df.loc[summary_df["policy_name"] == "Optimized_ROP_EOQ", "fill_rate"].values[0]
    base_fill = summary_df.loc[summary_df["policy_name"] == "Baseline_Fixed", "fill_rate"].values[0]
    assert opt_fill >= base_fill


# ---------------------------------------------------------------------------
# Additional Comprehensive Regression Tests
# ---------------------------------------------------------------------------

def test_hand_checked_backorder_deterministic_scenario():
    """Verify hand-calculated deterministic trace in backorder mode across 5 days.

    Setup:
    - Initial Stock: 10 units
    - Lead Time L: 2 days
    - Policy: ROP = 15, EOQ = 30
    - Demands: [20, 10, 5, 10, 5] (5 days)

    Day 0: Start=10, Receipts=0, Avail=10.
           Demand=20 -> Sales=10, Backorders=10. End=0.
           IP = 0 + 0 - 10 = -10 <= 15 -> Order Q_0 = 30 placed (arrives Day 2). On-Order=30.
    Day 1: Start=0, Receipts=0, Avail=0.
           Demand=10 -> Effective Demand = 10 + 10 = 20. Sales=0, Backorders=20. End=0.
           IP = 0 + 30 - 20 = 10 <= 15 -> Order Q_1 = 30 placed (arrives Day 3). On-Order=60.
    Day 2: Start=0, Receipts=30 (Q_0 arrives!). Avail=30. On-Order=30.
           Demand=5 -> Effective Demand = 5 + 20 = 25.
           Sales=25 (20 backlog cleared + 5 current). Backorders=0. End=5.
           IP = 5 + 30 - 0 = 35 > 15 -> No order placed.
    Day 3: Start=5, Receipts=30 (Q_1 arrives!). Avail=35. On-Order=0.
           Demand=10 -> Sales=10, Backorders=0. End=25.
           IP = 25 + 0 - 0 = 25 > 15 -> No order placed.
    Day 4: Start=25, Receipts=0. Avail=25.
           Demand=5 -> Sales=5, Backorders=0. End=20.
           IP = 20 > 15 -> No order placed.

    Totals:
    - Total Demand = 20 + 10 + 5 + 10 + 5 = 50
    - Total Sales = 10 + 0 + 25 + 10 + 5 = 50
    - Total Receipts = 30 + 30 = 60
    - Final On-Hand = 20
    - Initial Stock = 10
    - Net Stock = 20 - 10 = +10. Net Flow = 60 (Receipts) - 50 (Sales) = +10. (Reconciled!)
    """
    demands = [20.0, 10.0, 5.0, 10.0, 5.0]
    policy = InventoryPolicy(policy_name="BackorderTrace", rop=15.0, order_qty=30.0)

    sim = InventorySimulator(
        initial_stock=10.0,
        lead_time_days=2,
        backlog_allowed=True,
    )
    res = sim.run(actual_demand=demands, replenishment_policy=policy)
    daily = res.daily_history

    # Day 0
    assert daily.loc[0, "beginning_on_hand"] == 10.0
    assert daily.loc[0, "sales"] == 10.0
    assert daily.loc[0, "backorders"] == 10.0
    assert daily.loc[0, "ending_on_hand"] == 0.0
    assert daily.loc[0, "order_placed"] == 30.0

    # Day 1 (Backorder accumulates)
    assert daily.loc[1, "beginning_on_hand"] == 0.0
    assert daily.loc[1, "sales"] == 0.0
    assert daily.loc[1, "backorders"] == 20.0
    assert daily.loc[1, "ending_on_hand"] == 0.0
    assert daily.loc[1, "order_placed"] == 30.0

    # Day 2 (Q_0 arrives, clears backlog)
    assert daily.loc[2, "receipts"] == 30.0
    assert daily.loc[2, "available_stock"] == 30.0
    assert daily.loc[2, "beginning_backlog"] == 20.0
    assert daily.loc[2, "demand"] == 5.0
    assert daily.loc[2, "total_requested"] == 25.0
    assert daily.loc[2, "backlog_cleared_today"] == 20.0
    assert daily.loc[2, "current_demand_fulfilled_on_time"] == 5.0
    assert daily.loc[2, "sales"] == 25.0
    assert daily.loc[2, "backorders"] == 0.0
    assert daily.loc[2, "ending_on_hand"] == 5.0
    assert daily.loc[2, "order_placed"] == 0.0

    # Day 3
    assert daily.loc[3, "receipts"] == 30.0
    assert daily.loc[3, "sales"] == 10.0
    assert daily.loc[3, "ending_on_hand"] == 25.0

    # Day 4
    assert daily.loc[4, "sales"] == 5.0
    assert daily.loc[4, "ending_on_hand"] == 20.0

    s = res.summary_metrics
    assert s["total_demand"] == 50.0
    assert s["total_sales"] == 50.0
    assert s["total_immediate_sales"] == 30.0
    assert s["total_backlog_cleared"] == 20.0
    assert s["on_time_fill_rate"] == 30.0 / 50.0  # Exactly 60% on-time fill rate!
    assert s["physical_fulfillment_rate"] == 50.0 / 50.0  # 100% physical units fulfilled!
    assert s["final_backorders"] == 0.0
    assert s["final_on_hand"] == 20.0
    assert res.balance_reconciled is True


def test_receipt_arriving_on_day_with_demand_same_day_availability():
    """Verify orders arriving on day t are received BEFORE demand fulfillment."""
    # Start with 0 stock, an order of 50 units arrives on Day 0
    # Day 0 demand = 30 -> all 30 should be fulfilled immediately from the arriving receipt
    initial_orders = [{"arrival_day": 0, "quantity": 50.0}]
    demands = [30.0]
    policy = InventoryPolicy(policy_name="SameDayReceipt", rop=10.0, order_qty=20.0)

    sim = InventorySimulator(
        initial_stock=0.0,
        lead_time_days=2,
        initial_on_order=initial_orders,
    )
    res = sim.run(actual_demand=demands, replenishment_policy=policy)
    daily = res.daily_history

    assert daily.loc[0, "beginning_on_hand"] == 0.0
    assert daily.loc[0, "receipts"] == 50.0
    assert daily.loc[0, "available_stock"] == 50.0
    assert daily.loc[0, "sales"] == 30.0
    assert daily.loc[0, "lost_sales"] == 0.0
    assert daily.loc[0, "ending_on_hand"] == 20.0
    assert res.balance_reconciled is True


def test_multiple_orders_concurrently_in_transit():
    """Verify multiple in-flight orders arriving on consecutive days are correctly handled."""
    # Staggered pipeline orders: 20 arriving Day 1, 30 arriving Day 2
    initial_orders = [
        {"arrival_day": 1, "quantity": 20.0},
        {"arrival_day": 2, "quantity": 30.0},
    ]
    demands = [10.0, 15.0, 20.0]
    policy = InventoryPolicy(policy_name="MultiInTransit", rop=5.0, order_qty=25.0)

    sim = InventorySimulator(
        initial_stock=15.0,
        lead_time_days=3,
        initial_on_order=initial_orders,
    )
    res = sim.run(actual_demand=demands, replenishment_policy=policy)
    daily = res.daily_history

    assert daily.loc[1, "receipts"] == 20.0
    assert daily.loc[2, "receipts"] == 30.0
    assert res.balance_reconciled is True


def test_order_placed_on_final_day_does_not_break_conservation():
    """Verify an order placed on the final day (arriving after horizon) does not violate balance."""
    # Day 0 demand 10, Day 1 demand 20 on initial stock 25
    # Day 1: End stock = 5 <= ROP (10) -> Order placed arriving on Day 1 + 3 = Day 4 (T=2)
    demands = [10.0, 20.0]
    policy = InventoryPolicy(policy_name="FinalDayOrder", rop=10.0, order_qty=40.0)

    sim = InventorySimulator(initial_stock=25.0, lead_time_days=3)
    res = sim.run(actual_demand=demands, replenishment_policy=policy)
    daily = res.daily_history

    assert daily.loc[1, "order_placed"] == 40.0
    assert daily.loc[1, "receipts"] == 0.0

    s = res.summary_metrics
    assert s["total_orders_placed"] == 1
    assert s["total_units_ordered"] == 40.0
    assert s["total_units_received"] == 0.0
    assert s["final_on_order"] == 40.0
    assert s["final_on_hand"] == 0.0  # 25 - (10 + 15) = 0, 5 lost sales
    # Conservation holds: End (0) - Init (25) = -25. Flow: Receipts (0) - Sales (25) = -25.
    assert res.balance_reconciled is True


def test_inventory_position_prevents_premature_reordering():
    """Verify outstanding orders in IP prevent redundant order triggers."""
    # Day 0: Start=15, Demand=10 -> End=5 <= ROP(10). Order 50 placed (L=3, arrives Day 3).
    # Day 1: Start=5, Demand=2 -> End=3. IP = 3 + 50 = 53 > ROP(10) -> NO order placed!
    # Day 2: Start=3, Demand=1 -> End=2. IP = 2 + 50 = 52 > ROP(10) -> NO order placed!
    demands = [10.0, 2.0, 1.0]
    policy = InventoryPolicy(policy_name="IPTest", rop=10.0, order_qty=50.0)

    sim = InventorySimulator(initial_stock=15.0, lead_time_days=3)
    res = sim.run(actual_demand=demands, replenishment_policy=policy)
    daily = res.daily_history

    assert daily.loc[0, "order_placed"] == 50.0
    assert daily.loc[1, "order_placed"] == 0.0
    assert daily.loc[2, "order_placed"] == 0.0
    assert res.summary_metrics["total_orders_placed"] == 1
    assert res.balance_reconciled is True


def test_summary_metrics_consistent_with_daily_history():
    """Verify that all simulation summary metrics match daily history aggregations exactly."""
    demands = [15.0, 20.0, 5.0, 30.0, 10.0, 8.0, 12.0]
    policy = InventoryPolicy(policy_name="ConsistencyCheck", rop=25.0, order_qty=40.0)

    sim = InventorySimulator(
        initial_stock=30.0,
        lead_time_days=2,
        unit_cost=10.0,
        order_cost=50.0,
        holding_cost_annual_rate=0.20,
        shortage_penalty_per_unit=15.0,
        backlog_allowed=False,
    )
    res = sim.run(actual_demand=demands, replenishment_policy=policy)
    daily = res.daily_history
    s = res.summary_metrics

    # Verify flow totals
    assert abs(s["total_demand"] - daily["demand"].sum()) < 1e-9
    assert abs(s["total_sales"] - daily["sales"].sum()) < 1e-9
    assert abs(s["total_immediate_sales"] - daily["immediate_sales"].sum()) < 1e-9
    assert abs(s["total_lost_sales"] - daily["lost_sales"].sum()) < 1e-9
    assert s["stockout_days"] == int((daily["lost_sales"] > 0).sum())
    assert s["total_orders_placed"] == int((daily["order_placed"] > 0).sum())
    assert abs(s["total_units_ordered"] - daily["order_placed"].sum()) < 1e-9
    assert abs(s["total_units_received"] - daily["receipts"].sum()) < 1e-9
    assert abs(s["average_on_hand_inventory"] - daily["ending_on_hand"].mean()) < 1e-9

    # Verify cost totals and component additions
    assert abs(s["cost_holding"] - daily["holding_cost"].sum()) < 1e-9
    assert abs(s["cost_ordering"] - daily["ordering_cost"].sum()) < 1e-9
    assert abs(s["cost_purchase"] - daily["purchase_cost"].sum()) < 1e-9
    assert abs(s["cost_shortage_penalty"] - daily["shortage_cost"].sum()) < 1e-9

    # Total operating cost MUST equal sum of holding + ordering + shortage
    expected_operating = s["cost_holding"] + s["cost_ordering"] + s["cost_shortage_penalty"]
    assert abs(s["cost_operating_total"] - expected_operating) < 1e-9

    # Total operating cost MUST NOT include purchase spend
    assert s["cost_operating_total"] < s["cost_all_inclusive"]
    assert abs(s["cost_all_inclusive"] - (s["cost_operating_total"] + s["cost_purchase"])) < 1e-9

    # Fill rate calculation
    expected_fill_rate = s["total_immediate_sales"] / s["total_demand"]
    assert abs(s["fill_rate"] - expected_fill_rate) < 1e-9
    assert res.balance_reconciled is True


def test_backorder_penalty_modes_no_double_counting():
    """Verify backorder penalty modes ('one_time' vs 'per_day') and ensure no double counting."""
    # Scenario:
    # Day 0: Start=0, Demand=10 -> End=0, Backorders=10.
    # Day 1: Start=0, Demand=0  -> End=0, Backorders=10 (unmet from Day 0 carried over).
    # Day 2: Start=0, Receipts=20 arrives! Demand=0 -> End=10, Backorders=0 (backlog cleared).
    demands = [10.0, 0.0, 0.0]
    initial_orders = [{"arrival_day": 2, "quantity": 20.0}]
    policy = InventoryPolicy(policy_name="PenaltyModeTest", rop=0.0, order_qty=0.0)

    # 1. 'one_time' penalty mode: penalty charged only when new unmet demand occurs (Day 0: 10 units)
    sim_one_time = InventorySimulator(
        initial_stock=0.0,
        lead_time_days=2,
        unit_cost=10.0,
        shortage_penalty_per_unit=15.0,
        backlog_allowed=True,
        backorder_penalty_mode="one_time",
        initial_on_order=initial_orders,
    )
    res_one_time = sim_one_time.run(actual_demand=demands, replenishment_policy=policy)
    daily_ot = res_one_time.daily_history

    assert daily_ot.loc[0, "shortage_cost"] == 150.0  # 10 units * $15
    assert daily_ot.loc[1, "shortage_cost"] == 0.0    # No new unmet demand; not re-penalized
    assert daily_ot.loc[2, "shortage_cost"] == 0.0
    assert res_one_time.summary_metrics["cost_shortage_penalty"] == 150.0

    # 2. 'per_day' penalty mode: penalty charged each day backlog is open
    sim_per_day = InventorySimulator(
        initial_stock=0.0,
        lead_time_days=2,
        unit_cost=10.0,
        shortage_penalty_per_unit=15.0,
        backlog_allowed=True,
        backorder_penalty_mode="per_day",
        initial_on_order=initial_orders,
    )
    res_per_day = sim_per_day.run(actual_demand=demands, replenishment_policy=policy)
    daily_pd = res_per_day.daily_history

    assert daily_pd.loc[0, "shortage_cost"] == 150.0  # Day 0: 10 units open backlog * $15
    assert daily_pd.loc[1, "shortage_cost"] == 150.0  # Day 1: 10 units open backlog * $15
    assert daily_pd.loc[2, "shortage_cost"] == 0.0    # Day 2: backlog cleared, 0 units * $15
    assert res_per_day.summary_metrics["cost_shortage_penalty"] == 300.0

    # Physical inventory and backlog trajectories MUST be identical regardless of penalty mode
    assert (daily_ot["ending_on_hand"] == daily_pd["ending_on_hand"]).all()
    assert (daily_ot["backorders"] == daily_pd["backorders"]).all()
    assert (daily_ot["sales"] == daily_pd["sales"]).all()
    assert res_one_time.balance_reconciled is True
    assert res_per_day.balance_reconciled is True


def test_backorder_vs_lost_sales_timing_semantics():
    """Verify timing semantics and ensure unmet demand is lost in lost-sales mode and not backlogged."""
    demands = [20.0, 10.0]
    policy = InventoryPolicy(policy_name="SemanticsTest", rop=0.0, order_qty=0.0)

    # 1. Lost Sales Mode
    sim_lost = InventorySimulator(initial_stock=10.0, lead_time_days=2, backlog_allowed=False)
    res_lost = sim_lost.run(actual_demand=demands, replenishment_policy=policy)
    daily_lost = res_lost.daily_history
    s_lost = res_lost.summary_metrics

    # Day 0: start 10, demand 20 -> sales 10, lost sales 10, backorders 0, end 0
    assert daily_lost.loc[0, "sales"] == 10.0
    assert daily_lost.loc[0, "lost_sales"] == 10.0
    assert daily_lost.loc[0, "ending_backlog"] == 0.0
    assert daily_lost.loc[0, "ending_on_hand"] == 0.0

    # Day 1: start 0, demand 10 -> sales 0, lost sales 10, backorders 0, end 0
    assert daily_lost.loc[1, "sales"] == 0.0
    assert daily_lost.loc[1, "lost_sales"] == 10.0
    assert daily_lost.loc[1, "ending_backlog"] == 0.0
    assert daily_lost.loc[1, "ending_on_hand"] == 0.0

    # Summary: lost sales = 20, backorders = 0, fill rate = 10 / 30 = 1/3
    assert s_lost["total_demand"] == 30.0
    assert s_lost["total_sales"] == 10.0
    assert s_lost["total_lost_sales"] == 20.0
    assert s_lost["final_backorders"] == 0.0
    assert abs(s_lost["on_time_fill_rate"] - (10.0 / 30.0)) < 1e-9
    assert abs(s_lost["physical_fulfillment_rate"] - (10.0 / 30.0)) < 1e-9

    # 2. Backorder Mode
    sim_bo = InventorySimulator(initial_stock=10.0, lead_time_days=2, backlog_allowed=True)
    res_bo = sim_bo.run(actual_demand=demands, replenishment_policy=policy)
    daily_bo = res_bo.daily_history
    s_bo = res_bo.summary_metrics

    # Day 0: start 10, demand 20 -> sales 10, lost sales 0, ending_backlog 10, end 0
    assert daily_bo.loc[0, "sales"] == 10.0
    assert daily_bo.loc[0, "lost_sales"] == 0.0
    assert daily_bo.loc[0, "ending_backlog"] == 10.0
    assert daily_bo.loc[0, "ending_on_hand"] == 0.0

    # Day 1: start 0, demand 10, prior_backorders 10 -> effective demand 20. Sales 0, ending_backlog 20, end 0
    assert daily_bo.loc[1, "beginning_backlog"] == 10.0
    assert daily_bo.loc[1, "demand"] == 10.0
    assert daily_bo.loc[1, "total_requested"] == 20.0
    assert daily_bo.loc[1, "sales"] == 0.0
    assert daily_bo.loc[1, "ending_backlog"] == 20.0
    assert daily_bo.loc[1, "ending_on_hand"] == 0.0

    # Summary: lost sales = 0, final backorders = 20 (unmet at horizon, NOT eventually fulfilled)
    assert s_bo["total_demand"] == 30.0
    assert s_bo["total_sales"] == 10.0
    assert s_bo["total_lost_sales"] == 0.0
    assert s_bo["final_backorders"] == 20.0
    assert abs(s_bo["on_time_fill_rate"] - (10.0 / 30.0)) < 1e-9
    assert abs(s_bo["physical_fulfillment_rate"] - (10.0 / 30.0)) < 1e-9



