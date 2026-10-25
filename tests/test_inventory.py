"""Unit tests for inventory optimization policies, formulas, and uncertainty estimation."""

import math
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.retail_forecasting import inventory
from src.retail_forecasting.inventory import (
    InventoryPlanningUnit,
    calculate_daily_holding_cost_per_unit,
    calculate_eoq,
    calculate_inventory_policy,
    calculate_inventory_position,
    calculate_reorder_point,
    calculate_safety_stock,
    compute_autocorrelation_inflation_factor,
    estimate_series_demand_uncertainty,
)


# ---------------------------------------------------------------------------
# Module Interface & Dataclass Validation Tests
# ---------------------------------------------------------------------------

def test_inventory_module_importable():
    """Verify inventory module defines core policy formulas."""
    assert hasattr(inventory, "calculate_safety_stock")
    assert hasattr(inventory, "calculate_reorder_point")
    assert hasattr(inventory, "calculate_eoq")
    assert hasattr(inventory, "calculate_inventory_position")
    assert hasattr(inventory, "estimate_series_demand_uncertainty")
    assert hasattr(inventory, "calculate_inventory_policy")


def test_inventory_planning_unit_validation():
    """Verify InventoryPlanningUnit enforces strict validation constraints."""
    # Valid instance
    unit = InventoryPlanningUnit(
        store_nbr=1,
        family="GROCERY",
        daily_demand_mean=100.0,
        lead_time_days=7,
        service_level=0.95,
        unit_cost=10.0,
        order_cost=50.0,
        holding_cost_annual_rate=0.20,
    )
    assert unit.store_nbr == 1
    assert unit.service_level == 0.95

    # Negative store_nbr
    with pytest.raises(ValueError, match="store_nbr must be a positive integer"):
        InventoryPlanningUnit(store_nbr=0, family="GROCERY", daily_demand_mean=10.0)

    # Empty family
    with pytest.raises(ValueError, match="family must be a non-empty string"):
        InventoryPlanningUnit(store_nbr=1, family="   ", daily_demand_mean=10.0)

    # Negative daily demand
    with pytest.raises(ValueError, match="daily_demand_mean must be non-negative"):
        InventoryPlanningUnit(store_nbr=1, family="GROCERY", daily_demand_mean=-5.0)

    # Invalid lead time (0 or negative)
    with pytest.raises(ValueError, match="lead_time_days must be an integer >= 1"):
        InventoryPlanningUnit(store_nbr=1, family="GROCERY", daily_demand_mean=10.0, lead_time_days=0)

    # Invalid service level (<= 0.50 or >= 1.0)
    with pytest.raises(ValueError, match="service_level must be in"):
        InventoryPlanningUnit(store_nbr=1, family="GROCERY", daily_demand_mean=10.0, service_level=0.45)
    with pytest.raises(ValueError, match="service_level must be in"):
        InventoryPlanningUnit(store_nbr=1, family="GROCERY", daily_demand_mean=10.0, service_level=1.0)

    # Invalid unit cost (<= 0)
    with pytest.raises(ValueError, match="unit_cost must be strictly positive"):
        InventoryPlanningUnit(store_nbr=1, family="GROCERY", daily_demand_mean=10.0, unit_cost=0.0)

    # Invalid holding cost rate (<= 0)
    with pytest.raises(ValueError, match="holding_cost_annual_rate must be strictly positive"):
        InventoryPlanningUnit(
            store_nbr=1, family="GROCERY", daily_demand_mean=10.0, holding_cost_annual_rate=-0.05
        )


# ---------------------------------------------------------------------------
# Safety Stock & ROP Tests
# ---------------------------------------------------------------------------

def test_safety_stock_calculation():
    """Verify safety stock formula with deterministic and stochastic lead time."""
    # Deterministic lead time: SS = z * sigma_L
    # For CSL = 95%, z = 1.6448536...
    # If sigma_{D_L} = 20.0, SS = 1.6448536 * 20 = 32.89707
    ss_det = calculate_safety_stock(lead_time_demand_std=20.0, service_level=0.95)
    expected_z_95 = stats.norm.ppf(0.95)
    assert pytest.approx(ss_det, rel=1e-5) == expected_z_95 * 20.0

    # Stochastic lead time: SS = z * sqrt(L * sigma_D^2 + D^2 * sigma_L^2)
    # L = 4, sigma_D = 5, D = 50, sigma_L = 1
    # total_var = 4 * 25 + 2500 * 1 = 100 + 2500 = 2600
    # total_std = sqrt(2600) = 50.990195
    # SS = z * 50.990195
    ss_stoch = calculate_safety_stock(
        lead_time_demand_std=10.0,  # ignored when stochastic params provided
        service_level_z=1.645,
        lead_time=4.0,
        lead_time_std=1.0,
        demand_mean=50.0,
        demand_std=5.0,
    )
    expected_stoch = 1.645 * math.sqrt(4.0 * 25.0 + (50.0 ** 2) * 1.0)
    assert pytest.approx(ss_stoch, rel=1e-5) == expected_stoch


def test_safety_stock_zero_variability():
    """Verify safety stock collapses to 0 when lead-time demand error is zero."""
    ss = calculate_safety_stock(lead_time_demand_std=0.0, service_level=0.99)
    assert ss == 0.0


def test_safety_stock_service_levels():
    """Verify safety stock scales monotonically with cycle service level targets."""
    sigma = 15.0
    ss_90 = calculate_safety_stock(lead_time_demand_std=sigma, service_level=0.90)
    ss_95 = calculate_safety_stock(lead_time_demand_std=sigma, service_level=0.95)
    ss_99 = calculate_safety_stock(lead_time_demand_std=sigma, service_level=0.99)

    assert ss_90 < ss_95 < ss_99
    assert pytest.approx(ss_90, rel=1e-4) == stats.norm.ppf(0.90) * sigma
    assert pytest.approx(ss_99, rel=1e-4) == stats.norm.ppf(0.99) * sigma


def test_safety_stock_invalid_inputs_raise_errors():
    """Verify safety stock rejects negative standard deviations and invalid service levels."""
    with pytest.raises(ValueError, match="lead_time_demand_std must be non-negative"):
        calculate_safety_stock(lead_time_demand_std=-1.0)

    with pytest.raises(ValueError, match="service_level must be in"):
        calculate_safety_stock(lead_time_demand_std=5.0, service_level=0.30)

    with pytest.raises(ValueError, match="service_level_z must be non-negative"):
        calculate_safety_stock(lead_time_demand_std=5.0, service_level_z=-1.5)


def test_reorder_point_calculation():
    """Verify Reorder Point equals lead-time demand plus safety stock."""
    # Lead-time demand = 70.0, SS = 30.0 -> ROP = 100.0
    rop = calculate_reorder_point(demand_mean_lead_time=70.0, safety_stock=30.0)
    assert rop == 100.0

    # Boundary: zero safety stock
    assert calculate_reorder_point(demand_mean_lead_time=50.0, safety_stock=0.0) == 50.0

    # Invalid inputs
    with pytest.raises(ValueError, match="demand_mean_lead_time must be non-negative"):
        calculate_reorder_point(demand_mean_lead_time=-10.0, safety_stock=5.0)

    with pytest.raises(ValueError, match="safety_stock must be non-negative"):
        calculate_reorder_point(demand_mean_lead_time=10.0, safety_stock=-5.0)


def test_inventory_position_calculation():
    """Verify inventory position accounting for on-hand, pipeline, and backorders."""
    # On-hand = 40, On-order = 60, Backorders = 10 -> IP = 90
    ip = calculate_inventory_position(on_hand=40.0, on_order=60.0, backorders=10.0)
    assert ip == 90.0

    # Zero pipeline and zero backorders
    assert calculate_inventory_position(on_hand=50.0, on_order=0.0) == 50.0

    # Invalid inputs
    with pytest.raises(ValueError, match="on_hand must be non-negative"):
        calculate_inventory_position(on_hand=-5.0, on_order=0.0)
    with pytest.raises(ValueError, match="on_order must be non-negative"):
        calculate_inventory_position(on_hand=10.0, on_order=-1.0)
    with pytest.raises(ValueError, match="backorders must be non-negative"):
        calculate_inventory_position(on_hand=10.0, on_order=0.0, backorders=-2.0)


# ---------------------------------------------------------------------------
# Economic Order Quantity (EOQ) Tests
# ---------------------------------------------------------------------------

def test_eoq_hand_calculated_values():
    """Verify EOQ formula against hand-calculated analytical textbook benchmark.

    Annual Demand D = 1,000 units/year
    Order Setup Cost K = $50.00 / order
    Annual Holding Cost per Unit H = $2.00 / unit / year
    EOQ = sqrt(2 * 1000 * 50 / 2) = sqrt(50,000) = 223.6067977...
    """
    res = calculate_eoq(
        annual_demand=1000.0,
        order_cost=50.0,
        holding_cost_unit_year=2.0,
        min_order_qty=0.0,
        batch_multiple=1.0,
    )
    expected_unconstrained = math.sqrt(50000.0)
    assert pytest.approx(res["eoq_unconstrained"], rel=1e-6) == expected_unconstrained
    # With batch_multiple = 1.0, ceil(223.6068) = 224.0
    assert res["eoq_recommended"] == 224.0


def test_eoq_operational_rounding_and_moq():
    """Verify operational constraints MOQ and batch packaging multiples."""
    # D = 365, K = 10, H = 5 -> EOQ_raw = sqrt(2 * 365 * 10 / 5) = sqrt(1460) = 38.2099
    # If MOQ = 50.0 and batch_multiple = 25.0:
    # max(38.21, 50.0) = 50.0 -> ceil(50 / 25) * 25 = 50.0
    res = calculate_eoq(
        annual_demand=365.0,
        order_cost=10.0,
        holding_cost_unit_year=5.0,
        min_order_qty=50.0,
        batch_multiple=25.0,
    )
    assert pytest.approx(res["eoq_unconstrained"], rel=1e-4) == math.sqrt(1460.0)
    assert res["eoq_recommended"] == 50.0

    # If raw EOQ = 38.21, MOQ = 10, batch_multiple = 10:
    # max(38.21, 10) = 38.21 -> ceil(38.21 / 10) * 10 = 40.0
    res2 = calculate_eoq(
        annual_demand=365.0,
        order_cost=10.0,
        holding_cost_unit_year=5.0,
        min_order_qty=10.0,
        batch_multiple=10.0,
    )
    assert res2["eoq_recommended"] == 40.0


def test_eoq_zero_demand_and_zero_cost():
    """Verify boundary conditions: zero demand and zero order cost return 0."""
    res_zero_d = calculate_eoq(annual_demand=0.0, order_cost=50.0, holding_cost_unit_year=2.0)
    assert res_zero_d["eoq_unconstrained"] == 0.0
    assert res_zero_d["eoq_recommended"] == 0.0

    res_zero_k = calculate_eoq(annual_demand=1000.0, order_cost=0.0, holding_cost_unit_year=2.0)
    assert res_zero_k["eoq_unconstrained"] == 0.0
    assert res_zero_k["eoq_recommended"] == 0.0


def test_eoq_invalid_inputs_raise_errors():
    """Verify EOQ raises actionable ValueError for undefined / invalid inputs."""
    with pytest.raises(ValueError, match="annual_demand must be non-negative"):
        calculate_eoq(annual_demand=-100.0, order_cost=50.0, holding_cost_unit_year=2.0)

    with pytest.raises(ValueError, match="order_cost must be non-negative"):
        calculate_eoq(annual_demand=100.0, order_cost=-10.0, holding_cost_unit_year=2.0)

    with pytest.raises(ValueError, match="holding_cost_unit_year must be strictly positive"):
        calculate_eoq(annual_demand=100.0, order_cost=50.0, holding_cost_unit_year=0.0)

    with pytest.raises(ValueError, match="holding_cost_unit_year must be strictly positive"):
        calculate_eoq(annual_demand=100.0, order_cost=50.0, holding_cost_unit_year=-1.0)

    with pytest.raises(ValueError, match="batch_multiple must be >= 1.0"):
        calculate_eoq(annual_demand=100.0, order_cost=50.0, holding_cost_unit_year=2.0, batch_multiple=0.5)


# ---------------------------------------------------------------------------
# Autocorrelation & Lead-Time Compounding Tests
# ---------------------------------------------------------------------------

def test_autocorrelation_inflation_factor():
    """Verify lead-time autocorrelation variance inflation factor calculation."""
    # L = 1 -> alpha = 1.0
    assert compute_autocorrelation_inflation_factor(lead_time_days=1) == 1.0

    # L = 7 with rho = 0 -> alpha = 1.0 (pure iid assumption)
    assert compute_autocorrelation_inflation_factor(lead_time_days=7, rho_1=0.0) == 1.0

    # L = 7 with positive AR(1) rho = 0.25 -> alpha > 1.0
    alpha_7 = compute_autocorrelation_inflation_factor(lead_time_days=7, rho_1=0.25)
    assert alpha_7 > 1.0
    # Compounding factor should be approx ~ 1.25 - 1.35
    assert 1.20 <= alpha_7 <= 1.40


# ---------------------------------------------------------------------------
# Uncertainty Estimation Pipeline Tests
# ---------------------------------------------------------------------------

def test_estimate_series_demand_uncertainty_shrinkage_and_floor():
    """Verify Empirical Bayes shrinkage toward family variance and standard deviation floor."""
    # Create synthetic validation forecast dataset with 2 stores and 2 families
    # Store 1, Family HIGH_VOL: high variance
    # Store 2, Family HIGH_VOL: high variance
    # Store 1, Family LOW_VOL: zero variance (all 0 residuals)
    dates = pd.date_range("2023-01-01", periods=14)
    rows = []
    for d in dates:
        # High vol
        rows.append({"store_nbr": 1, "family": "BEVERAGES", "sales": 100.0, "forecast_rf": 90.0, "date": d})
        rows.append({"store_nbr": 2, "family": "BEVERAGES", "sales": 120.0, "forecast_rf": 100.0, "date": d})
        # Low vol (intermittent/zero)
        rows.append({"store_nbr": 1, "family": "BOOKS", "sales": 0.0, "forecast_rf": 0.0, "date": d})
        rows.append({"store_nbr": 2, "family": "BOOKS", "sales": 1.0, "forecast_rf": 0.0, "date": d})

    df = pd.DataFrame(rows)

    unc = estimate_series_demand_uncertainty(
        df,
        shrinkage_k=7,
        floor_std=0.05,
        lead_time_days=7,
        compounding_mode="ar1_analytical",
        ar1_rho=0.25,
    )

    assert len(unc) == 4
    # Store 1 BOOKS has 0 non-zero demand, sample std = 0.0
    books_1 = unc.loc[(1, "BOOKS")]
    assert books_1["nonzero_count"] == 0
    assert books_1["residual_std"] == 0.0
    # Because of shrinkage and floor, final_daily_std must be >= floor_std (0.05)
    assert books_1["final_daily_std"] >= 0.05
    assert books_1["quality_flag"] in ("FLOOR_APPLIED", "SPARSE_SHRUNK")
    assert books_1["lead_time_std"] > 0.0


def test_calculate_inventory_policy_pipeline():
    """Verify end-to-end bulk policy generation across multiple planning units."""
    forecast_df = pd.DataFrame({
        "store_nbr": [1, 1, 2],
        "family": ["GROCERY", "BEVERAGES", "GROCERY"],
        "forecast_rf": [150.0, 300.0, 200.0],
    })

    uncertainty_df = pd.DataFrame({
        "store_nbr": [1, 1, 2],
        "family": ["GROCERY", "BEVERAGES", "GROCERY"],
        "lead_time_std": [25.0, 40.0, 30.0],
        "quality_flag": ["RELIABLE", "RELIABLE", "RELIABLE"],
    }).set_index(["store_nbr", "family"])

    policy_df = calculate_inventory_policy(
        forecast_demand=forecast_df,
        series_uncertainty=uncertainty_df,
        lead_time_days=7,
        service_level=0.95,
        holding_cost_rate=0.20,
        order_cost=50.0,
        unit_cost=10.0,
        min_order_qty=10.0,
        batch_multiple=5.0,
    )

    assert len(policy_df) == 3
    assert "safety_stock" in policy_df.columns
    assert "reorder_point" in policy_df.columns
    assert "eoq_recommended" in policy_df.columns

    # Verify ROP = lead_time_demand + safety_stock
    for _, row in policy_df.iterrows():
        assert pytest.approx(row["reorder_point"]) == row["lead_time_demand"] + row["safety_stock"]
        assert row["safety_stock"] > 0.0
        assert row["eoq_recommended"] >= 10.0
        assert row["eoq_recommended"] % 5.0 == 0.0


# ---------------------------------------------------------------------------
# Purchase Order Recommendation Tests
# ---------------------------------------------------------------------------

def test_generate_purchase_orders_logic():
    """Verify purchase order trigger, quantity rounding, and urgency priority."""
    from src.retail_forecasting.recommendations import generate_purchase_orders

    policy_df = pd.DataFrame({
        "store_nbr": [1, 1, 2],
        "family": ["GROCERY", "BEVERAGES", "DAIRY"],
        "reorder_point": [100.0, 200.0, 50.0],
        "safety_stock": [30.0, 60.0, 15.0],
        "eoq_recommended": [150.0, 250.0, 80.0],
        "unit_cost": [10.0, 5.0, 8.0],
        "lead_time_days": [7, 7, 7],
    })

    # Unit 1: On-hand = 20 <= SS (30) <= ROP (100) -> Triggered, priority HIGH
    # Unit 2: On-hand = 0 <= ROP (200) -> Triggered, priority CRITICAL
    # Unit 3: On-hand = 60 > ROP (50) -> NOT Triggered, priority NONE
    inv_df = pd.DataFrame({
        "store_nbr": [1, 1, 2],
        "family": ["GROCERY", "BEVERAGES", "DAIRY"],
        "on_hand": [20.0, 0.0, 60.0],
        "on_order": [0.0, 0.0, 0.0],
        "backorders": [0.0, 0.0, 0.0],
    })

    po_df = generate_purchase_orders(
        current_inventory_df=inv_df,
        policy_df=policy_df,
        lead_time_days=7,
        current_date="2023-01-01",
    )

    assert len(po_df) == 3

    # Check Unit 2 (CRITICAL)
    crit = po_df[(po_df["store_nbr"] == 1) & (po_df["family"] == "BEVERAGES")].iloc[0]
    assert bool(crit["order_recommended"]) is True
    assert crit["urgency_priority"] == "CRITICAL"
    assert crit["recommended_order_qty"] == 250.0
    assert crit["estimated_purchase_cost"] == 250.0 * 5.0
    assert crit["expected_delivery_date"] == "2023-01-08"

    # Check Unit 1 (HIGH)
    high = po_df[(po_df["store_nbr"] == 1) & (po_df["family"] == "GROCERY")].iloc[0]
    assert bool(high["order_recommended"]) is True
    assert high["urgency_priority"] == "HIGH"
    assert high["recommended_order_qty"] == 150.0

    # Check Unit 3 (NONE)
    none_po = po_df[(po_df["store_nbr"] == 2) & (po_df["family"] == "DAIRY")].iloc[0]
    assert bool(none_po["order_recommended"]) is False
    assert none_po["urgency_priority"] == "NONE"
    assert none_po["recommended_order_qty"] == 0.0


def test_distinguish_zero_residuals_from_zero_demand_and_edge_cases():
    """Verify uncertainty estimation properly distinguishes zero residuals from zero actual demand."""
    dates = pd.date_range("2023-01-01", periods=10)
    rows = []

    # Series A (Store 1, GROCERY): Constant positive demand (50.0) with perfect prediction (50.0)
    # -> nonzero_demand_count = 10, nonzero_residual_count = 0, sample std = 0.0
    for d in dates:
        rows.append({"store_nbr": 1, "family": "GROCERY", "sales": 50.0, "forecast_rf": 50.0, "date": d})

    # Series B (Store 2, GROCERY): Positive demand with errors (variance anchor for GROCERY family)
    for i, d in enumerate(dates):
        rows.append({"store_nbr": 2, "family": "GROCERY", "sales": 50.0 + (i % 3) * 10.0, "forecast_rf": 50.0, "date": d})

    # Series C (Store 1, BOOKS): True zero demand (0.0) and zero prediction (0.0)
    # -> nonzero_demand_count = 0, nonzero_residual_count = 0, sample std = 0.0
    for d in dates:
        rows.append({"store_nbr": 1, "family": "BOOKS", "sales": 0.0, "forecast_rf": 0.0, "date": d})

    # Series D (Store 2, BOOKS): Single occasional sale (1.0) and zero prediction (0.0)
    for i, d in enumerate(dates):
        sales_val = 1.0 if i == 0 else 0.0
        rows.append({"store_nbr": 2, "family": "BOOKS", "sales": sales_val, "forecast_rf": 0.0, "date": d})

    df = pd.DataFrame(rows)

    unc = estimate_series_demand_uncertainty(
        df,
        shrinkage_k=7,
        floor_std=0.05,
        lead_time_days=7,
        compounding_mode="ar1_analytical",
        ar1_rho=0.25,
    )

    # Verify Series A (Active demand, zero residual)
    s_a = unc.loc[(1, "GROCERY")]
    assert s_a["nonzero_demand_count"] == 10
    assert s_a["nonzero_residual_count"] == 0
    assert s_a["residual_std"] == 0.0
    # Because demand was active (10 nonzeros >= 7), it was not sparse, but hit floor
    assert s_a["final_daily_std"] >= 0.05

    # Verify Series C (Zero demand, zero residual)
    s_c = unc.loc[(1, "BOOKS")]
    assert s_c["nonzero_demand_count"] == 0
    assert s_c["nonzero_residual_count"] == 0
    assert bool(s_c["is_sparse"]) is True
    # Fully shrunk to BOOKS family variance
    assert s_c["final_daily_std"] >= 0.05


def test_single_observation_and_nan_residual_handling():
    """Verify N < 2 triggers FAMILY_FALLBACK and NaNs in forecasts are dropped safely."""
    df = pd.DataFrame({
        "store_nbr": [1, 2, 2],
        "family": ["CLEANING", "CLEANING", "CLEANING"],
        "sales": [10.0, 20.0, 30.0],
        "forecast_rf": [10.0, 18.0, 32.0],
    })

    unc = estimate_series_demand_uncertainty(
        df,
        shrinkage_k=7,
        floor_std=0.05,
        lead_time_days=7,
    )

    # Store 1 has N = 1 (< 2)
    s1 = unc.loc[(1, "CLEANING")]
    assert s1["observation_count"] == 1
    assert bool(s1["family_fallback"]) is True
    assert s1["quality_flag"] == "FAMILY_FALLBACK"
    assert s1["final_daily_std"] > 0.0


def test_calculate_daily_holding_cost_per_unit_conversion():
    """Verify daily holding cost per unit conversion from annual rate and unit cost."""
    # Test configured baseline: h = 0.20, c = $10.00
    daily_h_20 = calculate_daily_holding_cost_per_unit(annual_holding_rate=0.20, unit_cost=10.0, days_per_year=365.0)
    expected_20 = 2.0 / 365.0
    assert abs(daily_h_20 - expected_20) < 1e-12
    assert abs(daily_h_20 - 0.005479452054794521) < 1e-12

    # Test alternate example: h = 0.25, c = $10.00
    daily_h_25 = calculate_daily_holding_cost_per_unit(annual_holding_rate=0.25, unit_cost=10.0, days_per_year=365.0)
    expected_25 = 2.5 / 365.0
    assert abs(daily_h_25 - expected_25) < 1e-12
    assert abs(daily_h_25 - 0.006849315068493151) < 1e-12

    # Known numerical example: 100 units held over 28 days at h=0.20, c=10.0
    total_holding_cost = 100.0 * 28.0 * daily_h_20
    assert abs(total_holding_cost - 15.342465753424657) < 1e-6

    # Invalid input validations
    with pytest.raises(ValueError, match="annual_holding_rate"):
        calculate_daily_holding_cost_per_unit(annual_holding_rate=0.0, unit_cost=10.0)
    with pytest.raises(ValueError, match="annual_holding_rate"):
        calculate_daily_holding_cost_per_unit(annual_holding_rate=-0.1, unit_cost=10.0)
    with pytest.raises(ValueError, match="unit_cost"):
        calculate_daily_holding_cost_per_unit(annual_holding_rate=0.20, unit_cost=0.0)
    with pytest.raises(ValueError, match="days_per_year"):
        calculate_daily_holding_cost_per_unit(annual_holding_rate=0.20, unit_cost=10.0, days_per_year=0.0)


def test_purchase_orders_reorder_thresholds_and_edge_cases():
    """Verify purchase order recommendations at, above, and below ROP thresholds."""
    from src.retail_forecasting.recommendations import generate_purchase_orders

    # Policy setup: ROP = 20.0, SS = 10.0, EOQ = 30.0
    policy_df = pd.DataFrame([
        {
            "store_nbr": 1,
            "family": "GROCERY I",
            "reorder_point": 20.0,
            "safety_stock": 10.0,
            "eoq_recommended": 30.0,
            "unit_cost": 10.0,
            "lead_time_days": 7,
        },
        {
            "store_nbr": 1,
            "family": "BEVERAGES",
            "reorder_point": 20.0,
            "safety_stock": 10.0,
            "eoq_recommended": 30.0,
            "unit_cost": 10.0,
            "lead_time_days": 7,
        },
        {
            "store_nbr": 1,
            "family": "CLEANING",
            "reorder_point": 20.0,
            "safety_stock": 10.0,
            "eoq_recommended": 30.0,
            "unit_cost": 10.0,
            "lead_time_days": 7,
        },
        {
            "store_nbr": 1,
            "family": "DAIRY",
            "reorder_point": 20.0,
            "safety_stock": 10.0,
            "eoq_recommended": 30.0,
            "unit_cost": 10.0,
            "lead_time_days": 7,
        },
        {
            "store_nbr": 1,
            "family": "POULTRY",
            "reorder_point": 20.0,
            "safety_stock": 10.0,
            "eoq_recommended": 30.0,
            "unit_cost": 10.0,
            "lead_time_days": 7,
        },
    ])

    # Case 1: IP = 21 > ROP (20) -> no order, NONE
    # Case 2: IP = 20 == ROP (20) -> order placed, NORMAL (on_hand = 20 >= SS)
    # Case 3: IP = 15 < ROP (20) -> order placed, NORMAL (on_hand = 15 >= SS)
    # Case 4: IP = 5 < ROP (20), on_hand = 5 < SS (10) -> order placed, HIGH
    # Case 5: IP = 0 < ROP (20), on_hand = 0 -> order placed, CRITICAL
    inv_df = pd.DataFrame([
        {"store_nbr": 1, "family": "GROCERY I", "on_hand": 21.0, "on_order": 0.0, "backorders": 0.0},
        {"store_nbr": 1, "family": "BEVERAGES", "on_hand": 20.0, "on_order": 0.0, "backorders": 0.0},
        {"store_nbr": 1, "family": "CLEANING", "on_hand": 15.0, "on_order": 0.0, "backorders": 0.0},
        {"store_nbr": 1, "family": "DAIRY", "on_hand": 5.0, "on_order": 0.0, "backorders": 0.0},
        {"store_nbr": 1, "family": "POULTRY", "on_hand": 0.0, "on_order": 0.0, "backorders": 0.0},
    ])

    po_df = generate_purchase_orders(current_inventory_df=inv_df, policy_df=policy_df, lead_time_days=7)
    res = po_df.set_index("family")

    # Above ROP:
    assert bool(res.loc["GROCERY I", "order_recommended"]) is False
    assert res.loc["GROCERY I", "recommended_order_qty"] == 0.0
    assert res.loc["GROCERY I", "urgency_priority"] == "NONE"

    # Exactly at ROP:
    assert bool(res.loc["BEVERAGES", "order_recommended"]) is True
    assert res.loc["BEVERAGES", "recommended_order_qty"] == 30.0
    assert res.loc["BEVERAGES", "urgency_priority"] == "NORMAL"

    # Below ROP, on_hand >= SS:
    assert bool(res.loc["CLEANING", "order_recommended"]) is True
    assert res.loc["CLEANING", "recommended_order_qty"] == 30.0
    assert res.loc["CLEANING", "urgency_priority"] == "NORMAL"

    # Below ROP, on_hand < SS:
    assert bool(res.loc["DAIRY", "order_recommended"]) is True
    assert res.loc["DAIRY", "recommended_order_qty"] == 30.0
    assert res.loc["DAIRY", "urgency_priority"] == "HIGH"

    # Below ROP, on_hand == 0:
    assert bool(res.loc["POULTRY", "order_recommended"]) is True
    assert res.loc["POULTRY", "recommended_order_qty"] == 30.0
    assert res.loc["POULTRY", "urgency_priority"] == "CRITICAL"

    # Case 6: Outstanding order sufficient to cover deficit
    # on_hand = 5.0, on_order = 25.0 -> IP = 30.0 > ROP (20.0) -> no order
    inv_sufficient = pd.DataFrame([
        {"store_nbr": 1, "family": "GROCERY I", "on_hand": 5.0, "on_order": 25.0, "backorders": 0.0}
    ])
    po_suff = generate_purchase_orders(
        current_inventory_df=inv_sufficient,
        policy_df=policy_df[policy_df["family"] == "GROCERY I"],
        lead_time_days=7,
    )
    assert bool(po_suff["order_recommended"].iloc[0]) is False
    assert po_suff["recommended_order_qty"].iloc[0] == 0.0
    assert po_suff["urgency_priority"].iloc[0] == "NONE"



