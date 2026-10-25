"""Inventory control policies and optimization formulas.

Calculates dynamic Safety Stock (SS), Reorder Points (ROP),
and Economic Order Quantities (EOQ) under service level constraints,
and estimates series-level demand uncertainty from out-of-sample forecast residuals.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data Classes & Input Validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InventoryPlanningUnit:
    """Validated inputs for a single store-family planning unit.

    Attributes
    ----------
    store_nbr : int
        Store identifier (> 0).
    family : str
        Product family identifier.
    daily_demand_mean : float
        Expected daily demand rate in units/day (>= 0).
    lead_time_days : int
        Replenishment lead time in calendar days (>= 1).
    service_level : float
        Target cycle service level CSL in (0.50, 1.0).
    unit_cost : float
        Unit purchase cost in $/unit (> 0).
    order_cost : float
        Fixed order/setup cost in $/order (>= 0).
    holding_cost_annual_rate : float
        Annual holding cost rate as fraction of unit cost (> 0, e.g. 0.20 = 20%/year).
    initial_on_hand : float
        Current on-hand physical stock in units (>= 0).
    initial_on_order : float
        Current pipeline stock in transit in units (>= 0).
    min_order_qty : float
        Minimum order quantity MOQ in units (>= 0).
    batch_multiple : float
        Batch packaging increment in units (>= 1.0).
    """

    store_nbr: int
    family: str
    daily_demand_mean: float
    lead_time_days: int = 7
    service_level: float = 0.95
    unit_cost: float = 10.0
    order_cost: float = 50.0
    holding_cost_annual_rate: float = 0.20
    initial_on_hand: float = 0.0
    initial_on_order: float = 0.0
    min_order_qty: float = 0.0
    batch_multiple: float = 1.0

    def __post_init__(self) -> None:
        """Validate input constraints on instantiation."""
        if not isinstance(self.store_nbr, (int, np.integer)) or self.store_nbr <= 0:
            raise ValueError(f"store_nbr must be a positive integer, got {self.store_nbr}")
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError(f"family must be a non-empty string, got {self.family!r}")
        if not np.isfinite(self.daily_demand_mean) or self.daily_demand_mean < 0:
            raise ValueError(
                f"daily_demand_mean must be non-negative and finite, got {self.daily_demand_mean}"
            )
        if not isinstance(self.lead_time_days, (int, np.integer)) or self.lead_time_days < 1:
            raise ValueError(
                f"lead_time_days must be an integer >= 1, got {self.lead_time_days}"
            )
        if not (0.50 < self.service_level < 1.0):
            raise ValueError(
                f"service_level must be in (0.50, 1.0), got {self.service_level}"
            )
        if not np.isfinite(self.unit_cost) or self.unit_cost <= 0:
            raise ValueError(f"unit_cost must be strictly positive and finite, got {self.unit_cost}")
        if not np.isfinite(self.order_cost) or self.order_cost < 0:
            raise ValueError(f"order_cost must be non-negative and finite, got {self.order_cost}")
        if not np.isfinite(self.holding_cost_annual_rate) or self.holding_cost_annual_rate <= 0:
            raise ValueError(
                f"holding_cost_annual_rate must be strictly positive and finite, got {self.holding_cost_annual_rate}"
            )
        if not np.isfinite(self.initial_on_hand) or self.initial_on_hand < 0:
            raise ValueError(
                f"initial_on_hand must be non-negative and finite, got {self.initial_on_hand}"
            )
        if not np.isfinite(self.initial_on_order) or self.initial_on_order < 0:
            raise ValueError(
                f"initial_on_order must be non-negative and finite, got {self.initial_on_order}"
            )
        if not np.isfinite(self.min_order_qty) or self.min_order_qty < 0:
            raise ValueError(
                f"min_order_qty must be non-negative and finite, got {self.min_order_qty}"
            )
        if not np.isfinite(self.batch_multiple) or self.batch_multiple < 1.0:
            raise ValueError(
                f"batch_multiple must be >= 1.0 and finite, got {self.batch_multiple}"
            )


# ---------------------------------------------------------------------------
# Uncertainty Estimation
# ---------------------------------------------------------------------------

def compute_autocorrelation_inflation_factor(
    lead_time_days: int,
    rho_1: float = 0.25,
    empirical_rhos: Optional[List[float]] = None,
) -> float:
    """Calculate the autocorrelation variance inflation factor alpha(L).

    In multi-day recursive forecasting, errors across consecutive days
    are positively autocorrelated, so:
    Var(E_L) = L * sigma^2 + 2 * sum_{k=1}^{L-1} (L - k) * Cov(e_t, e_{t+k})
    alpha(L) = sqrt(1 + 2 * sum_{k=1}^{L-1} (1 - k/L) * rho_k)

    Parameters
    ----------
    lead_time_days : int
        Replenishment lead time in calendar days (>= 1).
    rho_1 : float
        First-order autocorrelation parameter for AR(1) fallback (in [0, 0.9]).
    empirical_rhos : Optional[List[float]]
        Optional empirical autocorrelation coefficients [rho_1, ..., rho_{L-1}].

    Returns
    -------
    float
        Inflation multiplier alpha(L) >= 1.0 applied to sqrt(L).
    """
    if lead_time_days <= 1:
        return 1.0

    if empirical_rhos is not None and len(empirical_rhos) >= 1:
        inflation_sum = 0.0
        for k_idx, rho_k in enumerate(empirical_rhos[: lead_time_days - 1]):
            k = k_idx + 1
            weight = 1.0 - (k / float(lead_time_days))
            inflation_sum += 2.0 * weight * float(np.clip(rho_k, -0.5, 0.95))
        return float(np.sqrt(max(0.1, 1.0 + inflation_sum)))

    # Analytical AR(1) approximation: rho_k = rho_1^k
    rho = float(np.clip(rho_1, 0.0, 0.9))
    if rho == 0.0:
        return 1.0

    L = float(lead_time_days)
    inflation_sum = sum(2.0 * (1.0 - (k / L)) * (rho ** k) for k in range(1, lead_time_days))
    return float(np.sqrt(max(0.1, 1.0 + inflation_sum)))


def estimate_series_demand_uncertainty(
    forecast_df: pd.DataFrame,
    pred_col: str = "forecast_rf",
    actual_col: str = "sales",
    shrinkage_k: int = 7,
    floor_std: float = 0.05,
    lead_time_days: int = 7,
    compounding_mode: str = "ar1_analytical",
    ar1_rho: float = 0.25,
) -> pd.DataFrame:
    """Estimate series-level demand uncertainty from out-of-sample forecast errors.

    Applies hierarchical Empirical Bayes shrinkage toward the product family mean
    variance, applies a minimum standard deviation floor, and compounds daily error
    over the replenishment lead time.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Out-of-sample forecast DataFrame containing ['store_nbr', 'family', actual_col, pred_col].
    pred_col : str
        Name of model prediction column.
    actual_col : str
        Name of actual sales column.
    shrinkage_k : int
        Prior weight parameter for shrinkage toward family pooled variance (>= 0).
    floor_std : float
        Minimum daily residual standard deviation floor (> 0).
    lead_time_days : int
        Replenishment lead time in calendar days (>= 1).
    compounding_mode : str
        Method for lead time compounding: 'ar1_analytical', 'iid_sqrt_l', or 'empirical_recursive'.
    ar1_rho : float
        Assumed lag-1 error autocorrelation for analytical compounding.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by ['store_nbr', 'family'] with columns:
        ['residual_mean', 'residual_std', 'shrunk_std', 'final_daily_std',
         'lead_time_std', 'signed_bias_pct', 'observation_count', 'nonzero_count',
         'quality_flag', 'compounding_mode', 'autocorrelation_inflation'].
    """
    if forecast_df.empty:
        raise ValueError("forecast_df cannot be empty for uncertainty estimation.")
    required = ["store_nbr", "family", actual_col, pred_col]
    missing = [c for c in required if c not in forecast_df.columns]
    if missing:
        raise ValueError(f"forecast_df is missing required columns: {missing}")

    df = forecast_df.copy()
    # Drop rows with NaN in critical actual or prediction columns
    df = df.dropna(subset=[actual_col, pred_col])

    # Forecast error convention: e_t = y_t - y_hat_t (residual)
    df["residual"] = df[actual_col] - df[pred_col]
    df["abs_residual"] = df["residual"].abs()

    # Series-level statistics
    grouped = df.groupby(["store_nbr", "family"])

    def calc_series_stats(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        # N_demand_nonzero: Days with actual positive consumer sales
        n_demand_nonzero = int((g[actual_col] > 0).sum())
        # N_residual_nonzero: Days with non-zero forecast error
        n_residual_nonzero = int((g["abs_residual"] > 1e-6).sum())
        total_actual = float(g[actual_col].sum())
        total_pred = float(g[pred_col].sum())

        res = g["residual"].values
        mean_res = float(np.mean(res)) if n > 0 else 0.0

        # Sample standard deviation (ddof=1)
        if n >= 2:
            s_std = float(np.std(res, ddof=1))
        else:
            s_std = 0.0

        # Signed bias convention: (Pred - Actual) / Actual * 100%
        # Positive = over-forecasting, negative = under-forecasting
        if total_actual > 0:
            bias_pct = float((total_pred - total_actual) / total_actual * 100.0)
        else:
            bias_pct = 0.0 if total_pred == 0.0 else float("inf")

        return pd.Series({
            "observation_count": n,
            "nonzero_count": n_demand_nonzero,
            "nonzero_demand_count": n_demand_nonzero,
            "nonzero_residual_count": n_residual_nonzero,
            "residual_mean": mean_res,
            "residual_std": s_std,
            "signed_bias_pct": bias_pct,
            "total_actual": total_actual,
            "total_pred": total_pred,
        })

    series_df = grouped.apply(calc_series_stats, include_groups=False).reset_index()

    # Family-level pooled variance (unweighted average of series sample variances within family)
    family_vars = series_df.groupby("family")["residual_std"].apply(
        lambda s: float(np.mean(s ** 2)) if len(s) > 0 else (floor_std ** 2)
    ).to_dict()

    # Empirical Bayes shrinkage and flag classification
    shrunk_stds: List[float] = []
    final_stds: List[float] = []
    quality_flags: List[str] = []
    is_sparse_flags: List[bool] = []
    floor_applied_flags: List[bool] = []
    family_fallback_flags: List[bool] = []

    for _, row in series_df.iterrows():
        fam = str(row["family"])
        s_std = float(row["residual_std"])
        s_var = s_std ** 2
        fam_var = family_vars.get(fam, s_var)
        n_demand_nz = int(row["nonzero_demand_count"])
        n_obs = int(row["observation_count"])

        is_sparse = (n_demand_nz < 7)
        fam_fallback = (n_obs < 2)

        if fam_fallback:
            # Insufficient sample size for series variance (N < 2)
            shrunk_var = fam_var
            base_flag = "FAMILY_FALLBACK"
        elif shrinkage_k > 0:
            # Empirical Bayes shrinkage toward family mean variance
            w = float(n_demand_nz) / float(n_demand_nz + shrinkage_k)
            shrunk_var = w * s_var + (1.0 - w) * fam_var
            base_flag = "SPARSE_SHRUNK" if is_sparse else "RELIABLE"
        else:
            shrunk_var = s_var
            base_flag = "RELIABLE"

        shrunk_std = float(np.sqrt(max(0.0, shrunk_var)))
        shrunk_stds.append(shrunk_std)

        # Apply floor: prevent degenerate zero safety stock
        if shrunk_std < floor_std:
            final_std = float(floor_std)
            floor_applied = True
            # Prioritize flag: if fallback was applied, keep fallback; otherwise flag floor
            assigned_flag = "FLOOR_APPLIED" if base_flag != "FAMILY_FALLBACK" else "FAMILY_FALLBACK"
        else:
            final_std = shrunk_std
            floor_applied = False
            assigned_flag = base_flag

        final_stds.append(final_std)
        quality_flags.append(assigned_flag)
        is_sparse_flags.append(is_sparse)
        floor_applied_flags.append(floor_applied)
        family_fallback_flags.append(fam_fallback)

    series_df["shrunk_std"] = shrunk_stds
    series_df["final_daily_std"] = final_stds
    series_df["quality_flag"] = quality_flags
    series_df["is_sparse"] = is_sparse_flags
    series_df["floor_applied"] = floor_applied_flags
    series_df["family_fallback"] = family_fallback_flags

    # Lead-time error compounding
    actual_compounding_mode = compounding_mode
    if compounding_mode == "iid_sqrt_l":
        alpha = 1.0
    elif compounding_mode == "ar1_analytical":
        alpha = compute_autocorrelation_inflation_factor(lead_time_days, rho_1=ar1_rho)
    elif compounding_mode == "empirical_recursive":
        # If multi-step cumulative errors are not present in input, fall back to AR(1)
        actual_compounding_mode = "ar1_analytical (fallback: no multi-day error input)"
        alpha = compute_autocorrelation_inflation_factor(lead_time_days, rho_1=ar1_rho)
    else:
        alpha = compute_autocorrelation_inflation_factor(lead_time_days, rho_1=ar1_rho)

    series_df["autocorrelation_inflation"] = alpha
    series_df["compounding_mode"] = actual_compounding_mode
    # sigma_L = sigma_daily * sqrt(L) * alpha(L)
    series_df["lead_time_std"] = series_df["final_daily_std"] * np.sqrt(lead_time_days) * alpha

    return series_df.set_index(["store_nbr", "family"])


# ---------------------------------------------------------------------------
# Core Inventory Optimization Formulas
# ---------------------------------------------------------------------------

def calculate_safety_stock(
    lead_time_demand_std: float,
    service_level_z: Optional[float] = None,
    service_level: Optional[float] = None,
    lead_time: Optional[float] = None,
    lead_time_std: float = 0.0,
    demand_mean: float = 0.0,
    demand_std: Optional[float] = None,
) -> float:
    """Calculate dynamic Safety Stock (SS) under service level constraints.

    Supports both fixed lead time and stochastic lead time formulations:
    - Fixed lead time: SS = z * sigma_{D_L}
    - Stochastic lead time: SS = z * sqrt(L * sigma_D^2 + D^2 * sigma_L^2)

    Parameters
    ----------
    lead_time_demand_std : float
        Standard deviation of lead-time demand error sigma_{D_L} (>= 0).
    service_level_z : Optional[float]
        Standard normal critical value z = Phi^{-1}(CSL). If omitted, derived from service_level.
    service_level : Optional[float]
        Target cycle service level CSL in (0.50, 1.0) (e.g. 0.95 -> z = 1.645).
    lead_time : Optional[float]
        Average lead time in days (used when lead_time_std > 0).
    lead_time_std : float
        Standard deviation of lead time in days (>= 0).
    demand_mean : float
        Average daily demand rate in units/day (>= 0).
    demand_std : Optional[float]
        Daily demand standard deviation (used when lead_time_std > 0).

    Returns
    -------
    float
        Safety stock in units (>= 0.0).
    """
    if lead_time_demand_std < 0:
        raise ValueError(f"lead_time_demand_std must be non-negative, got {lead_time_demand_std}")

    # Determine z-factor
    if service_level_z is not None:
        z = float(service_level_z)
    elif service_level is not None:
        if not (0.50 < service_level < 1.0):
            raise ValueError(f"service_level must be in (0.50, 1.0), got {service_level}")
        z = float(stats.norm.ppf(service_level))
    else:
        # Default 95% CSL -> z = 1.644853
        z = float(stats.norm.ppf(0.95))

    if z < 0:
        raise ValueError(f"service_level_z must be non-negative for CSL >= 0.50, got {z}")

    # Stochastic lead time formulation if lead_time_std > 0 and demand_mean > 0
    if lead_time_std > 0 and demand_mean > 0 and lead_time is not None and demand_std is not None:
        total_var = float(lead_time) * (float(demand_std) ** 2) + (float(demand_mean) ** 2) * (float(lead_time_std) ** 2)
        total_std = math.sqrt(max(0.0, total_var))
        return float(z * total_std)

    # Standard fixed lead time formulation
    return float(z * float(lead_time_demand_std))


def calculate_reorder_point(
    demand_mean_lead_time: float,
    safety_stock: float,
) -> float:
    """Calculate Reorder Point (ROP).

    ROP = Expected Demand during Lead Time + Safety Stock

    Parameters
    ----------
    demand_mean_lead_time : float
        Expected cumulative demand over the replenishment lead time (>= 0).
    safety_stock : float
        Buffer stock to absorb forecast errors during lead time (>= 0).

    Returns
    -------
    float
        Reorder point in units (>= 0.0).
    """
    if not np.isfinite(demand_mean_lead_time) or demand_mean_lead_time < 0:
        raise ValueError(
            f"demand_mean_lead_time must be non-negative and finite, got {demand_mean_lead_time}"
        )
    if not np.isfinite(safety_stock) or safety_stock < 0:
        raise ValueError(
            f"safety_stock must be non-negative and finite, got {safety_stock}"
        )
    return float(demand_mean_lead_time + safety_stock)


def calculate_inventory_position(
    on_hand: float,
    on_order: float,
    backorders: float = 0.0,
) -> float:
    """Calculate current effective Inventory Position (IP).

    IP = On-Hand + On-Order - Backorders

    Parameters
    ----------
    on_hand : float
        Physical inventory in stock (>= 0).
    on_order : float
        Inventory currently on order / in transit (>= 0).
    backorders : float
        Unfilled customer orders awaiting stock (>= 0, default 0 for lost-sales).

    Returns
    -------
    float
        Inventory position in units.
    """
    if on_hand < 0:
        raise ValueError(f"on_hand must be non-negative, got {on_hand}")
    if on_order < 0:
        raise ValueError(f"on_order must be non-negative, got {on_order}")
    if backorders < 0:
        raise ValueError(f"backorders must be non-negative, got {backorders}")
    return float(on_hand + on_order - backorders)


def calculate_daily_holding_cost_per_unit(
    annual_holding_rate: float,
    unit_cost: float,
    days_per_year: float = 365.0,
) -> float:
    """Calculate the daily holding cost per unit from annual rate and unit cost.

    Standard formula:
        daily_holding_cost = (annual_holding_rate * unit_cost) / days_per_year

    Parameters
    ----------
    annual_holding_rate : float
        Annual holding cost rate as fraction of unit cost (> 0, e.g. 0.20 = 20%/year).
    unit_cost : float
        Unit purchase cost in $/unit (> 0, e.g. 10.0 = $10.00/unit).
    days_per_year : float
        Calendar days in a year (default 365.0).

    Returns
    -------
    float
        Daily holding cost per unit in $/unit/day.

    Examples
    --------
    >>> calculate_daily_holding_cost_per_unit(0.20, 10.0)
    0.005479452054794521  # ~$0.00548/unit/day
    >>> calculate_daily_holding_cost_per_unit(0.25, 10.0)
    0.006849315068493151  # ~$0.00685/unit/day
    """
    if not np.isfinite(annual_holding_rate) or annual_holding_rate <= 0:
        raise ValueError(
            f"annual_holding_rate must be strictly positive and finite, got {annual_holding_rate}"
        )
    if not np.isfinite(unit_cost) or unit_cost <= 0:
        raise ValueError(f"unit_cost must be strictly positive and finite, got {unit_cost}")
    if not np.isfinite(days_per_year) or days_per_year <= 0:
        raise ValueError(f"days_per_year must be strictly positive and finite, got {days_per_year}")

    return float((annual_holding_rate * unit_cost) / days_per_year)


def calculate_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost_unit_year: float,
    min_order_qty: float = 0.0,
    batch_multiple: float = 1.0,
) -> Dict[str, float]:
    """Calculate Economic Order Quantity (EOQ) with operational constraints.

    Balances annual order setup costs and annual inventory holding expenses:
    EOQ = sqrt(2 * D * K / H)

    Parameters
    ----------
    annual_demand : float
        Forecasted annual demand D in units/year (>= 0).
    order_cost : float
        Fixed setup cost per order K in $/order (>= 0).
    holding_cost_unit_year : float
        Annual holding cost per unit H = h * c in $/unit/year (> 0).
    min_order_qty : float
        Minimum order quantity MOQ constraint in units (>= 0).
    batch_multiple : float
        Packaging multiple increment in units (>= 1.0).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - 'eoq_unconstrained': float raw EOQ
        - 'eoq_recommended': float operationally rounded EOQ
        - 'annual_demand': float
        - 'order_cost': float
        - 'holding_cost_unit_year': float
        - 'min_order_qty': float
        - 'batch_multiple': float
    """
    if not np.isfinite(annual_demand) or annual_demand < 0:
        raise ValueError(f"annual_demand must be non-negative and finite, got {annual_demand}")
    if not np.isfinite(order_cost) or order_cost < 0:
        raise ValueError(f"order_cost must be non-negative and finite, got {order_cost}")
    if not np.isfinite(holding_cost_unit_year) or holding_cost_unit_year <= 0:
        raise ValueError(
            f"holding_cost_unit_year must be strictly positive and finite, got {holding_cost_unit_year}"
        )
    if min_order_qty < 0:
        raise ValueError(f"min_order_qty must be non-negative, got {min_order_qty}")
    if batch_multiple < 1.0:
        raise ValueError(f"batch_multiple must be >= 1.0, got {batch_multiple}")

    # Boundary case: zero demand or zero order cost
    if annual_demand == 0.0 or order_cost == 0.0:
        return {
            "eoq_unconstrained": 0.0,
            "eoq_recommended": 0.0,
            "annual_demand": float(annual_demand),
            "order_cost": float(order_cost),
            "holding_cost_unit_year": float(holding_cost_unit_year),
            "min_order_qty": float(min_order_qty),
            "batch_multiple": float(batch_multiple),
        }

    raw_eoq = math.sqrt((2.0 * float(annual_demand) * float(order_cost)) / float(holding_cost_unit_year))

    # Operational adjustments: MOQ and batch multiple
    constrained_q = max(raw_eoq, float(min_order_qty))
    rounded_q = math.ceil(constrained_q / float(batch_multiple)) * float(batch_multiple)

    return {
        "eoq_unconstrained": float(raw_eoq),
        "eoq_recommended": float(rounded_q),
        "annual_demand": float(annual_demand),
        "order_cost": float(order_cost),
        "holding_cost_unit_year": float(holding_cost_unit_year),
        "min_order_qty": float(min_order_qty),
        "batch_multiple": float(batch_multiple),
    }


# ---------------------------------------------------------------------------
# Vectorized / Bulk Inventory Policy Generation
# ---------------------------------------------------------------------------

def calculate_inventory_policy(
    forecast_demand: pd.DataFrame,
    series_uncertainty: pd.DataFrame,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    holding_cost_rate: float = 0.20,
    order_cost: float = 50.0,
    unit_cost: float = 10.0,
    min_order_qty: float = 10.0,
    batch_multiple: float = 5.0,
    demand_col: str = "forecast_rf",
) -> pd.DataFrame:
    """Calculate Reorder Points, Safety Stock, and EOQ across all store-family series.

    Parameters
    ----------
    forecast_demand : pd.DataFrame
        DataFrame with ['store_nbr', 'family', demand_col].
    series_uncertainty : pd.DataFrame
        DataFrame produced by estimate_series_demand_uncertainty indexed by ['store_nbr', 'family'].
    lead_time_days : int
        Replenishment lead time in days (>= 1).
    service_level : float
        Target Cycle Service Level CSL in (0.50, 1.0).
    holding_cost_rate : float
        Annual holding cost rate (fraction of unit cost).
    order_cost : float
        Fixed setup cost per order ($).
    unit_cost : float
        Unit valuation ($/unit).
    min_order_qty : float
        Minimum order quantity MOQ.
    batch_multiple : float
        Batch packaging increment.
    demand_col : str
        Name of forecast demand column.

    Returns
    -------
    pd.DataFrame
        Table with columns:
        ['store_nbr', 'family', 'daily_demand_mean', 'annual_demand',
         'lead_time_days', 'lead_time_demand', 'service_level', 'service_level_z',
         'lead_time_std', 'safety_stock', 'reorder_point',
         'eoq_unconstrained', 'eoq_recommended', 'unit_cost', 'order_cost',
         'holding_cost_annual_rate', 'holding_cost_unit_year', 'quality_flag'].
    """
    if forecast_demand.empty:
        raise ValueError("forecast_demand cannot be empty.")
    if demand_col not in forecast_demand.columns:
        raise ValueError(f"demand_col {demand_col!r} not found in forecast_demand.")

    # Calculate average daily demand per series
    mean_demand = forecast_demand.groupby(["store_nbr", "family"])[demand_col].mean().reset_index()
    mean_demand = mean_demand.rename(columns={demand_col: "daily_demand_mean"})

    # Merge with uncertainty statistics
    unc_df = series_uncertainty.reset_index() if isinstance(series_uncertainty.index, pd.MultiIndex) else series_uncertainty.copy()
    merged = pd.merge(mean_demand, unc_df, on=["store_nbr", "family"], how="left")

    # If some series lacked uncertainty estimates, fallback to family median
    if merged["lead_time_std"].isna().any():
        fam_fallback = merged.groupby("family")["lead_time_std"].transform("median")
        merged["lead_time_std"] = merged["lead_time_std"].fillna(fam_fallback).fillna(0.05 * np.sqrt(lead_time_days))
        merged["quality_flag"] = merged["quality_flag"].fillna("FALLBACK_ESTIMATE")

    z_score = float(stats.norm.ppf(service_level))
    h_unit_year = float(holding_cost_rate * unit_cost)

    policies: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        s_nbr = int(row["store_nbr"])
        fam = str(row["family"])
        d_mean = max(0.0, float(row["daily_demand_mean"]))
        annual_d = d_mean * 365.0
        lt_demand = d_mean * float(lead_time_days)
        lt_std = float(row["lead_time_std"])
        flag = str(row["quality_flag"])

        ss = calculate_safety_stock(lead_time_demand_std=lt_std, service_level_z=z_score)
        rop = calculate_reorder_point(demand_mean_lead_time=lt_demand, safety_stock=ss)

        eoq_dict = calculate_eoq(
            annual_demand=annual_d,
            order_cost=order_cost,
            holding_cost_unit_year=h_unit_year,
            min_order_qty=min_order_qty,
            batch_multiple=batch_multiple,
        )

        policies.append({
            "store_nbr": s_nbr,
            "family": fam,
            "daily_demand_mean": d_mean,
            "annual_demand": annual_d,
            "lead_time_days": lead_time_days,
            "lead_time_demand": lt_demand,
            "service_level": service_level,
            "service_level_z": z_score,
            "lead_time_std": lt_std,
            "safety_stock": ss,
            "reorder_point": rop,
            "eoq_unconstrained": eoq_dict["eoq_unconstrained"],
            "eoq_recommended": eoq_dict["eoq_recommended"],
            "unit_cost": unit_cost,
            "order_cost": order_cost,
            "holding_cost_annual_rate": holding_cost_rate,
            "holding_cost_unit_year": h_unit_year,
            "quality_flag": flag,
        })

    return pd.DataFrame(policies)
