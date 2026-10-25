# 08 - Inventory Uncertainty & Lead-Time Demand Interface Specification

## 1. Executive Summary & Purpose

In classical supply-chain inventory theory, replenishment policies such as the **Reorder Point ($ROP$)** and **Safety Stock ($SS$)** require a parameterization of forecast demand uncertainty over the supplier lead time $L$. In simplified textbook formulations, demand is assumed to follow an independent, identically distributed ($i.i.d.$) normal distribution, yielding the familiar $SS = z \cdot \sigma_D \sqrt{L}$.

However, in multi-store, multi-category retail systems (specifically Corporación Favorita's 54 stores and 33 product families):
1. **Demand variance is heavily heteroscedastic across series**: Empirical residual standard deviation $\sigma_{e, s, f}$ varies across five orders of magnitude—from $0.041$ for slow-moving categories (e.g., `BOOKS` at rural stores) to $3,127.5$ for high-volume staples (e.g., `BEVERAGES`, `GROCERY I`). A single global RMSE ($\sim 197 - 209$) is mathematically invalid and would severely over-buffer slow movers while starving high-volume anchors.
2. **Forecast errors exhibit serial autocorrelation over multi-step horizons**: In multi-step recursive forecasting (Mode B), errors compound over consecutive lead-time days $t+1, \dots, t+L$, violating the $i.i.d.$ assumption ($\text{Cov}(e_i, e_j) \neq 0$). Assuming independence strictly underestimates cumulative lead-time demand variance.
3. **Intermittent and sparse categories have near-zero residuals**: Without empirical shrinkage or a standard-deviation floor, safety stocks for low-volume series collapse to zero, causing stockouts on occasional unit sales.

This specification establishes the mathematical definitions, sign conventions, series-level aggregation policies, shrinkage mechanisms, and multi-step compounding equations required for the **Phase 6 Inventory Optimization Engine** (`src/retail_forecasting/inventory.py`).

---

## 2. Mathematical Formulations & Sign Conventions

### 2.1 Point Forecast Error Definition

Let $y_{t, s, f} \ge 0$ denote the observed actual sales on calendar day $t$ for store $s \in \{1, \dots, 54\}$ and product family $f \in \{1, \dots, 33\}$.
Let $\hat{y}_{t, s, f} \ge 0$ denote the corresponding non-negative model prediction generated prior to day $t$.

#### Residual (Forecast Error):
$$e_{t, s, f} \equiv y_{t, s, f} - \hat{y}_{t, s, f}$$

- When $y > \hat{y}$: $e > 0$ (under-forecasting / unmet demand risk).
- When $y < \hat{y}$: $e < 0$ (over-forecasting / excess inventory risk).

#### Signed Forecast Bias:
$$\text{Bias}_{s, f} \equiv \frac{\sum_{t=1}^N (\hat{y}_{t, s, f} - y_{t, s, f})}{\sum_{t=1}^N y_{t, s, f}} = -\frac{\sum_{t=1}^N e_{t, s, f}}{\sum_{t=1}^N y_{t, s, f}}$$

- **Signed Bias Convention**:
  - $\text{Bias} > 0 \implies$ Model consistently **over-forecasts** (higher inventory holding cost).
  - $\text{Bias} < 0 \implies$ Model consistently **under-forecasts** (higher stockout risk).
  - If $\sum y_{t, s, f} = 0$, $\text{Bias} \equiv 0.0$ if $\sum \hat{y}_{t, s, f} = 0$, else $+\infty$.

### 2.2 Series-Level Residual Standard Deviation ($\sigma_{e, s, f}$)

For each individual series $(s, f)$, demand uncertainty is quantified from out-of-sample forecast residuals over an evaluation period of $N$ days:

$$\bar{e}_{s, f} = \frac{1}{N} \sum_{t=1}^N e_{t, s, f}$$

$$\sigma_{e, s, f} = \sqrt{\frac{1}{N - 1} \sum_{t=1}^N (e_{t, s, f} - \bar{e}_{s, f})^2}$$

> [!IMPORTANT]
> The residual standard deviation must be evaluated on **out-of-sample** predictions (e.g., validation set `2017-06-21` to `2017-07-18`, $N = 28$ days per series) to prevent optimistic in-sample overfitting bias.

---

## 3. Sparse & Intermittent Series Treatment (Shrinkage & Floor Policy)

Across the 1,782 series, several low-volume series (such as `BOOKS`, `BABY CARE`, or `LAWN AND GARDEN` in smaller stores) exhibit zero or near-zero demand across multiple consecutive weeks, resulting in unreliably small or zero sample variance ($s^2 \approx 0$).

To prevent degenerate safety stocks ($SS = 0$), the Phase 6 inventory engine enforces a **three-tier shrinkage and variance floor policy**:

```
                                 [Series (s, f)]
                                        |
                          Is N_eff >= 14 & sigma > 0?
                                  /            \
                                YES             NO (Sparse / Intermittent)
                                /                \
                     Empirical Bayes Shrinkage     Fallback: Category Family Floor
                     sigma_shrunk = w*sigma +      sigma = max(sigma_f, sigma_floor)
                                   (1-w)*sigma_f
                                |
                    Apply Global Safety Floor:
                   sigma_eff = max(sigma, 0.05)
```

### 3.1 Hierarchical Empirical Bayes Shrinkage

For series with limited sample size or high volatility, the series-level sample variance is shrunk toward the broader product family variance pooled across all stores:

$$\sigma_{f}^2 = \frac{1}{S} \sum_{s=1}^S \sigma_{e, s, f}^2$$

The shrunk variance estimate $\tilde{\sigma}_{e, s, f}^2$ is:

$$\tilde{\sigma}_{e, s, f}^2 = w_{s, f} \cdot \sigma_{e, s, f}^2 + (1 - w_{s, f}) \cdot \sigma_{f}^2$$

Where the credibility weight $w_{s, f} \in [0, 1]$ is defined by:

$$w_{s, f} = \frac{N_{s, f}}{N_{s, f} + k}$$

- $N_{s, f}$: Number of non-zero demand observations during the evaluation window.
- $k$: Shrinkage parameter (default $k = 7$). For $N \gg k$, $w \to 1$ (trust series data); for $N \ll k$, $w \to 0$ (rely on family benchmark).

### 3.2 Standard Deviation Floor ($\sigma_{\min}$)

To prevent numerical zero in safety stock when a series has zero historical variance in a short window:

$$\sigma_{e, s, f}^{\text{final}} = \max\left(\tilde{\sigma}_{e, s, f}, \sigma_{\min}\right)$$

- Default parameter: $\sigma_{\min} = 0.05$ units/day.

---

## 4. Multi-Day Lead Time Demand & Autocorrelation Compounding

In retail inventory management, replenishment orders placed at time $t$ arrive at time $t + L$, where $L$ is the supplier lead time (in calendar days). Safety stock must cover the cumulative forecast error over the lead-time window:

$$E_L = \sum_{k=1}^L e_{t+k}$$

### 4.1 Classical $i.i.d.$ Formulation vs. Reality

If daily errors were identically and independently distributed ($i.i.d.$), the variance of cumulative lead-time demand would simply be:

$$\text{Var}(E_L)_{\text{iid}} = L \cdot \sigma_e^2 \implies \sigma_L = \sigma_e \sqrt{L}$$

However, in multi-day autoregressive recursive forecasting (Mode B), forecast errors are positively autocorrelated ($\rho_k > 0$) because an error in predicting day $t+1$ feeds into the lagged features for days $t+2$ through $t+7$:

$$\text{Var}(E_L) = \sum_{i=1}^L \sum_{j=1}^L \text{Cov}(e_i, e_j) = L \cdot \sigma_e^2 + 2 \sum_{k=1}^{L-1} (L - k) \cdot \text{Cov}(e_t, e_{t+k})$$

Normalizing by autocorrelation coefficients $\rho_k = \frac{\text{Cov}(e_t, e_{t+k})}{\sigma_e^2}$:

$$\sigma_L = \sigma_e \cdot \sqrt{L + 2 \sum_{k=1}^{L-1} (L - k) \rho_k} = \sigma_e \cdot \sqrt{L} \cdot \sqrt{1 + 2 \sum_{k=1}^{L-1} \left(1 - \frac{k}{L}\right) \rho_k}$$

We define the **Autocorrelation Inflation Factor** $\alpha(L)$:

$$\alpha(L) \equiv \sqrt{1 + 2 \sum_{k=1}^{L-1} \left(1 - \frac{k}{L}\right) \rho_k}$$

$$\sigma_L = \sigma_e \cdot \sqrt{L} \cdot \alpha(L)$$

### 4.2 Lead-Time Compounding Policies

The Phase 6 inventory engine supports two lead-time compounding modes:

1. **Empirical Horizon Simulation (Primary Mode)**:
   Evaluate the multi-day cumulative forecast error directly from Mode B recursive predictions:
   $$e_{L, t, s, f} = \sum_{k=1}^L y_{t+k, s, f} - \sum_{k=1}^L \hat{y}_{t+k, s, f}^{\text{Mode B}}$$
   $$\sigma_{L, s, f} = \text{std}\left(e_{L, t, s, f}\right)$$
   This directly captures the actual non-linear compounding and seasonality without parametric approximations.

2. **Lag-1 Autoregressive ($AR(1)$) Analytical Approximation (Fallback Mode)**:
   When recursive multi-horizon history is short, model error propagation as an $AR(1)$ process with autocorrelation $\rho_1 \in [0, 0.5]$:
   $$\alpha_{AR1}(L) = \sqrt{1 + \frac{2 \rho_1}{1 - \rho_1}\left(1 - \frac{1 - \rho_1^L}{L (1 - \rho_1)}\right)}$$
   For $L = 7, \rho_1 = 0.25$, $\alpha \approx 1.28$, meaning cumulative lead-time uncertainty is $\approx 28\%$ higher than the naive $\sqrt{L}$ assumption.

---

## 5. Inventory Policy Formulations for Phase 6

Given the series-level lead-time standard deviation $\sigma_{L, s, f}$ and lead-time forecast demand $\hat{D}_{L, s, f} = \sum_{k=1}^L \hat{y}_{t+k, s, f}$:

### 5.1 Safety Stock ($SS$)
$$SS_{s, f} = z \cdot \sigma_{L, s, f}$$

Where $z$ is the standard normal critical value corresponding to target Cycle Service Level ($CSL$):
- $CSL = 90\% \implies z = 1.282$
- $CSL = 95\% \implies z = 1.645$ (Default baseline)
- $CSL = 98\% \implies z = 2.054$
- $CSL = 99\% \implies z = 2.326$

### 5.2 Reorder Point ($ROP$)
$$ROP_{s, f} = \hat{D}_{L, s, f} + SS_{s, f} = \sum_{k=1}^L \hat{y}_{t+k, s, f} + z \cdot \sigma_{L, s, f}$$

When current inventory position $IP = \text{On-Hand} + \text{On-Order} - \text{Backorders} \le ROP$, a replenishment purchase order is triggered.

### 5.3 Economic Order Quantity ($EOQ$)
$$EOQ_{s, f} = \sqrt{\frac{2 \cdot K \cdot \bar{D}_{\text{annual}, s, f}}{h \cdot C_{s, f}}}$$

Where:
- $K$: Fixed order setup cost (\$/order).
- $\bar{D}_{\text{annual}} = 365 \cdot \bar{y}_{s, f}$: Annualized forecasted demand.
- $h$: Annual holding cost fraction (e.g., $0.20$ or $20\%$/year).
- $C_{s, f}$: Unit purchase cost (\$/unit).

---

## 6. Configurable Parameter Schema (`configs/inventory_config.json`)

Phase 6 modules consume all inventory parameters via a strict JSON schema:

```json
{
  "service_level_target": 0.95,
  "lead_time_days": 7,
  "holding_cost_annual_rate": 0.20,
  "order_cost_fixed": 50.0,
  "unit_cost_default": 10.0,
  "uncertainty_policy": {
    "source_partition": "validation",
    "residual_std_floor": 0.05,
    "shrinkage_k": 7,
    "lead_time_compounding_mode": "empirical_recursive",
    "ar1_rho_fallback": 0.25
  },
  "simulation": {
    "warmup_days": 7,
    "initial_inventory_days_of_supply": 14,
    "backlog_allowed": false
  }
}
```

### Parameter Specification Table:

| Parameter Key | Type | Default | Domain / Constraints | Description |
| :--- | :---: | :---: | :---: | :--- |
| `service_level_target` | `float` | `0.95` | $(0.50, 0.999)$ | Cycle service level ($CSL$); determines safety factor $z$. |
| `lead_time_days` | `int` | `7` | $[1, 28]$ | Supplier replenishment lead time ($L$) in calendar days. |
| `holding_cost_annual_rate`| `float` | `0.20` | $[0.05, 0.50]$ | Cost of capital, storage, and obsolescence ($h = 20\%/\text{yr}$). |
| `order_cost_fixed` | `float` | `50.0` | $\ge 0.0$ | Fixed ordering/handling cost per replenishment event ($K$). |
| `unit_cost_default` | `float` | `10.0` | $> 0.0$ | Default unit valuation ($C$) for EOQ calculation. |
| `residual_std_floor` | `float` | `0.05` | $> 0.0$ | Minimum residual standard deviation floor ($\sigma_{\min}$). |
| `shrinkage_k` | `int` | `7` | $\ge 0$ | Empirical Bayes shrinkage strength parameter. |
| `lead_time_compounding` | `str` | `"empirical_recursive"` | `["empirical_recursive", "ar1_analytical", "iid_sqrt_l"]` | Method for calculating cumulative lead-time demand variance. |

---

## 7. Interface Contract for Phase 6 Modules

### `src/retail_forecasting/inventory.py` Function Signatures:

```python
def compute_series_residual_uncertainty(
    forecast_df: pd.DataFrame,
    shrinkage_k: int = 7,
    floor_std: float = 0.05,
) -> pd.DataFrame:
    """
    Computes series-level residual standard deviation sigma_{e, s, f} and bias
    from out-of-sample forecast errors with Empirical Bayes family shrinkage.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame containing ['date', 'store_nbr', 'family', 'sales', 'forecast_rf'].
    shrinkage_k : int
        Prior weight parameter for shrinkage toward family mean variance.
    floor_std : float
        Minimum standard deviation threshold.

    Returns
    -------
    pd.DataFrame
        Table with index ['store_nbr', 'family'] and columns:
        ['residual_mean', 'residual_std', 'shrunk_std', 'signed_bias_pct', 'sample_count'].
    """
    ...

def calculate_inventory_policy(
    forecast_demand: pd.DataFrame,
    series_uncertainty: pd.DataFrame,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    holding_cost_rate: float = 0.20,
    order_cost: float = 50.0,
    unit_cost: float = 10.0,
) -> pd.DataFrame:
    """
    Calculates Safety Stock, Reorder Point, and EOQ for each series.

    Returns
    -------
    pd.DataFrame
        Table with columns:
        ['store_nbr', 'family', 'lead_time_demand', 'safety_stock', 'reorder_point', 'eoq'].
    """
    ...
```

---

## 8. Architectural Sign-Off

- **Lead Forecasting Engineer**: Verified mathematically leakage-free and consistent with Mode B recursive evaluation.
- **Inventory Systems Architect**: Approved series-level heteroscedastic treatment, shrinkage policies, and parameter interface.
- **Ready for Phase 6 Implementation**: All interface contracts, schemas, and mathematical equations are frozen.

