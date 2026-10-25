# 04 - Module Specifications

## 1. Modular Hierarchy

The system follows a strict modular structure. Each module is encapsulated with typed interfaces (Pydantic / Dataclasses) and decoupled from neighboring stages.

```
src/
├── data/                  # Ingestion, validation, and time-grid expansion
│   ├── ingestion.py
│   ├── validation.py
│   └── grid_builder.py
├── features/              # Feature transformers and leakage-free aggregators
│   ├── lag_features.py
│   ├── rolling_features.py
│   ├── calendar_features.py
│   └── pipeline.py
├── models/                # Machine learning forecasters and baselines
│   ├── base.py
│   ├── baselines.py
│   ├── tree_models.py
│   └── evaluator.py
├── inventory/             # Inventory policies and optimization algorithms
│   ├── safety_stock.py
│   ├── reorder_point.py
│   ├── eoq.py
│   └── policy_simulator.py
└── api/                   # Interface layer (CLI and REST endpoints)
    ├── schemas.py
    └── server.py
```

---

## 2. Module Functional Breakdown

### 2.1 Ingestion & Validation Module (`src/data/`)
* **`validation.py`**:
  * Implements schema validation (types, boundaries, missingness).
  * Validates non-negative quantities for sales and valid date ranges.
* **`grid_builder.py`**:
  * Constructs the canonical Store $\times$ SKU $\times$ Date tensor.
  * Solves intermittent zero-sales demand bias via explicit zero-filling.

### 2.2 Feature Engineering Module (`src/retail_forecasting/features.py`)
* **`generate_lag_features()`**:
  * Creates deterministic past-observation shifts: $y_{t-1}, y_{t-7}, y_{t-14}, y_{t-28}$ using exact calendar-day lookups.
  * Handles time-series gaps without imputing fake zero sales or displacing observations across time boundaries.
* **`generate_rolling_features()`**:
  * Computes rolling statistics (mean, sample standard deviation) over historical calendar windows of $W \in \{7, 14, 30\}$ days.
  * Ensures zero lookahead leakage via `closed='left'`, strictly excluding the target day observation ($t$).
  * Enforces minimum observations ($N \ge 2$) for sample standard deviation.
* **`generate_calendar_features()`**:
  * Extracts deterministic calendar signals: `day_of_week`, `day_of_month`, `month`, `quarter`, `year`, `is_weekend`.
  * Encodes continuous cyclical trigonometry: $\sin(2\pi \cdot t / T)$ and $\cos(2\pi \cdot t / T)$ for day of week and month.
* **`generate_holiday_features()`**:
  * Integrates Ecuador holiday registry (`holidays_events.csv`) with store locations (`stores.csv`).
  * Applies transferred holiday corrections (`transferred == True` $\to$ regular day; `Transfer` $\to$ holiday) and geographic scoping (`National`, `Regional`, `Local`).
* **`build_feature_pipeline()`**:
  * Orchestrates feature transformations, guarantees non-mutation of input data, enforces deterministic key sorting `['date', 'store_nbr', 'family']`, and produces comprehensive audit reporting.

### 2.3 Forecasting Engine & Metrics Modules (`src/retail_forecasting/`)
* **`forecasting.py`**:
  * `temporal_train_val_test_split()`: Partitions time-series chronologically into global non-overlapping train, validation, and holdout splits.
  * `SeasonalNaiveForecaster`: Baseline benchmark supporting one-step-ahead exact lag and fixed-origin weekly cycle repeating without lookahead leakage.
  * `RandomForestDemandForecaster`: Tabular demand forecaster utilizing 24 leakage-safe features, training-only imputer/encoder, and non-negative prediction clamping.
  * `predict_one_step()`: Mode A evaluation using historical feature states.
  * `predict_fixed_origin_recursive()`: Mode B autoregressive multi-day forecasting recursively feeding predictions into future lags.
* **`metrics.py`**:
  * Implements `calculate_mae()`, `calculate_rmse()`, `calculate_wape()`, and `calculate_bias()`.
  * `evaluate_forecast_metrics()`: Volume-weighted multi-series metric aggregation across product families, stores, and forecast horizons.

### 2.4 Inventory Optimization & Simulation Engine (`src/retail_forecasting/`)
* **`inventory.py`**:
  * `InventoryPlanningUnit`: Validated frozen dataclass enforcing non-negative demand, strictly positive costs, positive lead times, and valid service level bounds $(0.50, 1.0)$.
  * `estimate_series_demand_uncertainty()`: Calibrates series-level demand uncertainty from out-of-sample forecast residuals, applies Empirical Bayes shrinkage ($k = 7$) toward family mean variance, enforces minimum standard deviation floor ($\sigma_{\min} = 0.05$), and accounts for multi-day error autocorrelation ($\alpha(L)$).
  * `calculate_safety_stock()`: Computes dynamic safety stock:
    $$SS = z \cdot \sigma_L$$
    supporting both fixed lead time and stochastic lead time formulations ($SS = z \sqrt{L \sigma_D^2 + D^2 \sigma_L^2}$).
  * `calculate_reorder_point()`: Computes Reorder Point:
    $$ROP = \hat{D}_L + SS = L \cdot \bar{D} + SS$$
  * `calculate_inventory_position()`: Computes effective stock position:
    $$IP = \text{On-Hand} + \text{On-Order} - \text{Backorders}$$
  * `calculate_eoq()`: Solves classical Economic Order Quantity balancing annual ordering and holding costs:
    $$EOQ = \sqrt{\frac{2 \cdot D_{\text{annual}} \cdot K}{H}}$$
    and applies Minimum Order Quantity (MOQ) and batch multiple rounding.
  * `calculate_inventory_policy()`: Bulk vectorized computation of $SS$, $ROP$, and $EOQ$ for all planning units with full provenance.
* **`simulation.py`**:
  * `InventoryPolicy`: Typed dataclass configuring replenishment rules (`rop_eoq`, `order_up_to`, review period, MOQ, batch multiples).
  * `InventorySimulator`: Discrete-event daily simulation engine modeling order receipts, customer demand fulfillment, lost sales or backorders, review periods, and cost accounting.
  * Enforces the fundamental daily and cumulative inventory conservation invariant:
    $$I_t^{\text{end}} - I_{t-1}^{\text{end}} \equiv \text{Receipts}_t - \text{Sales}_t$$
    $$I_T^{\text{end}} - I_0 \equiv \sum_{t=0}^{T-1} \text{Receipts}_t - \sum_{t=0}^{T-1} \text{Sales}_t$$
  * `compare_inventory_policies()`: Runs multiple replenishment policies against identical demand series and compares fill rate, stockout days, and operating costs.
* **`recommendations.py`**:
  * `generate_purchase_orders()`: Evaluates inventory positions against ROPs, triggers purchase orders, applies batch rounding, estimates spend, and assigns urgency priorities (`CRITICAL`, `HIGH`, `NORMAL`, `NONE`).

