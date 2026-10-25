# Methodology and Scientific Approach

## 1. Machine Learning Forecasting Methodology

* Formulation: Supervised panel regression across 1,782 cross-sectional series defined by `(store_nbr, family)`.
* Feature Engineering: 24 predictive features including shifted demand lags (1, 7, 14, 28 days), trailing rolling window averages and standard deviations (7, 14, 30 days), cyclical sine/cosine calendar encodings, and store-specific holiday transfer mappings.
* Leakage Prevention: All lag features and rolling statistics are shifted into the past ($t-1$ or earlier). Categorical encoders and transformers are fitted strictly on the training partition (`2016-01-01` to `2017-06-20`).
* Chronological Splitting: Strict calendar partitioning:
  * Training: 2016-01-01 to 2017-06-20 (537 days)
  * Validation: 2017-06-21 to 2017-07-18 (28 days, 49,896 observations)
  * Holdout: 2017-07-19 to 2017-08-15 (28 days, 49,896 observations)
* Evaluation Modes:
  * Mode A (Rolling One-Step Walk-Forward): Daily evaluation with observed history updated through day $t-1$.
  * Mode B (Fixed-Origin Recursive Multi-Step): 14-day multi-step horizon (2017-07-19 to 2017-08-01) with predictions recursively feeding forward into future lag features without ground-truth access.
* Primary Metric: Weighted Absolute Percentage Error (WAPE), eliminating division-by-zero anomalies inherent in MAPE for retail series with intermittent or zero sales.

## 2. Statistical Inventory Optimization

* Uncertainty Calibration: Sample forecast residual standard deviation $s_i$ measured on validation errors ($y_t - \hat{y}_t$).
* Empirical Bayes Shrinkage: Shrinks series variance toward global pooled category variance with prior weight $k = 7$ and a minimum standard deviation floor of $\sigma_{\text{floor}} = 0.05$ units.
* Autoregressive Error Inflation: Multi-day lead-time demand variability incorporating AR(1) error correlation:
  $$\sigma_{L, i} = \sqrt{L \cdot \sigma_{\text{shrunk}, i}^2 \cdot \left[1 + \frac{2\rho}{1-\rho}\left(1 - \frac{1-\rho^L}{L(1-\rho)}\right)\right]}$$
* Dynamic Safety Stock: $SS_i = z \cdot \sigma_{L, i}$ with $z = 1.645$ for $CSL = 0.95$.
* Reorder Point: $ROP_i = (\bar{d}_i \cdot L) + SS_i$.
* Economic Order Quantity: $EOQ_i = \sqrt{\frac{2 K \bar{D}_i}{h \cdot c}}$, constrained by MOQ = 10 and batch multiples of 5 units.
* Inventory Position Trigger: Order placed when $IP_t = \text{On-Hand}_t + \text{On-Order}_t - \text{Backorders}_t \le ROP_i$.

## 3. Discrete-Event Simulation Mechanics

* Backtesting Horizon: 28-day out-of-sample holdout backtest across a representative benchmark cohort of 50 store-category series.
* Daily Execution Sequence:
  1. Arriving deliveries added to on-hand physical stock.
  2. Customer demand satisfied; unfulfilled demand recorded as lost sales.
  3. Daily carrying costs charged ($h \cdot c / 365 = $0.00547945/unit/day).
  4. Shortage penalties charged ($\pi = $15.00/unit).
  5. Inventory Position evaluated against ROP; replenishment orders dispatched if triggered.
* Conservation Invariant: Strict accounting conservation verified at every step:
  $$\text{Ending Stock}_t = \text{Starting Stock}_t + \text{Receipts}_t - \text{Fulfilled Sales}_t$$
