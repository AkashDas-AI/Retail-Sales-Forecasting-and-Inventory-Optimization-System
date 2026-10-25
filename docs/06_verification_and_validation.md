# 06 - Verification & Validation Framework

## 1. Overview & Validation Objectives

To ensure system correctness and business reliability, validation is divided into:
1. **Algorithmic Validation**: Proves that data transformations are mathematically sound, leakage-free, and reproducible.
2. **Predictive Validation**: Proves that the ML forecasting engine outperforms statistical baselines and produces unbiased predictions.
3. **Prescriptive (Inventory) Validation**: Proves through historical back-simulation that the recommended replenishment policies reduce inventory carrying costs while meeting target customer service levels.

---

## 2. Evaluation Metrics

### 2.1 Forecast Accuracy Metrics

| Metric | Formula | Purpose / Business Significance |
| :--- | :--- | :--- |
| **WAPE** (Weighted Absolute Percentage Error) | $$\frac{\sum_{i} \vert y_i - \hat{y}_i \vert}{\sum_{i} y_i}$$ | Primary retail metric. Avoids division-by-zero on zero-sales days and weights high-volume SKUs appropriately. |
| **MAE** (Mean Absolute Error) | $$\frac{1}{N} \sum_{i} \vert y_i - \hat{y}_i \vert$$ | Measures mean error magnitude in units. |
| **RMSE** (Root Mean Squared Error) | $$\sqrt{\frac{1}{N} \sum_{i} (y_i - \hat{y}_i)^2}$$ | Penalizes large errors heavily; sensitive to extreme peak demand. |
| **Forecast Bias** | $$\frac{\sum_{i} (\hat{y}_i - y_i)}{\sum_{i} y_i}$$ | Detects systematic over-forecasting (excess holding cost) or under-forecasting (stockout risk). |
| **Quantile Coverage** | $$\frac{1}{N} \sum_{i} \mathbb{I}(y_i \le \hat{y}^{(\tau)}_i)$$ | Verifies calibration of probabilistic quantiles (e.g. P90 forecast should cover $\approx 90\%$ of observations). |

### 2.2 Inventory Optimization & Financial Metrics

| Metric | Formula | Target Benchmark |
| :--- | :--- | :--- |
| **Cycle Service Level (CSL)** | $$\frac{\text{Cycles without Stockout}}{\text{Total Replenishment Cycles}}$$ | Target $\ge 95\%$ |
| **Item Fill Rate (IFR)** | $$\frac{\sum \text{Fulfilled Demand}}{\sum \text{Total Demand}}$$ | Target $\ge 98\%$ |
| **Stockout Days Ratio** | $$\frac{\text{Days with } (I_t = 0 \land y_t > 0)}{\text{Total Active Days}}$$ | Minimize ($< 2\%$) |
| **Total Inventory Cost** | $$\text{Holding Cost} + \text{Order Placement Cost} + \text{Stockout Penalty}$$ | Minimize vs baseline heuristic |

---

## 3. Automated Validation Checkpoints

Every step of implementation must pass the following deterministic verification protocols:

### Checkpoint A: Data & Grid Integrity
- [ ] No negative demand records after cleansing.
- [ ] Complete Cartesian temporal grid (no missing date gaps for active products).
- [ ] Target variable (`units_sold`) has 0 NaN values.

### Checkpoint B: Temporal Leakage Audit
- [ ] Feature timestamps are strictly $< (t + \text{horizon})$.
- [ ] Lag offset verification: minimum lag index $k \ge H$ for multi-step direct forecasting.
- [ ] Split separation: $\max(\text{Train Date}) < \min(\text{Val Date}) < \min(\text{Test Date})$.

### Checkpoint C: Forecasting Model Quality Gate
- [ ] Model training exhibits steady loss convergence across iterations.
- [ ] Machine Learning model achieves lower WAPE than the 7-day Seasonal Naive baseline:
  $$\text{WAPE}_{\text{ML}} < \text{WAPE}_{\text{Baseline}}$$
- [ ] Forecast Bias falls within safe business boundary: $-5\% \le \text{Bias} \le +5\%$.

### Checkpoint D: Inventory Policy Sanity Check
- [ ] Safety stock is strictly non-negative: $SS \ge 0$.
- [ ] Reorder Point satisfies: $ROP \ge SS$.
- [ ] Monte Carlo back-simulation achieves target service level without inventory runaway.

