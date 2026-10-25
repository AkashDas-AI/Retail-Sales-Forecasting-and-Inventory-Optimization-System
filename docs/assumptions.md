# Modeling & Supply Chain Assumptions: Corporación Favorita

## 1. Data Source Attribution & Grain

* **Primary Dataset**: Corporación Favorita Store Sales time-series forecasting dataset from Kaggle:
  `https://www.kaggle.com/competitions/store-sales-time-series-forecasting`
* **Modeling Grain**: The primary observational grain is strictly `(date, store_nbr, family)`.
* **Category vs. SKU Distinction**: In this dataset, `family` represents an aggregate product category (e.g. `GROCERY I`, `BEVERAGES`, `CLEANING`, `PRODUCE`, `MEATS`, `AUTOMOTIVE`), **not an individual product SKU**. Forecasts and downstream inventory parameters apply to category-level volume aggregates.
* **Required vs. Optional Source Files**:
  * **Required**: `data/raw/train.csv` (contains `id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion`).
  * **Optional**: `stores.csv`, `transactions.csv`, `holidays_events.csv`, `oil.csv` (used for exogenous enrichment if present, but not required to execute the primary sales loader).
* **Real-Data Validation Status**: Real-data pipeline validation is **completed and active**. The official Corporación Favorita dataset (`train.csv`, 3,000,888 rows) has been ingested, validated against schema constraints, and processed to `data/processed/sales_daily.csv` with zero row drop or corruption. Unit test fixtures remain in `tests/` for automated regression testing.

---

## 2. Promotion-Value Handling & Imputation Policy

* **Default Missing Value Preservation**:
  * Missing (`NaN` / `null`) values in `onpromotion` are **strictly preserved** as unobserved (`<NA>`) using pandas nullable integer (`Int64`) representation.
  * They are **not** silently converted or coerced to `0` during ingestion or safe preprocessing.
  * Any subsequent imputation (e.g., assuming unrecorded promotion implies no promotion) is an explicit downstream modeling decision, not a silent ingestion default.
  * Configurable via `configs/project_config.json` (`"preserve_missing_promotions": true`) or CLI flag `--impute-promotions-zero`.
* **Malformed Promotion Values**:
  * Non-numeric strings in `onpromotion` (e.g., text entries) are detected and recorded in `invalid_value_counts["malformed_onpromotion"]`.
  * Malformed or negative promotion values are treated as **fatal validation errors** and reject pipeline execution.

---

## 3. Fatal Validation Errors & Output Preservation

### 3.1 Taxonomy of Fatal Validation Failures
The following validation failures are classified as **fatal** and trigger immediate pipeline termination with a non-zero exit code:
1. **Missing Required Columns**: Absences of `id`, `date`, `store_nbr`, `family`, `sales`, or `onpromotion`.
2. **Invalid Dates**: Non-parseable or `NaT` date entries.
3. **Missing or Non-Positive Store Identifier**: Missing `store_nbr` or non-numeric/$\le 0$ identifiers.
4. **Missing or Blank Family**: Missing `family` or empty whitespace strings (`"   "`).
5. **Non-Numeric or Infinite Sales**: String sales values or `NaN`/$\pm\infty$ floats.
6. **Negative Sales**: Any `sales < 0.0`.
7. **Malformed / Negative Promotions**: Text values or values $< 0$ in `onpromotion`.
8. **Duplicate Modeling Keys**: Multiple rows sharing the exact same `(date, store_nbr, family)` tuple.

### 3.2 Non-Fatal Warnings
* Missing `onpromotion` values (recorded as advisory warning in data quality report; missing values preserved as `<NA>`).

### 3.3 Output File Preservation Guarantee
* The pipeline enforces a strict **fail-safe execution sequence**: all validation checks are completed and verified **before** any output files (`sales_daily.csv`) or quality reports (`data_quality_report.json`) are written or updated.
* If validation encounters any fatal error:
  * Processing immediately aborts with non-zero exit code (`1` or `2`).
  * Any pre-existing processed dataset or prior quality report remains **100% untouched and unmodified**.
  * No partial, corrupted, or misleading output files are published.

---

## 4. Missing-Date and Zero-Sales Policy (Mandatory Safeguards)

A missing row in `train.csv` cannot automatically be interpreted as zero customer demand, a stockout, or a store closure. Specifically:
1. **No Automatic Zero-Filling**: The pipeline does not impute `0.0` sales for unobserved dates.
2. **No Premature Cartesian Expansion**: A complete cross-product `(Dates × Stores × Families)` is intentionally avoided during initial ingestion.
3. **Store Life Cycle**: Stores opened at various points between 2013 and 2017. Unobserved dates prior to a store's opening represent non-existence, not zero sales.
4. **Assortment Eligibility**: Not all store formats carry every product family (e.g. hardware/automotive are absent from smaller format stores). Forcing absent families into the series with zero sales distorts demand variance ($\sigma_D$) and safety stock calculations.
5. **Censored Demand vs. True Zero Demand**: An observation of zero sales during store trading hours may reflect complete inventory depletion (censored demand) rather than zero consumer interest.
6. **Transactions Data as Supporting Signal**: When `transactions.csv` is present, zero transactions indicates the store was closed; it is treated as a supporting operational indicator, not definitive proof of cessation of demand.

---

## 5. Simulated Supply Chain Parameters

Because real-world commercial retailers treat inventory on hand, supplier replenishment lead times, and internal accounting costs as proprietary business secrets, the following parameters are modeled as configurable baseline assumptions:
* **Supplier Lead Time ($L$)**: Modeled as 7 days baseline (configurable in `configs/inventory_config.json`).
* **Annual Holding Cost Rate ($h$)**: Modeled as 20% of product value per annum.
* **Order Placement Cost ($K$)**: Modeled as \$50.00 administrative cost per order.
* **Target Service Level ($CSL$)**: Modeled at 95% cycle service level ($z = 1.645$).
* **Stockout Policy**: Unfulfilled demand is assumed lost (standard grocery retail lost sales model, no backorders).

---

## 6. Dataset Placement & Execution Guide

To run the data preparation pipeline on real data:

1. Download `train.csv` (and optionally `stores.csv`, `transactions.csv`, `holidays_events.csv`, `oil.csv`) from Kaggle.
2. Place the file(s) in the project directory:
   ```
   data/raw/train.csv
   ```
3. Run the data preparation script using the local virtual environment:
   ```powershell
   .\.venv\Scripts\python.exe scripts/prepare_data.py
   ```
4. Output will be generated at:
   - Cleaned Dataset: `data/processed/sales_daily.csv`
   - Data Quality Audit: `reports/metrics/data_quality_report.json`

---

## 7. Feature Engineering Policy & Leakage Prevention Safeguards

### 7.1 Calendar-Day Aligned Lags vs. Shifted Row Indexing
* Feature lags are computed strictly using **exact calendar date shifts** ($t - k$ calendar days), rather than naive row-index shifts (`shift(k)`).
* Why: Store sales time-series contain calendar discontinuities (e.g., annual store closures on New Year's Day Jan 1, earthquakes in April 2016, or newer store openings). Naive row shifting aligns $t-1$ row to $t-2$ or earlier calendar days, corrupting seasonal periodicity (e.g. day-of-week alignment).
* Missing Historical Dates: If a store/family was not observed exactly $k$ days prior, `sales_lag_kd` evaluates strictly to `NaN`. The pipeline does **not** substitute 0 or grab a non-matching date.

### 7.2 Calendar-Window Rolling Statistics
* Rolling aggregations (mean, standard deviation) for windows $W \in \{7, 14, 30\}$ days are computed on historical calendar windows $[t - W\text{ days}, t - 1\text{ day}]$ strictly excluding the target day observation ($t$) via `closed='left'`.
* Degrees of Freedom: Sample standard deviation ($N - 1$ denominator) evaluates to `NaN` when fewer than 2 valid historical observations fall within the window.

### 7.3 Holiday and Business Event Logic
* Ecuador's official holiday registry (`holidays_events.csv`) includes transferred holidays, bridge days, and transfer days.
* Transferred Holidays: When an official holiday is transferred to another date (`transferred == True`), the original date is a regular working day (`is_holiday = 0`). The celebratory date (`type == 'Transfer'`) is marked as an active holiday (`is_holiday = 1`).
* Locale Filtering: Store-level matching respects holiday scope (`National`, `Regional` matching store `state`, or `Local` matching store `city`).

### 7.4 Non-Mutation and Row Count Preservation
* The feature engineering pipeline preserves 100% of input rows (3,000,888 rows in = 3,000,888 rows out).
* The input DataFrame is never mutated in place. Output is deterministically sorted by `['date', 'store_nbr', 'family']`.
