# 01 - System Architecture

## 1. System Overview

The **Retail Sales Forecasting & Inventory Optimization System** is an enterprise-grade predictive and prescriptive platform designed to solve two core supply chain challenges:
1. **Accurate Demand Forecasting**: Predict store-SKU level demand across multiple forecast horizons (e.g., 7-day, 14-day, 30-day) capturing seasonality, promotions, price elasticity, and stockout events.
2. **Optimal Inventory Policy Generation**: Translate probabilistic demand forecasts into cost-minimizing inventory actions, including dynamic Safety Stock (SS), Reorder Points (ROP), and Economic Order Quantities (EOQ) under service level constraints.

---

## 2. Architecture Diagram

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data Ingestion & Storage Layer"]
        RAW_TX["Historical Transactions"] --> INGEST["Ingestion & Cleansing Engine"]
        RAW_ITEMS["Product & Category Metadata"] --> INGEST
        RAW_STORES["Store & Geography Metadata"] --> INGEST
        RAW_CAL["Calendar & Holiday Events"] --> INGEST
        RAW_INV["Current Inventory & Lead Times"] --> INGEST
        INGEST --> PARQUET_LAKE[("Validated Clean Data Store")]
    end

    subgraph Feature_Layer ["2. Feature Engineering & Temporal Processing"]
        PARQUET_LAKE --> FE["Feature Transformation Pipeline"]
        FE --> LAG_GEN["Lag & Rolling Aggregator (Non-leaking)"]
        FE --> CAL_FE["Temporal & Holiday Encoders"]
        FE --> PROMO_FE["Promotion & Price Elasticity Features"]
        LAG_GEN & CAL_FE & PROMO_FE --> FE_STORE[("Feature Store / Processed Datasets")]
    end

    subgraph Modeling_Layer ["3. Machine Learning & Forecasting Engine"]
        FE_STORE --> SPLIT["Temporal Splitter (Train / Val / Test)"]
        SPLIT --> MODEL_EXP["Model Training & Tuning Subsystem"]
        MODEL_EXP --> LGBP["LightGBM Regressor (Global Model)"]
        MODEL_EXP --> BASELINES["Statistical Baselines (ARIMA / ETS / Moving Avg)"]
        LGBP & BASELINES --> ENSEMBLE["Model Selector / Ensemble"]
        ENSEMBLE --> QUANTILE["Quantile / Distributional Forecasts (P10, P50, P90)"]
    end

    subgraph Inventory_Layer ["4. Inventory Optimization Engine"]
        QUANTILE --> OPTIM_ENGINE["Inventory Policy Optimizer"]
        PARQUET_LAKE -.->|"Lead Times & Holding Costs"| OPTIM_ENGINE
        OPTIM_ENGINE --> SS_CALC["Dynamic Safety Stock (SS)"]
        OPTIM_ENGINE --> ROP_CALC["Reorder Point (ROP)"]
        OPTIM_ENGINE --> ORDER_CALC["Purchase Order Recommendation (EOQ / s, S)"]
        OPTIM_ENGINE --> SIMULATOR["Monte Carlo Supply Chain Simulator"]
    end

    subgraph Serving_Layer ["5. Serving & Interface Layer"]
        OPTIM_ENGINE --> EXPORT["Reports & Decision Output (CSV / JSON / DB)"]
        EXPORT --> REST_API["REST API (FastAPI / Modular Engine)"]
        EXPORT --> DASHBOARD["Analytics & Inventory Dashboard"]
    end
```

---

## 3. Subsystem Architectural Details

### 3.1 Data Ingestion & Storage Layer
* **Responsibility**: Ingests heterogeneous data sources (transactions, metadata, inventory logs, supply lead times).
* **Guarantees**: Validates schema compliance, rejects anomalies, imputes missing dates (zero-sales imputation for active SKUs), and enforces referential integrity.
* **Storage Format**: Columnar formats (Parquet) partitioned by `store_id` or date for high-throughput reads.

### 3.2 Feature Engineering Layer
* **Responsibility**: Builds predictive signals strictly using historical timestamps prior to the prediction origin $T$.
* **Leakage Prevention**: All rolling windows, lags, and target-based encodings use shifting offsets (`shift(horizon)` or expanding past-only windows).
* **Feature Classes**:
  * Temporal / Calendar (Day of week, Month, Cyclical sin/cos encodings, Holiday flags).
  * Historical Demand (Lags: $t-7, t-14, t-28$; Rolling means/stds over 7, 14, 30, 60 days).
  * Product & Store Attributes (Category embeddings, Store type, Hierarchy groupings).
  * Dynamic Drivers (Price discount ratio, Promotional indicators).

### 3.3 Machine Learning & Forecasting Engine
* **Responsibility**: Generates point estimates and uncertainty bounds (e.g., P10, P50, P90).
* **Strategy**: Global tabular gradient-boosted trees (LightGBM/XGBoost) trained across all store-SKU series with categorical embeddings and hierarchical grouping, benchmarked against naive and classical time-series baselines.
* **Objective Functions**: Multi-objective evaluation including RMSE (mean expected demand) and Quantile Loss (Pinball Loss for asymmetrical over/under-prediction penalties).

### 3.4 Inventory Optimization Engine
* **Responsibility**: Translates probabilistic demand distributions into actionable supply chain triggers.
* **Core Logic**:
  * Calculates demand uncertainty over lead time ($\sigma_{L}$).
  * Computes safety stock using configured Service Level targets (Cycle Service Level $CSL \ge 95\%$).
  * Computes dynamic Reorder Point: $ROP = \mu_L + SS$.
  * Generates optimal order quantities considering batch constraints, holding costs, and order setup costs.

### 3.5 Serving & Presentation Layer
* **Responsibility**: Delivers forecast values, order recommendations, and validation metrics via structured APIs and reports.

