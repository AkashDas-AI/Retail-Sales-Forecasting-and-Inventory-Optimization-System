# System Architecture and Technical Design

## 1. Executive Architecture Summary

The **Retail Sales Forecasting and Inventory Optimization System** integrates supervised machine learning demand forecasting with statistical inventory control and discrete-event simulation to automate retail replenishment planning.

```mermaid
flowchart LR
    A["Raw Data Ingestion<br>(Corporacion Favorita)"] --> B["Data Validation &<br>Quality Gate"]
    B --> C["Feature Engineering<br>(Lags, Windows, Holidays)"]
    C --> D["Forecasting Engine<br>(Random Forest & Naive)"]
    D --> E["Inventory Policy<br>Optimization (s, Q)"]
    E --> F["Discrete-Event<br>Simulation Engine"]
    F --> G["Streamlit Decision<br>Support Dashboard"]
```

## 2. Component Boundaries and Directory Roles

1. `src/retail_forecasting/`: Reusable, tested production Python library implementing ingestion (`data.py`), schema validation (`validation.py`), feature generation (`features.py`), model training and recursive inference (`forecasting.py`), error metrics (`metrics.py`), policy formulation (`inventory.py`), purchase order requisitioning (`recommendations.py`), and discrete-event simulation (`simulation.py`).
2. `scripts/`: Command-line batch execution scripts orchestrating data preparation (`prepare_data.py`), feature engineering (`build_features.py`), model training (`train_model.py`), recommendation generation (`generate_recommendations.py`), and simulation backtesting (`run_simulation.py`).
3. `app/`: Multi-page Streamlit web application with cached loaders (`data_loader.py`), reusable Plotly visualizers (`components/charts.py`), standardized disclaimers (`components/disclaimers.py`), and six coordinated analytical views (`views/`).
4. `configs/`: Centralized JSON configurations defining operational paths and model hyperparameters (`project_config.json`) and inventory assumptions (`inventory_config.json`).
5. `data/`: Tiered local storage preserving raw benchmark inputs (`raw/`), interim artifacts (`interim/`), processed matrices (`processed/`), and testing fixtures (`sample/`).
6. `models/`: Model metadata signatures (`model_metadata.json`) documenting feature schemas and serialized model specifications.
7. `outputs/`: Generated forecast predictions (`forecasts/`), policy tables and purchase orders (`recommendations/`), and simulation results (`simulation/`).
8. `reports/metrics/`: Versioned evaluation audit reports (`forecast_evaluation_report.json`, `simulation_evaluation_report.json`, `inventory_planning_report.json`, `data_quality_report.json`).
9. `tests/`: 113 automated unit and integration tests verifying schema enforcement, temporal leakage prevention, model reproducibility, inventory conservation, and headless UI rendering.

## 3. Detailed Architecture Specifications

For deep-dive architectural specifications across individual subsystems, refer to:
* [`01_system_architecture.md`](01_system_architecture.md): Layer responsibilities, design patterns, and deployment topologies.
* [`02_data_pipeline_and_flow.md`](02_data_pipeline_and_flow.md): End-to-end data lifecycle, splitting logic, and recursive horizon feedback.
* [`03_dataset_specifications.md`](03_dataset_specifications.md): Relational schema definitions and entity relationships.
* [`04_module_specifications.md`](04_module_specifications.md): Function-by-function interface contracts and input/output deliverables.
* [`05_model_learning_and_optimization.md`](05_model_learning_and_optimization.md): Mathematical formulations, loss functions, and optimization algorithms.
* [`06_verification_and_validation.md`](06_verification_and_validation.md): Quality gates, backtesting protocols, and automated test suites.
* [`07_execution_log.md`](07_execution_log.md): Milestone delivery history, audits, and verification records.
* [`08_inventory_uncertainty_design.md`](08_inventory_uncertainty_design.md): Empirical Bayes shrinkage, lead-time error compounding, and policy derivations.
