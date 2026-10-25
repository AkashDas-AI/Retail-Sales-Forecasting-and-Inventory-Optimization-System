# Data Dictionary and Schema Reference: Corporacion Favorita

## 1. Primary Demand Transactions (`train.csv`)
* `id` (`INT`): Unique record identifier.
* `date` (`DATE`): Transaction observation date (`YYYY-MM-DD`).
* `store_nbr` (`INT`): Store identifier (1 to 54).
* `family` (`VARCHAR`): Product category/family (e.g., `GROCERY I`, `BEVERAGES`, `CLEANING`, `PRODUCE`, `MEATS`). Represents category aggregation, not individual SKU.
* `sales` (`FLOAT`): Total daily unit sales volume (target variable).
* `onpromotion` (`INT / Nullable Int64`): Count of promoted items in this product family on that date. Missing values are preserved as `<NA>` by default and not silently imputed as 0 during ingestion.

## 2. Store Hierarchy Metadata (`stores.csv`)
* `store_nbr` (`INT`): Store identifier (Primary Key).
* `city` (`VARCHAR`): Municipality (e.g., Quito, Guayaquil, Cuenca, Ambato).
* `state` (`VARCHAR`): Province or state.
* `type` (`VARCHAR`): Store format category (A, B, C, D, E).
* `cluster` (`INT`): Store cluster group (1 to 17) grouping similar store profiles.

## 3. Daily Transactions Metadata (`transactions.csv`)
* `date` (`DATE`): Observation date.
* `store_nbr` (`INT`): Store identifier.
* `transactions` (`INT`): Count of customer transactions at that store on that date.

## 4. Calendar and Holiday Events (`holidays_events.csv`)
* `date` (`DATE`): Date of event.
* `type` (`VARCHAR`): Event category (`Holiday`, `Event`, `Additional`, `Transfer`, `Bridge`, `Work Day`).
* `locale` (`VARCHAR`): Geographical scope (`National`, `Regional`, `Local`).
* `locale_name` (`VARCHAR`): Specific locality affected.
* `description` (`VARCHAR`): Holiday description (e.g., `Carnaval`, `Independencia de Cuenca`).
* `transferred` (`BOOLEAN`): True if official observance was moved to an alternate calendar date.

## 5. Macroeconomic Indicators (`oil.csv`)
* `date` (`DATE`): Date.
* `dcoilwtico` (`FLOAT`): Daily West Texas Intermediate (WTI) crude oil spot price.

## 6. Configured Inventory Parameters (Assumptions)
* `lead_time_days` (`INT`): Configured supplier delivery lead time ($L = 7$ days).
* `default_unit_cost_usd` (`FLOAT`): Wholesale unit cost ($c = $10.00).
* `default_holding_cost_annual_rate` (`FLOAT`): Annual inventory carrying rate ($h = 20\%$).
* `default_fixed_order_cost_usd` (`FLOAT`): Fixed administrative order setup fee ($K = $50.00/order).
* `stockout_penalty_multiplier` (`FLOAT`): Shortage penalty multiplier ($1.5\times$ unit cost $\implies \pi = $15.00/unit).
* `target_service_level` (`FLOAT`): Target Cycle Service Level ($CSL = 0.95 \implies z = 1.645$).
* `initial_inventory_days_of_supply` (`FLOAT`): Baseline stock on Day 0 (14 days of supply).
