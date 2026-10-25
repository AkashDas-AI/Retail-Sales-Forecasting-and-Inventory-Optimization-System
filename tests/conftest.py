"""Shared Pytest fixtures for Retail Sales Forecasting & Inventory Optimization tests."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_config_dict() -> dict:
    """Return a standard test configuration dictionary."""
    return {
        "target_service_level": 0.95,
        "lead_time_days": 7,
        "holding_cost_annual_rate": 0.20,
        "order_cost": 50.0,
    }


@pytest.fixture
def valid_favorita_df() -> pd.DataFrame:
    """Return a valid minimal DataFrame conforming to Corporación Favorita train.csv schema."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "date": ["2017-01-02", "2017-01-01", "2017-01-01", "2017-01-03", "2017-01-02"],
        "store_nbr": [1, 1, 2, 1, 2],
        "family": ["GROCERY I", "GROCERY I", "BEVERAGES", "BEVERAGES", "GROCERY I"],
        "sales": [15.5, 10.0, 25.0, 0.0, 8.2],
        "onpromotion": [1, 0, 3, 0, 0],
    })


@pytest.fixture
def favorita_df_with_null_promotions() -> pd.DataFrame:
    """Return a valid DataFrame where onpromotion contains legitimate missing values (NaN)."""
    return pd.DataFrame({
        "id": [10, 20, 30],
        "date": ["2017-01-01", "2017-01-02", "2017-01-03"],
        "store_nbr": [1, 1, 1],
        "family": ["GROCERY I", "GROCERY I", "GROCERY I"],
        "sales": [10.0, 15.0, 20.0],
        "onpromotion": [2, np.nan, 0],
    })


@pytest.fixture
def favorita_df_with_malformed_promotions() -> pd.DataFrame:
    """Return a DataFrame where onpromotion contains invalid non-numeric text values."""
    return pd.DataFrame({
        "id": [100, 200],
        "date": ["2017-01-01", "2017-01-02"],
        "store_nbr": [1, 1],
        "family": ["GROCERY I", "GROCERY I"],
        "sales": [10.0, 15.0],
        "onpromotion": [1, "malformed_promo_str"],
    })
