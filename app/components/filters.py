"""Reusable interactive filter components for Streamlit dashboard views."""

from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st


def render_store_family_selector(
    df: pd.DataFrame,
    default_store: int = 1,
    default_family: str = "GROCERY I",
    key_prefix: str = "",
) -> Tuple[int, str]:
    """Render coordinated store number and product family dropdown selectors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'store_nbr' and 'family' columns.
    default_store : int
        Default store number selection.
    default_family : str
        Default family selection.
    key_prefix : str
        Streamlit widget key prefix to prevent collision across tabs.

    Returns
    -------
    Tuple[int, str]
        Selected store number and product family string.
    """
    col1, col2 = st.columns(2)

    stores = sorted(df["store_nbr"].unique().astype(int))
    store_idx = stores.index(default_store) if default_store in stores else 0

    with col1:
        selected_store = st.selectbox(
            "Select Store",
            options=stores,
            index=store_idx,
            key=f"{key_prefix}_store_selector",
            help="Corporación Favorita retail store location (1-54)",
        )

    available_families = sorted(df[df["store_nbr"] == selected_store]["family"].unique())
    fam_idx = available_families.index(default_family) if default_family in available_families else 0

    with col2:
        selected_family = st.selectbox(
            "Select Product Family",
            options=available_families,
            index=fam_idx,
            key=f"{key_prefix}_family_selector",
            help="Kaggle Favorita merchandise category (33 categories)",
        )

    return int(selected_store), str(selected_family)


def render_date_range_selector(
    min_date_str: str,
    max_date_str: str,
    default_days: int = 180,
    key_prefix: str = "",
) -> Tuple[str, str]:
    """Render start and end date input selectors with sensible range defaults."""
    max_date = pd.to_datetime(max_date_str).date()
    min_date = pd.to_datetime(min_date_str).date()
    default_start = max(min_date, max_date - pd.Timedelta(days=default_days))

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_start_date",
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_end_date",
        )

    return str(start_date), str(end_date)
