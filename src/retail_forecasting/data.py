"""Data loading and dataset ingestion interfaces for Corporación Favorita.

Handles loading primary sales transactional records and optional supporting
metadata tables (stores, transactions, holidays, oil) with strict path resolution,
date parsing, and clear error diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

# Primary modeling columns expected in Corporación Favorita train.csv
REQUIRED_TRAIN_COLUMNS = [
    "id",
    "date",
    "store_nbr",
    "family",
    "sales",
    "onpromotion",
]


def resolve_project_root() -> Path:
    """Resolve the project root directory relative to this source file."""
    return Path(__file__).resolve().parent.parent.parent


def get_default_raw_sales_path() -> Path:
    """Resolve default raw train.csv path using project_config.json if available."""
    root = resolve_project_root()
    config_path = root / "configs" / "project_config.json"
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                raw_dir = cfg.get("data_paths", {}).get("raw_dir", "data/raw")
                return root / raw_dir / "train.csv"
        except Exception:
            pass
    return root / "data" / "raw" / "train.csv"


def load_raw_sales(
    file_path: Optional[Union[str, Path]] = None,
    parse_dates: bool = True,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load primary Corporación Favorita historical sales transactions from CSV.

    Args:
        file_path: Explicit path to train.csv. If None, resolved from configs/project_config.json.
        parse_dates: Whether to parse the 'date' column as datetime during read.
        nrows: Optional row limit for testing or exploratory inspection.

    Returns:
        pd.DataFrame: Raw transactional sales data.

    Raises:
        FileNotFoundError: If the specified or configured train.csv does not exist.
        ValueError: If file is empty or cannot be parsed.
    """
    if file_path is None:
        target_path = get_default_raw_sales_path()
    else:
        target_path = Path(file_path)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Primary training dataset not found at '{target_path}'. "
            "Please ensure 'train.csv' is placed in the designated directory "
            "(downloaded from Kaggle Corporación Favorita Store Sales competition) "
            "before initiating full pipeline execution."
        )

    if not target_path.is_file():
        raise ValueError(f"Specified path '{target_path}' is a directory, not a CSV file.")

    date_parser = ["date"] if parse_dates else False

    try:
        df = pd.read_csv(
            target_path,
            parse_dates=date_parser,
            nrows=nrows,
            low_memory=False,
        )
    except Exception as exc:
        raise ValueError(f"Failed to read CSV from '{target_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"The raw sales file at '{target_path}' is empty.")

    return df


def load_metadata(file_path: Union[str, Path], parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Load optional supporting metadata tables (stores, transactions, holidays, oil).

    Args:
        file_path: Path to the metadata CSV file.
        parse_dates: Optional list of column names to parse as datetimes.

    Returns:
        pd.DataFrame: Loaded metadata table.

    Raises:
        FileNotFoundError: If the specified metadata file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found at '{path}'.")

    return pd.read_csv(path, parse_dates=parse_dates or False)
