"""Unit tests for Corporación Favorita data ingestion, schema validation, and safe preprocessing."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.retail_forecasting import data, preprocessing, validation
from src.retail_forecasting.data import get_default_raw_sales_path, load_raw_sales
from src.retail_forecasting.preprocessing import clean_sales_records, preprocess_sales_pipeline
from src.retail_forecasting.validation import check_missing_temporal_gaps, validate_sales_schema


# ---------------------------------------------------------------------------
# Module and Interface Contract Tests
# ---------------------------------------------------------------------------

def test_data_module_importable():
    """Verify data module is importable and defines key function signatures."""
    assert hasattr(data, "load_raw_sales")
    assert hasattr(data, "load_metadata")
    assert hasattr(data, "get_default_raw_sales_path")


def test_validation_module_importable():
    """Verify validation module is importable and defines schema validators."""
    assert hasattr(validation, "validate_sales_schema")
    assert hasattr(validation, "check_missing_temporal_gaps")
    assert hasattr(validation, "ValidationReport")


# ---------------------------------------------------------------------------
# Detailed Schema Validation Tests
# ---------------------------------------------------------------------------

def test_valid_minimal_favorita_dataframe(valid_favorita_df):
    """Test 1: A valid minimal Favorita-style DataFrame passes validation cleanly."""
    report = validate_sales_schema(valid_favorita_df)
    assert report.is_valid is True
    assert len(report.fatal_errors) == 0
    assert report.row_count == 5
    assert report.store_count == 2
    assert report.family_count == 2
    assert report.duplicate_key_count == 0


def test_missing_required_columns(valid_favorita_df):
    """Test 2: Missing required columns are flagged as fatal errors and raise ValueError."""
    corrupted_df = valid_favorita_df.drop(columns=["sales"])
    report = validate_sales_schema(corrupted_df)
    assert report.is_valid is False
    assert any("Missing required columns" in err for err in report.fatal_errors)

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_sales_schema(corrupted_df, raise_on_error=True)


def test_invalid_dates(valid_favorita_df):
    """Test 3: Malformed or unparseable date values are flagged as fatal errors."""
    corrupted_df = valid_favorita_df.copy()
    corrupted_df.loc[0, "date"] = "not-a-valid-date"
    report = validate_sales_schema(corrupted_df)
    assert report.is_valid is False
    assert any("invalid or unparseable dates" in err for err in report.fatal_errors)
    assert report.invalid_value_counts.get("invalid_dates") == 1


def test_non_numeric_or_non_finite_sales(valid_favorita_df):
    """Test 4: Non-numeric and infinite sales values are flagged as fatal errors."""
    corrupted_df1 = valid_favorita_df.copy()
    corrupted_df1["sales"] = corrupted_df1["sales"].astype(object)
    corrupted_df1.loc[1, "sales"] = "invalid_sales_text"
    report1 = validate_sales_schema(corrupted_df1)
    assert report1.is_valid is False
    assert any("non-numeric" in err for err in report1.fatal_errors)
    assert report1.invalid_value_counts.get("non_numeric_sales") == 1

    corrupted_df2 = valid_favorita_df.copy()
    corrupted_df2.loc[1, "sales"] = np.inf
    report2 = validate_sales_schema(corrupted_df2)
    assert report2.is_valid is False
    assert any("non-finite" in err for err in report2.fatal_errors)
    assert report2.invalid_value_counts.get("infinite_sales") == 1


def test_negative_sales(valid_favorita_df):
    """Test 5: Negative sales values are flagged as fatal errors in retail demand modeling."""
    corrupted_df = valid_favorita_df.copy()
    corrupted_df.loc[2, "sales"] = -4.5
    report = validate_sales_schema(corrupted_df)
    assert report.is_valid is False
    assert any("negative sales" in err for err in report.fatal_errors)
    assert report.invalid_value_counts.get("negative_sales") == 1


def test_duplicate_modeling_keys_rejected(valid_favorita_df):
    """Test 6: Duplicate (date, store_nbr, family) keys are rejected; never silently dropped or aggregated."""
    corrupted_df = valid_favorita_df.copy()
    # Duplicate row 0
    corrupted_df = pd.concat([corrupted_df, corrupted_df.iloc[[0]]], ignore_index=True)
    assert len(corrupted_df) == 6

    report = validate_sales_schema(corrupted_df)
    assert report.is_valid is False
    assert report.duplicate_key_count == 1
    assert any("duplicate records" in err for err in report.fatal_errors)

    # Preprocessing must reject duplicate keys and refuse to silently aggregate or drop them
    with pytest.raises(ValueError, match="duplicate records"):
        clean_sales_records(corrupted_df)


def test_missing_store_or_family_identifiers(valid_favorita_df):
    """Test 7: Missing or invalid store identifiers and empty family categories are flagged."""
    corrupted_df1 = valid_favorita_df.copy()
    corrupted_df1["store_nbr"] = corrupted_df1["store_nbr"].astype(object)
    corrupted_df1.loc[0, "store_nbr"] = np.nan
    report1 = validate_sales_schema(corrupted_df1)
    assert report1.is_valid is False
    assert any("store_nbr" in err for err in report1.fatal_errors)

    corrupted_df2 = valid_favorita_df.copy()
    corrupted_df2.loc[0, "family"] = "   "
    report2 = validate_sales_schema(corrupted_df2)
    assert report2.is_valid is False
    assert any("blank/empty" in err for err in report2.fatal_errors)
    assert report2.invalid_value_counts.get("empty_family") == 1


# ---------------------------------------------------------------------------
# Promotion Handling Tests (Preservation & Malformation Audit)
# ---------------------------------------------------------------------------

def test_null_promotions_preserved_by_default(favorita_df_with_null_promotions):
    """Test 8: Missing onpromotion values are strictly preserved as <NA> and NOT coerced to zero."""
    clean_df, report = clean_sales_records(
        favorita_df_with_null_promotions,
        preserve_missing_promotions=True,
    )

    # Row 1 originally had NaN in onpromotion. It must remain missing (<NA>), not 0.
    assert pd.isna(clean_df.loc[1, "onpromotion"])
    assert clean_df["onpromotion"].dtype.name == "Int64"
    assert clean_df.loc[0, "onpromotion"] == 2
    assert clean_df.loc[2, "onpromotion"] == 0

    # Verify report records missing count accurately without fatal error
    assert report["missing_values_by_column"]["onpromotion"] == 1
    assert any("Missing 'onpromotion' values are preserved" in w for w in report["warnings_and_limitations"])


def test_malformed_promotions_detected_and_rejected(favorita_df_with_malformed_promotions):
    """Test 9: Malformed non-numeric promotion values are detected, counted, and flagged as fatal."""
    report = validate_sales_schema(favorita_df_with_malformed_promotions)
    assert report.is_valid is False
    assert report.invalid_value_counts.get("malformed_onpromotion") == 1
    assert any("malformed non-numeric values" in err for err in report.fatal_errors)

    with pytest.raises(ValueError, match="malformed non-numeric values"):
        clean_sales_records(favorita_df_with_malformed_promotions)


def test_explicit_zero_imputation_override(favorita_df_with_null_promotions):
    """Test 10: Explicit policy override (preserve_missing_promotions=False) imputes zero when requested."""
    clean_df, report = clean_sales_records(
        favorita_df_with_null_promotions,
        preserve_missing_promotions=False,
    )

    # With explicit override, NaN is filled with 0
    assert clean_df.loc[1, "onpromotion"] == 0
    assert any("explicit zero-imputation policy override" in t for t in report["transformations_performed"])


# ---------------------------------------------------------------------------
# Preprocessing, Sorting, Immutability & No Cartesian Expansion
# ---------------------------------------------------------------------------

def test_date_parsing_and_output_sorting(valid_favorita_df):
    """Test 11: Dates are normalized and records are deterministically sorted by grain."""
    clean_df, _ = clean_sales_records(valid_favorita_df)

    assert pd.api.types.is_datetime64_any_dtype(clean_df["date"])

    expected_order = valid_favorita_df.copy()
    expected_order["date"] = pd.to_datetime(expected_order["date"])
    expected_order = expected_order.sort_values(by=["date", "store_nbr", "family"]).reset_index(drop=True)

    pd.testing.assert_series_equal(clean_df["date"], expected_order["date"])
    pd.testing.assert_series_equal(clean_df["store_nbr"], expected_order["store_nbr"])
    pd.testing.assert_series_equal(clean_df["family"], expected_order["family"])


def test_raw_input_remains_unchanged(valid_favorita_df):
    """Test 12: The raw input DataFrame is preserved without in-place mutation."""
    original_copy = valid_favorita_df.copy(deep=True)
    _ = clean_sales_records(valid_favorita_df)

    pd.testing.assert_frame_equal(valid_favorita_df, original_copy)


def test_no_automatic_zero_filling_or_cartesian_expansion(valid_favorita_df):
    """Test 13: No premature Cartesian expansion or automatic zero-filling occurs."""
    clean_df, report = clean_sales_records(valid_favorita_df)

    assert len(clean_df) == 5
    assert report["output_row_count"] == 5

    limitations = " ".join(report["warnings_and_limitations"])
    assert "NOT zero-filled" in limitations
    assert "Cartesian expansion" in limitations


def test_data_quality_report_contents(valid_favorita_df):
    """Test 14: Data quality report contains all required schema metrics and distinguishes fatal vs warnings."""
    _, report = clean_sales_records(valid_favorita_df, input_filename="test_favorita.csv")

    assert report["input_filename"] == "test_favorita.csv"
    assert report["input_row_count"] == 5
    assert report["output_row_count"] == 5
    assert report["date_range"]["min_date"] == "2017-01-01"
    assert report["date_range"]["max_date"] == "2017-01-03"
    assert report["store_count"] == 2
    assert report["family_count"] == 2
    assert report["duplicate_key_count"] == 0
    assert "missing_values_by_column" in report
    assert "invalid_value_counts" in report
    assert "transformations_performed" in report
    assert "warnings_and_limitations" in report


def test_check_missing_temporal_gaps_reporting(valid_favorita_df):
    """Test 15: Temporal gap reporting distinguishes global calendar days from series observations."""
    gap_summary = check_missing_temporal_gaps(valid_favorita_df)

    assert gap_summary["global_min_date"] == "2017-01-01"
    assert gap_summary["global_max_date"] == "2017-01-03"
    assert gap_summary["total_calendar_days"] == 3
    assert gap_summary["total_active_series"] == 4
    assert gap_summary["min_observations_per_series"] >= 1


# ---------------------------------------------------------------------------
# CLI Execution Safety & Pre-existing Output File Preservation Tests
# ---------------------------------------------------------------------------

def test_missing_input_file_behavior():
    """Test 16: Missing input file raises FileNotFoundError with clear diagnostic guidance."""
    non_existent = Path("data/raw/non_existent_train.csv")
    with pytest.raises(FileNotFoundError, match="Primary training dataset not found"):
        load_raw_sales(file_path=non_existent)


def test_pipeline_preserves_existing_output_on_fatal_error(valid_favorita_df):
    """Test 17: A fatal validation error preserves existing output and report files completely."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        out_csv = tmp_path / "sales_daily.csv"
        rep_json = tmp_path / "data_quality_report.json"

        # 1. Establish an initial valid run
        clean_df, report = preprocess_sales_pipeline(
            input_df=valid_favorita_df,
            output_path=out_csv,
            report_path=rep_json,
        )
        assert out_csv.exists()
        assert rep_json.exists()
        initial_csv_content = out_csv.read_text(encoding="utf-8")
        initial_rep_content = rep_json.read_text(encoding="utf-8")

        # 2. Attempt a run with invalid data (negative sales)
        invalid_df = valid_favorita_df.copy()
        invalid_df.loc[0, "sales"] = -999.0

        with pytest.raises(ValueError, match="negative sales"):
            preprocess_sales_pipeline(
                input_df=invalid_df,
                output_path=out_csv,
                report_path=rep_json,
            )

        # 3. Assert pre-existing files were NOT modified or replaced by partial/corrupt output
        assert out_csv.read_text(encoding="utf-8") == initial_csv_content
        assert rep_json.read_text(encoding="utf-8") == initial_rep_content


def test_cli_execution_with_fatal_validation_failure():
    """Test 18: CLI prepare_data.py stops with non-zero exit code on invalid input and preserves outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_csv = tmp_path / "invalid_train.csv"
        out_csv = tmp_path / "sales_daily.csv"
        rep_json = tmp_path / "report.json"

        # Write invalid raw file (missing required 'family' column)
        pd.DataFrame({
            "id": [1],
            "date": ["2017-01-01"],
            "store_nbr": [1],
            "sales": [10.0],
            "onpromotion": [0],
        }).to_csv(raw_csv, index=False)

        # Create a pre-existing output file to verify preservation
        out_csv.write_text("pre_existing_valid_data", encoding="utf-8")

        # Run prepare_data.py via subprocess using local venv Python
        cmd = [
            sys.executable,
            "scripts/prepare_data.py",
            "--input", str(raw_csv),
            "--output", str(out_csv),
            "--report", str(rep_json),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Exit code must be non-zero (fatal error)
        assert result.returncode != 0
        assert "Missing required columns" in result.stderr or "Missing required columns" in result.stdout

        # Pre-existing output must remain untouched
        assert out_csv.read_text(encoding="utf-8") == "pre_existing_valid_data"
        # No misleading success report should have been written
        assert not rep_json.exists()


def test_cli_execution_success_with_relative_and_absolute_paths(valid_favorita_df):
    """Test 19: CLI prepare_data.py works with portable relative and absolute path arguments."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_csv = tmp_path / "train.csv"
        out_csv = tmp_path / "processed" / "sales_daily.csv"
        rep_json = tmp_path / "metrics" / "data_quality_report.json"

        valid_favorita_df.to_csv(raw_csv, index=False)

        cmd = [
            sys.executable,
            "scripts/prepare_data.py",
            "--input", str(raw_csv),
            "--output", str(out_csv),
            "--report", str(rep_json),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert out_csv.exists()
        assert rep_json.exists()

        # Verify output content
        loaded_clean = pd.read_csv(out_csv)
        assert len(loaded_clean) == 5
        report_data = json.loads(rep_json.read_text(encoding="utf-8"))
        assert report_data["output_row_count"] == 5
        assert report_data["duplicate_key_count"] == 0
