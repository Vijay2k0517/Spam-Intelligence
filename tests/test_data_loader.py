"""Unit tests for dataset loading, validation, encoding, and deduplication."""

import json
from pathlib import Path
import pytest
import pandas as pd
from app.ml.data_loader import load_and_validate_dataset, _normalize_label


def test_normalize_label():
    """Verify robust label normalization."""
    assert _normalize_label("ham") == "ham"
    assert _normalize_label("HAM") == "ham"
    assert _normalize_label("legit") == "ham"
    assert _normalize_label("0") == "ham"
    assert _normalize_label("spam") == "spam"
    assert _normalize_label("SPAM") == "spam"
    assert _normalize_label("1") == "spam"
    assert _normalize_label("unknown") is None
    assert _normalize_label(None) is None


def test_load_tsv_dataset(test_fixtures_paths):
    """Test loading valid TSV format dataset."""
    df, summary = load_and_validate_dataset(test_fixtures_paths["tsv"], deduplicate=False)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"label", "text", "label_id"}
    assert len(df) == 7
    assert summary.total_input_records == 7
    assert summary.invalid_records_count == 0


def test_load_tsv_deduplication(test_fixtures_paths):
    """Test exact duplicate removal policy."""
    df, summary = load_and_validate_dataset(test_fixtures_paths["tsv"], deduplicate=True)
    # The sample TSV has 7 rows, 1 is exact duplicate -> 6 retained
    assert len(df) == 6
    assert summary.duplicates_removed_count == 1
    assert summary.retained_records_count == 6
    assert summary.ham_count == 3
    assert summary.spam_count == 3


def test_load_csv_dataset(test_fixtures_paths):
    """Test loading CSV file with v1/v2 column headers."""
    df, summary = load_and_validate_dataset(test_fixtures_paths["csv"], deduplicate=False)
    assert len(df) == 4
    assert set(df["label"].unique()) == {"ham", "spam"}
    assert summary.total_input_records == 4


def test_malformed_tsv_handling(test_fixtures_paths):
    """Test rejection and counting of malformed and empty rows."""
    df, summary = load_and_validate_dataset(test_fixtures_paths["malformed"], deduplicate=False)
    # Malformed file has 6 lines:
    # 1 valid ham, 1 no tab, 1 empty, 1 invalid label, 1 missing text, 1 valid spam -> 2 valid
    assert len(df) == 2
    assert summary.invalid_records_count == 4
    assert summary.retained_records_count == 2
    assert set(df["label"]) == {"ham", "spam"}


def test_latin1_encoding_handling(test_fixtures_paths):
    """Test safe handling of Latin-1 encoded characters (e.g., currency symbols)."""
    df, summary = load_and_validate_dataset(test_fixtures_paths["latin1"])
    assert len(df) == 2
    assert "£1000" in df.iloc[0]["text"]
    assert "latin-1" in summary.encoding_used.lower()


def test_nonexistent_dataset():
    """Verify FileNotFoundError on missing file."""
    with pytest.raises(FileNotFoundError):
        load_and_validate_dataset("nonexistent_path/fake_dataset.tsv")


def test_save_summary_json(test_fixtures_paths):
    """Verify dataset summary JSON serialization."""
    _, summary = load_and_validate_dataset(test_fixtures_paths["tsv"])
    out_json = test_fixtures_paths["out_dir"] / "summary.json"
    summary.save_json(out_json)

    assert out_json.is_file()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "total_input_records" in data
    assert "spam_prevalence_pct" in data
    assert "reference_paper_comparison" in data
