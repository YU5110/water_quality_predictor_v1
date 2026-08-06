from pathlib import Path

import pandas as pd
import pytest

from src.core.file_parser import parse_file
from src.errors import AppError


def test_parse_csv(tmp_path: Path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"datetime": ["2024-01-01 00:00"], "出水COD_mgL": [1.0]}
    ).to_csv(path, index=False, encoding="utf-8-sig")
    raw = parse_file(path)
    assert raw.df["出水COD_mgL"].iloc[0] == 1.0


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(AppError):
        parse_file(tmp_path / "nope.csv")


def test_unsupported_extension_raises(tmp_path: Path):
    path = tmp_path / "data.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(AppError):
        parse_file(path)
