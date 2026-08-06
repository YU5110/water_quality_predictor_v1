import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.core.aliases import resolve_column_mapping
from src.core.preprocess import PreprocessSettings, clean_data
from src.ui.preprocess_dialog import (  # noqa: E402
    PreprocessSetupDialog,
    format_value,
    order_features,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def test_resolve_column_mapping_aliases():
    df = pd.DataFrame(
        columns=[
            "时间",
            "化学需氧量",
            "氨氮",
            "总氮",
            "总磷",
            "pH",
            "水温",
            "出水流量",
            "进水流量",
            "进水COD",
            "进水氨氮",
        ]
    )
    features = [
        "出水流量",
        "出水pH",
        "出水COD_mgL",
        "出水氨氮_mgL",
        "出水TN_mgL",
        "出水TP_mgL",
        "出水水温",
        "进水流量",
        "进水COD_mgL",
        "进水氨氮_mgL",
    ]
    mapping = resolve_column_mapping(df, features)
    assert mapping["出水COD_mgL"] == "化学需氧量"
    assert mapping["出水氨氮_mgL"] == "氨氮"
    assert mapping["出水TN_mgL"] == "总氮"
    assert mapping["出水TP_mgL"] == "总磷"
    assert mapping["出水pH"] == "pH"
    assert mapping["出水水温"] == "水温"
    assert mapping["出水流量"] == "出水流量"
    assert mapping["进水流量"] == "进水流量"
    assert mapping["进水COD_mgL"] == "进水COD"
    assert mapping["进水氨氮_mgL"] == "进水氨氮"


def test_clean_data_time_fill_outlier_interpolation():
    times = pd.date_range("2024-01-01", periods=50, freq="h")
    df = pd.DataFrame({"时间": times, "出水COD_mgL": 30.0, "出水pH": 7.0})
    df.loc[5, "出水pH"] = 15.0
    df.loc[10, "出水COD_mgL"] = np.nan
    df = df.drop(df.index[3])
    settings = PreprocessSettings(
        mapping={"出水COD_mgL": "出水COD_mgL", "出水pH": "出水pH"}
    )
    result = clean_data(df, ["出水COD_mgL", "出水pH", "hour", "month"], settings)
    assert result.report.original_rows == 49
    assert len(result.df) == 50
    assert result.report.outlier_counts.get("出水pH", 0) >= 1
    assert result.report.interpolated_counts.get("出水COD_mgL", 0) >= 1
    assert bool(result.outlier_mask["出水pH"].any())
    assert bool(result.interpolated_mask["出水COD_mgL"].any())
    assert result.df["hour"].isna().sum() == 0
    assert result.df["month"].isna().sum() == 0


def test_clean_data_duplicates():
    times = pd.date_range("2024-01-01", periods=5, freq="h")
    df = pd.DataFrame(
        {
            "时间": list(times) + [times[0]],
            "出水COD_mgL": [1.0] * 6,
            "出水pH": [7.0] * 6,
        }
    )
    settings = PreprocessSettings(
        mapping={"出水COD_mgL": "出水COD_mgL", "出水pH": "出水pH"},
        duplicate_strategy="first",
    )
    result = clean_data(df, ["出水COD_mgL", "出水pH", "hour", "month"], settings)
    assert result.report.duplicate_rows == 1
    assert len(result.df) == 5


def test_format_value_keeps_three_decimals_max():
    assert format_value(1.2) == "1.2"
    assert format_value(1.23456) == "1.235"
    assert format_value(30.0) == "30.0"
    assert format_value(float("nan")) == ""


def test_clean_data_flow_range_0_to_150000():
    times = pd.date_range("2024-01-01", periods=6, freq="h")
    df = pd.DataFrame(
        {
            "时间": times,
            "出水流量": [600.0, 150000.0, 200000.0, 0.0, 5000.0, 10.0],
            "出水COD_mgL": [30.0] * 6,
        }
    )
    settings = PreprocessSettings(
        mapping={"出水流量": "出水流量", "出水COD_mgL": "出水COD_mgL"}
    )
    result = clean_data(
        df, ["出水流量", "出水COD_mgL", "hour", "month"], settings
    )
    assert result.report.outlier_counts.get("出水流量", 0) == 1
    assert bool(result.outlier_mask["出水流量"].iloc[2])


def test_order_features_inlet_first():
    features = [
        "hour",
        "出水COD_mgL",
        "进水氨氮_mgL",
        "出水pH",
        "进水流量",
        "month",
    ]
    assert order_features(features) == [
        "进水氨氮_mgL",
        "进水流量",
        "出水COD_mgL",
        "出水pH",
        "hour",
        "month",
    ]


def test_setup_dialog_remembers_mapping(qapp):
    features = ["出水COD_mgL", "进水COD_mgL", "hour", "month"]
    columns = ["化学需氧量", "出水COD_mgL", "进水COD", "hour", "month"]
    suggestions = {
        "出水COD_mgL": "化学需氧量",
        "进水COD_mgL": "进水COD",
        "hour": None,
        "month": None,
    }
    previous = PreprocessSettings(
        mapping={
            "出水COD_mgL": "出水COD_mgL",
            "进水COD_mgL": "进水COD",
            "hour": None,
            "month": None,
        },
        z_score_threshold=5.5,
        max_interpolation_hours=20,
    )
    dialog = PreprocessSetupDialog(
        "x.csv",
        "unified",
        columns,
        features,
        suggestions,
        previous_settings=previous,
    )
    assert dialog._combos["出水COD_mgL"].currentText() == "出水COD_mgL"
    assert dialog._combos["进水COD_mgL"].currentText() == "进水COD"
    assert dialog.z_spin.value() == 5.5
    assert dialog.gap_spin.value() == 20
    dialog.close()
