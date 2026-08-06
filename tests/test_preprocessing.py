import numpy as np
import pandas as pd

from src.core.preprocessing import (
    apply_rolling_normalization,
    build_sequences,
    prepare_dataframe,
)


def _make_df(n=120):
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "datetime": times,
            "A": np.sin(np.arange(n) / 6.0) + 5,
            "B": np.cos(np.arange(n) / 5.0) + 3,
            "target": np.sin(np.arange(n) / 4.0) * 2 + 8,
            "hour": times.hour.astype(float),
            "month": times.month.astype(float),
        }
    )


def test_prepare_dataframe_sorts_and_deduplicates():
    df = pd.DataFrame(
        {
            "datetime": ["2024-01-02", "2024-01-01", "2024-01-01"],
            "A": [3.0, 1.0, 99.0],
        }
    )
    out = prepare_dataframe(df)
    assert len(out) == 2
    assert out["A"].tolist() == [1.0, 3.0]


def test_rolling_normalization_and_sequences():
    df = _make_df()
    norm_cols = ["A", "B", "target"]
    df_norm, stats = apply_rolling_normalization(
        df, norm_cols, window_days=1, min_days=1
    )
    assert set(stats) == set(norm_cols)
    x, idx, skipped = build_sequences(
        df_norm, ["A", "B", "target", "hour", "month"], ws=3
    )
    assert x.shape[1] == 3
    assert x.shape[2] == 5
    assert len(idx) == len(x)
    assert skipped == len(df) - 3 - len(x)
