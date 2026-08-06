from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """统一时间列、按时间排序并去重，返回可预测的干净时序表。"""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime"], keep="first").reset_index(drop=True)
    return df


def apply_rolling_normalization(
    df: pd.DataFrame,
    columns: list[str],
    window_days: int = 30,
    min_days: int = 7,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """因果滚动归一化，与训练脚本完全一致：只用过去 window_days 天的均值/标准差。"""
    df = df.copy()
    window_hours = int(window_days) * 24
    min_periods = int(min_days) * 24
    stats: dict[str, dict[str, np.ndarray]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].astype(float)
        rmean = s.rolling(window_hours, min_periods=min_periods, closed="left").mean()
        rstd = s.rolling(window_hours, min_periods=min_periods, closed="left").std()
        floor = 0.05 * float(s.std()) + 1e-8
        rstd_safe = rstd.where(rstd >= floor, floor)
        df[col] = (s - rmean) / rstd_safe
        stats[col] = {
            "mean": rmean.to_numpy(dtype=np.float64),
            "std": rstd_safe.to_numpy(dtype=np.float64),
        }
    return df, stats


def build_sequences(
    df: pd.DataFrame, features: list[str], ws: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """滑动窗口生成模型输入；窗口含缺失值的序列跳过，与训练一致。"""
    values = df[features].to_numpy(dtype=np.float64)
    n = len(df)
    x_list: list[np.ndarray] = []
    idx_list: list[int] = []
    for i in range(ws, n):
        window = values[i - ws : i]
        if np.isnan(window).any():
            continue
        x_list.append(window.astype(np.float32))
        idx_list.append(i)
    skipped = n - ws - len(x_list)
    if not x_list:
        return (
            np.empty((0, ws, len(features)), dtype=np.float32),
            np.array([], dtype=np.int64),
            int(skipped),
        )
    return np.stack(x_list), np.array(idx_list, dtype=np.int64), int(skipped)
