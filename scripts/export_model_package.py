#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把训练产物导出为软件可加载的模型包。

输入（训练流水线工作区）：
- config.json
- 数据预处理/预处理后水质数据.xlsx、特征池.json
- 最优超参数/最优超参数LSTM.json
- 模型预测结果/*_lstm_model.pth、评价指标汇总.json

输出：water_quality_predictor/models/<目标标签>/model.pth + metadata.json
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lstm import LSTMModel  # noqa: E402

WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_PATH = WORKSPACE_ROOT / "config.json"
PREPROCESSED_PATH = WORKSPACE_ROOT / "数据预处理" / "预处理后水质数据.xlsx"
FEATURE_POOL_PATH = WORKSPACE_ROOT / "数据预处理" / "特征池.json"
HP_PATH = WORKSPACE_ROOT / "最优超参数" / "最优超参数LSTM.json"
RESULT_DIR = WORKSPACE_ROOT / "模型预测结果"
OUTPUT_DIR = PROJECT_ROOT / "models"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_rolling_normalization(df: pd.DataFrame, cfg: dict):
    """因果滚动归一化，与训练脚本保持完全一致。"""
    rn = cfg["preprocessing"].get("rolling_norm", {})
    if not rn.get("enabled", False):
        return df, None
    window_hours = int(rn.get("window_days", 30)) * 24
    min_periods = int(rn.get("min_days", 7)) * 24
    cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ("hour", "month")
    ]
    stats = {}
    for c in cols:
        s = df[c].astype(float)
        rmean = s.rolling(window_hours, min_periods=min_periods, closed="left").mean()
        rstd = s.rolling(window_hours, min_periods=min_periods, closed="left").std()
        floor = 0.05 * float(s.std()) + 1e-8
        rstd_safe = rstd.where(rstd >= floor, floor)
        df[c] = (s - rmean) / rstd_safe
        stats[c] = pd.DataFrame(
            {"mean": rmean.values, "std": rstd_safe.values},
            index=df["datetime"].values,
        )
    return df, stats


def split_dataframe(df: pd.DataFrame, cfg: dict):
    """按相对时间窗口切分训练/验证/测试，与训练脚本一致。"""
    test_days = cfg["split"]["test_days"]
    val_days = cfg["split"]["val_days"]
    roll_days = cfg["split"].get("roll_days", 0)
    last = df["datetime"].iloc[-1]
    test_start = last - pd.Timedelta(hours=test_days * 24 - 1)
    test_mask = df["datetime"] >= test_start
    val_end = test_start - pd.Timedelta(hours=1)
    val_start = val_end - pd.Timedelta(hours=val_days * 24 - 1)
    val_mask = (df["datetime"] >= val_start) & (df["datetime"] <= val_end)
    if roll_days > 0:
        train_start = val_start - pd.Timedelta(days=roll_days)
        train_mask = (df["datetime"] >= train_start) & (df["datetime"] < val_start)
    else:
        train_mask = df["datetime"] < val_start
    return (
        df[train_mask].reset_index(drop=True),
        df[val_mask].reset_index(drop=True),
        df[test_mask].reset_index(drop=True),
    )


def standardize(X_train, X_val, X_test):
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0) + 1e-8
    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


def standardize_y(y_train, y_val, y_test):
    mean = np.nanmean(y_train)
    std = np.nanstd(y_train) + 1e-8
    return (
        (y_train - mean) / std,
        (y_val - mean) / std,
        (y_test - mean) / std,
        mean,
        std,
    )


def inverse_y(y_std, mean, std):
    return y_std * std + mean


def make_test_sequences(X_val, y_val, X_test, y_test, ws):
    """把验证集尾部拼到测试集前面，第一条预测对齐测试集第 0 小时。"""
    X_ext = np.concatenate([X_val[-ws:], X_test], axis=0)
    y_ext = np.concatenate([y_val[-ws:], y_test], axis=0)
    seqs_x, seqs_y, seq_idx = [], [], []
    n = len(y_ext)
    for i in range(n - ws):
        sx = X_ext[i : i + ws]
        sy = y_ext[i + ws]
        if np.isnan(sx).any() or np.isnan(sy):
            continue
        seqs_x.append(sx)
        seqs_y.append(sy)
        seq_idx.append(i)
    return (
        np.array(seqs_x, dtype=np.float32),
        np.array(seqs_y, dtype=np.float32),
        np.array(seq_idx, dtype=np.int64),
    )


def to_original_scale(y_std, y_mean, y_std_scale, roll_stats, target, test_dt, idx):
    """从标准化+滚动归一化空间还原为原始量纲。"""
    y = inverse_y(y_std, y_mean, y_std_scale)
    if roll_stats is not None:
        st = roll_stats[target]
        m = st.loc[test_dt[idx], "mean"].values
        s = st.loc[test_dt[idx], "std"].values
        y = y * s + m
    return y


def build_package(target, label, cfg, df, feature_pool, hp, metrics):
    include_lag = cfg["model"].get("include_target_lag", True)
    features = feature_pool if include_lag else [f for f in feature_pool if f != target]

    df_norm, roll_stats = apply_rolling_normalization(df.copy(), cfg)
    train_df, val_df, test_df = split_dataframe(df_norm, cfg)

    X_train = train_df[features].to_numpy(dtype=np.float32)
    y_train = train_df[target].to_numpy(dtype=np.float32)
    X_val = val_df[features].to_numpy(dtype=np.float32)
    y_val = val_df[target].to_numpy(dtype=np.float32)
    X_test = test_df[features].to_numpy(dtype=np.float32)
    y_test = test_df[target].to_numpy(dtype=np.float32)

    X_train_s, X_val_s, X_test_s, f_mean, f_std = standardize(X_train, X_val, X_test)
    y_train_s, y_val_s, y_test_s, y_mean, y_std = standardize_y(y_train, y_val, y_test)

    ws = int(hp["ws"])
    model = LSTMModel(
        input_dim=len(features),
        hidden_dim=int(hp["n1"]),
        n2=int(hp["n2"]),
        dropout=float(cfg["model"]["dropout"]),
        num_layers=int(hp.get("L", 2)),
    )
    pth_path = RESULT_DIR / f"{label}_lstm_model.pth"
    if not pth_path.exists():
        raise FileNotFoundError(f"缺少模型权重文件: {pth_path}")
    model.load_state_dict(
        torch.load(pth_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    X_seq, y_seq, idx = make_test_sequences(
        X_val_s, y_val_s, X_test_s, y_test_s, ws
    )
    if len(X_seq) == 0:
        raise RuntimeError(f"{label}: 测试集无法构造有效预测窗口")
    with torch.no_grad():
        pred_std = model(torch.tensor(X_seq)).numpy()
    pred = inverse_y(pred_std, y_mean, y_std)
    test_dt = test_df["datetime"].to_numpy()
    pred_orig = to_original_scale(pred, y_mean, y_std, roll_stats, target, test_dt, idx)

    smoke_idx = len(X_seq) - 1
    smoke_x = X_seq[smoke_idx].tolist()
    smoke_prediction = float(pred_orig[smoke_idx])
    metadata = {
        "schema_version": 1,
        "model_id": f"{label}_LSTM_v1",
        "target_label": label,
        "target": target,
        "features": features,
        "feature_means": {f: float(f_mean[i]) for i, f in enumerate(features)},
        "feature_stds": {f: float(f_std[i]) for i, f in enumerate(features)},
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "ws": ws,
        "hidden_dim": int(hp["n1"]),
        "n2": int(hp["n2"]),
        "num_layers": int(hp.get("L", 2)),
        "dropout": float(cfg["model"]["dropout"]),
        "rolling": cfg["preprocessing"].get("rolling_norm", {}),
        "metrics": metrics.get(label, {}).get("LSTM", {}),
        "data_range": {
            "start": str(df["datetime"].min()),
            "end": str(df["datetime"].max()),
        },
        "trained_at": str(Path(pth_path).stat().st_mtime),
        "smoke": {
            "x": smoke_x,
            "model_output": float(pred_std[smoke_idx]),
            "prediction": smoke_prediction,
        },
    }

    out_dir = OUTPUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pth_path, out_dir / "model.pth")
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[OK] {label}: 特征数={len(features)}, ws={ws}, 自检预测={smoke_prediction:.6f}")
    return out_dir


def main():
    cfg = load_config()
    df = pd.read_excel(PREPROCESSED_PATH, engine="openpyxl")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    with open(FEATURE_POOL_PATH, "r", encoding="utf-8") as f:
        feature_pool = json.load(f)["features"]
    with open(HP_PATH, "r", encoding="utf-8") as f:
        hp_all = json.load(f)

    metrics_path = RESULT_DIR / "评价指标汇总.json"
    metrics = (
        json.load(open(metrics_path, "r", encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )

    label_to_target = {v: k for k, v in cfg["target_labels"].items()}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, hp in hp_all.items():
        target = label_to_target[label]
        build_package(target, label, cfg, df, feature_pool, hp, metrics)

    print(f"\n模型包已导出到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
