from __future__ import annotations

import numpy as np
import torch

from src.core.models import PredictionResult, RawWaterData
from src.core.preprocessing import (
    apply_rolling_normalization,
    build_sequences,
    prepare_dataframe,
)
from src.errors import AppError
from src.model.model_package import ModelPackage

TIME_FEATURES = ("hour", "month")


class LSTMInferenceEngine:
    """用模型包做真实预测：滚动归一化 -> 标准化 -> LSTM -> 反变换。"""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def predict(self, raw: RawWaterData, pkg: ModelPackage) -> PredictionResult:
        df = prepare_dataframe(raw.df)
        if len(df) < pkg.ws + 1:
            raise AppError(
                f"有效数据不足：需要至少 {pkg.ws + 1} 行，当前 {len(df)} 行",
                code="SHORT_DATA",
            )

        rolling = pkg.rolling
        norm_columns = [
            c for c in pkg.features if c not in TIME_FEATURES and c in df.columns
        ]
        if rolling.get("enabled", True):
            df_norm, stats = apply_rolling_normalization(
                df,
                norm_columns,
                window_days=rolling.get("window_days", 30),
                min_days=rolling.get("min_days", 7),
            )
        else:
            df_norm = df.copy()
            stats = {
                c: {"mean": np.zeros(len(df)), "std": np.ones(len(df))}
                for c in norm_columns
            }

        x_seq, idx, skipped = build_sequences(df_norm, pkg.features, pkg.ws)
        if x_seq.shape[0] == 0:
            raise AppError(
                "无法构造预测窗口：所有窗口都包含缺失值，请检查数据中的空值",
                code="NO_SEQUENCE",
            )

        means = pkg.feature_means()
        stds = pkg.feature_stds()
        x_std = (x_seq - means) / stds

        model = pkg.get_model().to(self.device)
        model.eval()
        with torch.no_grad():
            pred_std = model(torch.tensor(x_std, device=self.device)).cpu().numpy()

        pred = pred_std * pkg.y_std + pkg.y_mean
        target_stats = stats[pkg.target_name]
        pred_orig = pred * target_stats["std"][idx] + target_stats["mean"][idx]
        pred_orig = np.maximum(pred_orig, 0.0)

        timestamps = df["datetime"].to_numpy()[idx]
        actuals = df[pkg.target_name].to_numpy(dtype=np.float64)[idx]
        return PredictionResult(
            target_label=pkg.target_label,
            target_name=pkg.target_name,
            timestamps=timestamps,
            predictions=pred_orig.astype(np.float64),
            skipped=int(skipped),
            model_id=pkg.model_id,
            actuals=actuals,
        )
