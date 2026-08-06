from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RawWaterData:
    path: Path
    df: pd.DataFrame


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PredictionResult:
    target_label: str
    target_name: str
    timestamps: np.ndarray
    predictions: np.ndarray
    skipped: int
    model_id: str
    actuals: np.ndarray | None = None


@dataclass
class CombinedResult:
    """“全部目标”预测结果：按目标顺序保存多个单目标结果。"""

    results: list[PredictionResult]
