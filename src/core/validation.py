from __future__ import annotations

import pandas as pd

from src.core.models import RawWaterData, ValidationResult
from src.model.model_package import ModelPackage


def validate(raw: RawWaterData, pkg: ModelPackage) -> ValidationResult:
    """校验导入数据是否满足模型包的列、行数和数值要求。"""
    errors: list[str] = []
    warnings: list[str] = []
    df = raw.df

    if "datetime" not in df.columns:
        errors.append("数据中缺少时间列 datetime")
    else:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        if dt.isna().any():
            errors.append("时间列存在无法解析的值")
        if dt.duplicated().any():
            warnings.append("时间列存在重复值，将按首次出现保留")

    missing = [c for c in pkg.features if c not in df.columns]
    if missing:
        errors.append(f"缺少模型需要的特征列：{'、'.join(missing)}")

    if len(df) < pkg.ws + 1:
        errors.append(f"数据行数不足：至少需要 {pkg.ws + 1} 行（窗口 {pkg.ws} + 目标时刻 1）")

    for col in pkg.features:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad = numeric.isna() & df[col].notna()
        if bad.any():
            errors.append(f"特征列 {col} 中存在无法转换为数值的内容")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)
