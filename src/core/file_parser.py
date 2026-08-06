from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.models import RawWaterData
from src.errors import AppError

SUPPORTED_EXTENSIONS = (".xlsx", ".csv")


def parse_file(path: str | Path) -> RawWaterData:
    """读取统一格式的 Excel/CSV 水质时间序列文件。"""
    p = Path(path)
    if not p.exists():
        raise AppError(f"文件不存在：{p}", code="FILE_NOT_FOUND")
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AppError(f"不支持的文件格式：{ext or '无扩展名'}，请选择 .xlsx 或 .csv", code="UNSUPPORTED_FILE")
    try:
        if ext == ".csv":
            df = pd.read_csv(p, encoding="utf-8-sig")
        else:
            df = pd.read_excel(p, engine="openpyxl")
    except Exception as exc:
        raise AppError(f"文件读取失败：{p.name}", code="PARSE_FAILED", detail=str(exc)) from exc
    if df is None or df.empty:
        raise AppError(f"文件中没有数据：{p.name}", code="EMPTY_FILE")
    df.columns = [str(c).strip() for c in df.columns]
    return RawWaterData(path=p, df=df)
