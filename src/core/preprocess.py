"""真实数据预处理：文件识别、时间对齐、异常值清洗、缺失值处理、报告。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.aliases import TIME_ALIASES, resolve_column_mapping
from src.errors import AppError

DEFAULT_PHYSICAL_RANGES = {
    "出水流量": [0, 150000],
    "出水pH": [0, 14],
    "出水COD_mgL": [0, 200],
    "出水氨氮_mgL": [0, 50],
    "出水TN_mgL": [0, 100],
    "出水TP_mgL": [0, 20],
    "出水水温": [0, 40],
    "进水流量": [0, 150000],
    "进水COD_mgL": [0, 5000],
    "进水氨氮_mgL": [0, 200],
}

DEFAULT_GROUP_HEADER_ROW = 2
DEFAULT_SUB_HEADER_ROW = 3
DEFAULT_DATA_START_ROW = 5
DEFAULT_ZSCORE_THRESHOLD = 4.0
DEFAULT_MAX_INTERPOLATION_HOURS = 12
DEFAULT_MISSING_RATE_THRESHOLD = 0.3

RAW_OUTLET_MAPPING = {
    "流量|累计流量(立方米)": "出水流量",
    "pH|监测值": "出水pH",
    "化学需氧量(毫克/升)|监测值": "出水COD_mgL",
    "氨氮(毫克/升)|监测值": "出水氨氮_mgL",
    "总氮(毫克/升)|监测值": "出水TN_mgL",
    "总磷(毫克/升)|监测值": "出水TP_mgL",
    "水温(摄氏度)|监测值": "出水水温",
}

RAW_INLET_MAPPING = {
    "流量|累计流量(立方米)": "进水流量",
    "化学需氧量(毫克/升)|监测值": "进水COD_mgL",
    "氨氮(毫克/升)|监测值": "进水氨氮_mgL",
}


@dataclass
class PreprocessSettings:
    mapping: dict[str, str | None] = field(default_factory=dict)
    z_score_threshold: float = DEFAULT_ZSCORE_THRESHOLD
    max_interpolation_hours: int = DEFAULT_MAX_INTERPOLATION_HOURS
    missing_rate_threshold: float = DEFAULT_MISSING_RATE_THRESHOLD
    duplicate_strategy: str = "first"
    group_header_row: int = DEFAULT_GROUP_HEADER_ROW
    sub_header_row: int = DEFAULT_SUB_HEADER_ROW
    data_start_row: int = DEFAULT_DATA_START_ROW


@dataclass
class PreprocessReport:
    source_type: str
    mapping: dict[str, str | None]
    original_rows: int
    final_rows: int
    time_start: str
    time_end: str
    outlier_counts: dict[str, int]
    interpolated_counts: dict[str, int]
    missing_counts: dict[str, int]
    high_missing_features: list[str]
    duplicate_rows: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreprocessResult:
    df: pd.DataFrame
    report: PreprocessReport
    outlier_mask: pd.DataFrame
    interpolated_mask: pd.DataFrame


def read_source(path: str | Path) -> tuple[str, pd.DataFrame]:
    """识别并读取数据源，返回 (文件类型, 原始数据表)。"""
    p = Path(path)
    if not p.exists():
        raise AppError(f"路径不存在：{p}", code="PATH_NOT_FOUND")
    if p.is_dir():
        return "raw_folder", _read_raw_folder(p)
    name = p.name
    if p.suffix.lower() in (".xlsx",) and ("进口" in name or "出口" in name):
        return "raw_file", _read_raw_file(p)
    return "unified", _read_unified(p)


def _read_unified(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, encoding="utf-8-sig")
        else:
            df = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        raise AppError(f"文件读取失败：{path.name}", code="PARSE_FAILED", detail=str(exc)) from exc
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_raw_file(path: Path) -> pd.DataFrame:
    marker = "出口" if "出口" in path.name else "进口"
    col_map = RAW_OUTLET_MAPPING if marker == "出口" else RAW_INLET_MAPPING
    return _parse_raw_excel(
        path,
        DEFAULT_GROUP_HEADER_ROW,
        DEFAULT_SUB_HEADER_ROW,
        DEFAULT_DATA_START_ROW,
        col_map,
    )


def _read_raw_folder(folder: Path) -> pd.DataFrame:
    files = sorted(list(folder.glob("*.xlsx")) + list(folder.glob("*.xls")))
    if not files:
        raise AppError(f"文件夹中没有 Excel 文件：{folder}", code="NO_RAW_FILES")
    outlet_frames = []
    inlet_frames = []
    for path in files:
        marker = "出口" if "出口" in path.name else "进口"
        col_map = RAW_OUTLET_MAPPING if marker == "出口" else RAW_INLET_MAPPING
        frame = _parse_raw_excel(
            path,
            DEFAULT_GROUP_HEADER_ROW,
            DEFAULT_SUB_HEADER_ROW,
            DEFAULT_DATA_START_ROW,
            col_map,
        )
        (outlet_frames if marker == "出口" else inlet_frames).append(frame)
    if not outlet_frames or not inlet_frames:
        raise AppError("文件夹中需要同时包含进口和出口 Excel 文件", code="MISSING_SIDE")
    outlet_df = pd.concat(outlet_frames, ignore_index=True)
    inlet_df = pd.concat(inlet_frames, ignore_index=True)
    merged = pd.merge(outlet_df, inlet_df, on="datetime", how="inner")
    return merged


def _parse_raw_excel(
    path: Path,
    group_row: int,
    sub_row: int,
    data_start_row: int,
    col_map: dict[str, str],
) -> pd.DataFrame:
    try:
        raw = pd.read_excel(path, header=None, engine="openpyxl")
    except Exception as exc:
        raise AppError(f"Excel 读取失败：{path.name}", code="PARSE_FAILED", detail=str(exc)) from exc
    groups = raw.iloc[group_row].astype(str).str.strip()
    subs = raw.iloc[sub_row].astype(str).str.strip()
    labels = [f"{g}|{s}" for g, s in zip(groups, subs)]
    data = raw.iloc[data_start_row:].reset_index(drop=True)
    dt = pd.to_datetime(data[0], errors="coerce")
    out = pd.DataFrame({"datetime": dt})
    for key, internal in col_map.items():
        if key in labels:
            out[internal] = pd.to_numeric(data[labels.index(key)], errors="coerce")
        else:
            out[internal] = np.nan
    return out


def clean_data(
    df: pd.DataFrame,
    features: list[str],
    settings: PreprocessSettings,
) -> PreprocessResult:
    """执行完整预处理，返回清洗后数据、颜色掩码和报告。"""
    out = df.copy()
    time_col = _find_time_column(out)
    if time_col is None:
        raise AppError("未找到时间列，请确认数据包含 时间/日期/采样时间 等时间列", code="NO_TIME_COLUMN")
    out = out.rename(columns={time_col: "datetime"})

    rename = {}
    for feature, source in settings.mapping.items():
        if source and source != feature and source in out.columns:
            rename[source] = feature
    if rename:
        out = out.rename(columns=rename)

    for feature in features:
        if feature in out.columns:
            out[feature] = pd.to_numeric(out[feature], errors="coerce")

    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"])
    original_rows = len(out)

    duplicate_rows = int(out["datetime"].duplicated().sum())
    if duplicate_rows:
        keep = "first" if settings.duplicate_strategy == "first" else "last"
        out = out.drop_duplicates(subset=["datetime"], keep=keep)
    out = out.sort_values("datetime").reset_index(drop=True)

    if len(out):
        full_range = pd.date_range(out["datetime"].min(), out["datetime"].max(), freq="h")
        out = out.set_index("datetime").reindex(full_range).reset_index()
        out = out.rename(columns={"index": "datetime"})
        out["datetime"] = pd.to_datetime(out["datetime"])

    for feature in features:
        if feature not in out.columns:
            out[feature] = np.nan

    outlier_mask = pd.DataFrame(False, index=out.index, columns=features)
    interpolated_mask = pd.DataFrame(False, index=out.index, columns=features)
    outlier_counts: dict[str, int] = {}
    numeric_cols = [f for f in features if f not in ("hour", "month")]

    for col, (lo, hi) in DEFAULT_PHYSICAL_RANGES.items():
        if col not in numeric_cols or col not in out.columns:
            continue
        mask = (out[col] < lo) | (out[col] > hi)
        mask = mask.fillna(False)
        outlier_mask.loc[mask, col] = True
        out.loc[mask, col] = np.nan
        outlier_counts[col] = int(mask.sum())

    for col in numeric_cols:
        if col not in out.columns:
            continue
        values = out[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        mean = float(np.mean(finite)) if finite.size else 0.0
        std = float(np.std(finite)) if finite.size else 0.0
        z = np.abs((values - mean) / (std + 1e-8))
        mask = np.isfinite(values) & (z > settings.z_score_threshold)
        outlier_mask.loc[mask, col] = True
        out.loc[mask, col] = np.nan
        outlier_counts[col] = outlier_counts.get(col, 0) + int(mask.sum())

    interpolated_counts: dict[str, int] = {}
    for col in numeric_cols:
        if col not in out.columns:
            continue
        s = out[col]
        before = s.isna()
        if not before.any():
            continue
        s2 = s.interpolate(
            method="linear",
            limit=settings.max_interpolation_hours,
            limit_area="inside",
        )
        values = s2.to_numpy(dtype=float)
        mask = pd.isna(values)
        if mask[0]:
            run_len = 0
            for v in mask:
                if v:
                    run_len += 1
                else:
                    break
            if run_len <= settings.max_interpolation_hours and run_len < len(mask):
                values[:run_len] = values[run_len]
        if mask[-1]:
            run_len = 0
            for v in mask[::-1]:
                if v:
                    run_len += 1
                else:
                    break
            if run_len <= settings.max_interpolation_hours and run_len < len(mask):
                values[-run_len:] = values[-run_len - 1]
        out[col] = values
        after = out[col].isna()
        filled = before & ~after
        interpolated_mask.loc[filled, col] = True
        interpolated_counts[col] = int(filled.sum())

    if "hour" in features:
        out["hour"] = out["datetime"].dt.hour.astype(float)
    if "month" in features:
        out["month"] = out["datetime"].dt.month.astype(float)

    final_cols = ["datetime"] + [f for f in features if f in out.columns]
    out = out[final_cols].reset_index(drop=True)
    outlier_mask = outlier_mask.loc[out.index]
    interpolated_mask = interpolated_mask.loc[out.index]

    missing_counts = {
        c: int(out[c].isna().sum()) for c in features if c in out.columns
    }
    high_missing = [
        c
        for c in features
        if c in out.columns
        and len(out) > 0
        and missing_counts[c] / len(out) > settings.missing_rate_threshold
    ]
    report = PreprocessReport(
        source_type="unified",
        mapping=dict(settings.mapping),
        original_rows=original_rows,
        final_rows=len(out),
        time_start=str(out["datetime"].min()) if len(out) else "-",
        time_end=str(out["datetime"].max()) if len(out) else "-",
        outlier_counts=outlier_counts,
        interpolated_counts=interpolated_counts,
        missing_counts=missing_counts,
        high_missing_features=high_missing,
        duplicate_rows=duplicate_rows,
    )
    return PreprocessResult(
        df=out,
        report=report,
        outlier_mask=outlier_mask,
        interpolated_mask=interpolated_mask,
    )


def _find_time_column(df: pd.DataFrame) -> str | None:
    columns = [str(c) for c in df.columns]
    for alias in TIME_ALIASES:
        for col in columns:
            if col.strip().lower() == alias.lower():
                return col
    for col in columns:
        if col.lower() in ("datetime", "时间", "日期"):
            return col
    return None
