"""常见水质指标名称与软件内部特征名的兼容映射。"""

from __future__ import annotations

TIME_ALIASES = [
    "时间",
    "日期",
    "采样时间",
    "监测时间",
    "时间点",
    "日期时间",
    "datetime",
    "date",
]

FEATURE_ALIASES: dict[str, list[str]] = {
    "出水流量": ["出水流量", "出口流量", "出水累计流量", "流量(出水)", "流量"],
    "出水pH": ["出水pH", "出口pH", "pH(出水)", "pH", "酸碱度"],
    "出水COD_mgL": [
        "出水COD_mgL",
        "出水COD",
        "出口COD",
        "出水化学需氧量",
        "出口化学需氧量",
        "化学需氧量",
        "COD",
    ],
    "出水氨氮_mgL": [
        "出水氨氮_mgL",
        "出水氨氮",
        "出口氨氮",
        "氨氮",
        "NH3-N",
        "NH₃-N",
        "NH3N",
    ],
    "出水TN_mgL": ["出水TN_mgL", "出水TN", "出口TN", "总氮", "TN"],
    "出水TP_mgL": ["出水TP_mgL", "出水TP", "出口TP", "总磷", "TP"],
    "出水水温": ["出水水温", "出口水温", "水温", "温度"],
    "进水流量": ["进水流量", "进口流量", "进水累计流量", "流量(进水)"],
    "进水COD_mgL": [
        "进水COD_mgL",
        "进水COD",
        "进口COD",
        "进水化学需氧量",
        "进口化学需氧量",
    ],
    "进水氨氮_mgL": ["进水氨氮_mgL", "进水氨氮", "进口氨氮"],
    "hour": ["hour", "小时"],
    "month": ["month", "月份"],
}


def _normalize(name: str) -> str:
    return "".join(str(name).strip().lower().split())


def resolve_column_mapping(
    df, features: list[str]
) -> dict[str, str | None]:
    """为每个必需特征自动建议对应源列，返回 {特征: 源列名或 None}。"""
    columns = [str(c) for c in df.columns]
    suggestions: dict[str, str | None] = {}
    used: set[str] = set()
    for feature in features:
        if feature in columns:
            suggestions[feature] = feature
            used.add(feature)
            continue
        match = _find_best_match(feature, [c for c in columns if c not in used])
        suggestions[feature] = match
        if match:
            used.add(match)
    return suggestions


def _find_best_match(feature: str, columns: list[str]) -> str | None:
    aliases = [feature] + list(FEATURE_ALIASES.get(feature, []))
    norm_aliases = {_normalize(a) for a in aliases}
    direction = "出水" if feature.startswith("出水") else (
        "进水" if feature.startswith("进水") else None
    )

    def _opposite_direction(col: str) -> bool:
        if direction is None:
            return False
        other = "进水" if direction == "出水" else "出水"
        return _normalize(col).startswith(other)

    candidates = [c for c in columns if not _opposite_direction(c)]

    exact = [c for c in candidates if _normalize(c) in norm_aliases]
    if exact:
        return min(exact, key=len)
    canonical = _normalize(feature)
    contains_canonical = [c for c in candidates if canonical in _normalize(c)]
    if contains_canonical:
        return min(contains_canonical, key=len)
    contains_alias = [
        c for c in candidates if any(a in _normalize(c) for a in norm_aliases)
    ]
    if contains_alias:
        return min(contains_alias, key=len)
    return None
