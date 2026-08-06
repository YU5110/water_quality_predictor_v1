from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


def project_root() -> Path:
    """返回项目根目录；打包成 exe 后返回 exe 所在目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    app_title: str
    app_version: str
    package_dir: Path
    logs_dir: Path
    default_target: str
    ui_language: str
    ui_style: str
    device: str

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        root = path.resolve().parent
        app_cfg = raw.get("app", {})
        model_cfg = raw.get("model", {})
        ui_cfg = raw.get("ui", {})
        paths_cfg = raw.get("paths", {})
        return cls(
            app_title=app_cfg.get("title", "污水厂出水水质预测软件 v1"),
            app_version=app_cfg.get("version", "0.1.0"),
            package_dir=(root / model_cfg.get("package_dir", "models")).resolve(),
            logs_dir=(root / paths_cfg.get("logs_dir", "logs")).resolve(),
            default_target=ui_cfg.get("default_target", "COD"),
            ui_language=ui_cfg.get("language", "zh-CN"),
            ui_style=ui_cfg.get("style", "简洁专业"),
            device=raw.get("device", "cpu"),
        )
