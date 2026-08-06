from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.errors import AppError
from src.model.lstm import LSTMModel

SCHEMA_VERSION = 1


@dataclass
class ModelPackage:
    """一个可整体加载的模型单元：权重 + 元数据。"""

    package_dir: Path
    metadata: dict
    model: LSTMModel | None = field(default=None, repr=False)

    @property
    def target_label(self) -> str:
        return str(self.metadata["target_label"])

    @property
    def target_name(self) -> str:
        return str(self.metadata["target"])

    @property
    def features(self) -> list[str]:
        return list(self.metadata["features"])

    @property
    def ws(self) -> int:
        return int(self.metadata["ws"])

    @property
    def y_mean(self) -> float:
        return float(self.metadata["y_mean"])

    @property
    def y_std(self) -> float:
        return float(self.metadata["y_std"])

    @property
    def rolling(self) -> dict:
        return dict(self.metadata.get("rolling", {}))

    @property
    def model_id(self) -> str:
        return str(self.metadata.get("model_id", self.target_label))

    def get_model(self) -> LSTMModel:
        if self.model is None:
            self.model = LSTMModel.from_metadata(self.metadata)
            state = torch.load(
                self.package_dir / "model.pth", map_location="cpu", weights_only=True
            )
            self.model.load_state_dict(state)
            self.model.eval()
        return self.model

    def feature_means(self) -> np.ndarray:
        return np.array(
            [float(self.metadata["feature_means"][f]) for f in self.features],
            dtype=np.float32,
        )

    def feature_stds(self) -> np.ndarray:
        return np.array(
            [float(self.metadata["feature_stds"][f]) for f in self.features],
            dtype=np.float32,
        )


class ModelPackageService:
    """扫描、加载并校验软件目录下的模型包。"""

    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir)

    def list_packages(self) -> list[str]:
        if not self.package_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.package_dir.iterdir()
            if p.is_dir() and (p / "metadata.json").exists()
        )

    def load_package(self, label: str) -> ModelPackage:
        pkg_dir = self.package_dir / label
        meta_path = pkg_dir / "metadata.json"
        if not meta_path.exists():
            raise AppError(f"模型包不存在：{label}", code="MODEL_PACKAGE_MISSING")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self._check_metadata(metadata, pkg_dir)
        return ModelPackage(package_dir=pkg_dir, metadata=metadata)

    def load_all(self) -> dict[str, ModelPackage]:
        return {label: self.load_package(label) for label in self.list_packages()}

    @staticmethod
    def _check_metadata(metadata: dict, pkg_dir: Path) -> None:
        if int(metadata.get("schema_version", 0)) != SCHEMA_VERSION:
            raise AppError(
                f"模型包版本不兼容：{metadata.get('schema_version')}", code="MODEL_SCHEMA"
            )
        required = [
            "target_label",
            "target",
            "features",
            "ws",
            "hidden_dim",
            "n2",
            "feature_means",
            "feature_stds",
        ]
        missing = [k for k in required if k not in metadata]
        if missing:
            raise AppError(
                f"模型包元数据缺少字段：{'、'.join(missing)}", code="MODEL_META"
            )
        if not (pkg_dir / "model.pth").exists():
            raise AppError("模型包缺少 model.pth 权重文件", code="MODEL_WEIGHTS")
