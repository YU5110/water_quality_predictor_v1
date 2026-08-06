from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.model.model_package import ModelPackageService


@dataclass
class SelfTestResult:
    ok: bool
    details: list[str] = field(default_factory=list)


class SelfTestService:
    """启动自检：加载真实模型包并用模型包内真实样本验证预测一致性。"""

    def __init__(self, model_service: ModelPackageService):
        self.model_service = model_service

    def run(self) -> SelfTestResult:
        details: list[str] = []
        labels = self.model_service.list_packages()
        if not labels:
            return SelfTestResult(ok=False, details=["未找到任何模型包，请先运行模型包导出脚本"])
        ok_all = True
        for label in labels:
            try:
                pkg = self.model_service.load_package(label)
                smoke = pkg.metadata.get("smoke")
                if not smoke:
                    details.append(f"{label}: 缺少自检样本，跳过")
                    ok_all = False
                    continue
                x = np.asarray(smoke["x"], dtype=np.float32)
                expected = float(smoke.get("model_output", smoke.get("prediction", 0.0)))
                model = pkg.get_model()
                with torch.no_grad():
                    pred_std = model(torch.tensor(x[None, ...])).numpy().ravel()[0]
                rel = abs(pred_std - expected) / max(abs(expected), 1e-6)
                ok = bool(np.isfinite(pred_std)) and rel < 1e-4
                details.append(
                    f"{label}: 自检{'通过' if ok else '失败'}（模型输出 {pred_std:.6f}，参考 {expected:.6f}）"
                )
                ok_all = ok_all and ok
            except Exception as exc:
                details.append(f"{label}: 自检异常 - {exc}")
                ok_all = False
        return SelfTestResult(ok=ok_all, details=details)
