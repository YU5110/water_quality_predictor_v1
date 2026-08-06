import json

import numpy as np
import pandas as pd
import torch

from src.core.file_parser import parse_file
from src.core.inference import LSTMInferenceEngine
from src.model.lstm import LSTMModel
from src.model.model_package import ModelPackageService


def _write_tiny_package(tmp_path):
    features = ["A", "B", "target"]
    ws = 3
    model = LSTMModel(input_dim=3, hidden_dim=4, n2=2, dropout=0.0, num_layers=1)
    torch.save(model.state_dict(), tmp_path / "model.pth")
    metadata = {
        "schema_version": 1,
        "target_label": "TEST",
        "target": "target",
        "features": features,
        "feature_means": {f: 0.0 for f in features},
        "feature_stds": {f: 1.0 for f in features},
        "y_mean": 0.0,
        "y_std": 1.0,
        "ws": ws,
        "hidden_dim": 4,
        "n2": 2,
        "num_layers": 1,
        "dropout": 0.0,
        "rolling": {"enabled": True, "window_days": 1, "min_days": 1},
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    return features, ws


def test_inference_engine(tmp_path):
    package_dir = tmp_path / "models"
    pkg_dir = package_dir / "TEST"
    pkg_dir.mkdir(parents=True)
    _write_tiny_package(pkg_dir)
    service = ModelPackageService(package_dir)
    pkg = service.load_package("TEST")

    times = pd.date_range("2024-01-01", periods=80, freq="h")
    data = pd.DataFrame(
        {
            "datetime": times,
            "A": np.sin(np.arange(80) / 6.0) + 5,
            "B": np.cos(np.arange(80) / 5.0) + 3,
            "target": np.sin(np.arange(80) / 4.0) * 2 + 8,
        }
    )
    csv_path = tmp_path / "data.csv"
    data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    raw = parse_file(csv_path)

    engine = LSTMInferenceEngine(device="cpu")
    result = engine.predict(raw, pkg)
    assert len(result.predictions) > 0
    assert np.isfinite(result.predictions).all()
    assert (result.predictions >= 0).all()
    assert result.target_label == "TEST"
