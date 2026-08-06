"""端到端验证：中文列名文件 -> 列名映射 -> 预处理 -> 预测。"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.aliases import resolve_column_mapping  # noqa: E402
from src.core.inference import LSTMInferenceEngine  # noqa: E402
from src.core.models import RawWaterData  # noqa: E402
from src.core.preprocess import PreprocessSettings, clean_data  # noqa: E402
from src.model.model_package import ModelPackageService  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    data_path = os.environ.get("WQ_DATA", "")
    if not data_path:
        print("SKIP: 需要设置 WQ_DATA 环境变量")
        return 0
    df = pd.read_excel(data_path, engine="openpyxl")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    renamed = df.rename(
        columns={
            "出水COD_mgL": "化学需氧量",
            "出水氨氮_mgL": "氨氮",
            "出水TN_mgL": "总氮",
            "出水TP_mgL": "总磷",
            "出水pH": "pH",
            "出水水温": "水温",
            "datetime": "时间",
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "aliased.csv"
        renamed.to_csv(csv_path, index=False, encoding="utf-8-sig")
        raw_df = pd.read_csv(csv_path, encoding="utf-8-sig")

        pkg = ModelPackageService(ROOT / "models").load_package("COD")
        features = pkg.features
        mapping = resolve_column_mapping(raw_df, features)
        assert mapping["出水COD_mgL"] == "化学需氧量"
        assert mapping["出水氨氮_mgL"] == "氨氮"
        assert mapping["出水TN_mgL"] == "总氮"
        assert mapping["出水TP_mgL"] == "总磷"
        assert mapping["出水pH"] == "pH"

        result = clean_data(raw_df, features, PreprocessSettings(mapping=mapping))
        assert set(features).issubset(set(result.df.columns))

        raw = RawWaterData(path=csv_path, df=result.df)
        pred = LSTMInferenceEngine(device="cpu").predict(raw, pkg)
        assert len(pred.predictions) > 0
        assert np.isfinite(pred.predictions).all()
        print("PREPROCESS_SMOKE_OK")
        print("rows =", len(pred.predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
