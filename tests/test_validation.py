import pandas as pd

from src.core.models import RawWaterData, ValidationResult
from src.core.validation import validate


class FakePackage:
    def __init__(self, features, ws):
        self.features = features
        self.ws = ws


def test_validation_missing_column():
    raw = RawWaterData(
        path=__import__("pathlib").Path("x.csv"),
        df=pd.DataFrame({"datetime": ["2024-01-01"], "A": [1.0]}),
    )
    pkg = FakePackage(features=["A", "B"], ws=3)
    result = validate(raw, pkg)
    assert isinstance(result, ValidationResult)
    assert not result.ok
    assert any("缺少" in e for e in result.errors)


def test_validation_short_data():
    raw = RawWaterData(
        path=__import__("pathlib").Path("x.csv"),
        df=pd.DataFrame({"datetime": ["2024-01-01", "2024-01-02"], "A": [1.0, 2.0]}),
    )
    pkg = FakePackage(features=["A"], ws=3)
    result = validate(raw, pkg)
    assert not result.ok
    assert any("行数不足" in e for e in result.errors)
