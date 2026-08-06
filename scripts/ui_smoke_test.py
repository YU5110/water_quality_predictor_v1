"""GUI 无头冒烟测试：构建主窗口、加载模型、导入数据、预测并检查结果表。"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.config import AppConfig, project_root  # noqa: E402
from src.core.inference import LSTMInferenceEngine  # noqa: E402
from src.core.models import CombinedResult  # noqa: E402
from src.logging_service import setup_logging  # noqa: E402
from src.model.model_package import ModelPackageService  # noqa: E402
from src.selftest import SelfTestService  # noqa: E402
from src.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    data_path = os.environ.get("WQ_DATA", "")
    if not data_path:
        print("SKIP: 需要设置 WQ_DATA 环境变量指向预处理数据文件")
        return 0

    root = project_root()
    config = AppConfig.load(root / "config.yaml")
    setup_logging(config.logs_dir)

    app = QApplication(sys.argv)
    model_service = ModelPackageService(config.package_dir)
    selftest = SelfTestService(model_service)
    win = MainWindow(config, model_service, selftest)
    win.show()

    assert win.target_combo.findText("NH₃-N") >= 0, "预测目标未显示 NH₃-N"
    assert win.target_combo.findText("NH3N") < 0, "预测目标仍显示 NH3N"
    win.target_combo.setCurrentText("NH₃-N")
    assert win._current_package().target_label == "NH3N", "NH₃-N 未映射回模型包"
    win.target_combo.setCurrentText("COD")

    win._on_file_selected(data_path)
    assert win.raw is not None, "文件未加载"

    pkg = win._current_package()
    assert pkg is not None, "没有可用模型包"

    result = LSTMInferenceEngine(device="cpu").predict(win.raw, pkg)
    win._on_prediction_ok(result)

    status = win.model_status.text()
    rows = win.result_panel.table.rowCount()
    assert " v1" in win.windowTitle(), "软件名称未带 v1"
    print("UI_SMOKE_OK")
    print("model_status =", status)
    print("target =", win.target_combo.currentText())
    print("result_rows =", rows)

    assert status == "模型自检通过"
    assert rows > 0

    index = win.target_combo.findText("全部")
    assert index >= 0, "缺少“全部”目标选项"
    win.target_combo.setCurrentIndex(index)
    packages = win._packages_for_current()
    assert len(packages) == 4, "全部目标应包含 4 个模型包"
    results = [
        LSTMInferenceEngine(device="cpu").predict(win.raw, p) for p in packages
    ]
    win._on_prediction_ok(CombinedResult(results=results))
    all_rows = win.result_panel.table.rowCount()
    headers = [
        win.result_panel.table.horizontalHeaderItem(c).text()
        for c in range(win.result_panel.table.columnCount())
    ]
    assert "COD实际值 (mg/L)" in headers, "全部目标缺少 COD 实际值列"
    assert "COD预测值 (mg/L)" in headers, "全部目标缺少 COD 预测值列"
    assert "NH₃-N实际值 (mg/L)" in headers, "NH3-N 未显示为下标"
    assert "NH₃-N预测值 (mg/L)" in headers
    first_time = win.result_panel.table.item(0, 0).text()
    assert first_time[4] == "-" and first_time[10] == " " and first_time[13] == ":"
    assert win.result_panel.table.item(0, 1).textAlignment() == Qt.AlignCenter

    selector = win.result_panel.chart_selector
    assert selector.count() == 4, "预测对比图应只有 4 个目标选项"
    assert selector.currentText() == "COD"
    selector.setCurrentText("NH₃-N")
    assert win.result_panel._title == "NH₃-N 出水预测对比", "图表选择器未切换目标"
    assert "$_{3}$" in win.result_panel._panels[0]["title"], "NH3 下标未使用数学排版"
    selector.setCurrentText("COD")
    colors = {line.get("color") for line in win.result_panel._panels[0]["lines"]}
    assert colors == {"#2e8b57", "#1f5f9e"}, "图表颜色应绿色实际值、蓝色预测值"
    linestyles = {
        line.get("linestyle", "-") for line in win.result_panel._panels[0]["lines"]
    }
    assert linestyles == {"-"}, "实际值应为绿色实线"

    win.result_panel.search_edit.setText("2025-04-08")
    win.result_panel._apply_search()
    visible = sum(
        1 for r in range(all_rows) if not win.result_panel.table.isRowHidden(r)
    )
    assert 0 < visible < all_rows, "时间搜索未过滤结果"
    win.result_panel._clear_search()

    win.result_panel._sort_by_column(2)
    first_v = float(win.result_panel.table.item(0, 2).text())
    last_v = float(win.result_panel.table.item(all_rows - 1, 2).text())
    assert first_v <= last_v, "升序排序失败"
    win.result_panel._sort_by_column(2)
    first_v = float(win.result_panel.table.item(0, 2).text())
    last_v = float(win.result_panel.table.item(all_rows - 1, 2).text())
    assert first_v >= last_v, "降序排序失败"
    win.result_panel._sort_by_column(2)
    assert win.result_panel.table.item(0, 0).text() == first_time, "排序复原失败"

    table = win.result_panel.table
    table.setCurrentCell(0, 0)
    assert table.selectedIndexes(), "点击后应显示选中"
    table._toggle_if_selected(table.currentIndex())
    assert not table.selectedIndexes(), "再次点击应取消选中"

    single_result = LSTMInferenceEngine(device="cpu").predict(win.raw, packages[0])
    win._on_prediction_ok(single_result)
    assert win.result_panel.chart_selector.count() == 1, "单目标时预测对比图应固定为当前目标"
    assert not win.result_panel.chart_selector.isEnabled()

    print("all_target_rows =", all_rows)
    print("visible_after_search =", visible)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
