from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.models import CombinedResult, PredictionResult
from src.model.model_package import ModelPackage

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

TARGET_DISPLAY = {"COD": "COD", "NH3N": "NH₃-N", "TN": "TN", "TP": "TP"}


def display_label(label: str) -> str:
    return TARGET_DISPLAY.get(label, label)


def chart_axes_title(label: str) -> str:
    """matplotlib 标题改用数学排版显示下标，避免字体缺字。"""
    return label.replace("₃", "$_{3}$")


def fmt_time(ts) -> str:
    """统一显示格式：2025-04-08 20:00:00。"""
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _downsample(series_x, series_y, max_points: int = 2500):
    """长序列均匀采样，降低拖动时的重绘开销。"""
    x = np.asarray(series_x)
    y = np.asarray(series_y)
    if x.size <= max_points:
        return x, y
    idx = np.unique(np.linspace(0, x.size - 1, max_points).astype(int))
    return x[idx], y[idx]


def draw_figure(fig: Figure, panels: list[dict], canvas: "ZoomableCanvas") -> None:
    """绘制对比图；当前版本始终只显示一张图。"""
    fig.clear()
    axes = [fig.add_subplot(111)]
    for ax, panel in zip(axes, panels):
        ax.set_title(panel["title"], fontsize=12)
        for line in panel["lines"]:
            xs, ys = _downsample(line["timestamps"], line["values"])
            ax.plot(
                xs,
                ys,
                color=line.get("color", "#1f5f9e"),
                linewidth=line.get("linewidth", 1.4),
                alpha=line.get("alpha", 1.0),
                linestyle=line.get("linestyle", "-"),
                label=line.get("label", ""),
            )
        if panel.get("ylabel"):
            ax.set_ylabel(panel["ylabel"])
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
    fig.set_constrained_layout(True)
    canvas.remember_view()
    canvas.draw_idle()


class ZoomableCanvas(FigureCanvas):
    """支持滚轮缩放、左键拖动平移、一键复原的画布；重绘做了节流以降低卡顿。"""

    def __init__(self, figure: Figure):
        super().__init__(figure)
        self._init_limits: dict = {}
        self._pan_press_display = None
        self._pan_limits = None
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def remember_view(self):
        self._init_limits = {
            ax: (tuple(ax.get_xlim()), tuple(ax.get_ylim()))
            for ax in self.figure.axes
        }

    def reset_view(self):
        for ax, (xlim, ylim) in self._init_limits.items():
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        factor = 0.85 if event.button == "up" else 1.18
        xdata, ydata = event.xdata, event.ydata
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(
            xdata - (xdata - xlim[0]) * factor,
            xdata + (xlim[1] - xdata) * factor,
        )
        ax.set_ylim(
            ydata - (ydata - ylim[0]) * factor,
            ydata + (ylim[1] - ydata) * factor,
        )
        self.draw_idle()

    def _on_press(self, event):
        if event.inaxes is None or event.button != 1:
            return
        self._pan_press_display = (event.x, event.y)
        self._pan_limits = {
            ax: (
                tuple(ax.get_xlim()),
                tuple(ax.get_ylim()),
                float(ax.bbox.width),
                float(ax.bbox.height),
            )
            for ax in self.figure.axes
        }

    def _on_motion(self, event):
        if self._pan_limits is None or event.x is None or event.y is None:
            return
        dx_px = event.x - self._pan_press_display[0]
        dy_px = event.y - self._pan_press_display[1]
        for ax, (xlim, ylim, width, height) in self._pan_limits.items():
            scale_x = (xlim[1] - xlim[0]) / max(width, 1.0)
            scale_y = (ylim[1] - ylim[0]) / max(height, 1.0)
            ax.set_xlim(xlim[0] - dx_px * scale_x, xlim[1] - dx_px * scale_x)
            ax.set_ylim(ylim[0] - dy_px * scale_y, ylim[1] - dy_px * scale_y)
        self.draw_idle()

    def _on_release(self, event):
        self._pan_press_display = None
        self._pan_limits = None


class PlotDialog(QDialog):
    """查看大图窗口，同样支持缩放、平移和复原。"""

    def __init__(self, title: str, panels: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"查看大图 - {title}")
        self.resize(1080, 680)
        self.figure = Figure(figsize=(9.5, 5.8), dpi=110)
        self.canvas = ZoomableCanvas(self.figure)
        draw_figure(self.figure, panels, self.canvas)

        reset_btn = QPushButton("图片复原")
        reset_btn.setObjectName("SecondaryButton")
        reset_btn.clicked.connect(self.canvas.reset_view)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        bar = QHBoxLayout()
        bar.addWidget(reset_btn)
        bar.addStretch(1)
        bar.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(bar)


class FileImportPanel(QWidget):
    file_selected = Signal(str)
    path_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setMinimumWidth(340)

        title = QLabel("数据导入")
        title.setObjectName("PanelTitle")
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择统一格式的 Excel / CSV 水质数据文件")
        self.browse_btn = QPushButton("选择文件")
        self.browse_btn.setObjectName("SecondaryButton")
        self.browse_btn.clicked.connect(self._browse)
        self.folder_btn = QPushButton("选择文件夹")
        self.folder_btn.setObjectName("SecondaryButton")
        self.folder_btn.clicked.connect(self._browse_folder)
        self.state_label = QLabel("未导入文件")

        row = QHBoxLayout()
        row.addWidget(self.path_edit, 1)
        row.addWidget(self.browse_btn)
        row.addWidget(self.folder_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(row)
        layout.addWidget(self.state_label)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择水质数据文件", "", "数据文件 (*.xlsx *.csv)"
        )
        if path:
            self.path_edit.setText(path)
            self.file_selected.emit(path)
            self.path_selected.emit(path)

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择进口/出口 Excel 文件夹")
        if path:
            self.path_edit.setText(path)
            self.path_selected.emit(path)

    def set_state(self, text: str):
        self.state_label.setText(text)


class ModelInfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")

        title = QLabel("模型信息")
        title.setObjectName("PanelTitle")
        self.form = QFormLayout()
        self.value_labels = {
            "目标指标": QLabel("-"),
            "模型ID": QLabel("-"),
            "输入特征数": QLabel("-"),
            "窗口长度WS": QLabel("-"),
            "隐藏层n1": QLabel("-"),
            "全连接层n2": QLabel("-"),
            "LSTM层数": QLabel("-"),
            "R2": QLabel("-"),
            "RMSE": QLabel("-"),
            "MAE": QLabel("-"),
        }
        for name, label in self.value_labels.items():
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setTextFormat(Qt.PlainText)
            self.form.addRow(name, label)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(self.form)

    def _set_values(self, text: dict[str, str]):
        for name, label in self.value_labels.items():
            label.setText(text[name])

    def show_package(self, pkg: ModelPackage):
        metrics = pkg.metadata.get("metrics", {})
        text = {
            "目标指标": display_label(pkg.target_label),
            "模型ID": pkg.model_id,
            "输入特征数": str(len(pkg.features)),
            "窗口长度WS": str(pkg.ws),
            "隐藏层n1": str(pkg.metadata.get("hidden_dim", "-")),
            "全连接层n2": str(pkg.metadata.get("n2", "-")),
            "LSTM层数": str(pkg.metadata.get("num_layers", "-")),
            "R2": f"{metrics.get('R2', '-'):.4f}"
            if isinstance(metrics.get("R2"), (int, float))
            else "-",
            "RMSE": f"{metrics.get('RMSE', '-'):.4f}"
            if isinstance(metrics.get("RMSE"), (int, float))
            else "-",
            "MAE": f"{metrics.get('MAE', '-'):.4f}"
            if isinstance(metrics.get("MAE"), (int, float))
            else "-",
        }
        self._set_values(text)

    def show_all_packages(self, packages: list[ModelPackage]):
        if not packages:
            self._set_values({name: "-" for name in self.value_labels})
            return
        ws_list = sorted(pkg.ws for pkg in packages)
        text = {
            "目标指标": f"全部（{len(packages)} 项）",
            "模型ID": " / ".join(display_label(pkg.target_label) for pkg in packages),
            "输入特征数": str(len(packages[0].features)),
            "窗口长度WS": " / ".join(str(ws) for ws in ws_list),
            "隐藏层n1": " / ".join(str(pkg.metadata.get("hidden_dim", "-")) for pkg in packages),
            "全连接层n2": " / ".join(str(pkg.metadata.get("n2", "-")) for pkg in packages),
            "LSTM层数": " / ".join(str(pkg.metadata.get("num_layers", "-")) for pkg in packages),
            "R2": "-",
            "RMSE": "-",
            "MAE": "-",
        }
        self._set_values(text)


class ToggleSelectTable(QTableWidget):
    """左键再次点击已选中的单元格时取消选中（去掉蓝色背景）。"""

    def _toggle_if_selected(self, index) -> bool:
        if index.isValid() and index in self.selectedIndexes():
            self.clearSelection()
            self.setCurrentIndex(QModelIndex())
            return True
        return False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            index = self.indexAt(event.pos())
            if self._toggle_if_selected(index):
                return
        super().mousePressEvent(event)


class ResultPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self._headers: list[str] = []
        self._original_rows: list[list[str]] = []
        self._sort_state: dict[int, int] = {}
        self._lines_by_target: dict[str, list[dict]] = {}
        self._panels: list[dict] = []
        self._title = "预测结果"

        title = QLabel("预测结果")
        title.setObjectName("PanelTitle")

        self.search_bar = QWidget()
        self.search_bar.setObjectName("SearchBar")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(
            "按时间搜索：年（2025）、月（2025-04）、日（2025-04-08）或小时（20:00）"
        )
        self.search_btn = QPushButton("搜索")
        self.search_btn.setObjectName("SecondaryButton")
        self.clear_search_btn = QPushButton("清空")
        self.clear_search_btn.setObjectName("SecondaryButton")
        self.search_count = QLabel("")
        self.search_count.setObjectName("SearchHint")
        search_row = QHBoxLayout(self.search_bar)
        search_row.addWidget(self.search_edit, 1)
        search_row.addWidget(self.search_btn)
        search_row.addWidget(self.clear_search_btn)
        search_row.addWidget(self.search_count)

        self.table = ToggleSelectTable(0, 3)
        self.table.setHorizontalHeaderLabels(["时间", "实际值 (mg/L)", "预测值 (mg/L)"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setWordWrap(False)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_cell_menu)
        self.table.horizontalHeader().sectionClicked.connect(self._sort_by_column)
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(
            self._show_header_menu
        )

        self.figure = Figure(figsize=(7, 4), dpi=100)
        self.canvas = ZoomableCanvas(self.figure)
        self.placeholder = QLabel("导入数据并开始预测后，这里显示预测结果")
        self.placeholder.setAlignment(Qt.AlignCenter)

        self.view_large_btn = QPushButton("查看大图")
        self.view_large_btn.setObjectName("SecondaryButton")
        self.reset_view_btn = QPushButton("图片复原")
        self.reset_view_btn.setObjectName("SecondaryButton")
        self.chart_selector_label = QLabel("预测对比图")
        self.chart_selector_label.setObjectName("SearchHint")
        self.chart_selector = QComboBox()
        self.chart_selector.setMinimumWidth(110)

        chart_toolbar = QHBoxLayout()
        chart_toolbar.addWidget(self.view_large_btn)
        chart_toolbar.addWidget(self.reset_view_btn)
        chart_toolbar.addStretch(1)
        chart_toolbar.addWidget(self.chart_selector_label)
        chart_toolbar.addWidget(self.chart_selector)

        chart_box = QWidget()
        chart_layout = QVBoxLayout(chart_box)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.addLayout(chart_toolbar)
        chart_layout.addWidget(self.canvas, 1)
        chart_layout.addWidget(self.placeholder, 1)
        self.canvas.hide()

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(chart_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([300, 200])

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.search_bar)
        layout.addWidget(splitter, 1)

        self.search_btn.clicked.connect(self._apply_search)
        self.clear_search_btn.clicked.connect(self._clear_search)
        self.search_edit.returnPressed.connect(self._apply_search)
        self.view_large_btn.clicked.connect(self._open_large_plot)
        self.reset_view_btn.clicked.connect(self.canvas.reset_view)
        self.chart_selector.currentTextChanged.connect(self._render_chart)

    def _set_rows(self, headers: list[str], rows: list[tuple]):
        self._headers = list(headers)
        self._original_rows = [list(r) for r in rows]
        self._sort_state = {}
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        for c in range(len(headers)):
            self.table.horizontalHeaderItem(c).setTextAlignment(Qt.AlignCenter)
        self._populate_rows(self._original_rows)
        self._update_header_arrows()
        self.search_count.setText(f"共 {len(rows)} 行")
        self.search_edit.clear()

    def _populate_rows(self, rows: list[list[str]]):
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()

    def show_result(self, result: PredictionResult):
        label = display_label(result.target_label)
        rows = []
        for i in range(len(result.predictions)):
            actual = result.actuals[i] if result.actuals is not None else np.nan
            rows.append(
                (
                    fmt_time(result.timestamps[i]),
                    "" if np.isnan(actual) else f"{actual:.3f}",
                    f"{result.predictions[i]:.3f}",
                )
            )
        self._set_rows([f"时间", f"{label}实际值 (mg/L)", f"{label}预测值 (mg/L)"], rows)

        lines = []
        if result.actuals is not None and not np.isnan(result.actuals).all():
            lines.append(
                {
                    "timestamps": result.timestamps,
                    "values": result.actuals,
                    "label": "实际值",
                    "alpha": 0.85,
                }
            )
        lines.append(
            {
                "timestamps": result.timestamps,
                "values": result.predictions,
                "label": "预测值",
            }
        )
        self._lines_by_target = {label: lines}
        self.chart_selector.blockSignals(True)
        self.chart_selector.clear()
        self.chart_selector.addItem(label)
        self.chart_selector.setCurrentIndex(0)
        self.chart_selector.setEnabled(False)
        self.chart_selector.blockSignals(False)
        self._render_chart()

    def show_all(self, combined: CombinedResult):
        labels = [display_label(r.target_label) for r in combined.results]
        all_times = set()
        for result in combined.results:
            all_times.update(pd.to_datetime(result.timestamps))
        ordered_times = sorted(all_times)

        per_target = []
        for result in combined.results:
            mapping = {}
            actuals = result.actuals if result.actuals is not None else [np.nan] * len(result.predictions)
            for ts, actual, pred in zip(result.timestamps, actuals, result.predictions):
                mapping[pd.Timestamp(ts)] = (actual, pred)
            per_target.append(mapping)

        rows = []
        for ts in ordered_times:
            row = [fmt_time(ts)]
            for mapping in per_target:
                entry = mapping.get(ts)
                if entry is None:
                    row.extend(["", ""])
                else:
                    actual, pred = entry
                    row.append("" if np.isnan(actual) else f"{actual:.3f}")
                    row.append(f"{pred:.3f}")
            rows.append(tuple(row))
        headers = ["时间"]
        for label in labels:
            headers.append(f"{label}实际值 (mg/L)")
            headers.append(f"{label}预测值 (mg/L)")
        self._set_rows(headers, rows)

        self._lines_by_target = {}
        for result in combined.results:
            label = display_label(result.target_label)
            lines = []
            if result.actuals is not None and not np.isnan(result.actuals).all():
                lines.append(
                    {
                    "timestamps": result.timestamps,
                    "values": result.actuals,
                    "label": "实际值",
                    "alpha": 0.85,
                }
            )
            lines.append(
                {
                    "timestamps": result.timestamps,
                    "values": result.predictions,
                    "label": "预测值",
                }
            )
            self._lines_by_target[label] = lines

        self.chart_selector.blockSignals(True)
        self.chart_selector.clear()
        self.chart_selector.addItems(labels)
        self.chart_selector.setCurrentIndex(0)
        self.chart_selector.setEnabled(True)
        self.chart_selector.blockSignals(False)
        self._render_chart()

    def _render_chart(self):
        if not self._lines_by_target:
            return
        target_labels = list(self._lines_by_target)
        selection = self.chart_selector.currentText()
        if selection not in target_labels:
            selection = target_labels[0]
        lines = []
        for line in self._lines_by_target[selection]:
            color = "#2e8b57" if line.get("label") == "实际值" else "#1f5f9e"
            lines.append({**line, "color": color})
        title = f"{selection} 出水预测对比"
        self._panels = [
            {
                "title": chart_axes_title(title),
                "ylabel": "浓度 (mg/L)",
                "lines": lines,
            }
        ]
        self._title = title
        draw_figure(self.figure, self._panels, self.canvas)
        self.canvas.show()
        self.placeholder.hide()

    def _apply_search(self):
        query = self.search_edit.text().strip()
        total = self.table.rowCount()
        visible = 0
        for row in range(total):
            item = self.table.item(row, 0)
            match = item is not None and (not query or query in item.text())
            self.table.setRowHidden(row, not match)
            if match:
                visible += 1
        self.search_count.setText(f"显示 {visible} / {total} 行")

    def _clear_search(self):
        self.search_edit.clear()
        for row in range(self.table.rowCount()):
            self.table.setRowHidden(row, False)
        self.search_count.setText(f"共 {self.table.rowCount()} 行")

    def _sort_key(self, value: str, col: int):
        if col == 0:
            try:
                return (1, pd.Timestamp(value).timestamp())
            except Exception:
                return (0, value)
        try:
            return (1, float(value))
        except ValueError:
            return (0, "")

    def _sort_by_column(self, col: int):
        state = (self._sort_state.get(col, 0) + 1) % 3
        self._sort_state[col] = state
        rows = list(self._original_rows)
        if state == 1:
            rows.sort(key=lambda r: self._sort_key(r[col], col))
        elif state == 2:
            rows.sort(key=lambda r: self._sort_key(r[col], col), reverse=True)
        self._populate_rows(rows)
        self._update_header_arrows()
        self._apply_search()

    def _update_header_arrows(self):
        for col, base in enumerate(self._headers):
            arrow = {0: "", 1: " ▲", 2: " ▼"}.get(self._sort_state.get(col, 0), "")
            item = self.table.horizontalHeaderItem(col)
            if item is None:
                item = QTableWidgetItem(base)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setHorizontalHeaderItem(col, item)
            item.setText(base + arrow)

    def _copy_selected(self):
        indexes = self.table.selectedIndexes()
        if not indexes:
            return
        rows = sorted({idx.row() for idx in indexes})
        cols = sorted({idx.column() for idx in indexes})
        lines = []
        for r in rows:
            cells = []
            for c in cols:
                item = self.table.item(r, c)
                cells.append(item.text() if item is not None else "")
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))

    def _show_cell_menu(self, pos):
        index = self.table.indexAt(pos)
        if index.isValid():
            self.table.setCurrentCell(index.row(), index.column())
        menu = QMenu(self)
        action = QAction("复制", menu)
        action.triggered.connect(self._copy_selected)
        menu.addAction(action)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _show_header_menu(self, pos):
        col = self.table.horizontalHeader().logicalIndexAt(pos)
        if col < 0 or col >= len(self._headers):
            return
        menu = QMenu(self)
        action = QAction("复制表头", menu)
        base = self._headers[col]
        action.triggered.connect(lambda: QApplication.clipboard().setText(base))
        menu.addAction(action)
        menu.exec(self.table.horizontalHeader().mapToGlobal(pos))

    def _open_large_plot(self):
        if not self._panels:
            return
        dialog = PlotDialog(self._title, self._panels, self)
        dialog.exec()

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self._copy_selected()
            event.accept()
            return
        super().keyPressEvent(event)
