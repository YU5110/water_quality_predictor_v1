from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.config import AppConfig
from src.core.aliases import resolve_column_mapping
from src.core.file_parser import parse_file
from src.core.inference import LSTMInferenceEngine
from src.core.models import CombinedResult, RawWaterData
from src.core.preprocess import clean_data, read_source
from src.core.validation import validate
from src.errors import AppError
from src.model.model_package import ModelPackage, ModelPackageService
from src.selftest import SelfTestService
from src.ui.panels import FileImportPanel, ModelInfoPanel, ResultPanel, display_label
from src.ui.preprocess_dialog import PreprocessReportDialog, PreprocessSetupDialog
from src.ui.styles import QSS


class PredictionWorker(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        engine: LSTMInferenceEngine,
        raw,
        packages: list[ModelPackage],
        parent=None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.raw = raw
        self.packages = packages

    def run(self):
        try:
            results = [self.engine.predict(self.raw, pkg) for pkg in self.packages]
            if len(results) == 1:
                self.finished_ok.emit(results[0])
            else:
                self.finished_ok.emit(CombinedResult(results=results))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: AppConfig,
        model_service: ModelPackageService,
        selftest: SelfTestService,
    ):
        super().__init__()
        self.config = config
        self.model_service = model_service
        self.selftest = selftest
        self.logger = logging.getLogger("water_quality_predictor")
        self.packages: dict[str, ModelPackage] = {}
        self._label_to_key: dict[str, str] = {}
        self.raw = None
        self.worker: PredictionWorker | None = None
        self.engine = LSTMInferenceEngine(device=config.device)

        self._build_ui()
        self.setStyleSheet(QSS)
        self._load_models()

    def _build_ui(self):
        self.setWindowTitle(self.config.app_title)
        self.resize(1180, 720)

        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(64)
        title_label = QLabel(self.config.app_title)
        title_label.setObjectName("AppTitle")
        target_hint = QLabel("预测目标")
        target_hint.setObjectName("HeaderLabel")
        self.target_combo = QComboBox()
        self.model_status = QLabel("模型未加载")
        self.model_status.setObjectName("HeaderLabel")
        header_layout = QHBoxLayout(header)
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(target_hint)
        header_layout.addWidget(self.target_combo)
        header_layout.addSpacing(16)
        header_layout.addWidget(self.model_status)
        header_layout.addSpacing(8)

        self.file_panel = FileImportPanel()
        self.info_panel = ModelInfoPanel()
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.setMinimumHeight(36)
        self.result_panel = ResultPanel()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.file_panel)
        left_layout.addWidget(self.info_panel)
        left_layout.addWidget(self.predict_btn)
        left_layout.addStretch(1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.result_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 800])

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.addWidget(header)
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")

        self.file_panel.file_selected.connect(self._on_file_selected)
        self.file_panel.path_selected.connect(self._on_path_selected)
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        self.predict_btn.clicked.connect(self._on_predict_clicked)

    def _load_models(self):
        labels = self.model_service.list_packages()
        if not labels:
            self.model_status.setText("无模型包")
            QMessageBox.warning(
                self,
                "模型未就绪",
                "未找到模型包。请先运行 scripts/export_model_package.py 生成模型包后重试。",
            )
            return
        for label in labels:
            try:
                self.packages[label] = self.model_service.load_package(label)
            except AppError as exc:
                self.logger.error("模型包加载失败 %s: %s", label, exc.user_message)
        if not self.packages:
            self.model_status.setText("模型加载失败")
            return

        self._label_to_key = {
            display_label(key): key for key in sorted(self.packages)
        }
        self.target_combo.addItem("全部")
        self.target_combo.addItems(list(self._label_to_key))
        default = self.config.default_target
        index = self.target_combo.findText(display_label(default))
        self.target_combo.setCurrentIndex(index if index >= 0 else 0)

        test = self.selftest.run()
        if test.ok:
            self.model_status.setText("模型自检通过")
        else:
            self.model_status.setText("模型自检异常")
        for line in test.details:
            self.logger.info(line)

        self._on_target_changed()

    def _current_package(self) -> ModelPackage | None:
        label = self.target_combo.currentText()
        key = self._label_to_key.get(label, label)
        return self.packages.get(key)

    def _packages_for_current(self) -> list[ModelPackage]:
        label = self.target_combo.currentText()
        if label == "全部":
            return list(self.packages.values())
        key = self._label_to_key.get(label, label)
        pkg = self.packages.get(key)
        return [pkg] if pkg else []

    def _on_target_changed(self):
        label = self.target_combo.currentText()
        if label == "全部":
            self.info_panel.show_all_packages(list(self.packages.values()))
        else:
            pkg = self._current_package()
            if pkg is None:
                return
            self.info_panel.show_package(pkg)
        if self.raw is not None:
            self._validate_current()

    def _on_file_selected(self, path: str):
        try:
            self.raw = parse_file(path)
        except AppError as exc:
            self.raw = None
            self.file_panel.set_state("文件读取失败")
            QMessageBox.critical(self, "文件读取失败", exc.user_message)
            return
        self._validate_current()

    def _features_for_preprocess(self) -> list[str]:
        packages = list(self.packages.values())
        if not packages:
            return []
        return packages[0].features

    def _on_path_selected(self, path: str):
        """导入文件/文件夹后自动进入预处理流程。"""
        self.raw = None
        self.file_panel.set_state("正在读取数据...")
        try:
            source_type, df = read_source(path)
        except AppError as exc:
            self.file_panel.set_state("数据读取失败")
            QMessageBox.critical(self, "数据读取失败", exc.user_message)
            return

        features = self._features_for_preprocess()
        if not features:
            QMessageBox.warning(self, "模型未就绪", "请先生成模型包")
            return

        columns = [str(c) for c in df.columns]
        suggestions = resolve_column_mapping(df, features)
        last_settings = None
        while True:
            setup = PreprocessSetupDialog(
                path,
                source_type,
                columns,
                features,
                suggestions,
                previous_settings=last_settings,
                parent=self,
            )
            if setup.exec() != QDialog.Accepted:
                self.file_panel.set_state("已取消预处理")
                return
            settings = setup.settings()
            last_settings = settings
            try:
                result = clean_data(df, features, settings)
            except AppError as exc:
                self.file_panel.set_state("预处理失败")
                QMessageBox.critical(self, "预处理失败", exc.user_message)
                return

            if result.report.high_missing_features:
                text = (
                    "以下特征缺失率较高：\n"
                    + "、".join(result.report.high_missing_features)
                    + "\n\n是否仍然继续预测？"
                )
                choice = QMessageBox.question(
                    self,
                    "高缺失率特征",
                    text,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if choice != QMessageBox.Yes:
                    continue

            report_dialog = PreprocessReportDialog(path, result, self)
            if report_dialog.exec() != QDialog.Accepted:
                self.file_panel.set_state("已返回调整")
                continue
            self.raw = RawWaterData(path=Path(path), df=result.df)
            self.file_panel.set_state(f"预处理完成：{result.report.final_rows} 行")
            self.logger.info(
                "预处理完成 rows=%d outliers=%d interpolated=%d",
                result.report.final_rows,
                sum(result.report.outlier_counts.values()),
                sum(result.report.interpolated_counts.values()),
            )
            self._validate_current()
            return

    def _validate_current(self):
        if self.raw is None:
            return
        packages = self._packages_for_current()
        if not packages:
            return
        results = [validate(self.raw, pkg) for pkg in packages]
        ok = all(r.ok for r in results)
        errors = [e for r in results for e in r.errors]
        warnings = [w for r in results for w in r.warnings]
        if ok:
            self.file_panel.set_state("文件校验通过")
            self.logger.info("文件校验通过: %s", self.raw.path.name)
        else:
            self.file_panel.set_state("文件校验未通过")
            self.logger.warning("文件校验失败: %s", "；".join(errors))
            QMessageBox.warning(self, "数据校验未通过", "\n".join(errors))
        if warnings:
            self.statusBar().showMessage("；".join(warnings))

    def _on_predict_clicked(self):
        if self.worker is not None and self.worker.isRunning():
            return
        if self.raw is None:
            QMessageBox.information(self, "提示", "请先选择水质数据文件")
            return
        packages = self._packages_for_current()
        if not packages:
            QMessageBox.warning(self, "提示", "没有可用的模型包")
            return
        checks = [validate(self.raw, pkg) for pkg in packages]
        if not all(c.ok for c in checks):
            errors = [e for c in checks for e in c.errors]
            QMessageBox.warning(self, "数据校验未通过", "\n".join(errors))
            return

        self.predict_btn.setEnabled(False)
        self.predict_btn.setText("预测中...")
        self.statusBar().showMessage("正在预测，请稍候")
        self.worker = PredictionWorker(self.engine, self.raw, packages, self)
        self.worker.finished_ok.connect(self._on_prediction_ok)
        self.worker.failed.connect(self._on_prediction_failed)
        self.worker.start()

    def _on_prediction_ok(self, result):
        self.predict_btn.setEnabled(True)
        self.predict_btn.setText("开始预测")
        if isinstance(result, CombinedResult):
            self.result_panel.show_all(result)
            total = sum(len(r.predictions) for r in result.results)
            skipped = sum(r.skipped for r in result.results)
            labels = "、".join(r.target_label for r in result.results)
            self.statusBar().showMessage(
                f"预测完成（全部目标）：{labels}，共 {total} 条，跳过 {skipped} 条"
            )
            self.logger.info(
                "预测完成 targets=%s rows=%d skipped=%d",
                labels,
                total,
                skipped,
            )
        else:
            self.result_panel.show_result(result)
            self.statusBar().showMessage(
                f"预测完成：{result.target_label}，共 {len(result.predictions)} 条，跳过 {result.skipped} 条"
            )
            self.logger.info(
                "预测完成 label=%s rows=%d skipped=%d model=%s",
                result.target_label,
                len(result.predictions),
                result.skipped,
                result.model_id,
            )

    def _on_prediction_failed(self, message: str):
        self.predict_btn.setEnabled(True)
        self.predict_btn.setText("开始预测")
        self.logger.error("预测失败: %s", message)
        QMessageBox.critical(self, "预测失败", message)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(2000)
        event.accept()
