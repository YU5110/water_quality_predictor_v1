import sys

from PySide6.QtWidgets import QApplication

from src.config import AppConfig, project_root
from src.logging_service import setup_logging
from src.model.model_package import ModelPackageService
from src.selftest import SelfTestService
from src.ui.main_window import MainWindow


def main() -> int:
    root = project_root()
    config = AppConfig.load(root / "config.yaml")
    setup_logging(config.logs_dir)

    app = QApplication(sys.argv)
    app.setApplicationName(config.app_title)
    app.setApplicationVersion(config.app_version)

    model_service = ModelPackageService(config.package_dir)
    selftest = SelfTestService(model_service)
    window = MainWindow(config, model_service, selftest)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
