QSS = """
QMainWindow {
    background: #f2f5f9;
}
QWidget {
    font-family: "Microsoft YaHei";
    font-size: 13px;
    color: #1f2d3d;
}
QFrame#Header {
    background: #1f5f9e;
    border: none;
}
QLabel#AppTitle {
    color: #ffffff;
    font-size: 18px;
    font-weight: bold;
}
QLabel#HeaderLabel {
    color: #eaf2fb;
}
QWidget#Panel {
    background: #ffffff;
    border: 1px solid #d7e0ea;
    border-radius: 6px;
}
QLabel#PanelTitle {
    font-size: 14px;
    font-weight: bold;
    color: #1f5f9e;
}
QLineEdit {
    background: #ffffff;
    border: 1px solid #c5d2df;
    border-radius: 4px;
    padding: 6px 8px;
}
QLineEdit:focus {
    border: 1px solid #1f5f9e;
}
QPushButton {
    background: #1f5f9e;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 7px 14px;
}
QPushButton:hover {
    background: #2a6fb5;
}
QPushButton:disabled {
    background: #9db8d2;
}
QPushButton#SecondaryButton {
    background: #ffffff;
    color: #1f5f9e;
    border: 1px solid #1f5f9e;
}
QComboBox {
    background: #ffffff;
    border: 1px solid #c5d2df;
    border-radius: 4px;
    padding: 5px 8px;
    min-width: 90px;
}
QTableWidget {
    background: #ffffff;
    gridline-color: #e4eaf1;
    border: 1px solid #d7e0ea;
}
QTableWidget::item:selected {
    background: rgba(31, 95, 158, 0.8);
    color: #10233a;
}
QHeaderView::section {
    background: #e8eef5;
    border: none;
    border-bottom: 1px solid #d7e0ea;
    padding: 6px;
    font-weight: bold;
}
QStatusBar {
    background: #e8eef5;
    color: #3a4a5c;
}
QWidget#SearchBar {
    background: #e3eef9;
    border: 1px solid #c5d7e9;
    border-radius: 4px;
}
QLabel#SearchHint {
    color: #2a5c8f;
}
QDialog {
    background: #f2f5f9;
}
"""
