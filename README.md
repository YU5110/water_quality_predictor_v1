# 污水厂出水水质预测软件 v1

基于已有 LSTM 训练产物构建的本地 Windows 桌面软件。界面为中文，数据只在本机处理，不联网、不上传。

## 功能

- 导入统一格式的 Excel / CSV 水质时间序列文件（`datetime` + 特征列）
- 支持导入进口/出口 Excel 文件夹，自动按时间合并
- 数据预处理：列名别名自动匹配与手动映射、时间对齐与补全、物理范围 + Z-score 异常值清洗、缺失值插值、hour/month 自动补充
- 预处理报告：异常/插补/缺失数量、高缺失特征提示；预览表和导出的 Excel 用颜色标记（黄=异常值、浅蓝=插补值、红=仍缺失）
- 在“全部”或 COD、TP、TN、NH3N 四个出水指标间切换预测目标，可一次性输出所有指标
- 使用真实 LSTM 模型包预测并显示数值表格、实际/预测对比曲线
- 表格支持按时间（年/月/日/小时）搜索、单元格选中复制；曲线支持滚轮缩放、左键拖动和复原
- 启动自检：用模型包内真实样本验证模型可加载且预测一致
- 本地滚动日志，异常以中文提示，不崩溃退出

## 目录结构

```text
water_quality_predictor/
├─ app.py                    # 启动入口
├─ config.yaml               # 软件配置
├─ requirements.txt
├─ models/                   # 模型包（导出脚本生成）
├─ src/
│  ├─ core/                  # 解析、校验、预处理、推理
│  ├─ model/                 # LSTM 结构、模型包服务
│  ├─ ui/                    # PySide6 界面
│  ├─ config.py
│  ├─ errors.py
│  ├─ logging_service.py
│  └─ selftest.py
├─ scripts/
│  └─ export_model_package.py
└─ tests/
```

## 快速开始

1. 安装依赖：

```powershell
pip install -r requirements.txt
```

2. 生成模型包（读取训练流水线产物，输出到 `models/`）：

```powershell
python scripts/export_model_package.py
```

3. 启动软件：

```powershell
python app.py
```

4. 导入 `../数据预处理/预处理后水质数据.xlsx`（或同格式 CSV），选择预测目标后点击“开始预测”。

## 数据格式

文件需要包含：

- 时间列：`datetime`，可被 pandas 解析（如 `2024-06-01 00:00`）
- 模型特征列：以 `models/<目标>/metadata.json` 中 `features` 为准

缺列、缺历史窗口、数值无法解析都会给出中文提示；窗口含缺失值的序列会跳过并在状态栏显示跳过数量。

## 与训练端的一致性

推理与 `LSTM模型预测污水厂出水水质.py` 使用同一套口径：

- 30 天因果滚动归一化（`closed="left"`，只用过去数据）
- 按训练集拟合的特征均值/标准差标准化
- 与训练相同的 LSTM 结构、窗口长度和缺失跳过规则
- 预测完成后按目标列的滚动均值/标准差还原为原始量纲

## 后续规划（v2+）

- 导出 Excel / CSV / 图片报告
- 排放限值达标判断与预警
- 预测历史记录
- 模型包管理：不同水厂切换/更新模型
- 软件内重新训练 LSTM 并保存为模型包
