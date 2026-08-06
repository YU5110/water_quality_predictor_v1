"""确保从项目根目录导入 src 包。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
