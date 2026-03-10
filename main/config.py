"""
项目运行时配置
"""

from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimized_config import (
    BASE_DIR, DATA_DIR, OUTPUT_DIR,
    MODEL_CONFIG, GATE_CONFIG, UPPER_GRAPH_CONFIG,
    TRAIN_CONFIG, LOSS_CONFIG, SUSPICIOUS_SCORE_CONFIG,
    VULNERABILITY_CONFIG, INHERITANCE_CONFIG,
    DATA_CONFIG, EVAL_CONFIG, OUTPUT_CONFIG
)

# 设备配置
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# 随机种子
RANDOM_SEED = 42

# 数据路径
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "samples"
DEFAULT_LABEL_FILE = PROJECT_ROOT / "data" / "labels.xlsx"

# 输出路径
MODEL_SAVE_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

# 确保输出目录存在
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

