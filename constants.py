"""
Constants for the project.
"""

# mBART-one-to-many: 尝试从多语言到英语的优化版本
# 理论上在目标语言生成上可能更专注
MODEL_CHECKPOINT: str = "facebook/mbart-large-50-one-to-many-mmt"

# mBART语言代码
SRC_LANG: str = "zh_CN"  # 中文
TGT_LANG: str = "en_XX"  # 英文

# 序列最大长度
MAX_INPUT_LENGTH: int = 128
MAX_TARGET_LENGTH: int = 128

# Output directory
OUTPUT_DIR: str = "./results"
