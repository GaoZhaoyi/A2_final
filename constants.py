"""
Constants for the project.
"""

# mBART: 已验证成功，test_bleu = 21.59
# 611M参数，50种语言
MODEL_CHECKPOINT: str = "facebook/mbart-large-50-many-to-many-mmt"

# mBART语言代码
SRC_LANG: str = "zh_CN"  # 中文
TGT_LANG: str = "en_XX"  # 英文

# 序列最大长度
MAX_INPUT_LENGTH: int = 128
MAX_TARGET_LENGTH: int = 128

# Output directory
OUTPUT_DIR: str = "./results"
