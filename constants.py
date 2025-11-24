"""
Constants for the project.
"""

# M2M-100: Facebook大规模多语言翻译模型
# 1.2B参数，100种语言，比mBART更强
MODEL_CHECKPOINT: str = "facebook/m2m100_1.2B"

# M2M-100语言代码
SRC_LANG: str = "zh"  # 中文
TGT_LANG: str = "en"  # 英文

# 序列最大长度
MAX_INPUT_LENGTH: int = 128
MAX_TARGET_LENGTH: int = 128

# Output directory
OUTPUT_DIR: str = "./results"
