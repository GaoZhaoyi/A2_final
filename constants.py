"""
Constants for the project.
"""

# NLLB-200: Meta的多语言翻译模型，零样本BLEU 25.04
MODEL_CHECKPOINT: str = "facebook/nllb-200-distilled-600M"

# NLLB语言代码
SRC_LANG: str = "zho_Hans"  # 简体中文
TGT_LANG: str = "eng_Latn"  # 英文

# 序列最大长度
MAX_INPUT_LENGTH: int = 128
MAX_TARGET_LENGTH: int = 128

# Output directory
OUTPUT_DIR: str = "./results"
