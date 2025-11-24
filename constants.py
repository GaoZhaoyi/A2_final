"""
Constants for the project.
"""

# mT5-base: Google多语言T5模型
# 580M参数，101种语言，架构稳定，兼容性好
MODEL_CHECKPOINT: str = "google/mt5-base"

# mT5不需要语言代码，而是使用提示词前缀
SRC_LANG: str = "zh"  # 仅用于数据处理
TGT_LANG: str = "en"  # 仅用于数据处理

# T5特定的前缀
PREFIX = "translate Chinese to English: "

# 序列最大长度
MAX_INPUT_LENGTH: int = 128
MAX_TARGET_LENGTH: int = 128
    
# Output directory
OUTPUT_DIR: str = "./results"
