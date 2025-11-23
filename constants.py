"""
Constants for the machine translation project.
针对RTX 4080S优化的中文到英文翻译配置。
"""

# Model checkpoint - 使用Helsinki-NLP的opus-mt模型，专门针对zh->en翻译
# 优势：模型小(~300M)，训练快，架构稳定，易于达到BLEU 24-25
MODEL_CHECKPOINT: str = "Helsinki-NLP/opus-mt-zh-en"

# 序列最大长度
MAX_INPUT_LENGTH: int = 128
MAX_TARGET_LENGTH: int = 128

# Output directory
OUTPUT_DIR: str = "./results"
