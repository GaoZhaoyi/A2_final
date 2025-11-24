#!/bin/bash
# 运行main.py并消除警告信息

# 消除tokenizers并行警告
export TOKENIZERS_PARALLELISM=false

# 消除CUDA确定性警告
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 运行训练
python main.py
