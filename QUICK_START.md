# 快速入门指南

## 前置条件
确保已安装所有依赖：
```bash
uv sync
# 或
pip install -e .
```

## 测试设置（强烈推荐第一步）
在运行完整训练之前，先测试你的设置：
```bash
python test_setup.py
```

这将验证：
- ✅ 所有依赖已安装
- ✅ 模型和分词器正确加载
- ✅ 数据集可访问
- ✅ 测试数据集未被修改
- ✅ GPU可用且配置正确
- ✅ 前向传播正常工作

## 运行训练
```bash
# 标准运行
python main.py

# 使用uv运行
uv run python main.py

# 监控GPU使用情况（在另一个终端）
nvidia-smi -l 1
```

## 预期输出
```
加载数据集...
总训练样本数: 5,500,000+
预计训练时间: 约3.5-4小时 (RTX 4080S, 2轮)
模型: facebook/nllb-200-distilled-600M
使用BF16训练，梯度检查点已启用

第1轮/共2轮:
  [训练进度，每2000步显示BLEU分数]
  步数 2000:  BLEU ~19-20
  步数 10000: BLEU ~22-23
  步数 30000: BLEU ~23-24 (第1轮结束)
  
第2轮/共2轮:
  [继续训练...]
  步数 60000:  BLEU ~24-25
  步数 90000:  BLEU ~25-26
  步数 114000: BLEU ~25-27 (第2轮结束)

加载最佳模型...
测试指标: {'test_bleu': 25.x-27.x, ...}
```

## 训练时间估计
- **✅ RTX 4080S (16GB) - 当前优化配置**: 约3.5-4小时
- **单张V100 (16GB)**: 约5-6小时（使用当前配置）
- **单张A100 (40GB)**: 约3-4小时（使用当前配置）
- **单张RTX 3090/4090**: 约4-5小时（使用当前配置）

## 内存要求
- **GPU显存**: 14-16GB（使用梯度检查点 + BF16）
- **系统内存**: 推荐32GB+
- **硬盘空间**: 约50GB（数据集 + 检查点）

## 💡 RTX 4080S 专用优化
当前配置已针对 **RTX 4080S (16GB)** 在 **Linux** 系统下优化：
- ✅ 使用BF16混合精度（比FP16更适合RTX 40系列）
- ✅ 优化批次大小：24（配合4步梯度累积）
- ✅ 训练2轮（约550万样本）
- ✅ 预计时间：3.5-4小时
- ✅ 预期BLEU：25-27

详见 `RTX4080S_配置说明.md`

## 如果遇到显存不足（OOM）
编辑 `trainer.py`：
```python
per_device_train_batch_size=8,  # 从16减少
gradient_accumulation_steps=16,  # 从8增加
```

## 如果训练太慢
编辑 `constants.py` 或 `dataset.py`：
- 使用较小的数据集子集
- 将MAX_INPUT_LENGTH减少到192
- 将num_train_epochs减少到2

## 各轮预期BLEU分数（RTX 4080S配置）
- **第1轮后（约2小时）**: 约23-24 BLEU
- **第2轮后（约4小时）**: 约25-27 BLEU ✅ 达标！

## 监控训练
训练器将：
- 每100步记录日志
- 每2000步评估一次（优化以节省时间）
- 每3000步保存检查点
- 保留最佳的1个检查点（节省磁盘空间）

## Linux下监控GPU（推荐）
```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# 或查看温度和功耗
nvidia-smi dmon -s puct -d 2
```

## 输出文件
```
results/
├── checkpoint-2000/
├── checkpoint-4000/
├── checkpoint-best/  (BLEU最佳模型)
├── trainer_state.json
└── training_args.bin
```

## 问题排查

### 问题：数据集下载很慢
**解决方案**：HuggingFace数据集首次下载后会本地缓存。请等待初始下载完成。

### 问题：CUDA显存不足
**解决方案**：减少批次大小或增加梯度累积步数（见上文）。

### 问题：BLEU分数很低（<20）
**解决方案**： 
- 确保训练完成所有3轮
- 检查FP16是否已启用
- 验证是否在使用GPU

### 问题：测试数据集验证失败
**解决方案**：不要在dataset.py中修改测试数据集。WMT19的验证集必须保持不变。

## 提交作业
使用你的学号创建zip文件：
```bash
# Windows PowerShell
Compress-Archive -Path dataset.py,model.py,trainer.py,evaluation.py,constants.py,utils.py,main.py,pyproject.toml -DestinationPath 30300xxxxx.zip

# 或手动压缩这些文件：
# - main.py
# - dataset.py  
# - model.py
# - trainer.py
# - evaluation.py
# - utils.py
# - constants.py
# - pyproject.toml
```

**重要**：不要包含：
- 大型检查点文件
- 数据集缓存
- results目录
- 虚拟环境文件夹

## 联系方式
如果遇到问题，请查阅：
1. README.md - 完整文档
2. OPTIMIZATION_NOTES.md - 技术细节
3. Canvas上的课程材料
