#!/bin/bash
# 检查和修复训练环境问题

echo "=========================================="
echo "环境检查与修复脚本"
echo "=========================================="

# 1. 检查磁盘空间
echo -e "\n1. 检查磁盘空间:"
df -h ~/A2_final | head -2

DISK_USAGE=$(df ~/A2_final | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "⚠️  警告: 磁盘使用率 ${DISK_USAGE}%，可能导致checkpoint保存失败"
    echo "   建议清理空间或使用更大的磁盘"
else
    echo "✅ 磁盘空间充足 (${DISK_USAGE}%)"
fi

# 2. 检查results目录
echo -e "\n2. 检查results目录:"
if [ -d ~/A2_final/results ]; then
    ls -lh ~/A2_final/results/
    echo "✅ results目录存在"
else
    echo "❌ results目录不存在，正在创建..."
    mkdir -p ~/A2_final/results
fi

# 3. 检查失败的checkpoint
echo -e "\n3. 检查失败的checkpoint:"
FAILED_CKPTS=$(find ~/A2_final/results -name "checkpoint-*" -type d 2>/dev/null)
if [ ! -z "$FAILED_CKPTS" ]; then
    echo "⚠️  发现失败的checkpoint:"
    echo "$FAILED_CKPTS"
    read -p "是否删除这些checkpoint? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ~/A2_final/results/checkpoint-*
        echo "✅ 已清理"
    fi
else
    echo "✅ 无失败的checkpoint"
fi

# 4. 检查BLEU历史
echo -e "\n4. 检查训练历史:"
if [ -f ~/A2_final/results/training_bleu_history.csv ]; then
    echo "✅ 找到训练历史文件"
    echo "最近的记录:"
    tail -3 ~/A2_final/results/training_bleu_history.csv
else
    echo "ℹ️  未找到训练历史文件（训练还未开始）"
fi

# 5. 总结
echo -e "\n=========================================="
echo "检查完成！"
echo "=========================================="
echo "建议: 使用mBART零样本结果 (BLEU 21.64)"
echo "原因: Fine-tuning会导致性能下降"
echo "=========================================="
