"""
查看训练过程中的BLEU历史记录
从CSV文件读取并可视化训练进度
"""
import csv
import os
from pathlib import Path

def view_training_history(csv_path="./results/training_bleu_history.csv"):
    """
    读取并显示训练历史
    """
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        print(f"   Training history will be saved here once training starts.")
        return
    
    print("=" * 80)
    print("📊 Training BLEU History")
    print("=" * 80)
    print()
    
    # 读取CSV
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("⚠️  No training data found yet.")
        return
    
    # 显示表格
    print(f"{'Epoch':<8} {'Step':<8} {'Eval BLEU':<12} {'Test BLEU':<12} {'Diff':<8} {'Time':<20}")
    print("-" * 80)
    
    for row in rows:
        epoch = row.get('epoch', 'N/A')
        step = row.get('step', 'N/A')
        eval_bleu = row.get('eval_bleu', 'N/A')
        test_bleu = row.get('test_bleu', 'N/A')
        diff = row.get('difference', 'N/A')
        timestamp = row.get('timestamp', 'N/A')
        
        print(f"{epoch:<8} {step:<8} {eval_bleu:<12} {test_bleu:<12} {diff:<8} {timestamp:<20}")
    
    print("-" * 80)
    print(f"Total evaluations: {len(rows)}")
    
    # 统计信息
    if rows:
        latest = rows[-1]
        best_eval = max((float(r['eval_bleu']) for r in rows if r['eval_bleu']), default=0)
        best_test = max((float(r['test_bleu']) for r in rows if r['test_bleu']), default=0)
        
        print()
        print("📈 Summary:")
        print(f"   Latest Eval BLEU:  {latest.get('eval_bleu', 'N/A')}")
        print(f"   Latest Test BLEU:  {latest.get('test_bleu', 'N/A')}")
        print(f"   Best Eval BLEU:    {best_eval:.2f}")
        print(f"   Best Test BLEU:    {best_test:.2f}")
        print()
        
        # 趋势判断
        if len(rows) >= 2:
            try:
                recent_test = [float(r['test_bleu']) for r in rows[-3:] if r['test_bleu']]
                if len(recent_test) >= 2:
                    if recent_test[-1] > recent_test[0]:
                        print("✅ Trend: Improving")
                    elif recent_test[-1] < recent_test[0]:
                        print("⚠️  Trend: Declining (possible overfitting)")
                    else:
                        print("➡️  Trend: Stable")
            except:
                pass
    
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数指定路径
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "./results/training_bleu_history.csv"
    
    view_training_history(csv_path)
