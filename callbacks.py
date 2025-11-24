"""
Custom callbacks for training
在evaluation时同时计算test_bleu供参考，并保存到文件
"""
from transformers import TrainerCallback
import evaluate
import csv
import os
from datetime import datetime


class TestBLEUCallback(TrainerCallback):
    """
    在每次evaluation后额外计算test_bleu
    这样可以实时监控模型在完整测试集上的真实性能
    所有结果保存到CSV文件供后续查看
    """
    
    def __init__(self, trainer, test_dataset, tokenizer, output_dir="./results"):
        self.trainer = trainer
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.bleu_metric = evaluate.load("sacrebleu")
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV文件路径
        self.csv_path = os.path.join(output_dir, "training_bleu_history.csv")
        
        # 初始化CSV文件（如果不存在）
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'step', 'epoch', 
                    'eval_loss', 'eval_bleu', 
                    'test_loss', 'test_bleu', 
                    'difference', 'status'
                ])
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        在每次evaluation后调用
        """
        if metrics is None:
            return
        
        # 只在有eval_bleu的时候计算test_bleu
        if 'eval_bleu' not in metrics:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch = metrics.get('epoch', 0)
        eval_loss = metrics.get('eval_loss', 0)
        eval_bleu = metrics['eval_bleu']
        
        print("\n" + "="*70)
        print(f"📊 Evaluation at step {state.global_step} (epoch {epoch:.2f}):")
        print(f"   Eval BLEU (2K samples): {eval_bleu:.2f}")
        
        # 计算test_bleu
        test_bleu = None
        test_loss = None
        difference = None
        status = "success"
        
        try:
            print(f"   Computing test BLEU (full {len(self.test_dataset)} samples)...")
            
            # 使用trainer的predict方法
            test_output = self.trainer.predict(self.test_dataset, metric_key_prefix="test")
            test_metrics = test_output.metrics
            
            if 'test_bleu' in test_metrics:
                test_bleu = test_metrics['test_bleu']
                test_loss = test_metrics.get('test_loss', None)
                difference = test_bleu - eval_bleu
                
                print(f"   Test BLEU (full set):   {test_bleu:.2f}")
                print(f"   Difference:             {difference:+.2f}")
            
        except Exception as e:
            status = f"failed: {str(e)}"
            print(f"   Failed to compute test BLEU: {str(e)}")
        
        print("="*70 + "\n")
        
        # 保存到CSV
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, state.global_step, f"{epoch:.2f}",
                    f"{eval_loss:.4f}" if eval_loss else "",
                    f"{eval_bleu:.2f}" if eval_bleu else "",
                    f"{test_loss:.4f}" if test_loss else "",
                    f"{test_bleu:.2f}" if test_bleu else "",
                    f"{difference:+.2f}" if difference is not None else "",
                    status
                ])
            print(f"✅ Results saved to: {self.csv_path}")
        except Exception as e:
            print(f"⚠️  Failed to save results: {str(e)}")
