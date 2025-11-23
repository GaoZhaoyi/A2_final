"""
Custom callbacks for training
在evaluation时同时计算test_bleu供参考
"""
from transformers import TrainerCallback
import evaluate


class TestBLEUCallback(TrainerCallback):
    """
    在每次evaluation后额外计算test_bleu
    这样可以实时监控模型在完整测试集上的真实性能
    """
    
    def __init__(self, trainer, test_dataset, tokenizer):
        self.trainer = trainer
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.bleu_metric = evaluate.load("sacrebleu")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        在每次evaluation后调用
        """
        if metrics is None:
            return
        
        # 只在有eval_bleu的时候计算test_bleu
        if 'eval_bleu' not in metrics:
            return
        
        print("\n" + "="*70)
        print(f"📊 Evaluation at step {state.global_step} (epoch {metrics.get('epoch', 0):.2f}):")
        print(f"   Eval BLEU (2K samples): {metrics['eval_bleu']:.2f}")
        
        # 计算test_bleu
        try:
            print(f"   Computing test BLEU (full {len(self.test_dataset)} samples)...")
            
            # 使用trainer的predict方法
            test_output = self.trainer.predict(self.test_dataset, metric_key_prefix="test")
            test_metrics = test_output.metrics
            
            if 'test_bleu' in test_metrics:
                print(f"   Test BLEU (full set):   {test_metrics['test_bleu']:.2f}")
                print(f"   Difference:             {test_metrics['test_bleu'] - metrics['eval_bleu']:.2f}")
            
        except Exception as e:
            print(f"   Failed to compute test BLEU: {str(e)}")
        
        print("="*70 + "\n")
