import os
# 消除警告信息
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 消除tokenizers并行警告
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 消除CUDA确定性警告

from dataset import build_dataset, preprocess_data
from model import initialize_model, initialize_tokenizer
from trainer import build_trainer
from utils import not_change_test_dataset, set_random_seeds


def main():
    """
    Main function to execute model training and evaluation.
    """
    # Set random seeds for reproducibility
    set_random_seeds()

    # Initialize tokenizer and model
    model = initialize_model()

    # Initialize tokenizer
    tokenizer = initialize_tokenizer()

    raw_datasets = build_dataset()

    assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"

    # Load and preprocess datasets
    tokenized_datasets = preprocess_data(raw_datasets, tokenizer)

    # Build and train the model
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
    )
    
    # 训练前先评估零样本 BLEU（作为基准）
    print("\n" + "="*50)
    print("Zero-shot evaluation (before training):")
    print("="*50)
    zero_shot_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="zero_shot",
    )
    print(f"Zero-shot BLEU: {zero_shot_metrics['zero_shot_bleu']:.2f}")
    print("="*50 + "\n")
    
    # Train the model
    trainer.train()

    # Evaluate the model on the test dataset
    print("\n" + "="*50)
    print("Fine-tuned evaluation (after training):")
    print("="*50)
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="test",
    )
    print(f"Fine-tuned BLEU: {test_metrics['test_bleu']:.2f}")
    print("="*50)
    
    # 对比
    print(f"\nBLEU Change: {zero_shot_metrics['zero_shot_bleu']:.2f} -> {test_metrics['test_bleu']:.2f} ({test_metrics['test_bleu'] - zero_shot_metrics['zero_shot_bleu']:+.2f})")


if __name__ == "__main__":
    main()