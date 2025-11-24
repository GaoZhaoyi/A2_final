import os
# 消除警告信息
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 消除tokenizers并行警告
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 消除CUDA确定性警告

from pathlib import Path
from dataset import build_dataset, preprocess_data
from model import initialize_model, initialize_tokenizer
from trainer import build_trainer
from utils import not_change_test_dataset, set_random_seeds
from constants import OUTPUT_DIR


def get_latest_checkpoint(output_dir):
    """
    获取最新的checkpoint路径，用于断点续训。
    
    Args:
        output_dir: 输出目录路径
    
    Returns:
        str or None: 最新checkpoint的路径，如果不存在则返回None
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    # 查找所有checkpoint目录
    checkpoints = [
        d for d in output_path.iterdir() 
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    
    if not checkpoints:
        return None
    
    # 按checkpoint编号排序，获取最新的
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    latest = checkpoints[-1]
    
    print(f"\n🔄 检测到checkpoint: {latest.name}")
    print(f"   将从此处继续训练...")
    return str(latest)


def main():
    """
    Main function to execute model training and evaluation.
    支持断点续训：如果检测到checkpoint，会自动从最新checkpoint继续训练。
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
    
    # 检测是否有checkpoint可以恢复
    resume_from_checkpoint = get_latest_checkpoint(OUTPUT_DIR)
    
    if resume_from_checkpoint:
        print(f"✅ 从checkpoint恢复训练")
    else:
        print(f"🆕 开始新的训练")
    
    # 开始训练（如果有checkpoint会自动恢复）
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Evaluate the model on the test dataset
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="test",
    )
    print("Test Metrics:", test_metrics)


if __name__ == "__main__":
    main()