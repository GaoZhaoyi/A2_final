from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import DataCollatorForSeq2Seq

from constants import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset for Chinese to English translation.
    针对RTX 4080S优化，使用WMT19数据集的合理子集。

    Returns:
        The dataset.

    NOTE: You can replace this with your own dataset. Make sure to include
    the `validation` split and ensure that it is the same as the test split from the WMT19 dataset.
    """
    # Load WMT19 zh-en dataset
    wmt19 = load_dataset("wmt19", "zh-en")
    
    # 修复验证集问题：使用官方验证集，避免虚高的eval_bleu
    # mBART超保守策略：极小学习率 + 极少数据，微调不破坏预训练
    # 零样本21.64，目标保持或轻微提升到22+
    total_train_size = 10000  # 回到1万样本，避免过度微调
    validation_size = 500     # 验证集只作为训练参考，不需要太大
    
    # 随机采样1万条作为训练集，避免顺序选取的分布偏差
    import random
    random.seed(42)
    train_indices = random.sample(range(len(wmt19["train"])), total_train_size)
    train_dataset = wmt19["train"].select(train_indices)
    
    # 使用官方验证集随机抽取500条，避免位置偏差
    import random
    random.seed(42)  # 保证可复现
    val_indices = random.sample(range(len(wmt19["validation"])), validation_size)
    validation_dataset = wmt19["validation"].select(val_indices)
    
    print(f"训练样本数: {len(train_dataset):,}")
    print(f"验证样本数: {len(validation_dataset):,}")
    print(f"测试样本数: {len(wmt19['validation']):,} (完整官方验证集)")

    # 测试集保持不变
    test_dataset = wmt19["validation"]
    
    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })


def create_data_collator(tokenizer, model):
    """
    Create standard data collator for MarianMT model.
    
    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    """
    Preprocess the data for NLLB/M2M100 model.
    直接使用 tokenizer 的 text_target 参数，确保正确生成所有必要字段。

    Args:
        examples: Examples.
        tokenizer: Tokenizer object.
        max_input_length: Maximum input length.
        max_target_length: Maximum target length.

    Returns:
        Model inputs.
    """
    # 提取中文输入和英文目标
    inputs = [ex["zh"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True,
        padding=False  # DataCollator会处理padding
    )
    
    # Tokenize targets (使用 text_target 参数)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length, 
            truncation=True,
            padding=False
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.

    Args:
        raw_datasets: Raw datasets.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized datasets.
    """
    tokenized_datasets: DatasetDict = raw_datasets.map(
        function=lambda examples: preprocess_function(
            examples=examples,
            tokenizer=tokenizer,
            max_input_length=MAX_INPUT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    return tokenized_datasets
