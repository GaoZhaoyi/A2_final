from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create training arguments for NLLB ultra-conservative fine-tuning.
    极限保守策略：学习率5e-8，避免破坏NLLB预训练知识
    目标：在NLLB零样本基础上保持或提升性能

    Returns:
        TrainingArguments instance。

    NOTE: You are free to change this。
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        learning_rate=5e-8,  # 极限小学习率，最后尝试
        per_device_train_batch_size=8,   # 较小batch，mBART较大
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,   # 有效batch=64
        weight_decay=0.02,  # 增强正则化
        save_strategy="no",  # 禁用checkpoint保存，节省磁盘空间
        num_train_epochs=1,  # 只1轮，避免过度训练
        predict_with_generate=True,
        fp16=False,
        bf16=True,  # RTX 4080S支持BF16
        logging_steps=100,
        eval_steps=125,  # 1万样本约156步，评估一次
        load_best_model_at_end=False,  # 不保存checkpoint时无法加载最佳模型
        warmup_ratio=0.2,   # 更多的warmup，保护预训练模型
        lr_scheduler_type="linear",  # linear调度器更稳定
        seed=42,
        report_to="none",
        label_smoothing_factor=0.1,  # 适度label smoothing
        generation_max_length=128,
        generation_num_beams=4,
        
        # 基本配置
        max_grad_norm=1.0,
        
        # 系统优化
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    return training_args


def create_data_collator(tokenizer, model):
    """
    Create custom data collator for NLLB model.
    NLLB需要保留手动创建的decoder_input_ids，不能让DataCollator自动生成。
    
    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        Custom data collator function.

    NOTE: You are free to change this.
    """
    import torch
    
    def custom_collator(features):
        """自定义collator，保留decoder_input_ids"""
        # 提取所有字段
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        decoder_input_ids = [f["decoder_input_ids"] for f in features]
        
        # Pad input_ids
        max_input_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        attention_mask = []
        for ids in input_ids:
            padding_len = max_input_len - len(ids)
            padded_input_ids.append(ids + [tokenizer.pad_token_id] * padding_len)
            attention_mask.append([1] * len(ids) + [0] * padding_len)
        
        # Pad decoder_input_ids
        max_decoder_len = max(len(ids) for ids in decoder_input_ids)
        padded_decoder_input_ids = []
        for ids in decoder_input_ids:
            padding_len = max_decoder_len - len(ids)
            padded_decoder_input_ids.append(ids + [tokenizer.pad_token_id] * padding_len)
        
        # Pad labels (use -100 for padding)
        max_label_len = max(len(ids) for ids in labels)
        padded_labels = []
        for ids in labels:
            padding_len = max_label_len - len(ids)
            padded_labels.append(ids + [-100] * padding_len)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "decoder_input_ids": torch.tensor(padded_decoder_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
    
    return custom_collator


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.
    包含TestBLEUCallback，在每次eval时同时计算test_bleu

    Args:
        model: Model for sequence-to-sequence tasks.
        tokenizer: Tokenizer object.
        tokenized_datasets: Tokenized datasets.

    Returns:
        Trainer object for training and evaluation.

    NOTE: You are free to change this. But make sure the trainer is the same as the model.
    """
    data_collator = create_data_collator(tokenizer, model)
    training_args: TrainingArguments = create_training_arguments()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )
    
    return trainer
