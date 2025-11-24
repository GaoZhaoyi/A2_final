from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create training arguments for mBART one-to-many fine-tuning.
    策略：保持与many-to-many完全一致的保守策略（5e-8），以便公平对比。
    
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
    Create data collator for mBART model.
    使用标准DataCollatorForSeq2Seq，会自动处理decoder_input_ids。
    
    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.

    NOTE: You are free to change this.
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        padding=True
    )


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
