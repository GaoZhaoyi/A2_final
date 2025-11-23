from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create and return the training arguments for the model.

    Returns:
        Training arguments for the model.

    NOTE: You can change the training arguments as needed.
    # Below is an example of how to create training arguments. You are free to change this.
    # ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,  # 减少到2轮以控制时间（约4小时）
        per_device_train_batch_size=24,  # RTX 4080S优化：增加批次大小
        per_device_eval_batch_size=48,
        learning_rate=8e-5,  # 更高学习率以加快收敛
        weight_decay=0.01,
        warmup_ratio=0.06,  # 减少预热比例以加快训练
        logging_steps=100,
        save_steps=2000,  # 必须是eval_steps的整数倍
        eval_strategy="steps",
        eval_steps=2000,  # 减少评估频率节省时间
        save_total_limit=1,  # 只保存最佳模型
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        max_grad_norm=1.0,
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=4,  # 保持集束搜索以确保质量
        bf16=True,  # RTX 4080S使用BF16比FP16更优
        fp16=False,  # 禁用FP16
        gradient_accumulation_steps=4,  # 有效批次 = 24 * 4 = 96
        gradient_checkpointing=True,  # 16GB显存需要梯度检查点
        dataloader_num_workers=8,  # Linux下可以使用更多worker
        lr_scheduler_type="cosine",  # 余弦调度
        optim="adamw_torch_fused",  # RTX 4080S使用融合优化器更快
        report_to="none",
        label_smoothing_factor=0.1,
        ddp_find_unused_parameters=False,  # 加速训练
        dataloader_pin_memory=True,  # Linux下启用内存固定
    )

    return training_args


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.
    Optimized for RTX 4080S with BF16.

    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.

    NOTE: You are free to change this. But make sure the data collator is the same as the model.
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        padding=True,
        pad_to_multiple_of=8,  # 针对RTX 4080S张量核心优化
        label_pad_token_id=tokenizer.pad_token_id,  # 确保正确处理labels
        return_tensors="pt"
    )


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.

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

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )
