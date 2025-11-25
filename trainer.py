from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from constants import OUTPUT_DIR, TGT_LANG
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create training arguments for NLLB-200 fine-tuning.
    NLLB零样本BLEU 25.04，轻度微调以保持或提升性能。
    
    Returns:
        TrainingArguments instance.

    NOTE: You are free to change this.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        learning_rate=2e-5,  # NLLB标准学习率
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,   # 有效batch=32
        weight_decay=0.01,
        save_strategy="no",  # 禁用checkpoint保存，节省磁盘空间
        num_train_epochs=3,  # 3轮微调
        predict_with_generate=True,
        fp16=False,
        bf16=True,  # 使用BF16混合精度
        logging_steps=100,
        eval_steps=500,
        load_best_model_at_end=False,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        label_smoothing_factor=0.1,
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
    Create data collator for NLLB model.
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
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id,
    )


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.
    设置NLLB的forced_bos_token_id以指定目标语言。

    Args:
        model: Model for sequence-to-sequence tasks.
        tokenizer: Tokenizer object.
        tokenized_datasets: Tokenized datasets.

    Returns:
        Trainer object for training and evaluation.

    NOTE: You are free to change this. But make sure the trainer is the same as the model.
    """
    # 设置NLLB的目标语言token
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    model.config.forced_bos_token_id = forced_bos_token_id
    print(f"NLLB forced_bos_token_id: {forced_bos_token_id} ({TGT_LANG})")
    
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
