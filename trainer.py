from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create training arguments for mT5 fine-tuning.
    mT5-base策略：使用标准学习率5e-5，T5架构通常比mBART更稳定。
    
    Returns:
        TrainingArguments instance。

    NOTE: You are free to change this。
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        learning_rate=5e-5,  # mT5标准学习率
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        save_strategy="no",
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=False,
        bf16=True,
        logging_steps=100,
        eval_steps=125,
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
    Create data collator for mT5 model.
    使用标准DataCollatorForSeq2Seq。
    
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
