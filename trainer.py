from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create training arguments for RTX 4080S (16GB VRAM).
    针对opus-mt-zh-en模型和4小时训练目标优化。

    Returns:
        TrainingArguments instance.

    NOTE: You are free to change this.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        learning_rate=5e-5,  # opus-mt模型使用标准学习率
        per_device_train_batch_size=32,  # opus-mt更小，可以用更大batch
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,  # 有效batch=64
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,  # 3轮训练以达到BLEU 24-25
        predict_with_generate=True,
        fp16=False,
        bf16=True,  # RTX 4080S支持BF16
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=500,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        label_smoothing_factor=0.1,
        generation_max_length=128,
        generation_num_beams=4,
        
        # 系统优化
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    return training_args


def create_data_collator(tokenizer, model):
    """
    Create standard data collator for MarianMT model.
    
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
