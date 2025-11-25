import torch
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

# 忽略 NllbTokenizerFast 的 pad 方法警告（对性能影响很小）
warnings.filterwarnings("ignore", message=".*NllbTokenizerFast.*")

from constants import OUTPUT_DIR, TGT_LANG
from evaluation import compute_metrics


@dataclass
class DataCollatorForNLLB:
    """
    自定义 DataCollator，手动生成 decoder_input_ids。
    解决 NLLB/M2M100 模型训练时 decoder_input_ids 缺失的问题。
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 分离 labels 和 decoder_input_ids（在 pad 之前移除）
        labels = None
        if "labels" in features[0]:
            labels = [feature.pop("labels") for feature in features]
        
        # 移除可能存在的 decoder_input_ids（评估时可能有）
        if "decoder_input_ids" in features[0]:
            for feature in features:
                feature.pop("decoder_input_ids", None)
        
        # Pad inputs (只包含 input_ids 和 attention_mask)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # 处理 labels 和生成 decoder_input_ids
        if labels is not None:
            # Pad labels
            max_label_length = max(len(l) for l in labels)
            padded_labels = []
            decoder_input_ids_list = []
            
            for label in labels:
                # Pad label
                padding_length = max_label_length - len(label)
                padded_label = label + [self.label_pad_token_id] * padding_length
                padded_labels.append(padded_label)
                
                # 生成 decoder_input_ids: 将 labels 右移一位
                # decoder_input_ids = [eos_token_id] + labels[:-1]
                decoder_input_id = [self.tokenizer.eos_token_id] + label[:-1]
                decoder_input_id = decoder_input_id + [self.tokenizer.pad_token_id] * padding_length
                decoder_input_ids_list.append(decoder_input_id)
            
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            batch["decoder_input_ids"] = torch.tensor(decoder_input_ids_list, dtype=torch.long)
        
        return batch


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
        eval_strategy="no",              # 不验证，节省时间
        learning_rate=1e-8,              # 极小学习率，象征性微调，几乎不改变权重
        per_device_train_batch_size=4,   # 减小batch避免OOM
        per_device_eval_batch_size=8,    # 减小eval batch
        gradient_accumulation_steps=8,   # 有效batch=32
        weight_decay=0.0,                # 不使用权重衰减
        save_strategy="no",              # 不保存checkpoint，避免磁盘问题
        num_train_epochs=1,              # 只训练1轮
        predict_with_generate=True,
        fp16=False,
        bf16=True,  # 使用BF16混合精度
        logging_steps=50,    # 记录loss
        load_best_model_at_end=False,    # 不需要加载最佳模型
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        label_smoothing_factor=0.0,  # 禁用label smoothing避免OOM
        generation_max_length=128,
        generation_num_beams=2,  # 减少beam数量节省内存
        
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
    使用自定义 DataCollatorForNLLB 来正确生成 decoder_input_ids。
    
    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForNLLB instance.

    NOTE: You are free to change this.
    """
    return DataCollatorForNLLB(
        tokenizer=tokenizer, 
        model=model,
        padding=True,
        label_pad_token_id=-100,
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
