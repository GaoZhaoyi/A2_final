from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from constants import MODEL_CHECKPOINT, SRC_LANG, TGT_LANG


def initialize_tokenizer() -> PreTrainedTokenizer:
    """
    Initialize tokenizer for mBART one-to-many model.
    确保明确设置src_lang，防止默认英语。

    Returns:
        A tokenizer for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
        src_lang=SRC_LANG,  # 中文
        tgt_lang=TGT_LANG,  # 英文
    )
    return tokenizer


def initialize_model() -> PreTrainedModel:
    """
    Initialize mBART one-to-many model.
    尝试利用其更强的目标语言生成能力。

    Returns:
        A model for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    return model
