from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from constants import MODEL_CHECKPOINT, SRC_LANG, TGT_LANG


def initialize_tokenizer() -> PreTrainedTokenizer:
    """
    Initialize tokenizer for NLLB model.
    NLLB-200需要设置源语言和目标语言代码。

    Returns:
        A tokenizer for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
        src_lang=SRC_LANG,  # 简体中文
        tgt_lang=TGT_LANG,  # 英文
    )
    return tokenizer


def initialize_model() -> PreTrainedModel:
    """
    Initialize NLLB model for Chinese to English translation.
    NLLB-200-distilled-600M: 600M参数，Meta 2022最新多语言翻译模型

    Returns:
        A model for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )
    
    # Enable gradient checkpointing for memory efficiency on RTX 4080S
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    return model
