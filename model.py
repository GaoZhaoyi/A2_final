from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from constants import MODEL_CHECKPOINT, SRC_LANG, TGT_LANG


def initialize_tokenizer() -> PreTrainedTokenizer:
    """
    Initialize tokenizer for M2M-100 model.
    M2M-100需要设置源语言和目标语言代码。

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
    Initialize M2M-100 model for Chinese to English translation.
    M2M-100-1.2B: 1.2B参数，Facebook大规模多语言翻译模型

    Returns:
        A model for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )
    
    # M2M-100需要设置decoder_start_token_id为eos_token_id（根据官方文档）
    # 并设置forced_bos_token_id为目标语言ID
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # M2M-100使用eos_token_id作为decoder的起始token
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    
    # 设置目标语言token（英语）
    tokenizer.tgt_lang = TGT_LANG
    model.config.forced_bos_token_id = tokenizer.get_lang_id(TGT_LANG)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    return model
