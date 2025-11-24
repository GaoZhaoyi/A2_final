from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from constants import MODEL_CHECKPOINT


def initialize_tokenizer() -> PreTrainedTokenizer:
    """
    Initialize tokenizer for mT5 model.
    mT5不需要设置源语言和目标语言代码，而是使用前缀。

    Returns:
        A tokenizer for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
    )
    return tokenizer


def initialize_model() -> PreTrainedModel:
    """
    Initialize mT5 model for Chinese to English translation.
    mT5-base: 580M参数，Google多语言T5模型

    Returns:
        A model for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )
    
    # mT5不需要特殊的decoder设置，它使用标准的T5架构
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    return model
