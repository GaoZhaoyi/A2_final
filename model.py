from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel, AutoModelForSeq2SeqLM

from constants import MODEL_CHECKPOINT, SOURCE_LANG, TARGET_LANG


def initialize_tokenizer() -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """
    Initialize a tokenizer for sequence-to-sequence tasks.

    Returns:
        A tokenizer for sequence-to-sequence tasks.

    NOTE: You are free to change this. But make sure the tokenizer is the same as the model.
    """
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
        src_lang=SOURCE_LANG,
        tgt_lang=TARGET_LANG
    )
    return tokenizer


def initialize_model() -> PreTrainedModel:
    """
    Initialize a model for sequence-to-sequence tasks. You are free to change this,
    not only seq2seq models, but also other models like BERT, or even LLMs.

    Returns:
        A model for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    import torch
    
    # 直接使用device_map加载到GPU，避免meta device问题
    # 注意：确保缓存中只有model.safetensors，没有pytorch_model.bin
    if torch.cuda.is_available():
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_CHECKPOINT,
            device_map="auto",  # 自动分配到可用GPU
            torch_dtype=torch.float32  # 明确指定数据类型
        )
    else:
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_CHECKPOINT,
            low_cpu_mem_usage=False
        )
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    return model
