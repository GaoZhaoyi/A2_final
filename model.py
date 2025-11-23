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
    
    # 首先尝试使用safetensors加载（避免PyTorch版本检查）
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
        use_safetensors=True,
        low_cpu_mem_usage=False
    )
    
    # 立即将模型移到GPU，避免Trainer初始化时的meta device问题
    if torch.cuda.is_available():
        # 检查是否有meta device上的参数
        has_meta = any(p.device.type == 'meta' for p in model.parameters())
        if has_meta:
            # 如果有meta参数，先移到CPU再移到GPU
            device = torch.device("cuda")
            model = model.to_empty(device=device)
            # 重新加载权重到GPU
            model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path=MODEL_CHECKPOINT,
                use_safetensors=True,
                device_map={"": 0}  # 直接加载到GPU 0
            )
        else:
            # 没有meta参数，直接移到GPU
            model = model.cuda()
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    return model
