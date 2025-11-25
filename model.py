from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from constants import MODEL_CHECKPOINT, SRC_LANG, TGT_LANG


def initialize_tokenizer() -> PreTrainedTokenizer:
    """
    Initialize tokenizer for NLLB-200 model.
    设置源语言为简体中文。

    Returns:
        A tokenizer for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    # NLLB/M2M100 需要在初始化时传入语言参数
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
        src_lang=SRC_LANG,  # zho_Hans
        tgt_lang=TGT_LANG,  # eng_Latn
    )
    return tokenizer


def initialize_model() -> PreTrainedModel:
    """
    Initialize NLLB-200 model.
    零样本BLEU 25.04，微调后可进一步提升。

    Returns:
        A model for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )
    
    # 注意：NLLB 不启用 gradient_checkpointing，可能导致 decoder_input_ids 问题
    
    return model
