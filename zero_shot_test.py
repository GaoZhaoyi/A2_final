"""
零样本测试脚本：测试多个高级翻译模型在 WMT19 zh-en 测试集上的 BLEU 分数
用于选择最佳基础模型进行微调
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import evaluate
from tqdm import tqdm
import gc

# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SAMPLES = None  # None 表示使用完整测试集 3981 条
BATCH_SIZE = 8  # 根据显存调整

# 要测试的模型列表（都是高下载量的热门模型）
MODELS_TO_TEST = [
    {
        "name": "Helsinki-NLP/opus-mt-zh-en",
        "type": "marianmt",
        "src_lang": None,  # MarianMT 不需要设置语言代码
        "tgt_lang": None,
    },
    {
        "name": "facebook/nllb-200-distilled-600M",
        "type": "nllb",
        "src_lang": "zho_Hans",  # NLLB 使用特殊语言代码
        "tgt_lang": "eng_Latn",
    },
    {
        "name": "facebook/m2m100_418M",
        "type": "m2m100",
        "src_lang": "zh",
        "tgt_lang": "en",
    },
    {
        "name": "facebook/mbart-large-50-many-to-many-mmt",
        "type": "mbart",
        "src_lang": "zh_CN",
        "tgt_lang": "en_XX",
    },
    {
        "name": "facebook/mbart-large-50-one-to-many-mmt",
        "type": "mbart",
        "src_lang": "zh_CN",
        "tgt_lang": "en_XX",
    },
]


def load_test_data(num_samples = TEST_SAMPLES):
    """加载 WMT19 测试集"""
    wmt19 = load_dataset("wmt19", "zh-en")
    if num_samples is None:
        test_data = wmt19["validation"]  # 完整测试集
        print(f"加载 WMT19 zh-en 完整测试集 ({len(test_data)} 条)...")
    else:
        test_data = wmt19["validation"].select(range(min(num_samples, len(wmt19["validation"]))))
        print(f"加载 WMT19 zh-en 测试集 (前 {num_samples} 条)...")
    
    sources = [ex["zh"] for ex in test_data["translation"]]
    references = [[ex["en"]] for ex in test_data["translation"]]
    
    print(f"测试样本数: {len(sources)}")
    print(f"示例输入: {sources[0][:50]}...")
    print(f"示例参考: {references[0][0][:50]}...")
    return sources, references


def test_model(model_config: dict, sources: list, references: list) -> dict:
    """测试单个模型的零样本翻译性能"""
    model_name = model_config["name"]
    model_type = model_config["type"]
    
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"模型类型: {model_type}")
    print(f"{'='*60}")
    
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 设置源语言（如果需要）
        if model_type == "nllb":
            tokenizer.src_lang = model_config["src_lang"]
        elif model_type == "mbart":
            tokenizer.src_lang = model_config["src_lang"]
        elif model_type == "m2m100":
            tokenizer.src_lang = model_config["src_lang"]
        
        # 加载模型
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)
        model.eval()
        
        # 获取目标语言的 forced_bos_token_id
        forced_bos_token_id = None
        if model_type == "nllb":
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(model_config["tgt_lang"])
        elif model_type == "mbart":
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(model_config["tgt_lang"])
        elif model_type == "m2m100":
            forced_bos_token_id = tokenizer.get_lang_id(model_config["tgt_lang"])
        
        print(f"forced_bos_token_id: {forced_bos_token_id}")
        
        # 批量翻译
        predictions = []
        for i in tqdm(range(0, len(sources), BATCH_SIZE), desc="翻译中"):
            batch = sources[i:i + BATCH_SIZE]
            
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(DEVICE)
            
            with torch.no_grad():
                if forced_bos_token_id is not None:
                    outputs = model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=128,
                        num_beams=4,
                    )
                else:
                    outputs = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4,
                    )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)
        
        # 计算 BLEU
        metric = evaluate.load("sacrebleu")
        result = metric.compute(predictions=predictions, references=references)
        bleu_score = result["score"]
        
        print(f"\n✓ BLEU 分数: {bleu_score:.2f}")
        print(f"示例翻译: {predictions[0][:80]}...")
        
        # 清理显存
        del model, tokenizer
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "model": model_name,
            "bleu": bleu_score,
            "status": "success",
            "sample_output": predictions[0]
        }
        
    except Exception as e:
        print(f"✗ 错误: {str(e)}")
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return {
            "model": model_name,
            "bleu": 0,
            "status": "failed",
            "error": str(e)
        }


def main():
    print("="*60)
    print("零样本翻译模型测试 (中文 → 英文)")
    print(f"设备: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("="*60)
    
    # 加载测试数据
    sources, references = load_test_data()
    
    # 测试每个模型
    results = []
    for model_config in MODELS_TO_TEST:
        result = test_model(model_config, sources, references)
        results.append(result)
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    print(f"{'模型名称':<45} {'BLEU':>8} {'状态':>10}")
    print("-"*60)
    
    # 按 BLEU 分数排序
    results.sort(key=lambda x: x["bleu"], reverse=True)
    
    for r in results:
        status = "✓" if r["status"] == "success" else "✗"
        print(f"{r['model']:<45} {r['bleu']:>8.2f} {status:>10}")
    
    print("-"*60)
    if results and results[0]["status"] == "success":
        print(f"\n🏆 推荐模型: {results[0]['model']} (BLEU: {results[0]['bleu']:.2f})")
    
    # 不保存文件，只打印结果
    print("\n测试完成！")


if __name__ == "__main__":
    main()
