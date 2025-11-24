"""
测试mBART预训练模型的零样本BLEU分数
快速验证mBART是否比opus-mt更好
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import time
from tqdm import tqdm

def main():
    # 加载mBART模型
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    print(f"Loading mBART model: {model_name}")
    print("=" * 60)
    print("⚠️  Note: mBART is 611M parameters, loading may take a few minutes...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        src_lang="zh_CN",
        tgt_lang="en_XX"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    model.cuda()
    
    print("✅ Model loaded successfully!")

    # 加载WMT19验证集（完整官方验证集）
    print("\nLoading WMT19 zh-en validation set...")
    wmt19 = load_dataset("wmt19", "zh-en")
    test_data = wmt19["validation"]  # 使用完整验证集（约4000条）
    
    num_samples = len(test_data)
    print(f"Total test samples: {num_samples}")
    print("=" * 60)

    # 准备数据
    sources = [ex["zh"] for ex in test_data["translation"]]
    references = [[ex["en"]] for ex in test_data["translation"]]

    # 翻译
    print("\nTranslating with mBART...")
    predictions = []
    batch_size = 8  # mBART较大，可能需要较小的batch
    
    start_time = time.time()
    
    # 设置强制的BOS token为目标语言
    forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]
    
    for i in tqdm(range(0, len(sources), batch_size)):
        batch = sources[i:i+batch_size]
        
        # mBART tokenization
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                forced_bos_token_id=forced_bos_token_id  # 强制输出英文
            )
        
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)
    
    runtime = time.time() - start_time

    # 计算BLEU
    print("\nComputing BLEU score...")
    bleu_metric = evaluate.load("sacrebleu")
    bleu_result = bleu_metric.compute(predictions=predictions, references=references)

    # 输出格式与main.py保持一致
    print("\n" + "="*60)
    test_metrics = {
        'test_bleu': bleu_result['score'],
        'test_runtime': runtime,
        'test_samples_per_second': num_samples / runtime,
        'model': 'mBART-large-50 (zero-shot)',
        'note': 'No fine-tuning, 611M parameters'
    }
    print("Test Metrics:", test_metrics)
    print("="*60)
    
    # 与opus-mt对比
    print("\n📊 Comparison with opus-mt:")
    print(f"   opus-mt-zh-en (77M):  BLEU = 19.92")
    print(f"   mBART-large (611M):   BLEU = {bleu_result['score']:.2f}")
    print(f"   Improvement:          {bleu_result['score'] - 19.92:+.2f}")
    print("="*60)

    # 展示一些样例
    print("\n样例翻译 (前5条)：")
    print("-" * 60)
    for i in range(min(5, len(predictions))):
        print(f"\n[{i+1}]")
        print(f"源文: {sources[i]}")
        print(f"预测: {predictions[i]}")
        print(f"参考: {references[i][0]}")
        print("-" * 60)

if __name__ == "__main__":
    main()
