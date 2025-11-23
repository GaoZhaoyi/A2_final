"""
使用预训练模型（零样本）在WMT19完整验证集上评估
输出格式与main.py的Test Metrics保持一致
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import time
from tqdm import tqdm

def main():
    # 加载预训练模型
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    print(f"Loading pretrained model: {model_name}")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    model.cuda()

    # 加载WMT19验证集（完整官方验证集）
    print("Loading WMT19 zh-en validation set...")
    wmt19 = load_dataset("wmt19", "zh-en")
    test_data = wmt19["validation"]  # 使用完整验证集（约4000条）
    
    num_samples = len(test_data)
    print(f"Total test samples: {num_samples}")
    print("=" * 60)

    # 准备数据
    sources = [ex["zh"] for ex in test_data["translation"]]
    references = [[ex["en"]] for ex in test_data["translation"]]

    # 翻译
    print("\nTranslating...")
    predictions = []
    batch_size = 8
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(sources), batch_size)):
        batch = sources[i:i+batch_size]
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
                early_stopping=True
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
        'model': 'pretrained (zero-shot)',
        'note': 'No fine-tuning'
    }
    print("Test Metrics:", test_metrics)
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
