"""
测试预训练模型的零样本BLEU分数
用于验证模型在不fine-tune的情况下的基准性能
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import numpy as np
from tqdm import tqdm

# 加载预训练模型
model_name = "Helsinki-NLP/opus-mt-zh-en"
print(f"加载预训练模型: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
model.cuda()

# 加载WMT19验证集（官方验证集）
print("加载WMT19 zh-en验证集...")
wmt19 = load_dataset("wmt19", "zh-en")
test_data = wmt19["validation"].select(range(500))  # 测试500条

# 准备数据
sources = [ex["zh"] for ex in test_data["translation"]]
references = [[ex["en"]] for ex in test_data["translation"]]

# 翻译
print("开始翻译...")
predictions = []
batch_size = 8

for i in tqdm(range(0, len(sources), batch_size)):
    batch = sources[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    predictions.extend(batch_preds)

# 计算BLEU
print("计算BLEU分数...")
bleu_metric = evaluate.load("sacrebleu")
bleu_result = bleu_metric.compute(predictions=predictions, references=references)

print("\n" + "="*60)
print(f"预训练模型零样本BLEU: {bleu_result['score']:.2f}")
print("="*60)

# 展示一些样例
print("\n样例翻译：")
for i in range(min(5, len(predictions))):
    print(f"\n源文: {sources[i]}")
    print(f"预测: {predictions[i]}")
    print(f"参考: {references[i][0]}")
