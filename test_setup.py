"""
快速验证脚本，在完整训练前测试设置。
该脚本验证所有组件是否正常工作。
"""

import torch
from dataset import build_dataset, preprocess_data
from model import initialize_model, initialize_tokenizer
from trainer import build_trainer
from utils import not_change_test_dataset, set_random_seeds

def test_setup():
    """测试所有组件是否正常工作。"""
    print("="*50)
    print("测试作业设置")
    print("="*50)
    
    # 1. 测试随机种子设置
    print("\n1. 设置随机种子...")
    set_random_seeds()
    print("✓ 随机种子设置成功")
    
    # 2. 测试分词器初始化
    print("\n2. 初始化分词器...")
    tokenizer = initialize_tokenizer()
    print(f"✓ 分词器已加载: {tokenizer.__class__.__name__}")
    print(f"  - 词汇表大小: {len(tokenizer)}")
    print(f"  - 模型类型: MarianMT (中文→英文专用)")
    
    # 3. 测试模型初始化
    print("\n3. 初始化模型...")
    model = initialize_model()
    print(f"✓ 模型已加载: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 模型大小: 约{total_params / 1e9:.2f}B参数")
    
    # 4. 测试数据集加载
    print("\n4. 加载数据集...")
    raw_datasets = build_dataset()
    print(f"✓ 数据集已加载")
    print(f"  - 训练样本: {len(raw_datasets['train']):,}")
    print(f"  - 验证样本: {len(raw_datasets['validation']):,}")
    print(f"  - 测试样本: {len(raw_datasets['test']):,}")
    
    # 5. 验证测试数据集完整性
    print("\n5. 验证测试数据集...")
    assert not_change_test_dataset(raw_datasets), "❌ 测试数据集已被修改！"
    print("✓ 测试数据集未改变")
    
    # 6. 测试数据预处理
    print("\n6. 测试数据预处理...")
    # 使用小子集测试
    from datasets import DatasetDict
    test_subset = raw_datasets["train"].select(range(100))
    raw_subset = DatasetDict({"train": test_subset, "validation": test_subset, "test": test_subset})
    tokenized_subset = preprocess_data(
        raw_subset,
        tokenizer
    )
    print(f"✓ 数据预处理正常工作")
    sample = tokenized_subset["train"][0]
    print(f"  - 输入ID形状: {len(sample['input_ids'])}")
    print(f"  - 标签ID形状: {len(sample['labels'])}")
    
    # 7. 测试分词示例
    print("\n7. 测试分词示例...")
    test_zh = "今天天气很好。"
    test_en = "The weather is very nice today."
    input_ids = tokenizer(test_zh, return_tensors="pt").input_ids
    label_ids = tokenizer(text_target=test_en, return_tensors="pt").input_ids
    print(f"✓ 分词正常工作")
    print(f"  - 中文输入: {test_zh}")
    print(f"  - 分词后: {input_ids.shape}")
    print(f"  - 英文目标: {test_en}")
    print(f"  - 分词后: {label_ids.shape}")
    
    # 8. 测试GPU可用性
    print("\n8. 检查GPU可用性...")
    if torch.cuda.is_available():
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  - CUDA版本: {torch.version.cuda}")
    else:
        print("⚠ 无可用GPU - 训练会非常慢！")
    
    # 9. 测试FP16支持
    print("\n9. 检查FP16支持...")
    if torch.cuda.is_available():
        try:
            with torch.cuda.amp.autocast():
                dummy = torch.randn(1, 10).cuda()
            print("✓ FP16/AMP已支持")
        except:
            print("⚠ 此GPU不支持FP16/AMP")
    
    # 10. 测试前向传播
    print("\n10. 测试模型前向传播...")
    try:
        model.eval()
        with torch.no_grad():
            test_input = tokenizer(test_zh, return_tensors="pt")
            if torch.cuda.is_available():
                model = model.cuda()
                test_input = {k: v.cuda() for k, v in test_input.items()}
            output = model.generate(**test_input, max_length=50, num_beams=2)
            translated = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"✓ 前向传播成功")
            print(f"  - 输入: {test_zh}")
            print(f"  - 输出: {translated}")
    except Exception as e:
        print(f"⚠ 前向传播失败: {e}")
    
    print("\n" + "="*50)
    print("所有测试完成！")
    print("="*50)
    print("\n✅ 设置已就绪，可以开始训练！")
    print("\n启动训练，运行：")
    print("  python main.py")
    print("\n或使用uv：")
    print("  uv run python main.py")

if __name__ == "__main__":
    test_setup()
