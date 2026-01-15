import json

def calculate_accuracy(json_file):
    """
    计算模型预测的正确率
    
    Args:
        json_file: JSONL文件路径
    
    Returns:
        准确率统计信息
    """
    total = 0
    correct = 0
    wrong = 0
    
    # 分类统计
    forward_total = 0
    forward_correct = 0
    reverse_total = 0
    reverse_correct = 0
    
    # 存储错误案例
    errors = []
    
    # 读取JSONL文件
    with open(json_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # 获取预测结果和标签
                response = data.get('response', '').strip()
                labels = data.get('labels', '').strip()
                label_text = data.get('label', '')
                video_path = data.get('videos', [''])[0]
                
                total += 1
                
                # 判断是否正确
                is_correct = (response == labels)
                
                if is_correct:
                    correct += 1
                else:
                    wrong += 1
                    errors.append({
                        'line': line_num,
                        'video': video_path,
                        'predicted': response,
                        'ground_truth': labels,
                        'label': label_text
                    })
                
                # 分类统计
                if labels == 'A':  # Forward
                    forward_total += 1
                    if is_correct:
                        forward_correct += 1
                elif labels == 'B':  # Reverse
                    reverse_total += 1
                    if is_correct:
                        reverse_correct += 1
                        
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_num} 行JSON解析错误: {e}")
                continue
    
    # 计算准确率
    overall_accuracy = (correct / total * 100) if total > 0 else 0
    forward_accuracy = (forward_correct / forward_total * 100) if forward_total > 0 else 0
    reverse_accuracy = (reverse_correct / reverse_total * 100) if reverse_total > 0 else 0
    
    # 打印结果
    print("=" * 60)
    print("📊 准确率统计结果")
    print("=" * 60)
    print(f"\n总体统计:")
    print(f"  总样本数: {total}")
    print(f"  正确数量: {correct}")
    print(f"  错误数量: {wrong}")
    print(f"  总体准确率: {overall_accuracy:.2f}%")
    
    print(f"\n分类统计:")
    print(f"  Forward (A - 从左到右):")
    print(f"    样本数: {forward_total}")
    print(f"    正确数: {forward_correct}")
    print(f"    准确率: {forward_accuracy:.2f}%")
    
    print(f"  Reverse (B - 从右到左):")
    print(f"    样本数: {reverse_total}")
    print(f"    正确数: {reverse_correct}")
    print(f"    准确率: {reverse_accuracy:.2f}%")
    
    # 显示错误案例
    if errors:
        print(f"\n❌ 错误案例 ({len(errors)} 个):")
        print("-" * 60)
        for i, error in enumerate(errors[:10], 1):  # 只显示前10个
            print(f"{i}. 行号: {error['line']}")
            print(f"   视频: {error['video']}")
            print(f"   预测: {error['predicted']} | 真实: {error['ground_truth']}")
            print(f"   标签: {error['label']}")
            print()
        
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个错误案例未显示")
    
    print("=" * 60)
    
    return {
        'total': total,
        'correct': correct,
        'wrong': wrong,
        'overall_accuracy': overall_accuracy,
        'forward_accuracy': forward_accuracy,
        'reverse_accuracy': reverse_accuracy,
        'errors': errors
    }

if __name__ == "__main__":
    # 设置JSONL文件路径
    jsonl_file = "/root/zhuweijie/result/Qwen2.5-VL-7B-Instruct/infer_result/20260104-161014.jsonl"
    
    # 计算准确率
    results = calculate_accuracy(jsonl_file)
    
    # # 可选：保存错误案例到文件
    # if results['errors']:
    #     error_file = "/root/zhuweijie/Chirality_in_Action/test_dataset/error_cases.json"
    #     with open(error_file, 'w', encoding='utf-8') as f:
    #         json.dump(results['errors'], f, indent=2, ensure_ascii=False)
    #     print(f"\n💾 错误案例已保存到: {error_file}")