import json
import re
import sys

def extract_answer(response):
    """
    严格提取逻辑（与 mismatch 文件提取逻辑一致）：
    只移除 <think> 标签，去除首尾空白和句号。
    不做任何猜测或正则搜索。
    """
    if not response:
        return None
    
    # 移除 <think> 标签内容
    response_clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # 去除空白字符
    response_clean = response_clean.strip()
    
    # 去除末尾句号 (兼容 "A." 这种情况)
    response_clean = response_clean.rstrip('.')
    
    # 再次去除可能剩余的空白
    response_clean = response_clean.strip()
    
    return response_clean.upper()

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
    unparseable = 0  # 无法解析的答案
    
    # 分类统计
    forward_total = 0
    forward_correct = 0
    reverse_total = 0
    reverse_correct = 0
    
    # 存储错误案例
    errors = []
    unparseable_cases = []
    
    # 读取JSONL文件
    with open(json_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # 获取原始响应和标签
                raw_response = data.get('response', '')
                labels = data.get('labels', '').strip()
                label_text = data.get('label', '')
                video_path = data.get('videos', [''])[0]
                
                # 提取答案
                predicted_answer = extract_answer(raw_response)
                
                # 如果无法提取答案
                if predicted_answer is None:
                    unparseable += 1
                    unparseable_cases.append({
                        'line': line_num,
                        'video': video_path,
                        'raw_response': raw_response[:200],  # 只保存前200字符
                        'ground_truth': labels,
                        'label': label_text
                    })
                    continue
                
                total += 1
                
                # 判断是否正确
                is_correct = (predicted_answer == labels)
                
                if is_correct:
                    correct += 1
                else:
                    wrong += 1
                    errors.append({
                        'line': line_num,
                        'video': video_path,
                        'predicted': predicted_answer,
                        'ground_truth': labels,
                        'label': label_text,
                        'raw_response': raw_response[:200]  # 保存部分原始响应
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
    print("=" * 70)
    print("📊 准确率统计结果")
    print("=" * 70)
    print(f"\n总体统计:")
    print(f"  有效样本数: {total}")
    print(f"  正确数量: {correct}")
    print(f"  错误数量: {wrong}")
    print(f"  无法解析: {unparseable}")
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
        print("-" * 70)
        for i, error in enumerate(errors[:10], 1):
            print(f"{i}. 行号: {error['line']}")
            print(f"   视频: {error['video']}")
            print(f"   预测: {error['predicted']} | 真实: {error['ground_truth']}")
            print(f"   标签: {error['label']}")
            print(f"   响应片段: {error['raw_response'][:100]}...")
            print()
        
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个错误案例未显示")
    
    # 显示无法解析的案例
    if unparseable_cases:
        print(f"\n⚠️  无法解析的案例 ({len(unparseable_cases)} 个):")
        print("-" * 70)
        for i, case in enumerate(unparseable_cases[:5], 1):
            print(f"{i}. 行号: {case['line']}")
            print(f"   视频: {case['video']}")
            print(f"   真实标签: {case['ground_truth']}")
            print(f"   响应片段: {case['raw_response'][:100]}...")
            print()
        
        if len(unparseable_cases) > 5:
            print(f"   ... 还有 {len(unparseable_cases) - 5} 个案例未显示")
    
    print("=" * 70)
    
    return {
        'total': total,
        'correct': correct,
        'wrong': wrong,
        'unparseable': unparseable,
        'overall_accuracy': overall_accuracy,
        'forward_accuracy': forward_accuracy,
        'reverse_accuracy': reverse_accuracy,
        'errors': errors,
        'unparseable_cases': unparseable_cases
    }

if __name__ == "__main__":
    # 支持命令行参数
    if len(sys.argv) > 1:
        jsonl_file = sys.argv[1]
    else:
        # 默认路径
        jsonl_file = "/root/zhuweijie/result/Qwen3-VL-8B-Thinking/infer_result/20260104-192900.jsonl"
    
    print(f"📁 读取文件: {jsonl_file}\n")
    
    # 计算准确率
    results = calculate_accuracy(jsonl_file)
    
    # 保存详细报告
    import os
    base_dir = os.path.dirname(jsonl_file)
    
    # 保存错误案例
    if results['errors']:
        error_file = os.path.join(base_dir, "error_cases.json")
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(results['errors'], f, indent=2, ensure_ascii=False)
        print(f"\n💾 错误案例已保存到: {error_file}")
    
    # 保存无法解析的案例
    if results['unparseable_cases']:
        unparseable_file = os.path.join(base_dir, "unparseable_cases.json")
        with open(unparseable_file, 'w', encoding='utf-8') as f:
            json.dump(results['unparseable_cases'], f, indent=2, ensure_ascii=False)
        print(f"💾 无法解析的案例已保存到: {unparseable_file}")
    
    # 保存总结报告
    summary_file = os.path.join(base_dir, "accuracy_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("📊 准确率统计报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"总体统计:\n")
        f.write(f"  有效样本数: {results['total']}\n")
        f.write(f"  正确数量: {results['correct']}\n")
        f.write(f"  错误数量: {results['wrong']}\n")
        f.write(f"  无法解析: {results['unparseable']}\n")
        f.write(f"  总体准确率: {results['overall_accuracy']:.2f}%\n\n")
        f.write(f"分类统计:\n")
        f.write(f"  Forward (A): {results['forward_accuracy']:.2f}%\n")
        f.write(f"  Reverse (B): {results['reverse_accuracy']:.2f}%\n")
    
    print(f"💾 统计报告已保存到: {summary_file}")