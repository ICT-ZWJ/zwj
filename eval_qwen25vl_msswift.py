"""
Qwen2.5-VL 评测脚本 - 适配 ms-swift 框架
使用 Video-R1 风格的提示词和评估流程

使用方法:
    # 使用 ms-swift 推理
    python eval_qwen25vl_msswift.py \
        --model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
        --dataset tempcompass \
        --data_root ./Tempcompass \
        --use_sample
    
    # 使用 LoRA 微调后的模型
    python eval_qwen25vl_msswift.py \
        --model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
        --adapters "output/vx-xxx/checkpoint-xxx" \
        --dataset tempcompass
"""

import os
import json
import re
import argparse
from tqdm import tqdm
from datetime import datetime
import torch


# ===================== Video-R1 风格的提示词模板 =====================

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
}


# ===================== 数据加载函数 =====================

def load_tempcompass_data(data_path, video_dir):
    """加载 Tempcompass 数据集并转换为 Video-R1 格式"""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    processed_data = []
    for item in raw_data:
        video_id = item['video_id']
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        
        # 检查视频是否存在
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        # 提取答案字母
        answer = item['answer']
        answer_letter = answer.strip()[0] if answer else ""
        
        processed_data.append({
            'path': video_path,
            'data_type': 'video',
            'problem': item['question'],
            'problem_type': 'multiple choice',
            'options': [],
            'solution': f"<answer>{answer_letter}</answer>",
            'gt_answer': answer,  # 保留完整答案用于调试
            'dim': item.get('dim', 'unknown'),
            'video_id': video_id
        })
    
    return processed_data


def load_video_r1_format_data(json_path, video_base_dir=None):
    """加载 Video-R1 格式的评测数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果指定了 video_base_dir，更新视频路径
    if video_base_dir:
        for item in data:
            if item.get('path', '').startswith('./'):
                item['path'] = os.path.join(video_base_dir, item['path'][2:])
    
    return data


# ===================== 评估函数 =====================

def extract_think(output_str):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def reward_fn(sample, model_output, question_type):
    """计算准确率"""
    try:
        output_ans = extract_answer(model_output)
        if output_ans == '':
            # 尝试直接从输出中提取答案
            output_ans = model_output.strip()
        
        gt_ans = extract_answer(sample.get("solution", ""))
        
        if question_type == "multiple choice":
            # 提取选项字母
            output_letter = output_ans.strip().upper()[:1] if output_ans else ""
            gt_letter = gt_ans.strip().upper()[:1] if gt_ans else ""
            
            if output_letter == gt_letter:
                return 1.0
            # 也检查完整答案匹配
            if sample.get('gt_answer', '') in output_ans:
                return 1.0
            return 0.0
        else:
            return 0.0
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        return 0.0


# ===================== ms-swift 推理封装 =====================

def create_swift_engine(model_path, adapters=None, max_pixels=200704, nframes=32):
    """创建 ms-swift 推理引擎"""
    try:
        from swift.llm import InferEngine, PtEngine, InferRequest
        from swift.plugin import InferStats
        
        # 设置环境变量控制视频采样
        os.environ['MAX_PIXELS'] = str(max_pixels)
        os.environ['VIDEO_MAX_PIXELS'] = str(max_pixels * nframes)
        os.environ['FPS_MAX_FRAMES'] = str(nframes)
        
        # 创建推理引擎
        if adapters:
            engine = PtEngine(model_path, adapters=[adapters])
        else:
            engine = PtEngine(model_path)
        
        return engine, InferRequest
        
    except ImportError:
        print("ms-swift 未安装，尝试使用 transformers 直接推理...")
        return None, None


def create_transformers_model(model_path, adapters=None):
    """使用 transformers 创建模型（备用方案）"""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 加载 LoRA adapters
    if adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapters)
    
    return model, processor


def infer_with_swift(engine, InferRequest, video_path, question, max_tokens=1024):
    """使用 ms-swift 进行推理"""
    request = InferRequest(
        messages=[{
            "role": "user",
            "content": f"<video>{question}"
        }],
        videos=[video_path]
    )
    
    response = engine.infer([request])[0]
    return response.choices[0].message.content


def infer_with_transformers(model, processor, video_path, question, max_tokens=1024, nframes=32):
    """使用 transformers 进行推理"""
    from qwen_vl_utils import process_vision_info
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "nframes": nframes},
            {"type": "text", "text": question}
        ]
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


# ===================== 主评测逻辑 =====================

def run_evaluation(args):
    """运行评测"""
    
    print("=" * 60)
    print("Qwen2.5-VL 评测 (ms-swift / transformers)")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"Adapters: {args.adapters or 'None'}")
    print(f"数据集: {args.dataset}")
    print(f"数据目录: {args.data_root}")
    print("=" * 60)
    
    # 加载数据
    if args.dataset == 'tempcompass':
        if args.use_sample:
            data_path = os.path.join(args.data_root, 'tempcompass_sample.json')
        else:
            data_path = os.path.join(args.data_root, 'tempcompass_full.json')
        video_dir = os.path.join(args.data_root, 'videos')
        data = load_tempcompass_data(data_path, video_dir)
    else:
        data_path = os.path.join(args.data_root, f'eval_{args.dataset}.json')
        data = load_video_r1_format_data(data_path, args.data_root)
    
    print(f"加载了 {len(data)} 个样本")
    
    # 限制样本数量（用于测试）
    if args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"限制为 {len(data)} 个样本进行测试")
    
    # 尝试加载 ms-swift 引擎
    engine, InferRequest = create_swift_engine(
        args.model_path, 
        args.adapters,
        args.max_pixels,
        args.nframes
    )
    
    use_swift = engine is not None
    
    if not use_swift:
        print("使用 transformers 进行推理...")
        model, processor = create_transformers_model(args.model_path, args.adapters)
    
    # 输出路径
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(args.output_dir, f"{args.output_name}_{timestamp}.json")
    
    # 评测循环
    results = []
    correct_count = 0
    dim_stats = {}
    
    for i, sample in enumerate(tqdm(data, desc="评测中")):
        video_path = sample['path']
        
        # 构建问题（带 Video-R1 提示词模板）
        if sample["problem_type"] == 'multiple choice' and sample.get("options"):
            question = sample['problem'] + "\nOptions:\n" + "\n".join(sample["options"])
        else:
            question = sample['problem']
        
        full_question = QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[sample['problem_type']]
        
        # 推理
        try:
            if use_swift:
                output = infer_with_swift(engine, InferRequest, video_path, full_question, args.max_tokens)
            else:
                output = infer_with_transformers(model, processor, video_path, full_question, args.max_tokens, args.nframes)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            output = "<answer>error</answer>"
        
        # 评估
        think_chain = extract_think(output)
        final_ans = extract_answer(output)
        if final_ans == "":
            final_ans = output
        
        reward = reward_fn(sample, output, sample['problem_type'])
        correct = reward == 1.0
        
        if correct:
            correct_count += 1
        
        # 按维度统计
        dim = sample.get('dim', 'unknown')
        if dim not in dim_stats:
            dim_stats[dim] = {'correct': 0, 'total': 0}
        dim_stats[dim]['total'] += 1
        if correct:
            dim_stats[dim]['correct'] += 1
        
        # 保存结果
        result = {
            'video_id': sample.get('video_id', ''),
            'video_path': video_path,
            'question': sample['problem'],
            'gt_answer': sample.get('gt_answer', extract_answer(sample.get('solution', ''))),
            'output': output,
            'prediction': final_ans,
            'correct': correct,
            'reward': reward,
            'dim': dim
        }
        if think_chain:
            result['reasoning'] = think_chain
        
        results.append(result)
        
        # 定期保存
        if (i + 1) % 50 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'results': results,
                    'progress': f"{i+1}/{len(data)}"
                }, f, ensure_ascii=False, indent=2)
    
    # 计算最终指标
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total,
        'dim_accuracy': {
            dim: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            for dim, stats in dim_stats.items()
        }
    }
    
    # 保存最终结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'metrics': metrics,
            'config': vars(args)
        }, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评测完成!")
    print(f"结果保存至: {output_path}")
    print(f"\n总体准确率: {accuracy*100:.2f}% ({correct_count}/{total})")
    print("\n各维度准确率:")
    for dim, acc in sorted(metrics['dim_accuracy'].items()):
        dim_total = dim_stats[dim]['total']
        dim_correct = dim_stats[dim]['correct']
        print(f"  {dim}: {acc*100:.2f}% ({dim_correct}/{dim_total})")
    print("=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 评测 (ms-swift 适配)")
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="模型路径或 ModelScope/HuggingFace 模型名")
    parser.add_argument('--adapters', type=str, default=None,
                        help="LoRA adapters 路径 (ms-swift 训练的 checkpoint)")
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='tempcompass',
                        choices=['tempcompass', 'mvbench', 'videomme', 'videommmu', 'vsibench', 'mmvu'],
                        help="评测数据集")
    parser.add_argument('--data_root', type=str, default='./Tempcompass',
                        help="数据集根目录")
    parser.add_argument('--use_sample', action='store_true',
                        help="使用 sample 数据集进行快速测试")
    parser.add_argument('--max_samples', type=int, default=-1,
                        help="最大样本数 (-1 表示全部)")
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./result/eval_msswift',
                        help="输出目录")
    parser.add_argument('--output_name', type=str, default='qwen25vl',
                        help="输出文件名前缀")
    
    # 推理参数
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help="最大生成 token 数")
    parser.add_argument('--max_pixels', type=int, default=200704,
                        help="每帧最大像素数")
    parser.add_argument('--nframes', type=int, default=32,
                        help="最大帧数")
    
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()