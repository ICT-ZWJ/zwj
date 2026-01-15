"""
Qwen2.5-VL 评测脚本 - 使用 Video-R1 风格的提示词和评估流程

支持的数据集:
- Tempcompass (本地已有)
- 其他 Video-R1 支持的 benchmark

使用方法:
    python eval_qwen25vl_videor1_style.py \
        --model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
        --dataset tempcompass \
        --output_name "qwen25vl_tempcompass"
"""

import os
import json
import re
import argparse
from tqdm import tqdm
from datetime import datetime
import torch

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


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


# ===================== 数据处理函数 =====================

def load_tempcompass_data(data_path, video_dir):
    """加载 Tempcompass 数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    processed_data = []
    for item in raw_data:
        video_id = item['video_id']
        # 处理 reverse 视频
        if video_id.endswith('_reverse'):
            video_filename = f"{video_id}.mp4"
        else:
            video_filename = f"{video_id}.mp4"
        
        video_path = os.path.join(video_dir, video_filename)
        
        processed_data.append({
            'path': video_path,
            'data_type': 'video',
            'problem': item['question'],
            'problem_type': 'multiple choice',  # Tempcompass 都是选择题
            'options': [],  # 选项已经包含在 question 中
            'solution': f"<answer>{item['answer']}</answer>",
            'dim': item.get('dim', 'unknown'),
            'video_id': video_id
        })
    
    return processed_data


def load_video_r1_format_data(json_path):
    """加载 Video-R1 格式的评测数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ===================== 评估函数 (来自 Video-R1) =====================

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


def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception:
        return None


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    conditions = rel_error < (1 - thresholds)
    mra = conditions.float().mean()
    return mra.item()


def reward_fn(sample, model_output, question_type):
    """计算奖励/准确率"""
    try:
        output_ans = extract_answer(model_output)
        if output_ans == '':
            output_ans = model_output
        gt_ans = extract_answer(sample.get("solution", ""))
        
        if question_type == "multiple choice":
            # 提取选项字母进行比较
            output_letter = output_ans.strip().upper()
            gt_letter = gt_ans.strip().upper()
            
            # 处理 "A. xxx" 格式
            if len(output_letter) > 1 and output_letter[1] in '.、:：':
                output_letter = output_letter[0]
            if len(gt_letter) > 1 and gt_letter[1] in '.、:：':
                gt_letter = gt_letter[0]
            
            # 也处理完整答案匹配
            if output_letter == gt_letter:
                return 1.0
            elif gt_ans.strip() in output_ans or output_ans in gt_ans.strip():
                return 1.0
            else:
                return 0.0
                
        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return mean_relative_accuracy(out_number, gt_number)
            
        else:
            return 0.0
    except Exception:
        return 0.0


# ===================== 主评测逻辑 =====================

def run_evaluation(args):
    """运行评测"""
    
    print(f"Loading model from: {args.model_path}")
    
    # 初始化 vLLM
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    
    # 加载数据
    if args.dataset == 'tempcompass':
        if args.use_sample:
            data_path = os.path.join(args.data_root, 'tempcompass_sample.json')
        else:
            data_path = os.path.join(args.data_root, 'tempcompass_full.json')
        video_dir = os.path.join(args.data_root, 'videos')
        data = load_tempcompass_data(data_path, video_dir)
    else:
        # Video-R1 格式数据
        data_path = os.path.join(args.data_root, f'eval_{args.dataset}.json')
        data = load_video_r1_format_data(data_path)
    
    print(f"Loaded {len(data)} samples from {args.dataset}")
    
    # 构建消息
    messages = []
    for x in data:
        if x["problem_type"] == 'multiple choice' and x.get("options"):
            question = x['problem'] + "\nOptions:\n"
            for op in x["options"]:
                question += op + "\n"
        else:
            question = x['problem']
        
        msg = [{
            "role": "user",
            "content": [
                {
                    "type": x['data_type'],
                    x['data_type']: x['path'],
                    "max_pixels": args.max_pixels,
                    "nframes": args.nframes,
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x['problem_type']]
                }
            ]
        }]
        messages.append(msg)
    
    # 输出路径
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(args.output_dir, f"{args.output_name}_{timestamp}.json")
    
    # 断点续评
    final_output = []
    start_idx = 0
    if args.resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                print(f"Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")
    
    # 批量推理
    mean_acc = []
    mean_mra = []
    dim_acc = {}  # 按维度统计准确率
    
    BSZ = args.batch_size
    
    for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
        batch_messages = messages[i:i + BSZ]
        batch_data = data[i:i + BSZ]
        
        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                   for msg in batch_messages]
        
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                batch_messages, return_video_kwargs=True
            )
            
            image_idx = 0
            video_idx = 0
            llm_inputs = []
            
            for idx, prompt in enumerate(prompts):
                mm_type = batch_messages[idx][0]['content'][0]['type']
                sample_mm_data = {}
                sample_video_kw = {}
                
                if mm_type == 'image':
                    sample_mm_data["image"] = image_inputs[image_idx]
                    image_idx += 1
                elif mm_type == 'video':
                    sample_mm_data["video"] = video_inputs[video_idx]
                    for key, value in video_kwargs.items():
                        sample_video_kw[key] = value[video_idx]
                    video_idx += 1
                
                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                })
            
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            batch_output_text = [out.outputs[0].text for out in outputs]
            
        except Exception as e:
            print(f'Error processing batch starting at {i}: {e}')
            batch_output_text = ['<answer>error</answer>'] * len(batch_data)
        
        # 处理结果
        for j, (sample, model_output) in enumerate(zip(batch_data, batch_output_text)):
            think_chain = extract_think(model_output)
            final_ans = extract_answer(model_output)
            if final_ans == "":
                final_ans = model_output
            
            sample["output"] = model_output
            sample["prediction"] = final_ans
            q_type = sample.get("problem_type", "")
            sample["reward"] = reward_fn(sample, model_output, q_type)
            sample['correct'] = sample["reward"] == 1.0
            
            # 统计
            if sample['problem_type'] != 'regression':
                mean_acc.append(sample["reward"])
            else:
                mean_mra.append(sample["reward"])
            
            # 按维度统计
            dim = sample.get('dim', 'unknown')
            if dim not in dim_acc:
                dim_acc[dim] = []
            dim_acc[dim].append(sample["reward"])
            
            if think_chain:
                sample["process"] = f"<think>{think_chain}</think>"
            
            final_output.append(sample)
        
        # 定期保存
        if (i // BSZ + 1) % 10 == 0 or i + BSZ >= len(messages):
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(final_output)} samples to {output_path}")
            except Exception as e:
                print(f"Error writing to output file: {e}")
    
    # 计算最终指标
    final_metrics = {
        'mean_acc': torch.tensor(mean_acc).mean().item() if mean_acc else 0.0,
        'mean_mra': torch.tensor(mean_mra).mean().item() if mean_mra else 0.0,
        'total_samples': len(final_output),
        'correct_samples': sum(1 for x in final_output if x.get('correct', False)),
        'dim_accuracy': {dim: torch.tensor(accs).mean().item() for dim, accs in dim_acc.items()}
    }
    
    # 保存最终结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": final_output, 
            "metrics": final_metrics
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("评测完成!")
    print(f"结果保存至: {output_path}")
    print(f"\n总体准确率: {final_metrics['mean_acc']*100:.2f}%")
    print(f"正确数/总数: {final_metrics['correct_samples']}/{final_metrics['total_samples']}")
    print("\n各维度准确率:")
    for dim, acc in sorted(final_metrics['dim_accuracy'].items()):
        print(f"  {dim}: {acc*100:.2f}%")
    print("="*60)
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 评测 (Video-R1 风格)")
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="模型路径")
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='tempcompass',
                        choices=['tempcompass', 'mvbench', 'videomme', 'videommmu', 'vsibench', 'mmvu'],
                        help="评测数据集")
    parser.add_argument('--data_root', type=str, default='./Tempcompass',
                        help="数据集根目录")
    parser.add_argument('--use_sample', action='store_true',
                        help="使用 sample 数据集进行测试")
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./result/eval_videor1_style',
                        help="输出目录")
    parser.add_argument('--output_name', type=str, default='qwen25vl',
                        help="输出文件名前缀")
    parser.add_argument('--resume', action='store_true',
                        help="从断点续评")
    
    # 推理参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help="批处理大小")
    parser.add_argument('--max_model_len', type=int, default=16384,
                        help="最大模型长度")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8,
                        help="GPU 显存利用率")
    parser.add_argument('--temperature', type=float, default=0.1,
                        help="采样温度")
    parser.add_argument('--top_p', type=float, default=0.001,
                        help="Top-p 采样")
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help="最大生成 token 数")
    
    # 视频参数
    parser.add_argument('--max_pixels', type=int, default=200704,
                        help="每帧最大像素数")
    parser.add_argument('--nframes', type=int, default=32,
                        help="最大帧数")
    
    args = parser.parse_args()
    
    run_evaluation(args)


if __name__ == "__main__":
    main()