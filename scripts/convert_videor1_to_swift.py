"""
将 Video-R1 评测 JSON 转换为 ms-swift 格式

使用方法:
    python scripts/convert_videor1_to_swift.py
"""

import json
import os

# Video-R1 风格的提示词模板
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


def convert_videor1_to_swift(input_json, output_jsonl, video_base_dir):
    """
    将 Video-R1 格式转换为 ms-swift 格式
    
    Video-R1 格式:
    {
        "problem": "问题",
        "options": ["A. xxx", "B. xxx"],
        "problem_type": "multiple choice",
        "solution": "<answer>A</answer>",
        "path": "./Evaluation/TempCompass/xxx.mp4"
    }
    
    ms-swift 格式:
    {"messages": [{"role": "user", "content": "<video>问题"}], "videos": ["视频路径"]}
    """
    
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    swift_data = []
    
    for item in data:
        # 构建问题文本
        problem_type = item.get('problem_type', 'multiple choice')
        
        if problem_type == 'multiple choice' and item.get('options'):
            question = item['problem'] + "\nOptions:\n" + "\n".join(item['options'])
        else:
            question = item['problem']
        
        # 添加 Video-R1 风格的提示词
        full_question = QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE.get(problem_type, TYPE_TEMPLATE['free-form'])
        
        # 构建视频路径
        original_path = item.get('path', '')
        # ./Evaluation/TempCompass/xxx.mp4 -> video_base_dir/TempCompass/xxx.mp4
        if original_path.startswith('./Evaluation/'):
            video_path = os.path.join(video_base_dir, original_path[13:])  # 去掉 "./Evaluation/"
        else:
            video_path = original_path
        
        # ms-swift 格式
        swift_item = {
            "messages": [
                {"role": "user", "content": f"<video>{full_question}"}
            ],
            "videos": [video_path],
            # 保留原始信息用于评估
            "_meta": {
                "video_id": item.get('video_id', ''),
                "solution": item.get('solution', ''),
                "dim": item.get('dim', ''),
                "problem_id": item.get('problem_id', 0),
                "gt_answer": item.get('answer', '')
            }
        }
        
        swift_data.append(swift_item)
    
    # 保存为 JSONL 格式
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in swift_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成: {len(swift_data)} 条记录")
    print(f"输出文件: {output_jsonl}")
    
    return len(swift_data)


if __name__ == "__main__":
    # 路径配置
    INPUT_JSON = "/root/zhuweijie/Video-R1/src/r1-v/Evaluation/eval_tempcompass.json"
    OUTPUT_JSONL = "/root/zhuweijie/Video-R1/src/r1-v/Evaluation/eval_tempcompass_swift.jsonl"
    VIDEO_BASE_DIR = "/root/zhuweijie/Video-R1/src/r1-v/Evaluation"
    
    convert_videor1_to_swift(INPUT_JSON, OUTPUT_JSONL, VIDEO_BASE_DIR)