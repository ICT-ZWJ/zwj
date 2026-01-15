#!/bin/bash
# TempCompass 评测脚本 - 使用 ms-swift + vLLM 加速 + Video-R1 风格 Prompt
#
# 使用方法:
#   bash scripts/eval_tempcompass_swift_vllm.sh

set -e

# ===================== 配置区域 =====================

# 模型选择
# MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL="Qwen/Qwen3-VL-8B-Thinking"

# GPU 配置
GPUS="0,1,2,3"

# 视频采样参数 (Video-R1 推荐配置)
IMAGE_MAX_TOKEN_NUM=512
VIDEO_MAX_TOKEN_NUM=256
FPS_MAX_FRAMES=16

# 推理参数
MAX_NEW_TOKENS=1024

# ===== vLLM 相关参数（核心改动）=====
INFER_BACKEND="vllm"

# 张量并行：通常设为你可见 GPU 数量，或其中一部分（例如 8 卡就设 8）
VLLM_TP_SIZE=4

# vLLM 显存利用率
VLLM_GPU_MEM_UTIL=0.90

# vLLM 最大上下文长度（多模态 + prompt 较长时需要大一点；过大可能更吃显存）
VLLM_MAX_MODEL_LEN=16384

# 多模态每个 prompt 的 image/video 限制（按你参考写法）
VLLM_LIMIT_MM_PER_PROMPT='{"image": 5, "video": 2}'

# （可选）像素上限：很多 VL 模型会用这个环境变量限制输入分辨率/帧
MAX_PIXELS=1003520

# 数据路径
VAL_DATASET="/root/zhuweijie/Video-R1/src/r1-v/Evaluation/eval_tempcompass_swift.jsonl"

# 输出路径
OUTPUT_DIR="/root/zhuweijie/result/eval_tempcompass_videor1"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/tempcompass_${TIMESTAMP}.jsonl"

# ===================== 准备工作 =====================

echo "======================================"
echo "TempCompass 评测 (Video-R1 风格 + vLLM)"
echo "======================================"
echo "模型: $MODEL"
echo "GPU: $GPUS"
echo "帧数: $FPS_MAX_FRAMES"
echo "数据集: $VAL_DATASET"
echo "输出: $OUTPUT_FILE"
echo "vLLM TP: $VLLM_TP_SIZE"
echo "vLLM 显存利用率: $VLLM_GPU_MEM_UTIL"
echo "vLLM max_model_len: $VLLM_MAX_MODEL_LEN"
echo "vLLM limit_mm_per_prompt: $VLLM_LIMIT_MM_PER_PROMPT"
echo "MAX_PIXELS: $MAX_PIXELS"
echo "======================================"

mkdir -p "$OUTPUT_DIR"

# 检查数据集是否存在，不存在则转换
if [ ! -f "$VAL_DATASET" ]; then
    echo "转换数据集为 ms-swift 格式..."
    python /root/zhuweijie/scripts/convert_videor1_to_swift.py
fi

# ===================== 运行推理 (vLLM) =====================

CUDA_VISIBLE_DEVICES=$GPUS \
MAX_PIXELS=$MAX_PIXELS \
IMAGE_MAX_TOKEN_NUM=$IMAGE_MAX_TOKEN_NUM \
VIDEO_MAX_TOKEN_NUM=$VIDEO_MAX_TOKEN_NUM \
FPS_MAX_FRAMES=$FPS_MAX_FRAMES \
swift infer \
    --model "$MODEL" \
    --val_dataset "$VAL_DATASET" \
    --stream false \
    --infer_backend "$INFER_BACKEND" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --max_batch_size 1 \
    --result_path "$OUTPUT_FILE" \
    --vllm_gpu_memory_utilization $VLLM_GPU_MEM_UTIL \
    --vllm_tensor_parallel_size $VLLM_TP_SIZE \
    --vllm_max_model_len $VLLM_MAX_MODEL_LEN \
    --vllm_limit_mm_per_prompt "$VLLM_LIMIT_MM_PER_PROMPT"

echo "======================================"
echo "推理完成!"
echo "结果保存至: $OUTPUT_FILE"
echo "======================================"

# ===================== 计算准确率 =====================
echo "计算准确率..."

python -c "
import json
import re

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''

results = []
with open('$OUTPUT_FILE', 'r') as f:
    for line in f:
        results.append(json.loads(line))

with open('/root/zhuweijie/Video-R1/src/r1-v/Evaluation/eval_tempcompass.json', 'r') as f:
    gt_data = json.load(f)

correct = 0
total = len(results)
dim_stats = {}

for result, gt in zip(results, gt_data):
    response = result.get('response', '')
    pred = extract_answer(response)
    gt_answer = extract_answer(gt.get('solution', ''))

    pred_letter = pred.strip().upper()[:1] if pred else ''
    gt_letter = gt_answer.strip().upper()[:1] if gt_answer else ''

    is_correct = (pred_letter == gt_letter)
    if is_correct:
        correct += 1

    dim = gt.get('dim', 'unknown')
    dim_stats.setdefault(dim, {'correct': 0, 'total': 0})
    dim_stats[dim]['total'] += 1
    if is_correct:
        dim_stats[dim]['correct'] += 1

print(f'\n总体准确率: {correct}/{total} = {correct/total*100:.2f}%')
print('\n各维度准确率:')
for dim, stats in sorted(dim_stats.items()):
    acc = stats['correct'] / stats['total'] * 100
    print(f'  {dim}: {stats[\"correct\"]}/{stats[\"total\"]} = {acc:.2f}%')
"

echo "======================================"
echo "评测完成!"
echo "======================================"
