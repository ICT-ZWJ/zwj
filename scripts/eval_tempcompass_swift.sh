#!/bin/bash
# TempCompass 评测脚本 - 使用 ms-swift 和 Video-R1 风格 Prompt
#
# 使用方法:
#   bash scripts/eval_tempcompass_swift.sh

# ===================== 配置区域 =====================

# 模型选择 (取消注释你要使用的模型)
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL="Qwen/Qwen3-VL-8B-Thinking"

# GPU 配置
GPUS="0,1,2,3,4,5,6,7"  # 使用的 GPU

# 视频采样参数 (Video-R1 推荐配置)
IMAGE_MAX_TOKEN_NUM=512
VIDEO_MAX_TOKEN_NUM=256   # 增大以支持更多帧
FPS_MAX_FRAMES=32         # 帧数 (Video-R1 推荐 16/32/64)

# 推理参数
MAX_NEW_TOKENS=1024       # 最大生成长度 (需要足够长以容纳推理过程)
INFER_BACKEND="pt"        # pt 或 vllm

# 数据路径
VAL_DATASET="/root/zhuweijie/Video-R1/src/r1-v/Evaluation/eval_tempcompass_swift.jsonl"

# 输出路径
OUTPUT_DIR="/root/zhuweijie/result/eval_tempcompass_videor1"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/tempcompass_${TIMESTAMP}.jsonl"

# ===================== 准备工作 =====================

echo "======================================"
echo "TempCompass 评测 (Video-R1 风格)"
echo "======================================"
echo "模型: $MODEL"
echo "GPU: $GPUS"
echo "帧数: $FPS_MAX_FRAMES"
echo "数据集: $VAL_DATASET"
echo "输出: $OUTPUT_FILE"
echo "======================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查数据集是否存在，不存在则转换
if [ ! -f "$VAL_DATASET" ]; then
    echo "转换数据集为 ms-swift 格式..."
    python /root/zhuweijie/scripts/convert_videor1_to_swift.py
fi

# ===================== 运行推理 =====================

CUDA_VISIBLE_DEVICES=$GPUS \
IMAGE_MAX_TOKEN_NUM=$IMAGE_MAX_TOKEN_NUM \
VIDEO_MAX_TOKEN_NUM=$VIDEO_MAX_TOKEN_NUM \
FPS_MAX_FRAMES=$FPS_MAX_FRAMES \
swift infer \
    --model "$MODEL" \
    --val_dataset "$VAL_DATASET" \
    --stream false \
    --infer_backend "$INFER_BACKEND" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --max_batch_size 16 \
    --result_path "$OUTPUT_FILE"

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

# 读取结果
results = []
with open('$OUTPUT_FILE', 'r') as f:
    for line in f:
        results.append(json.loads(line))

# 读取原始数据获取答案
with open('/root/zhuweijie/Video-R1/src/r1-v/Evaluation/eval_tempcompass.json', 'r') as f:
    gt_data = json.load(f)

# 计算准确率
correct = 0
total = len(results)
dim_stats = {}

for i, (result, gt) in enumerate(zip(results, gt_data)):
    response = result.get('response', '')
    pred = extract_answer(response)
    gt_answer = extract_answer(gt.get('solution', ''))
    
    # 提取选项字母
    pred_letter = pred.strip().upper()[:1] if pred else ''
    gt_letter = gt_answer.strip().upper()[:1] if gt_answer else ''
    
    is_correct = (pred_letter == gt_letter)
    if is_correct:
        correct += 1
    
    # 按维度统计
    dim = gt.get('dim', 'unknown')
    if dim not in dim_stats:
        dim_stats[dim] = {'correct': 0, 'total': 0}
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