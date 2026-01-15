#!/bin/bash
# Qwen2.5-VL 评测脚本 - ms-swift 框架适配
# 
# 数据集下载位置说明:
# ====================
# 
# 1. 评测 JSON 文件 (必须下载):
#    来源: https://huggingface.co/datasets/Video-R1/Video-R1-eval
#    放置: ./Video-R1/src/r1-v/Evaluation/
#    文件: eval_mvbench.json, eval_tempcompass.json, eval_videomme.json 等
#
# 2. 各 Benchmark 视频数据:
#    - TempCompass: 你已有 ./Tempcompass/videos/ (无需下载)
#    - MVBench: https://github.com/OpenGVLab/Ask-Anything
#    - VideoMME: https://video-mme.github.io/
#    - VideoMMMU: https://huggingface.co/datasets/lmms-lab/Video-MMMU
#    - VSI-Bench: https://vision-x-nyu.github.io/VSI-Bench/
#
# 使用方法:
# =========
#    bash run_eval_msswift.sh

# ===================== 配置区域 =====================

# 模型路径 (本地路径或 ModelScope/HuggingFace 模型名)
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# LoRA Adapters 路径 (如果有微调模型，填写 checkpoint 路径)
# 例如: ADAPTERS="output/vx-xxx/checkpoint-xxx"
ADAPTERS=""

# 数据集选择: tempcompass, mvbench, videomme, videommmu, vsibench, mmvu
DATASET="tempcompass"

# 数据集根目录
# - tempcompass: ./Tempcompass (视频在 ./Tempcompass/videos/)
# - 其他: ./Video-R1/src/r1-v/Evaluation/
DATA_ROOT="./Tempcompass"

# 输出目录
OUTPUT_DIR="./result/eval_msswift"

# 输出文件名前缀
OUTPUT_NAME="qwen25vl_tempcompass"

# 是否使用 sample 数据集 (快速测试用)
USE_SAMPLE="--use_sample"  # 注释掉此行使用完整数据集

# 最大样本数 (-1 表示全部)
MAX_SAMPLES=100  # 设为 -1 评测全部

# 视频参数
MAX_PIXELS=200704     # 每帧最大像素
NFRAMES=32            # 最大帧数 (16/32/64)

# 最大生成 token
MAX_TOKENS=1024

# ===================== 运行评测 =====================

echo "======================================"
echo "Qwen2.5-VL 评测 (ms-swift 适配)"
echo "======================================"
echo "模型: $MODEL_PATH"
echo "Adapters: ${ADAPTERS:-None}"
echo "数据集: $DATASET"
echo "数据目录: $DATA_ROOT"
echo "最大帧数: $NFRAMES"
echo "======================================"

# 构建命令
CMD="python eval_qwen25vl_msswift.py \
    --model_path \"$MODEL_PATH\" \
    --dataset \"$DATASET\" \
    --data_root \"$DATA_ROOT\" \
    --output_dir \"$OUTPUT_DIR\" \
    --output_name \"$OUTPUT_NAME\" \
    --max_samples $MAX_SAMPLES \
    --max_tokens $MAX_TOKENS \
    --max_pixels $MAX_PIXELS \
    --nframes $NFRAMES"

# 添加可选参数
if [ -n "$ADAPTERS" ]; then
    CMD="$CMD --adapters \"$ADAPTERS\""
fi

if [ -n "$USE_SAMPLE" ]; then
    CMD="$CMD $USE_SAMPLE"
fi

# 执行
eval $CMD

echo "======================================"
echo "评测完成!"
echo "======================================"