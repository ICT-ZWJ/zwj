#!/bin/bash
# Qwen2.5-VL 评测脚本 - Video-R1 风格
# 使用方法: bash run_eval_qwen25vl.sh

# ===================== 配置区域 =====================

# 模型路径 (本地路径或 HuggingFace 模型名)
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# 数据集选择: tempcompass, mvbench, videomme, videommmu, vsibench, mmvu
DATASET="tempcompass"

# Tempcompass 数据根目录
DATA_ROOT="./Tempcompass"

# 输出目录
OUTPUT_DIR="./result/eval_videor1_style"

# 输出文件名前缀
OUTPUT_NAME="qwen25vl_tempcompass"

# 推理参数
BATCH_SIZE=8          # 批处理大小，根据显存调整
MAX_MODEL_LEN=16384   # 最大模型长度
GPU_MEMORY=0.8        # GPU 显存利用率
TEMPERATURE=0.1       # 采样温度 (Video-R1 默认 0.1)
TOP_P=0.001           # Top-p (Video-R1 默认 0.001，非常低保证稳定输出)
MAX_TOKENS=1024       # 最大生成 token

# 视频参数
MAX_PIXELS=200704     # 每帧最大像素 (256 × 28 × 28)
NFRAMES=32            # 最大帧数 (16/32/64)

# 是否使用 sample 数据集 (测试用)
USE_SAMPLE=""  # 设置为 "--use_sample" 启用

# 是否断点续评
RESUME=""  # 设置为 "--resume" 启用

# ===================== 运行评测 =====================

export DECORD_EOF_RETRY_MAX=20480

echo "======================================"
echo "Qwen2.5-VL 评测 (Video-R1 风格)"
echo "======================================"
echo "模型: $MODEL_PATH"
echo "数据集: $DATASET"
echo "数据目录: $DATA_ROOT"
echo "输出目录: $OUTPUT_DIR"
echo "批处理大小: $BATCH_SIZE"
echo "最大帧数: $NFRAMES"
echo "======================================"

python eval_qwen25vl_videor1_style.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME" \
    --batch_size $BATCH_SIZE \
    --max_model_len $MAX_MODEL_LEN \
    --gpu_memory_utilization $GPU_MEMORY \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_tokens $MAX_TOKENS \
    --max_pixels $MAX_PIXELS \
    --nframes $NFRAMES \
    $USE_SAMPLE \
    $RESUME

echo "======================================"
echo "评测完成!"
echo "======================================"