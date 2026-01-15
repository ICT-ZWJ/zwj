# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# IMAGE_MAX_TOKEN_NUM=512 \
# VIDEO_MAX_TOKEN_NUM=64 \
# FPS_MAX_FRAMES=16 \
# swift infer \
#     --model Qwen/Qwen3-VL-8B-Thinking \
#     --val_dataset /root/zhuweijie/Chirality_in_Action/test_dataset/all_merged.json \
#     --stream false\
#     --infer_backend pt \
#     --max_new_tokens 2048\

cd /root/zhuweijie

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=512 \
VIDEO_MAX_TOKEN_NUM=256 \
FPS_MAX_FRAMES=32 \
swift infer \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --val_dataset /root/zhuweijie/Video-R1/src/r1-v/Evaluation/eval_tempcompass_swift.jsonl \
    --stream false \
    --infer_backend pt \
    --max_new_tokens 2048