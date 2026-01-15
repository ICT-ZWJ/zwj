CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=512 \
VIDEO_MAX_TOKEN_NUM=64 \
FPS_MAX_FRAMES=16 \
swift infer \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --val_dataset /root/zhuweijie/Chirality_in_Action/test_dataset/all_merged.json \
    --stream false\
    --infer_backend pt \
    --max_new_tokens 2048