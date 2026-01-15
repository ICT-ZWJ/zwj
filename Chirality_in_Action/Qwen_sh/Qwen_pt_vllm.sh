NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=1003520 \
swift infer \
    --model Qwen/Qwen3-VL-8B-Thinking \
    --infer_backend pt \
    --val_dataset /root/zhuweijie/Chirality_in_Action/test_dataset/all_merged.json \
    --max_batch_size 16 \
    --max_new_tokens 512

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# MAX_PIXELS=1003520 \
# swift infer \
#     --model Qwen/Qwen3-VL-8B-Thinking \
#     --infer_backend vllm \
#     --val_dataset /root/zhuweijie/Chirality_in_Action/test_dataset/all_merged.json \
#     --vllm_gpu_memory_utilization 0.9 \
#     --vllm_tensor_parallel_size 2 \
#     --vllm_max_model_len 32768 \
#     --max_new_tokens 2048 \
#     --vllm_limit_mm_per_prompt '{"image": 5, "video": 2}'