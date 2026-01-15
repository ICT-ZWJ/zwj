import torch
import sys
import time
import random
import argparse

# ---------------- 参数解析 ----------------
parser = argparse.ArgumentParser()
parser.add_argument("gpu_id", type=int, help="GPU ID")
parser.add_argument("--target_util", type=int, default=85, help="目标利用率 (例如 85 表示 80–90%)")
parser.add_argument("--target_mem", type=int, default=10, help="目标显存 (MB)")
parser.add_argument("--max_mem", type=int, default=50, help="显存上限 (MB)")
args = parser.parse_args()

gpu_id = args.gpu_id
target_util = args.target_util
target_mem_mb = args.target_mem
max_mem_mb = args.max_mem

device = torch.device(f"cuda:{gpu_id}")
torch.cuda.set_device(device)
torch.manual_seed(7)

# 显存限制（按百分比）
torch.cuda.set_per_process_memory_fraction(0.01, device)  # 约等于1%显存

# ---------------- 动态选择 N ----------------
def estimate_mem_mb(N):
    # 只计算 features 和 weights
    bytes_needed = 2 * (N * N * 4)  # 两个 N×N float32 矩阵
    return bytes_needed / 1024 / 1024

N = 32
while estimate_mem_mb(N) < target_mem_mb and estimate_mem_mb(N*2) < max_mem_mb:
    N *= 2

print(f"[INFO] Using N={N}, estimated memory {estimate_mem_mb(N):.2f} MB")

# ---------------- 初始化数据 ----------------
features = torch.randn(N, N, device=device)
weights = torch.randn(N, N, device=device)

print(f"[INFO] Actual allocated memory: {torch.cuda.memory_allocated(device)/1024/1024:.2f} MB")

# ---------------- 主循环 ----------------
# 根据 target_util 动态设置基础 workload
base_steps = int((target_util - 70) * 50)  # 粗略映射

while True:
    with torch.no_grad():
        y = features

        # 在 [base_steps*0.8, base_steps*1.2] 范围内随机 workload
        steps = random.randint(int(base_steps * 0.8), int(base_steps * 1.2))
        for _ in range(steps):
            y = torch.matmul(y, weights)
            y = torch.sigmoid(y)

    torch.cuda.synchronize()

    # 随机 sleep，制造波动
    if random.random() < 0.3:
        time.sleep(random.uniform(0.002, 0.01))