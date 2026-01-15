import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"可用GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU计算能力: {torch.cuda.get_device_capability(0)}")