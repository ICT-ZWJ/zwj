import torchvision
import torch
import os

# 使用之前看到的一个视频路径作为样本
video_path = "/root/zhuweijie/Chirality_in_Action/ssv2_valid_dataset/Approaching_with_your_camera/106553.mp4"

print(f"Checking torchvision version: {torchvision.__version__}")
print(f"Checking video: {video_path}")

if not os.path.exists(video_path):
    print(f"File not found: {video_path}")
    # 尝试找一个存在的视频
    base_dir = "/root/zhuweijie/Chirality_in_Action/ssv2_valid_dataset"
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                print(f"Found alternative video: {video_path}")
                break
        if video_path != "/root/zhuweijie/Chirality_in_Action/ssv2_valid_dataset/Approaching_with_your_camera/106553.mp4":
            break

try:
    # 模拟报错代码中的调用方式
    video, audio, info = torchvision.io.read_video(video_path, pts_unit='sec')
    print("Success!")
    print(f"Info keys: {info.keys()}")
    print(f"Info content: {info}")
    
    if 'video_fps' in info:
        print(f"video_fps: {info['video_fps']}")
    else:
        print("ERROR: 'video_fps' key is missing!")
except Exception as e:
    print(f"Error reading video: {e}")