import json

# 定义视频ID和对应的标签
video_data = {
    # Closing door - 对应第一组ID
    "closing": {
        "label": "Closing door",
        "answer": "A",
        "ids": [
            139370, 143971, 24384, 43004, 18971, 197774, 129432, 151821, 
            87379, 102791, 124242, 82636, 151541, 128813, 142710, 208684, 
            203240, 194380, 220290
        ]
    },
    # Opening door - 对应第二组ID
    "opening": {
        "label": "Opening door",
        "answer": "B",
        "ids": [
            211653, 157167, 207122, 23995, 73233, 177502, 67287, 152453, 
            42860, 116099, 124039, 4372, 54551, 177365, 99508, 114056, 
            81100, 155015, 29554, 157951, 156401, 96935, 217420, 166636, 
            96366, 78066, 106135, 49005, 47953, 15959, 55006
        ]
    }
}

# 生成JSON数据
json_data = []

base_path = "/root/zhuweijie/Chirality_in_Action/test_dataset/close_open_door"

# 处理 Closing 视频
for video_id in video_data["closing"]["ids"]:
    entry = {
        "messages": [
            {
                "role": "system",
                "content": "You are a video action recognition assistant. Please watch the video carefully and select the correct action from the given options. Only answer with the option letter (A or B)."
            },
            {
                "role": "user",
                "content": "<video>Which action?\nA. Closing door\nB. Opening door"
            },
            {
                "role": "assistant",
                "content": video_data["closing"]["answer"]
            }
        ],
        "videos": [f"{base_path}/{video_id}.mp4"],
        "label": video_data["closing"]["label"],
        "answer": video_data["closing"]["answer"]
    }
    json_data.append(entry)

# 处理 Opening 视频
for video_id in video_data["opening"]["ids"]:
    entry = {
        "messages": [
            {
                "role": "system",
                "content": "You are a video action recognition assistant. Please watch the video carefully and select the correct action from the given options. Only answer with the option letter (A or B)."
            },
            {
                "role": "user",
                "content": "<video>Which action?\nA. Closing door\nB. Opening door"
            },
            {
                "role": "assistant",
                "content": video_data["opening"]["answer"]
            }
        ],
        "videos": [f"{base_path}/{video_id}.mp4"],
        "label": video_data["opening"]["label"],
        "answer": video_data["opening"]["answer"]
    }
    json_data.append(entry)

# 保存为JSON文件
output_file = "/root/zhuweijie/Chirality_in_Action/test_dataset/close_open_door/close_door.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"✅ JSON文件已生成: {output_file}")
print(f"📊 总共生成 {len(json_data)} 条数据")
print(f"   - Closing door (A): {len(video_data['closing']['ids'])} 个视频")
print(f"   - Opening door (B): {len(video_data['opening']['ids'])} 个视频")