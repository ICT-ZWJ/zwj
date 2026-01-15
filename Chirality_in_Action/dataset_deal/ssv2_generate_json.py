import json

# 定义视频ID和对应的标签
video_data = {
    # id_forward 列的视频 - 标签是 verb_forward
    "forward": {
        "label": "Pulling [something] from left to right",
        "answer": "A",
        "ids": [
            216055, 62874, 103661, 54495, 197762, 72618, 92158, 196602, 
            47325, 18481, 129654, 188239, 98886, 140370, 32201, 77208, 
            5139, 99830, 169771, 184572, 69324, 148138, 162511, 86575, 
            86333, 198107, 130186, 48710, 57358, 129488, 17519, 140317, 
            218468, 133541, 152497, 163441, 217557, 84967, 88927, 185743, 
            6848, 111829, 64431, 133624, 182229, 214050, 99643
        ]
    },
    # id_reverse 列的视频 - 标签是 verb_reverse
    "reverse": {
        "label": "Pulling [something] from right to left",
        "answer": "B",
        "ids": [
            52114, 147014, 48456, 29941, 90646, 110620, 121844, 29373, 
            23202, 181899, 115717, 167711, 130178, 175308, 62873, 24815, 
            90357, 143775, 77092, 158690, 208718, 114180, 28528, 
            156523, 205686, 205013, 168785, 220302, 60477
        ]
    }
}

# 生成JSON数据
json_data = []

base_path = "/root/zhuweijie/Chirality_in_Action/test_dataset"

# 处理 forward 视频
for video_id in video_data["forward"]["ids"]:
    entry = {
        "messages": [
            {
                "role": "system",
                "content": "You are a video action recognition assistant. Please watch the video carefully and select the correct action from the given options. Only answer with the option letter (A or B)."
            },
            {
                "role": "user",
                "content": "<video>Which action?\nA. Pulling [something] from left to right\nB. Pulling [something] from right to left"
            },
            {
                "role": "assistant",
                "content": video_data["forward"]["answer"]
            }
        ],
        "videos": [f"{base_path}/{video_id}.mp4"],
        "label": video_data["forward"]["label"],
        "answer": video_data["forward"]["answer"]
    }
    json_data.append(entry)

# 处理 reverse 视频
for video_id in video_data["reverse"]["ids"]:
    entry = {
        "messages": [
            {
                "role": "system",
                "content": "You are a video action recognition assistant. Please watch the video careful and select the correct action from the given options. Only answer with the option letter (A or B)."
            },
            {
                "role": "user",
                "content": "<video>Which action?\nA. Pulling [something] from left to right\nB. Pulling [something] from right to left"
            },
            {
                "role": "assistant",
                "content": video_data["reverse"]["answer"]
            }
        ],
        "videos": [f"{base_path}/{video_id}.mp4"],
        "label": video_data["reverse"]["label"],
        "answer": video_data["reverse"]["answer"]
    }
    json_data.append(entry)

# 保存为JSON文件
output_file = "/root/zhuweijie/Chirality_in_Action/test_dataset/annotations.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"✅ JSON文件已生成: {output_file}")
print(f"📊 总共生成 {len(json_data)} 条数据")
print(f"   - Forward (A): {len(video_data['forward']['ids'])} 个视频")
print(f"   - Reverse (B): {len(video_data['reverse']['ids'])} 个视频")