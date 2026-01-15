import os

def clean_folder_name(verb, noun):
    folder_name = verb.replace('[something]', '').replace("['something']", '').replace("['object']", '').replace("['door']", '').replace("['bottle']", '').replace("['book']", '').replace("['purse']", '').replace("['drawer']", '')
    folder_name = "".join([c if c.isalnum() or c.isspace() else "" for c in folder_name])
    folder_name = "_".join(folder_name.split())
    if noun != "something":
        folder_name += f"_{noun}"
    return folder_name

def process_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip header
    data_lines = lines[1:]

    script_content = []
    script_content.append("import json")
    script_content.append("import os")
    script_content.append("")
    script_content.append("base_output_dir = '/root/zhuweijie/Chirality_in_Action/test_dataset'")
    script_content.append("")
    script_content.append("# 定义所有分类的数据")
    script_content.append("categories = [")

    for line in data_lines:
        if not line.strip(): continue
        parts = line.strip().split('\t')
        if len(parts) < 6: continue
        
        verb_fwd = parts[0]
        verb_rev = parts[1]
        noun = parts[2].replace("['", "").replace("']", "")
        
        ids_fwd = [int(x.strip()) for x in parts[4].split(';') if x.strip()]
        ids_rev = [int(x.strip()) for x in parts[5].split(';') if x.strip()]
        
        folder_name = clean_folder_name(verb_fwd, noun)
        
        # 决定输出 JSON 文件名，通常使用第一个单词或 cleaned folder name 的一部分
        # 为了避免文件名过长，我们可以直接使用 folder_name.lower()
        json_filename = folder_name.lower() + ".json"

        category_entry = {
            "folder_name": folder_name,
            "json_filename": json_filename,
            "option_a": {
                "label": verb_fwd,
                "answer": "A",
                "ids": ids_fwd
            },
            "option_b": {
                "label": verb_rev,
                "answer": "B",
                "ids": ids_rev
            }
        }
        
        # 将字典转换为格式化的字符串添加到脚本中
        script_content.append("    {")
        script_content.append(f"        'folder_name': '{folder_name}',")
        script_content.append(f"        'json_filename': '{json_filename}',")
        script_content.append("        'option_a': {")
        script_content.append(f"            'label': {repr(verb_fwd)},")
        script_content.append(f"            'answer': 'A',")
        script_content.append(f"            'ids': {repr(ids_fwd)}")
        script_content.append("        },")
        script_content.append("        'option_b': {")
        script_content.append(f"            'label': {repr(verb_rev)},")
        script_content.append(f"            'answer': 'B',")
        script_content.append(f"            'ids': {repr(ids_rev)}")
        script_content.append("        }")
        script_content.append("    },")

    script_content.append("]")
    script_content.append("")
    
    # 添加生成逻辑
    script_content.append("# 遍历每个分类生成对应的 JSON 文件")
    script_content.append("for cat in categories:")
    script_content.append("    json_data = []")
    script_content.append("    base_path = os.path.join(base_output_dir, cat['folder_name'])")
    script_content.append("")
    script_content.append("    # 处理 Option A (正向)")
    script_content.append("    for video_id in cat['option_a']['ids']:")
    script_content.append("        entry = {")
    script_content.append("            'messages': [")
    script_content.append("                {")
    script_content.append("                    'role': 'system',")
    script_content.append("                    'content': 'You are a video action recognition assistant. Please watch the video carefully and select the correct action from the given options. Only answer with the option letter (A or B).'")
    script_content.append("                },")
    script_content.append("                {")
    script_content.append("                    'role': 'user',")
    script_content.append("                    'content': f'<video>Which action?\\nA. {cat[\"option_a\"][\"label\"]}\\nB. {cat[\"option_b\"][\"label\"]}'")
    script_content.append("                },")
    script_content.append("                {")
    script_content.append("                    'role': 'assistant',")
    script_content.append("                    'content': cat['option_a']['answer']")
    script_content.append("                }")
    script_content.append("            ],")
    script_content.append("            'videos': [f'{base_path}/{video_id}.mp4'],")
    script_content.append("            'label': cat['option_a']['label'],")
    script_content.append("            'answer': cat['option_a']['answer']")
    script_content.append("        }")
    script_content.append("        json_data.append(entry)")
    script_content.append("")
    script_content.append("    # 处理 Option B (反向)")
    script_content.append("    for video_id in cat['option_b']['ids']:")
    script_content.append("        entry = {")
    script_content.append("            'messages': [")
    script_content.append("                {")
    script_content.append("                    'role': 'system',")
    script_content.append("                    'content': 'You are a video action recognition assistant. Please watch the video carefully and select the correct action from the given options. Only answer with the option letter (A or B).'")
    script_content.append("                },")
    script_content.append("                {")
    script_content.append("                    'role': 'user',")
    script_content.append("                    'content': f'<video>Which action?\\nA. {cat[\"option_a\"][\"label\"]}\\nB. {cat[\"option_b\"][\"label\"]}'")
    script_content.append("                },")
    script_content.append("                {")
    script_content.append("                    'role': 'assistant',")
    script_content.append("                    'content': cat['option_b']['answer']")
    script_content.append("                }")
    script_content.append("            ],")
    script_content.append("            'videos': [f'{base_path}/{video_id}.mp4'],")
    script_content.append("            'label': cat['option_b']['label'],")
    script_content.append("            'answer': cat['option_b']['answer']")
    script_content.append("        }")
    script_content.append("        json_data.append(entry)")
    script_content.append("")
    script_content.append("    # 保存为JSON文件")
    script_content.append("    output_file = os.path.join(base_output_dir, cat['json_filename'])")
    script_content.append("    with open(output_file, 'w', encoding='utf-8') as f:")
    script_content.append("        json.dump(json_data, f, indent=2, ensure_ascii=False)")
    script_content.append("")
    script_content.append("    print(f'✅ JSON文件已生成: {output_file}')")
    script_content.append("    print(f'   - 包含数据: {len(json_data)} 条')")

    # Write to file
    with open("Chirality_in_Action/dataset_deal/ssv2_generate_json_folder.py", "w") as out:
        out.write("\n".join(script_content))

if __name__ == "__main__":
    process_file("ids_data.txt")