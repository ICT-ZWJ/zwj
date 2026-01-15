import os
from transformers import AutoTokenizer, AutoModel

def download_models_to_specific_path():
    # 1. 设置目标绝对路径 (使用 r前缀表示原始字符串，防止Windows路径中的反斜杠转义问题)
    base_save_directory = r"E:\ITS\ISTS-PLM\PLMs"

    # 2. 定义要下载的模型列表 (使用 Hugging Face 的完整 ID)
    # 这里包含了您要求的 bert 和 gpt2，您也可以在列表中添加其他模型
    models = [
        "google-bert/bert-base-uncased",  # 对应 BERT
        "openai-community/gpt2"           # 对应 GPT-2
    ]

    print(f"准备将模型下载到: {base_save_directory}\n")

    for model_id in models:
        try:
            # 获取模型的简短名称用于创建子文件夹 (例如: bert-base-uncased)
            model_name = model_id.split("/")[-1]
            
            # 拼接完整的保存路径: E:\ITS\ISTS-PLM\PLMs\bert-base-uncased
            save_path = os.path.join(base_save_directory, model_name)

            # 如果路径不存在，则创建
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            print(f"正在处理: {model_id} ...")

            # --- 下载 Tokenizer ---
            print(f"  正在下载 Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # --- 下载 Model ---
            # 使用 AutoModel 可以适应各种架构 (BERT, GPT, RoBERTa, T5 等)
            print(f"  正在下载 Model 权重...")
            model = AutoModel.from_pretrained(model_id)

            # --- 保存到指定文件夹 ---
            print(f"  正在保存到: {save_path}")
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)

            print(f"✅ 成功: {model_name} 已就绪。\n")

        except Exception as e:
            print(f"❌ 错误: 下载 {model_id} 失败。原因: {e}\n")

    print("所有下载任务完成。")

if __name__ == "__main__":
    download_models_to_specific_path()