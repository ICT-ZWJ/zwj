#!/usr/bin/env python3
"""
TempCompass 数据集读取示例脚本
"""

import pandas as pd
import json

def read_parquet_file(parquet_path):
    """读取 parquet 文件并返回 DataFrame"""
    try:
        df = pd.read_parquet(parquet_path)
        return df
    except Exception as e:
        print(f"读取 parquet 文件失败: {e}")
        return None

def explore_dataframe(df):
    """探索 DataFrame 的基本信息"""
    print("=" * 80)
    print("数据集基本信息")
    print("=" * 80)
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n前 5 行数据预览:")
    print(df.head())
    print("\n" + "=" * 80)

def save_as_json(df, output_path, max_rows=None):
    """将 DataFrame 保存为 JSON 文件"""
    if max_rows:
        df_to_save = df.head(max_rows)
    else:
        df_to_save = df
    
    json_data = df_to_save.to_dict(orient='records')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"已保存 {len(json_data)} 条记录到 {output_path}")

def main():
    parquet_file = "test-00000-of-00001 (1).parquet"
    
    print("正在读取 TempCompass 数据集...")
    df = read_parquet_file(parquet_file)
    
    if df is None:
        return
    
    explore_dataframe(df)
    
    if 'dim' in df.columns:
        print("\n数据集中的所有维度 (dim):")
        print(df['dim'].value_counts())
    
    print("\n正在保存完整数据为 JSON...")
    save_as_json(df, "tempcompass_full.json")
    
    print("\n正在保存前 100 条数据作为示例...")
    save_as_json(df, "tempcompass_sample.json", max_rows=100)
    
    if 'video_id' in df.columns:
        unique_videos = df['video_id'].nunique()
        print(f"\n唯一视频数量: {unique_videos}")
        print(f"前 10 个视频 ID:")
        print(df['video_id'].head(10).tolist())

if __name__ == "__main__":
    main()