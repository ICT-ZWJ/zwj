#!/usr/bin/env python3
"""
TempCompass 数据集工具函数
提供常用的数据处理和分析功能
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional

class TempCompassDataset:
    """TempCompass 数据集处理类"""
    
    def __init__(self, parquet_path: str = "test-00000-of-00001 (1).parquet"):
        """初始化数据集"""
        self.df = pd.read_parquet(parquet_path)
        self.video_dir = Path("videos")
        
    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        return {
            "总样本数": len(self.df),
            "唯一视频数": self.df['video_id'].nunique(),
            "维度分布": self.df['dim'].value_counts().to_dict(),
            "列名": list(self.df.columns)
        }
    
    def filter_by_dimension(self, dim: str) -> pd.DataFrame:
        """按维度筛选数据"""
        return self.df[self.df['dim'] == dim]
    
    def get_video_questions(self, video_id: str) -> pd.DataFrame:
        """获取特定视频的所有问题"""
        return self.df[self.df['video_id'] == video_id]
    
    def get_video_path(self, video_id: str) -> str:
        """获取视频文件路径"""
        video_file = self.video_dir / f"{video_id}.mp4"
        if video_file.exists():
            return str(video_file)
        return None
    
    def export_by_dimension(self, output_dir: str = "data_by_dim"):
        """按维度导出数据到单独的JSON文件"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for dim in self.df['dim'].unique():
            dim_data = self.filter_by_dimension(dim)
            output_file = output_path / f"{dim}.json"
            
            json_data = dim_data.to_dict(orient='records')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"已保存 {len(json_data)} 条 {dim} 数据到 {output_file}")
    
    def get_train_test_split(self, test_ratio: float = 0.2, random_state: int = 42):
        """按视频ID进行训练/测试集划分（确保同一视频的所有问题在同一集合中）"""
        unique_videos = self.df['video_id'].unique()
        
        from sklearn.model_selection import train_test_split
        train_videos, test_videos = train_test_split(
            unique_videos, 
            test_size=test_ratio, 
            random_state=random_state
        )
        
        train_df = self.df[self.df['video_id'].isin(train_videos)]
        test_df = self.df[self.df['video_id'].isin(test_videos)]
        
        return train_df, test_df
    
    def create_video_dataset_mapping(self, output_file: str = "video_mapping.json"):
        """创建视频到问题的映射"""
        mapping = {}
        
        for video_id in self.df['video_id'].unique():
            questions = self.get_video_questions(video_id)
            mapping[video_id] = {
                "video_path": f"videos/{video_id}.mp4",
                "num_questions": len(questions),
                "dimensions": questions['dim'].unique().tolist(),
                "questions": questions[['question', 'answer', 'dim']].to_dict('records')
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        print(f"已保存视频映射到 {output_file}")
        return mapping
    
    def analyze_question_patterns(self):
        """分析问题模式"""
        print("\n=== 问题模式分析 ===")
        
        # 问题长度分布
        self.df['question_length'] = self.df['question'].str.len()
        print(f"\n平均问题长度: {self.df['question_length'].mean():.1f} 字符")
        print(f"最短问题: {self.df['question_length'].min()} 字符")
        print(f"最长问题: {self.df['question_length'].max()} 字符")
        
        # 选项数量分析
        self.df['num_options'] = self.df['question'].str.count('\n[A-Z]\.')
        print(f"\n选项数量分布:")
        print(self.df['num_options'].value_counts().sort_index())
        
        # 按维度统计
        print(f"\n各维度的平均问题数:")
        for dim in self.df['dim'].unique():
            dim_data = self.filter_by_dimension(dim)
            avg_questions = len(dim_data) / dim_data['video_id'].nunique()
            print(f"  {dim}: {avg_questions:.2f} 问题/视频")


def main():
    """主函数示例"""
    print("正在加载 TempCompass 数据集...")
    dataset = TempCompassDataset()
    
    # 显示统计信息
    print("\n数据集统计信息:")
    stats = dataset.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 分析问题模式
    dataset.analyze_question_patterns()
    
    # 按维度导出
    print("\n按维度导出数据...")
    dataset.export_by_dimension()
    
    # 创建视频映射
    print("\n创建视频到问题的映射...")
    dataset.create_video_dataset_mapping()
    
    # 示例：获取某个视频的所有问题
    sample_video = dataset.df['video_id'].iloc[0]
    print(f"\n示例：视频 {sample_video} 的问题:")
    video_questions = dataset.get_video_questions(sample_video)
    for idx, row in video_questions.head(3).iterrows():
        print(f"\n问题 {idx + 1}:")
        print(f"  维度: {row['dim']}")
        print(f"  问题: {row['question'][:100]}...")
        print(f"  答案: {row['answer']}")


if __name__ == "__main__":
    main()