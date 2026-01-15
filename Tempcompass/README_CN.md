# TempCompass 数据集完整指南

## 📊 数据集概览

TempCompass 是一个专注于**视频时序理解**的多选题数据集，用于评估视觉语言模型对视频中时序信息的理解能力。

### 核心统计
- ✅ **总样本数**: 1,580 条
- 🎬 **唯一视频数**: 410 个
- 📁 **数据维度**: 5 个类别
- 📝 **平均问题长度**: 136.3 字符
- 🎯 **平均问题数/视频**: 约 3.0-3.4 个

## 📂 项目结构

```
Tempcompass/
├── videos/                              # 410个视频文件
│   ├── {video_id}.mp4                  # 原始视频
│   ├── {video_id}_reverse.mp4          # 反向播放（测试方向）
│   └── {video_id}_concat_{n}.mp4       # 拼接视频（测试顺序）
│
├── test-00000-of-00001 (1).parquet     # 原始标注数据
│
├── 生成的数据文件/
│   ├── tempcompass_full.json           # 完整数据（1,580条）
│   ├── tempcompass_sample.json         # 示例数据（100条）
│   ├── video_mapping.json              # 视频->问题映射
│   └── data_by_dim/                    # 按维度分类的数据
│       ├── action.json                 # 338条
│       ├── direction.json              # 335条
│       ├── speed.json                  # 317条
│       ├── order.json                  # 302条
│       └── attribute_change.json       # 288条
│
├── 脚本工具/
│   ├── read_tempcompass.py             # 基础数据读取
│   ├── data_utils.py                   # 高级数据处理工具
│   └── 使用说明.md                      # 详细使用文档
│
└── README_CN.md                         # 本文件
```

## 🎯 五大测试维度

| 维度 | 样本数 | 占比 | 说明 | 示例 |
|------|--------|------|------|------|
| **action** | 338 | 21.4% | 动作识别 | "What is the man doing? A. dunking B. dribbling" |
| **direction** | 335 | 21.2% | 方向理解 | "What direction is the man moving? A. left to right B. towards camera" |
| **speed** | 317 | 20.1% | 速度判断 | "How fast is the object moving? A. fast B. slow" |
| **order** | 302 | 19.1% | 时序顺序 | "What happens first? A. opens door B. closes door" |
| **attribute_change** | 288 | 18.2% | 属性变化 | "What changes in the video? A. color B. size" |

## 🚀 快速开始

### 1. 环境准备

使用 **swift** conda 环境：

```bash
# 激活环境
conda activate swift

# 确认 pandas 已安装
conda run -n swift pip install pandas pyarrow
```

### 2. 基础数据读取

```bash
cd Tempcompass
conda run -n swift python read_tempcompass.py
```

这会生成：
- `tempcompass_full.json` - 完整数据集
- `tempcompass_sample.json` - 示例数据

### 3. 高级数据处理

```bash
conda run -n swift python data_utils.py
```

这会生成：
- 按维度分类的 JSON 文件（`data_by_dim/`）
- 视频到问题的映射（`video_mapping.json`）
- 详细的数据分析报告

## 💻 使用示例

### Python 代码示例

```python
import pandas as pd
import json

# 方式1：读取 Parquet 文件
df = pd.read_parquet("test-00000-of-00001 (1).parquet")

# 方式2：读取生成的 JSON
with open('tempcompass_full.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 按维度筛选
action_data = df[df['dim'] == 'action']
print(f"动作识别问题数: {len(action_data)}")

# 获取特定视频的所有问题
video_questions = df[df['video_id'] == '1034419625']
print(f"视频 1034419625 有 {len(video_questions)} 个问题")

# 统计每个视频的问题数
video_counts = df.groupby('video_id').size()
print(f"平均每个视频有 {video_counts.mean():.2f} 个问题")
```

### 使用工具类

```python
from data_utils import TempCompassDataset

# 初始化数据集
dataset = TempCompassDataset()

# 获取统计信息
stats = dataset.get_stats()
print(stats)

# 按维度筛选
action_df = dataset.filter_by_dimension('action')

# 获取视频路径
video_path = dataset.get_video_path('1034419625')
print(f"视频路径: {video_path}")

# 导出分类数据
dataset.export_by_dimension()
```

## 📋 数据格式

### Parquet/JSON 字段说明

```json
{
  "video_id": "1034419625",           // 视频ID（不含扩展名）
  "question": "What is the man...?",  // 多选题问题（包含选项A/B/C/D）
  "answer": "A. dunking...",          // 正确答案
  "dim": "action"                     // 问题维度类别
}
```

### 问题格式分析

- **选项数量分布**:
  - 2个选项: 6题 (0.4%)
  - 3个选项: 900题 (57.0%)
  - 4个选项: 674题 (42.6%)

- **问题长度**:
  - 最短: 57 字符
  - 最长: 456 字符
  - 平均: 136.3 字符

## 🎥 视频命名规则

1. **原始视频**: `{video_id}.mp4`
   - 例: `1034419625.mp4`

2. **反向视频**: `{video_id}_reverse.mp4`
   - 用途: 测试方向理解能力
   - 例: `1034419625_reverse.mp4`

3. **拼接视频**: `{video_id}_concat_{n}.mp4`
   - 用途: 测试时序顺序理解
   - 例: `1034419625_concat_0.mp4`

4. **组合视频**: `{video_id1}_{video_id2}_{n}.mp4`
   - 用途: 测试多视频片段理解
   - 例: `1024867412_1034049020_0.mp4`

## 🔧 常见任务

### 1. 按维度训练模型

```python
# 只使用动作识别数据
df_train = df[df['dim'] == 'action']

# 或使用工具类导出
dataset.export_by_dimension('train_data')
```

### 2. 视频问答对构建

```python
# 构建视频-问题对
for idx, row in df.iterrows():
    video_path = f"videos/{row['video_id']}.mp4"
    question = row['question']
    answer = row['answer']
    # 加载视频和问题进行训练...
```

### 3. 评估模型性能

```python
# 按维度评估准确率
for dim in df['dim'].unique():
    dim_data = df[df['dim'] == dim]
    # 评估模型在该维度的表现...
    print(f"{dim} 维度准确率: {accuracy:.2%}")
```

## 📈 数据洞察

### 维度分布
- 五个维度的样本数较为均衡（18-21%）
- 每个视频平均有 3-3.4 个问题
- 不同维度可能测试同一视频的不同方面

### 视频特点
- 410个唯一视频ID
- 许多视频有多个变体（原始、反向、拼接）
- 视频变体用于测试模型对时序信息的敏感度

### 问题特点
- 多数为3-4选项的多选题
- 问题覆盖动作、方向、速度、顺序、属性变化
- 设计用于全面评估视频时序理解能力

## 🛠️ 可用工具

### 1. `read_tempcompass.py`
基础数据读取和转换工具

**功能**:
- 读取 Parquet 文件
- 转换为 JSON 格式
- 显示基本统计信息

### 2. `data_utils.py`
高级数据处理工具类

**功能**:
- 按维度筛选和导出
- 视频-问题映射
- 数据统计分析
- 训练/测试集划分

## 📚 相关资源

- **原始数据**: `test-00000-of-00001 (1).parquet`
- **完整文档**: `使用说明.md`
- **视频文件**: `videos/` 目录（410个MP4文件）

## ⚠️ 注意事项

1. **文件名特殊字符**: Parquet 文件名包含空格和括号，在shell中使用时需要注意转义
2. **环境选择**: 必须使用 **swift** conda 环境，已安装所需依赖
3. **数据完整性**: 部分视频ID对应多个视频文件（原始、反向、拼接等）
4. **内存占用**: 完整数据集较小（1580条），可以一次性加载到内存

## 🎓 推荐工作流

1. **探索阶段**: 运行 `read_tempcompass.py` 了解数据结构
2. **分析阶段**: 运行 `data_utils.py` 生成详细分析
3. **开发阶段**: 使用生成的 JSON 文件或直接读取 Parquet
4. **训练阶段**: 按维度或按视频划分训练/测试集
5. **评估阶段**: 按维度评估模型性能，识别薄弱环节

## 📞 支持

如有问题，请参考：
- `使用说明.md` - 详细使用文档
- `data_utils.py` - 工具类源码和示例
- 生成的 JSON 文件 - 可直接查看数据格式

---

**最后更新**: 2026-01-13  
**环境要求**: Python 3.10+, pandas, pyarrow  
**推荐环境**: conda 环境 `swift`