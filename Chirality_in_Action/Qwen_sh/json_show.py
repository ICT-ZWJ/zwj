import streamlit as st
import json
import os
import pandas as pd
import random
import time
from pathlib import Path

# ==========================================
# 0. 全局配置与状态初始化
# ==========================================

st.set_page_config(page_title="JSONL 数据浏览器 (增强版)", layout="wide", page_icon="⚡")

# 单次加载的最大扫描行数
MAX_SCAN_LINES_PER_LOAD = 50000 

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = []
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "表格概览"

# --- 正式状态 (用于渲染，仅在点击按钮后更新) ---
if 'data_cols_count' not in st.session_state:
    st.session_state.data_cols_count = 2 
if 'json_collapse_all' not in st.session_state:
    st.session_state.json_collapse_all = True # 默认折叠

# --- 临时配置状态 (用于侧边栏控件，实时更新) ---
if 'data_cols_count_config' not in st.session_state:
    st.session_state.data_cols_count_config = st.session_state.data_cols_count
if 'json_collapse_all_config' not in st.session_state:
    st.session_state.json_collapse_all_config = st.session_state.json_collapse_all

# ==========================================
# 1. 预设路径配置
# ==========================================

JSONL_FILES = [
    # "/root/code-yck/tmp/feed/pretrian/video/tmp/video_tags/bilibili_video_2024_2days_1000_digg_count.jsonl",
    # "/root/code-yck/tmp/feed/pretrian/video/tmp/video_tags/douyin_video_2024_7days_3000_digg_count.jsonl",
    # "/root/code-yck/tmp/feed/pretrian/video/tmp/video_tags/video_middle_quality",
    # "/root/code-yck/tmp/feed/pretrian/video/tmp/video_tags/xiaohongshu_video_digg_count.jsonl",
    # "/root/code-yck/tmp/eb/text2image/clip_quchong",
    # "/root/code-yck/tmp/feed/rl/songhe",
    # "/root/code-yck/tmp/feed/test/eb5",
    # # "/root/code-yck/tmp/feed/sft/ruixi",
    "/root/zhuweijie/result/response_answer/tempcompass_qwen2.5.wrong.jsonl",
    "/root/zhuweijie/result/eval_tempcompass_videor1/tempcompass_qwen3.jsonl",
    "/root/zhuweijie/Chirality_in_Action/Test_Result/Qwen3/Qwen3_ssv2_mismatch.jsonl",
    "/root/zhuweijie/result/response_answer/tempcompass_qwen2.5.wrong.jsonl.by_dim/action.jsonl",
    "/root/zhuweijie/result/response_answer/tempcompass_qwen2.5.wrong.jsonl.by_dim/attribute_change.jsonl",
    "/root/zhuweijie/result/response_answer/tempcompass_qwen2.5.wrong.jsonl.by_dim/direction.jsonl",
    "/root/zhuweijie/result/response_answer/tempcompass_qwen2.5.wrong.jsonl.by_dim/order.jsonl",
    "/root/zhuweijie/result/response_answer/tempcompass_qwen2.5.wrong.jsonl.by_dim/speed.jsonl"

    # "/root/zhuweijie/Chirality_in_Action/Test_Result/Qwen3/Qwen3_ssv2.jsonl",
    # "/root/zhuweijie/Chirality_in_Action/Test_Result/Qwen3/ssv2_folder.jsonl",
    # "/root/zhuweijie/Chirality_in_Action/Test_Result/Qwen3/ssv2_pulling.jsonl",
    # "/root/zhuweijie/Chirality_in_Action/Test_Result/Qwen3/ssv2_close_door.jsonl"
]

# ==========================================
# 2. 核心逻辑函数 (省略，与原代码相同)
# ==========================================

def is_url(text):
    if not isinstance(text, str): return False
    return text.startswith(('http://', 'https://'))

def is_image(text):
    if not isinstance(text, str): return False
    # 支持本地路径和URL
    text_lower = text.lower()
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.ico')
    base_path = text_lower.split('?')[0].split('#')[0]
    
    if base_path.endswith(IMAGE_EXTENSIONS): return True
    
    # 针对 URL 的特殊参数检测
    if is_url(text):
        return any(x in text_lower for x in ['f=jpeg', 'format=jpg', 'fmt='])
    return False

def is_video(text):
    if not isinstance(text, str): return False
    # 支持本地路径和URL
    VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.webm', '.ogg', '.mkv')
    base_path = text.lower().split('?')[0].split('#')[0]
    return base_path.endswith(VIDEO_EXTENSIONS)

def extract_media(data, media_list=None, current_path=""):
    """
    递归遍历 JSON 数据，提取所有图片和视频链接，并记录提取路径。
    """
    if media_list is None:
        media_list = []
        
    if isinstance(data, dict):
        for k, v in data.items():
            # 构建新的路径，对于字典使用 .key 格式
            new_path = f"{current_path}.{k}" if current_path else k
            extract_media(v, media_list, new_path)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            # 构建新的路径，对于列表使用 [index] 格式
            new_path = f"{current_path}[{i}]"
            extract_media(item, media_list, new_path)
    elif isinstance(data, str):
        if is_image(data):
            # 记录媒体类型、URL和提取路径
            media_list.append({"type": "image", "url": data, "source": current_path})
        elif is_video(data):
            # 记录媒体类型、URL和提取路径
            media_list.append({"type": "video", "url": data, "source": current_path})
            
    return media_list

def is_valid_data_file(file_path):
    path = Path(file_path)
    allowed_extensions = {'.jsonl', '.json', '.txt', ''}
    if path.suffix.lower() not in allowed_extensions: return False
    binary_extensions = {'.jpg', '.png', '.gif', '.zip', '.mp4', '.pyc', '.pkl', '.parquet'}
    if path.suffix.lower() in binary_extensions: return False
    return True

def expand_paths(paths):
    expanded_files = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists(): continue
        if path.is_file():
            if is_valid_data_file(path):
                expanded_files.append(str(path))
        elif path.is_dir():
            extensions = ['*.jsonl', '*.json', '*.txt']
            files_found = []
            for ext in extensions:
                files_found.extend(list(path.rglob(ext)))
            for item in path.glob('*'): 
                 if item.is_file() and not item.suffix and is_valid_data_file(item):
                     files_found.append(item)
            for f in sorted(list(set(files_found))):
                expanded_files.append(str(f))
    return expanded_files

def load_data(files, target_count, filter_expr):
    expanded_files = expand_paths(files)
    if not expanded_files:
        st.warning("⚠️ 未找到有效文件")
        return []

    compiled_filter = None
    if filter_expr and filter_expr.strip() != "True":
        try:
            # 允许用户表达式访问 data，以及 len, str, int, get 等常用函数
            compiled_filter = compile(filter_expr, '<string>', 'eval')
        except Exception as e:
            st.error(f"❌ 过滤表达式语法错误: {e}")
            return []

    collected_pool = []
    total_scanned = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    # 动态调整每个文件扫描的行数限制，确保总扫描行数不超过 MAX_SCAN_LINES_PER_LOAD
    lines_per_file_limit = max(MAX_SCAN_LINES_PER_LOAD // len(expanded_files), 1000)
    start_time = time.time()

    for f_idx, file_path in enumerate(expanded_files):
        try:
            file_name = os.path.basename(file_path)
            status_text.text(f"正在扫描 ({f_idx+1}/{len(expanded_files)}): {file_name} ...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_scanned_count = 0
                for line in f:
                    total_scanned += 1
                    file_scanned_count += 1
                    
                    if total_scanned % 2000 == 0:
                        progress = min(total_scanned / MAX_SCAN_LINES_PER_LOAD, 1.0)
                        progress_bar.progress(progress)
                    
                    if total_scanned >= MAX_SCAN_LINES_PER_LOAD: break
                    if file_scanned_count > lines_per_file_limit: break

                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        keep = True
                        if compiled_filter:
                            # 过滤逻辑：在安全的命名空间中执行用户代码
                            safe_globals = {"__builtins__": None}
                            safe_locals = {"data": data, "len": len, "str": str, "int": int, "get": data.get}
                            try:
                                keep = eval(compiled_filter, safe_globals, safe_locals)
                            except Exception:
                                keep = False
                        if keep:
                            collected_pool.append(data)
                    except json.JSONDecodeError:
                        continue
                if total_scanned >= MAX_SCAN_LINES_PER_LOAD: break
        except Exception as e:
            st.error(f"读取文件 {file_name} 出错: {e}")

    progress_bar.empty()
    status_text.empty()
    
    elapsed = time.time() - start_time
    st.caption(f"⏱️ 耗时 {elapsed:.2f}s | 扫描行数: {total_scanned} | 命中数据: {len(collected_pool)}")

    if not collected_pool: return []
    if len(collected_pool) > target_count:
        return random.sample(collected_pool, target_count)
    return collected_pool

# ==========================================
# 3. 页面布局 (UI Layout)
# ==========================================

st.title("⚡ JSONL Server Data Viewer")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 控制面板")
    
    st.subheader("1. 数据源")
    selected_files = st.multiselect(
        "选择文件/文件夹",
        JSONL_FILES,
        default=JSONL_FILES if JSONL_FILES else None,
        format_func=lambda x: f"📁 {os.path.basename(x)}" if os.path.isdir(x) else f"📄 {os.path.basename(x)}"
    )
    
    st.subheader("2. 采样设置")
    sample_count = st.number_input("采样数量", min_value=1, max_value=2000, value=100)

    st.subheader("3. 自定义过滤")
    user_filter_code = st.text_area("Python 表达式 (data)", value="True", height=80)
    
    st.subheader("4. 展示布局配置 (需点击加载生效)")
    
    # *** 修改点：将滑块值存入临时状态 ***
    st.session_state.data_cols_count_config = st.slider(
        "数据卡片列数 (排)",
        min_value=1,
        max_value=4, 
        value=st.session_state.data_cols_count_config, # 使用临时状态作为默认值
        help="详细卡片模式下，数据卡片（Expander）的水平展示列数。默认为双排。"
    )
    
    # *** 修改点：将复选框值转换为临时状态 (默认折叠) ***
    # 复选框：True = 默认全展开； False = 默认全折叠
    # 内部状态：True = 全部折叠； False = 全部展开
    is_expanded = st.checkbox(
        "JSON 默认全展开",
        value=not st.session_state.json_collapse_all_config, # 使用临时状态作为默认值
        help="勾选后，所有 JSON 数据默认完全展开；否则默认全部折叠（只显示根节点）。"
    )
    st.session_state.json_collapse_all_config = not is_expanded
    # ----------------------------------------------------
    
    st.divider()
    apply_btn = st.button("🚀 加载/刷新数据", type="primary", use_container_width=True)

# --- 主体逻辑 ---

if apply_btn:
    with st.spinner('正在极速扫描与解析...'):
        new_dataset = load_data(selected_files, sample_count, user_filter_code)
        
    st.session_state.dataset = new_dataset
    st.session_state.data_loaded = True
    
    # *** 关键修改点：点击按钮时，将临时配置同步到正式状态 ***
    st.session_state.data_cols_count = st.session_state.data_cols_count_config
    st.session_state.json_collapse_all = st.session_state.json_collapse_all_config
    # -----------------------------------------------------------

# --- 展示逻辑 ---
if st.session_state.data_loaded:
    dataset = st.session_state.dataset
    
    if not dataset:
        st.warning(f"⚠️ 未找到符合条件的数据。")
    else:
        st.success(f"✅ 已加载 {len(dataset)} 条数据")
        
        # 顶部视图切换
        st.session_state.view_mode = st.radio(
            "模式", ["表格概览", "详细卡片"], 
            horizontal=True,
            label_visibility="collapsed"
        )
        st.divider()

        # 表格模式 (仅展示基础文本概览)
        if st.session_state.view_mode == "表格概览":
            df_data = []
            for item in dataset:
                # 简单扁平化用于表格展示
                row = {k: str(v)[:50] + "..." if isinstance(v, (dict, list)) else v for k, v in item.items()}
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # 表格概览功能：填入完整的数据，默认仅展示前10条
            display_limit = 10
            
            if len(df) > display_limit:
                 st.caption(f"仅展示前 {display_limit} 条数据（共 {len(df)} 条）")

            # 默认仅展示前 N 条数据
            st.dataframe(df.head(display_limit), use_container_width=True)


        # 详细卡片模式 (实现数据卡片的多排展示)
        else:
            # *** 关键：从正式状态读取配置 ***
            data_cols_count = st.session_state.data_cols_count
            collapse_all = st.session_state.json_collapse_all
            default_expanded = not collapse_all
            # --------------------------------
            
            # 1. 创建用于容纳数据卡片的列
            cols = st.columns(data_cols_count)
            
            for idx, item in enumerate(dataset):
                # 2. 确定当前数据卡片应放入哪一列
                col_index = idx % data_cols_count
                
                with cols[col_index]:
                    # 获取标题
                    title_candidates = [item.get(k) for k in ['id', 'title', 'query', 'prompt'] if k in item]
                    title = str(title_candidates[0]) if title_candidates else f"Item #{idx+1}"
                    if len(title) > 100: title = title[:100] + "..."

                    # 确保 expader 标题显示完整
                    with st.expander(f"📄 {title}", expanded=True):
                        
                        # 1. 媒体展示区 (仅图片和视频) - 媒体内部的列数固定为 2 列
                        media_items = extract_media(item)
                        if media_items:
                            # 媒体内部的列数固定为 2 列
                            media_cols = st.columns(min(len(media_items), 2)) 
                            
                            for i, media in enumerate(media_items):
                                with media_cols[i % 2]:
                                    # 增加字段提取来源
                                    caption = f"Source: `{media['source']}`"
                                    if media['type'] == 'image':
                                        st.image(media['url'], caption=caption, use_column_width=True)
                                    elif media['type'] == 'video':
                                        # Streamlit video 暂不支持 caption 属性，故在 video 上方显示
                                        st.markdown(f"**Video** {caption}")
                                        st.video(media['url'])
                            st.divider()
                        
                        # 2. 原始 JSON 展示区
                        st.markdown("#### 📝 原始数据")
                        
                        # 使用从正式状态读取的配置
                        st.json(item, expanded=default_expanded)

else:
    st.info("👈 请点击左侧 '加载/刷新数据' 开始")