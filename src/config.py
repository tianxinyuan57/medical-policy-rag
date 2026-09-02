"""项目配置：模型、路径、参数集中管理"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- DeepSeek API ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHAT_MODEL = "deepseek-chat"

# --- 路径 ---
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

# --- Embedding 模型 ---
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # 本地中文 embedding，免费、离线可跑

# --- RAG 参数 ---
CHUNK_SIZE = 500       # 每个文本片段的字符数
CHUNK_OVERLAP = 100    # 相邻片段的重叠字符数
TOP_K = 3              # 检索返回的最相关片段数
