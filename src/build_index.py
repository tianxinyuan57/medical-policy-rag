"""阶段 2：RAG 建库 —— 切分 → 向量化 → 存储

这是整个 RAG 系统的地基。理解这个文件，你就理解了 RAG 离线建库的全部。

====================================================================
面试必背：RAG 建库三步走
====================================================================

第 1 步：加载文档（Load）
    把 PDF / TXT 等文件读成纯文本。
    → 面试点：不同格式用不同 loader，PDF 扫描件需要 OCR。

第 2 步：切分（Chunk / Split）
    把长文档切成小片段。
    → 面试高频题：为什么要切？切多大？为什么要重叠？

第 3 步：向量化并存储（Embed + Store）
    把每个小片段转成向量（一串数字），存进向量数据库。
    → 面试点：embedding 是什么？为什么用本地模型？向量库怎么选？

====================================================================
"""

import sys
import os
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import DATA_DIR, CHROMA_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from smart_splitter import smart_split_documents


# =====================================================
# 第 1 步：加载文档
# =====================================================
#
# 【面试知识点】
# Q: 你的系统支持哪些文件格式？
# A: 目前支持 TXT 和 PDF。TXT 用 TextLoader 直接读，PDF 用 PyPDFLoader
#    逐页提取文字。如果是扫描件（图片 PDF），PyPDFLoader 读不出来，
#    需要加 OCR（比如 pytesseract），这是一个已知局限。
#
# 【为什么 LangChain 的 Loader 有用？】
# 它不只是读文件内容，还会自动附带 metadata（文件名、页码等）。
# 这些 metadata 后面做引用溯源时非常关键——告诉用户"这段来自哪个文件"。

def load_documents():
    """从 data/ 目录加载所有政策文件。

    Returns:
        list[Document]: LangChain Document 对象列表，
                        每个对象有 .page_content（文本）和 .metadata（来源信息）
    """
    # 找到项目根目录下的 data 文件夹
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, DATA_DIR)

    docs = []
    file_count = {"txt": 0, "pdf": 0}

    for name in sorted(os.listdir(data_path)):
        path = os.path.join(data_path, name)

        if name.endswith(".pdf"):
            loader = PyPDFLoader(path)
            loaded = loader.load()
            docs.extend(loaded)
            file_count["pdf"] += 1
            print(f"  📄 PDF: {name} → {len(loaded)} 页")

        elif name.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            loaded = loader.load()
            docs.extend(loaded)
            file_count["txt"] += 1
            print(f"  📄 TXT: {name} → {len(loaded)} 个文档")

    total = file_count["txt"] + file_count["pdf"]
    print(f"\n  📊 共加载 {total} 个文件（TXT: {file_count['txt']}, PDF: {file_count['pdf']}）")
    print(f"  📊 共 {len(docs)} 个原始文档片段")
    return docs


# =====================================================
# 第 2 步：切分文档（Chunking）
# =====================================================
#
# 【面试必考题：为什么要切分？】
# 两个原因：
# 1. 模型有上下文长度限制（token limit），长文档塞不进去
# 2. 检索需要小颗粒才精准——你搜"器官捐献的年龄限制"，如果整份条例
#    是一个 chunk，检索到了也找不到重点；切成小段后，能精准命中那一条
#
# 【面试必考题：chunk 切多大？】
# 太大 → 检索到的段落里夹杂无关内容（噪音），浪费 token，还更贵
# 太小 → 一句话被切断，丢失上下文，模型看到残句答不好
# 经验值：中文政策文件 300-500 字是合理起点
# 我们选 500 字，因为政策条款一般一条在 100-300 字，500 字能包完整的条款
#
# 【面试必考题：为什么要重叠（overlap）？】
# 假设第三条正好被切在两个 chunk 的边界——前半句在 chunk A 末尾，
# 后半句在 chunk B 开头。没有重叠的话，两个 chunk 都不完整。
# 设 100 字重叠，相邻 chunk 有 100 字相同内容，关键信息不会被切断。
#
# 【RecursiveCharacterTextSplitter 是什么？】
# LangChain 提供的智能切分器。它按 separators 列表的优先级依次尝试：
# 先试 "\n\n"（段落），再试 "\n"（换行），再试 "。"（句子）……
# 尽量在自然断点切，不会切在词的中间。
# 对中文，我们要加"。""；"这些中文标点作为切分点。

def split_documents(docs):
    """把文档切分成小片段。

    Args:
        docs: load_documents() 返回的 Document 列表
    Returns:
        list[Document]: 切分后的 chunk 列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # 每个 chunk 最大 500 字符
        chunk_overlap=CHUNK_OVERLAP, # 相邻 chunk 重叠 100 字符
        separators=[
            "\n\n",  # 优先在段落间切
            "\n",    # 其次在换行处切
            "。",    # 中文句号
            "；",    # 中文分号
            "，",    # 中文逗号
            " ",     # 空格
            "",      # 最后逐字符切（兜底）
        ],
        # 注意 separators 的顺序很重要！
        # 排在前面的优先使用 → 尽量保持语义完整
    )
    chunks = splitter.split_documents(docs)
    return chunks


# =====================================================
# 第 3 步：向量化并存储
# =====================================================
#
# 【面试必考题：Embedding 是什么？】
# 把一段文字映射成一个向量（一串数字，比如 384 维）。
# 训练目标：语义相近的文本 → 向量距离近。
# 比如"禁止买卖器官"和"不得交易人体器官"的向量会很接近，
# 即使字面上没有重叠的词。
#
# 检索时：把用户的问题也转成向量 → 找距离最近的 chunk 向量 → 返回对应文本。
# 这就是"语义检索"——不是关键词匹配，而是理解"意思"。
#
# 【面试必考题：为什么用本地 bge 模型不用 API？】
# 1. 免费：不用为每次建库付钱
# 2. 离线可跑：不依赖网络
# 3. 中文效果好：bge-small-zh 是专门为中文训练的
# 4. 数据量小没必要用 API：我就几份文件、几百个 chunk
# 代价：第一次要下载模型（约 100MB），占一点本地算力
#
# 【面试必考题：向量库为什么选 Chroma？】
# 我的数据量：几份文件、几百个 chunk
# Chroma：本地零配置、开箱即用、支持持久化，完美适配快速原型
# FAISS：更偏底层，缺持久化和元数据管理，要自己封装
# Milvus：为千万级、高并发生产环境设计，我这个规模用它杀鸡用牛刀
# → 如果未来数据涨到百万级，我会考虑迁到 Milvus

def build_vectorstore(chunks):
    """把 chunk 转成向量并存入 Chroma。

    Args:
        chunks: split_documents() 返回的 chunk 列表
    Returns:
        Chroma: 可以用来检索的向量库对象
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_path = os.path.join(project_root, CHROMA_DIR)

    # 如果向量库已存在，先删除重建（确保数据一致）
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print("  🗑️  已清除旧的向量库")

    print(f"  🧠 加载 embedding 模型: {EMBED_MODEL}")
    print("     （第一次运行会下载模型，约 100MB，请等待...）")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},  # 用 CPU 跑，兼容性最好
        encode_kwargs={"normalize_embeddings": True},  # 归一化，提高检索质量
    )

    print("  📦 正在将文本片段向量化并存入 Chroma...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path,
    )

    return vectorstore


# =====================================================
# 主流程：串起 加载 → 切分 → 存储
# =====================================================

def build():
    """完整的建库流程。"""
    print("=" * 60)
    print("🏗️  RAG 向量库建设")
    print("=" * 60)

    # 第 1 步
    print("\n📂 第 1 步：加载文档")
    print("-" * 40)
    start = time.time()
    docs = load_documents()
    print(f"  ⏱️  耗时: {time.time() - start:.1f}s")

    # 第 2 步
    print(f"\n✂️  第 2 步：智能切分文档（按条款/序号结构，非固定字数）")
    print("-" * 40)
    start = time.time()
    chunks = smart_split_documents(docs)
    print(f"  ⏱️  耗时: {time.time() - start:.1f}s")

    # 展示前 3 个 chunk 的样子，帮助理解切分效果
    print(f"\n  👀 预览前 3 个片段：")
    for i, chunk in enumerate(chunks[:3]):
        source = os.path.basename(chunk.metadata.get("source", "未知"))
        preview = chunk.page_content[:80].replace("\n", " ")
        print(f"     [{i+1}] 来源: {source}")
        print(f"         内容: {preview}...")
        print(f"         长度: {len(chunk.page_content)} 字符")
        print()

    # 第 3 步
    print(f"🧮 第 3 步：向量化并存入 Chroma")
    print("-" * 40)
    start = time.time()
    vectorstore = build_vectorstore(chunks)
    print(f"  ⏱️  耗时: {time.time() - start:.1f}s")

    # 验证：试一次检索
    print(f"\n🔍 验证：试一次检索")
    print("-" * 40)
    test_query = "器官捐献有什么年龄限制？"
    print(f"  测试问题: {test_query}")
    results = vectorstore.similarity_search(test_query, k=3)
    for i, doc in enumerate(results):
        source = os.path.basename(doc.metadata.get("source", "未知"))
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [Top-{i+1}] 来源: {source}")
        print(f"           内容: {preview}...")
        print()

    print("=" * 60)
    print("✅ 向量库建设完成！")
    print(f"   共 {len(chunks)} 个片段已存入 {CHROMA_DIR}/")
    print(f"   可以开始阶段 3（检索 + 问答）了")
    print("=" * 60)


# =====================================================
# 增量更新：只处理新增的文件
# =====================================================
#
# 【面试必考题：文件更新或新增怎么办？】
#
# 答：不需要全部重建。Chroma 支持增量添加。
# 我的做法是：
# 1. 检查向量库里已经有哪些文件（通过 metadata 里的 source 字段）
# 2. 扫描 data/ 目录，找出新增的文件
# 3. 只对新文件走 加载→切分→向量化→加入 的流程
#
# 什么时候必须全部重建？
# - 改了 chunk_size 或 chunk_overlap（切法变了）
# - 换了 embedding 模型（向量空间变了，旧新向量不兼容）
# - 删除了某份文件（需要清掉对应的 chunk）

def update():
    """增量更新：只处理 data/ 里新增的文件，加入已有向量库。"""
    print("=" * 60)
    print("🔄 向量库增量更新")
    print("=" * 60)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_path = os.path.join(project_root, CHROMA_DIR)
    data_path = os.path.join(project_root, DATA_DIR)

    if not os.path.exists(chroma_path):
        print("  ⚠️  向量库不存在，需要先完整建库！")
        print("  运行: python src/build_index.py --mode full")
        return

    # 连接已有向量库
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
    )

    # 查看向量库里已有哪些文件
    existing_docs = vectorstore.get()
    existing_sources = set()
    if existing_docs and existing_docs.get("metadatas"):
        for meta in existing_docs["metadatas"]:
            if meta and "source" in meta:
                existing_sources.add(os.path.basename(meta["source"]))
    print(f"\n  📦 向量库已有 {len(existing_sources)} 个文件的数据：")
    for s in sorted(existing_sources):
        print(f"     · {s}")

    # 扫描 data/ 目录，找新文件
    all_files = set(f for f in os.listdir(data_path) if f.endswith((".txt", ".pdf")))
    new_files = all_files - existing_sources

    if not new_files:
        print(f"\n  ✅ 没有新文件需要处理，知识库已是最新！")
        print(f"     当前共 {vectorstore._collection.count()} 个片段")
        return

    print(f"\n  🆕 发现 {len(new_files)} 个新文件：")
    for f in sorted(new_files):
        print(f"     · {f}")

    # 只加载新文件
    new_docs = []
    for name in sorted(new_files):
        path = os.path.join(data_path, name)
        if name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        loaded = loader.load()
        new_docs.extend(loaded)
        print(f"  📄 已加载: {name} → {len(loaded)} 个文档")

    # 智能切分新文件
    new_chunks = smart_split_documents(new_docs)
    print(f"\n  ✂️  智能切分完成")

    # 加入向量库（不删除已有数据）
    print(f"  📦 正在向量化并加入已有向量库...")
    vectorstore.add_documents(new_chunks)

    total = vectorstore._collection.count()
    print(f"\n  ✅ 增量更新完成！")
    print(f"     新增 {len(new_chunks)} 个片段")
    print(f"     向量库总计 {total} 个片段")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 向量库建设/更新")
    parser.add_argument("--mode", choices=["full", "update"], default="full",
                        help="full=全量重建（默认），update=只处理新增文件")
    args = parser.parse_args()

    if args.mode == "update":
        update()
    else:
        build()
