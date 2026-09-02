"""阶段 4（续）：调优实验 —— 面试黄金素材

====================================================================
这个脚本记录一个完整的"发现问题→定位→调优→验证"过程。
面试时这就是你讲故事的素材。
====================================================================

实验问题：跨文件对比题
  "医师和护士在紧急情况下的责任有什么不同？"
  答案分散在《医师法》和《护士条例》两份文件里。

实验变量：Top-K（检索片段数）
  假设：K=3 可能只找到一边的内容，K=5 能覆盖两边

实验设计：
  1. 先用 K=3 跑一次，看检索了哪些文件
  2. 再用 K=5 跑一次，对比检索结果
  3. 对比两次回答的完整度
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBED_MODEL
from llm import ask_llm
from rag import RAG_SYSTEM_PROMPT

# 加载向量库
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
chroma_path = os.path.join(project_root, CHROMA_DIR)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

QUESTION = "医师和护士在紧急情况下的责任有什么不同？"

def run_with_k(k: int) -> dict:
    """用指定的 Top-K 运行一次 RAG，返回检索和生成结果。"""
    print(f"\n{'='*65}")
    print(f"🔬 实验组：Top-K = {k}")
    print(f"{'='*65}")
    print(f"   问题：{QUESTION}")

    # 1. 检索
    results = vectorstore.similarity_search(QUESTION, k=k)

    # 2. 分析检索结果
    sources = []
    for i, doc in enumerate(results):
        src = os.path.basename(doc.metadata.get("source", "")).replace(".txt", "").replace(".pdf", "")
        sources.append(src)
        print(f"   片段 {i+1}: 《{src}》 → {doc.page_content[:60]}...")

    unique_sources = list(dict.fromkeys(sources))  # 去重但保序
    has_physician = any("医师" in s for s in sources)
    has_nurse = any("护士" in s for s in sources)

    print(f"\n   📋 检索到的文件：{unique_sources}")
    print(f"   ✅ 包含医师法: {'是' if has_physician else '否'}")
    print(f"   ✅ 包含护士条例: {'是' if has_nurse else '否'}")
    print(f"   ✅ 双文件覆盖: {'是 ✅' if (has_physician and has_nurse) else '否 ❌'}")

    # 3. 生成回答
    context_blocks = []
    for i, doc in enumerate(results):
        src = os.path.basename(doc.metadata.get("source", "")).replace(".txt", "").replace(".pdf", "")
        context_blocks.append(f"[片段{i+1}｜来源：《{src}》]\n{doc.page_content}")
    context = "\n\n".join(context_blocks)
    user_prompt = f"【参考原文】\n{context}\n\n【问题】\n{QUESTION}"

    answer = ask_llm(RAG_SYSTEM_PROMPT, user_prompt)
    print(f"\n   💬 回答：\n{answer}")

    return {
        "k": k,
        "sources": sources,
        "unique_sources": unique_sources,
        "has_physician": has_physician,
        "has_nurse": has_nurse,
        "both_covered": has_physician and has_nurse,
        "answer": answer,
    }


if __name__ == "__main__":
    print("📊 调优实验：Top-K 对跨文件检索的影响")
    print("   目标：验证增大 K 能否改善跨文件对比问题的回答质量")

    # 对照组：K=3
    result_k3 = run_with_k(3)

    # 实验组：K=5
    result_k5 = run_with_k(5)

    # 汇总对比
    print(f"\n{'='*65}")
    print("📊 实验结论")
    print(f"{'='*65}")
    print(f"   | 指标         | K=3           | K=5           |")
    print(f"   |-------------|---------------|---------------|")
    print(f"   | 文件覆盖     | {'双文件 ✅' if result_k3['both_covered'] else '单文件 ❌'}    | {'双文件 ✅' if result_k5['both_covered'] else '单文件 ❌'}    |")
    print(f"   | 来源数       | {len(result_k3['unique_sources'])} 个文件       | {len(result_k5['unique_sources'])} 个文件       |")
    print(f"   | 总片段数     | {len(result_k3['sources'])} 个片段       | {len(result_k5['sources'])} 个片段       |")

    if result_k5["both_covered"] and not result_k3["both_covered"]:
        print("\n   ✅ 结论：增大 Top-K 显著改善了跨文件检索的覆盖度。")
        print("   📝 建议：对比类问题可以动态调整 K 值，或使用 K=5 作为默认值。")
    elif result_k3["both_covered"] and result_k5["both_covered"]:
        print("\n   ✅ 结论：K=3 已经能覆盖双文件，K=5 提供了额外冗余。")
        print("   📝 建议：当前 K=3 对此类问题已足够，保持不变可节省 token。")
    else:
        print("\n   ⚠️ 结论：两组都未完全覆盖，可能需要更大的 K 或改进 embedding。")
