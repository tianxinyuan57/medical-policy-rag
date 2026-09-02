"""阶段 3：RAG 检索 + 生成 + 引用溯源

这是整个项目最核心的文件——把检索和生成串起来，实现完整的问答。

====================================================================
面试必背：RAG = Retrieval-Augmented Generation（检索增强生成）
====================================================================

完整七步链路（面试要能完整背出来）：
  离线建库：① 加载文档 → ② 切分 → ③ 向量化 → ④ 存入向量库
  在线问答：⑤ 问题向量化 → ⑥ 检索 Top-K → ⑦ 拼入 Prompt → 模型生成

本文件负责在线问答部分（步骤 ⑤⑥⑦）。
====================================================================
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBED_MODEL, TOP_K
from llm import ask_llm


# =====================================================
# 加载向量库（离线建好的，直接读取）
# =====================================================
#
# 【面试知识点】
# 向量库持久化在磁盘上，每次启动不需要重新建。
# 这里只需要加载 embedding 模型（把问题也转成向量用同一个模型），
# 然后连上已有的 Chroma 数据库即可。
#
# 为什么问题和文档要用同一个 embedding 模型？
# 因为向量空间必须一致，才能比较距离。
# 用 A 模型编码文档、B 模型编码问题，两组向量不在同一个空间里，
# 算出来的距离没有意义。

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
chroma_path = os.path.join(project_root, CHROMA_DIR)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
)
vectorstore = Chroma(
    persist_directory=chroma_path,
    embedding_function=embeddings,
)


# =====================================================
# RAG 系统提示词 —— 这是生成质量的关键
# =====================================================
#
# 【面试知识点：为什么 RAG 的 Prompt 要特别设计？】
#
# 普通的 Prompt：模型用自己的知识回答
# RAG 的 Prompt：明确告诉模型"只看我给你的原文，不要用你自己的知识"
#
# 这是防幻觉的核心机制。不这么写的话，模型看到检索原文后
# 可能会"自由发挥"，把自己知道的东西混进去，用户分不清
# 哪些是原文里的、哪些是模型编的。
#
# 三条核心约束：
# 1. "只依据参考原文" —— 锁定信息来源
# 2. "未能找到就说未找到" —— 给模型安全退路，防止硬编
# 3. "标注引用来源" —— 让答案可核对、可追溯

RAG_SYSTEM_PROMPT = """你是医疗政策问答助手。请严格遵守以下规则：

1. 只依据下面提供的【参考原文】回答，不得使用参考原文之外的任何知识。
2. 如果参考原文中没有相关信息，明确回答"根据现有政策文件，未能找到相关规定"，不要猜测。
3. 回答时引用具体条款（如"根据《XX法》第X条"），让答案可追溯。
4. 回答末尾用"📎 引用来源"标注你参考了哪些文件片段。
5. 语言简洁准确，像政策解读而非学术论文。
"""


# =====================================================
# 核心函数：answer() —— RAG 问答的完整流程
# =====================================================

def answer(question: str, verbose: bool = False) -> str:
    """RAG 问答：检索 → 拼 Prompt → 生成回答。

    这个函数包含 RAG 在线问答的全部逻辑：
    1. 用向量检索找到与问题最相关的 Top-K 个原文片段
    2. 把这些片段格式化后拼进 Prompt
    3. 让大模型基于这些原文生成回答

    Args:
        question: 用户的问题
        verbose:  是否打印检索细节（调试/排障用）
    Returns:
        模型基于检索原文生成的回答
    """

    # ---- 第 ⑤⑥ 步：向量化问题 + 检索 Top-K ----
    #
    # similarity_search 内部做了两件事：
    # 1. 把 question 用 embedding 模型转成向量
    # 2. 在 Chroma 里找余弦距离最近的 k 个片段
    #
    # 【面试必考题：Top-K 怎么定？】
    # K 太小 → 可能漏掉相关片段，答案不完整
    # K 太大 → 引入太多不相关内容（噪音），模型抓不住重点，还更贵
    # 我们设 K=3，因为政策问题通常 1-3 条相关条款就能答清楚

    results = vectorstore.similarity_search(question, k=TOP_K)

    if verbose:
        print(f"\n🔍 检索到 {len(results)} 个相关片段：")
        for i, doc in enumerate(results):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            print(f"   [{i+1}] 来源: {source}")
            print(f"       前80字: {doc.page_content[:80]}...")
            print()

    # ---- 第 ⑦ 步：拼接 Prompt（Augment） ----
    #
    # 把检索到的原文片段格式化，加上编号和来源信息，
    # 这样模型在回答时能引用"片段1""片段2"，我们也能核对。
    #
    # 【面试知识点：引用溯源怎么做？】
    # 每个 chunk 自带 metadata（来源文件名），
    # 我们在拼 Prompt 时把文件名标注在每段前面，
    # 再要求模型在回答末尾标注引用了哪些片段。
    # 这样答案可核对——这是项目的差异化亮点。

    context_blocks = []
    for i, doc in enumerate(results):
        source = os.path.basename(doc.metadata.get("source", "未知"))
        # 去掉文件后缀，显示更干净
        source_name = source.replace(".txt", "").replace(".pdf", "")
        context_blocks.append(
            f"[片段{i+1}｜来源：《{source_name}》]\n{doc.page_content}"
        )
    context = "\n\n".join(context_blocks)

    # 拼成最终的 user prompt
    user_prompt = f"【参考原文】\n{context}\n\n【问题】\n{question}"

    if verbose:
        print(f"📝 Prompt 长度: {len(user_prompt)} 字符")
        print(f"   （包含 {len(results)} 个参考片段）\n")

    # ---- 调用大模型生成回答 ----
    return ask_llm(RAG_SYSTEM_PROMPT, user_prompt)


# =====================================================
# 批量问答：一次测试多个问题
# =====================================================

def batch_answer(questions: list[str], verbose: bool = False) -> list[dict]:
    """批量回答问题，返回结构化结果。

    Args:
        questions: 问题列表
        verbose:   是否打印每题的检索细节
    Returns:
        [{"question": ..., "answer": ...}, ...]
    """
    results = []
    for i, q in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"❓ 问题 {i+1}: {q}")
        print(f"{'='*60}")
        ans = answer(q, verbose=verbose)
        print(f"\n💬 回答：\n{ans}")
        results.append({"question": q, "answer": ans})
    return results


# =====================================================
# 交互式问答（命令行）
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医疗政策 RAG 问答系统")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示检索细节（调试模式）")
    parser.add_argument("--test", action="store_true",
                        help="运行预设测试问题")
    args = parser.parse_args()

    print("=" * 60)
    print("🏥 医疗政策智能问答系统")
    print("   基于公开政策文件 · 答案可溯源 · 拒绝编造")
    print(f"   知识库：{vectorstore._collection.count()} 个片段")
    print("=" * 60)

    if args.test:
        # 预设测试问题：涵盖不同场景
        test_questions = [
            # 1. 答案明确在文件里的问题
            "献血的年龄范围是多少？",
            "什么情况下可以对精神障碍患者实施强制住院？",
            "医疗纠纷有哪些合法的解决途径？",
            # 2. 需要跨概念理解的问题
            "医师和护士在紧急情况下的责任有什么不同？",
            # 3. 文件里没有的问题（测试拒答能力）
            "新冠疫情期间各地的隔离补贴标准是多少？",
        ]
        batch_answer(test_questions, verbose=args.verbose)
    else:
        # 交互式问答
        print("\n输入问题开始提问（输入 q 退出，输入 v 切换详细模式）：\n")
        verbose_mode = args.verbose
        while True:
            q = input("❓ 你的问题：").strip()
            if q.lower() == "q":
                print("👋 再见！")
                break
            if q.lower() == "v":
                verbose_mode = not verbose_mode
                print(f"   详细模式: {'开启' if verbose_mode else '关闭'}")
                continue
            if not q:
                continue

            print("\n⏳ 正在检索并生成回答...\n")
            result = answer(q, verbose=verbose_mode)
            print(f"💬 回答：\n{result}\n")
            print("-" * 60)
