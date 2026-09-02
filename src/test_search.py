"""测试向量库检索质量 —— 覆盖多个主题，确认跨文件检索正常"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBED_MODEL

# 加载向量库
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
chroma_path = os.path.join(project_root, CHROMA_DIR)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

# 测试问题 —— 涵盖不同政策文件
test_queries = [
    ("献血的年龄要求是什么？", "献血法"),
    ("什么是假药？", "药品管理法"),
    ("甲类传染病有哪些？", "传染病防治法"),
    ("护士需要什么资格才能执业？", "护士条例"),
    ("疫苗接种可以收费吗？", "疫苗管理法"),
    ("精神障碍患者可以被强制住院吗？", "精神卫生法"),
    ("医疗纠纷可以通过哪些途径解决？", "医疗纠纷预防和处理条例"),
    ("医师在紧急情况下可以拒绝救治吗？", "医师法"),
    ("医疗机构可以使用非卫生技术人员吗？", "医疗机构管理条例"),
    ("器官移植手术可以收取哪些费用？", "人体器官移植条例"),
]

print("=" * 70)
print("🔍 向量库检索质量测试")
print(f"   知识库：{vectorstore._collection.count()} 个片段")
print("=" * 70)

correct = 0
total = len(test_queries)

for i, (query, expected_source) in enumerate(test_queries):
    results = vectorstore.similarity_search(query, k=3)
    top1_source = os.path.basename(results[0].metadata.get("source", ""))
    top1_preview = results[0].page_content[:80].replace("\n", " ")

    # 检查 Top-3 里有没有命中期望的来源
    all_sources = [os.path.basename(r.metadata.get("source", "")) for r in results]
    hit = any(expected_source in s for s in all_sources)
    if hit:
        correct += 1

    status = "✅" if hit else "❌"
    print(f"\n  [{i+1}] {status} {query}")
    print(f"      期望来源: {expected_source}")
    print(f"      Top-1:    {top1_source}")
    print(f"      内容预览: {top1_preview}...")
    if not hit:
        print(f"      Top-3 来源: {all_sources}")

print(f"\n{'=' * 70}")
print(f"📊 检索准确率: {correct}/{total} ({correct/total*100:.0f}%)")
print(f"{'=' * 70}")
