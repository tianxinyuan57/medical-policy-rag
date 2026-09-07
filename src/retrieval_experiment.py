"""检索策略对比实验：向量 vs BM25 vs 混合 vs 混合+重排

====================================================================
实验设计
====================================================================

对比 4 种检索策略在同一批测试题上的表现：
  A. 仅向量检索（基线）
  B. 仅 BM25 检索
  C. 混合检索（RRF 融合，不重排）
  D. 混合检索 + Cross-Encoder 重排

评估指标：
  · Hit@K      —— Top-K 里是否命中期望来源（召回能力）
  · MRR        —— 第一个正确结果的排名倒数（排序质量）
  · Precision@K—— Top-K 里正确来源的占比（精度）

====================================================================
为什么要做这个实验？（面试要点）
====================================================================

"我加了混合检索" 是陈述。
"我做了 A/B 对比，混合检索把 MRR 从 0.72 提到 0.89" 是证据。

面试官关心的是：你怎么知道改进有效？有没有翻车的情况？
这个实验回答的就是这个问题。
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.documents import Document
from hybrid_retriever import get_retriever


# =====================================================
# 测试集：扩展到 20 题，覆盖不同检索难度
# =====================================================
#
# 设计思路：
# - 语义型：问法和原文用词不同，考向量检索
# - 术语型：含专有名词，考 BM25
# - 混合型：既要理解语义又要精确匹配术语

RETRIEVAL_TEST_SET = [
    # ---- 语义型（问法≠原文用词，向量检索应占优）----
    {"q": "无偿献血的年龄范围是多少？", "src": "献血法", "kind": "语义"},
    {"q": "什么样的药算是假的？", "src": "药品管理法", "kind": "语义"},
    {"q": "捐器官有年龄要求吗？", "src": "人体器官移植条例", "kind": "语义"},
    {"q": "病人能不能看自己的病历？", "src": ["医疗纠纷预防和处理条例", "医疗机构病历管理规定"], "kind": "语义"},
    {"q": "打疫苗要交钱吗？", "src": "疫苗管理法", "kind": "语义"},
    {"q": "护士想上岗需要满足什么？", "src": "护士条例", "kind": "语义"},

    # ---- 术语型（含专有名词，BM25 应占优）----
    {"q": "医疗保障基金飞行检查的启动情形", "src": "医疗保障基金飞行检查管理暂行办法", "kind": "术语"},
    {"q": "DIP 病种分值付费的分组方案", "src": ["按病组和病种分值付费2.0版分组方案", "疾病诊断相关分组DRG付费技术规范"], "kind": "术语"},
    {"q": "跨省异地就医直接结算的备案流程", "src": ["基本医疗保险跨省异地就医直接结算经办规程", "关于进一步做好基本医疗保险跨省异地就医直接结算工作的通知"], "kind": "术语"},
    {"q": "定点医药机构相关人员医保支付资格管理", "src": ["医疗保障定点医药机构相关人员医保支付资格管理经办规程", "关于建立定点医药机构人员医保支付资格管理制度的指导意见"], "kind": "术语"},
    {"q": "长期处方的最长开具时限", "src": "长期处方管理规范", "kind": "术语"},
    {"q": "医疗机构检查检验结果互认的范围", "src": "医疗机构检查检验结果互认管理办法", "kind": "术语"},
    {"q": "电子病历应用管理的分级要求", "src": "电子病历应用管理规范", "kind": "术语"},

    # ---- 混合型（既要语义又要术语）----
    {"q": "医生在急救时不救人要负什么责任？", "src": "医师法", "kind": "混合"},
    {"q": "跟师傅学中医的人怎么才能拿到行医资格？", "src": ["中医药法", "传统医学师承和确有专长人员医师资格考核考试办法"], "kind": "混合"},
    {"q": "医院乱收费骗医保会怎么处罚？", "src": ["医疗保障基金使用监督管理条例", "关于办理医保骗保刑事案件若干问题的指导意见"], "kind": "混合"},
    {"q": "麻醉药品处方有什么特殊管理规定？", "src": ["麻醉药品和精神药品管理条例", "医疗机构麻醉药品、第一类精神药品管理规定", "处方管理办法"], "kind": "混合"},
    {"q": "甲类传染病有哪些？发现了要多久上报？", "src": "传染病防治法", "kind": "混合"},
    {"q": "医疗废物怎么分类处理？", "src": "医疗废物管理条例", "kind": "混合"},
    {"q": "个人健康信息被泄露了怎么办？", "src": "个人信息保护法", "kind": "混合"},
]


# =====================================================
# 评估指标
# =====================================================

def _norm_source(doc: Document) -> str:
    """从 Document 提取来源文件名（去后缀）。"""
    src = doc.metadata.get("source", "")
    return os.path.basename(src).replace(".txt", "").replace(".pdf", "")


def _is_hit(source_name: str, expected) -> bool:
    """判断来源是否匹配期望（支持 str 或 list）。"""
    if isinstance(expected, str):
        expected = [expected]
    return any(e in source_name for e in expected)


def evaluate_strategy(retrieve_fn, test_set: list[dict], top_k: int = 5) -> dict:
    """评估一个检索策略。

    Args:
        retrieve_fn: 接受 (query, top_k) 返回 list[Document] 的函数
        test_set: 测试集
        top_k: 检索片段数

    Returns:
        {"hit_rate":…, "mrr":…, "precision":…, "by_kind":…, "details":[…]}
    """
    hits = 0
    reciprocal_ranks = []
    precisions = []
    details = []
    by_kind = {}

    for tc in test_set:
        q, expected, kind = tc["q"], tc["src"], tc["kind"]

        t0 = time.time()
        docs = retrieve_fn(q, top_k)
        elapsed = time.time() - t0

        sources = [_norm_source(d) for d in docs]
        hit_flags = [_is_hit(s, expected) for s in sources]

        # Hit@K
        hit = any(hit_flags)
        if hit:
            hits += 1

        # MRR：第一个命中的排名倒数
        rr = 0.0
        for rank, flag in enumerate(hit_flags, start=1):
            if flag:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Precision@K
        prec = sum(hit_flags) / len(hit_flags) if hit_flags else 0.0
        precisions.append(prec)

        # 分类统计
        if kind not in by_kind:
            by_kind[kind] = {"total": 0, "hit": 0, "mrr_sum": 0.0}
        by_kind[kind]["total"] += 1
        by_kind[kind]["hit"] += 1 if hit else 0
        by_kind[kind]["mrr_sum"] += rr

        details.append({
            "question": q,
            "kind": kind,
            "expected": expected,
            "sources": sources,
            "hit": hit,
            "rr": round(rr, 3),
            "precision": round(prec, 3),
            "elapsed": round(elapsed, 3),
        })

    n = len(test_set)
    for k, v in by_kind.items():
        v["hit_rate"] = round(v["hit"] / v["total"], 3)
        v["mrr"] = round(v["mrr_sum"] / v["total"], 3)
        del v["mrr_sum"]

    return {
        "hit_rate": round(hits / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "precision": round(sum(precisions) / n, 3),
        "avg_latency": round(sum(d["elapsed"] for d in details) / n, 3),
        "by_kind": by_kind,
        "details": details,
    }


# =====================================================
# 主实验
# =====================================================

def run_experiment(top_k: int = 5):
    print("=" * 76)
    print("🧪 检索策略对比实验")
    print(f"   测试集: {len(RETRIEVAL_TEST_SET)} 题 | Top-K = {top_k}")
    print("=" * 76)

    print("\n正在加载检索器...")
    retriever = get_retriever()
    print(f"✅ 就绪（知识库 {len(retriever.corpus_texts)} 个片段）\n")

    strategies = {
        "A. 仅向量检索（基线）": lambda q, k: retriever.retrieve_vector_only(q, k),
        "B. 仅 BM25 检索": lambda q, k: retriever.retrieve_bm25_only(q, k),
        "C. 混合检索（RRF）": lambda q, k: HybridNoRerank(retriever).retrieve(q, k),
        "D. 混合检索 + 重排": lambda q, k: retriever.retrieve(q, k),
    }

    results = {}
    for name, fn in strategies.items():
        print(f"⏳ 评估: {name} ...")
        results[name] = evaluate_strategy(fn, RETRIEVAL_TEST_SET, top_k)
        r = results[name]
        print(f"   Hit@{top_k}={r['hit_rate']:.0%}  MRR={r['mrr']:.3f}  "
              f"P@{top_k}={r['precision']:.3f}  延迟={r['avg_latency']*1000:.0f}ms")

    # ---- 汇总表 ----
    print(f"\n{'='*76}")
    print("📊 总体对比")
    print(f"{'='*76}")
    print(f"{'策略':<24} {'Hit@K':>8} {'MRR':>8} {'P@K':>8} {'延迟':>10}")
    print("─" * 76)
    for name, r in results.items():
        print(f"{name:<24} {r['hit_rate']:>7.0%} {r['mrr']:>8.3f} "
              f"{r['precision']:>8.3f} {r['avg_latency']*1000:>8.0f}ms")

    # ---- 分类对比 ----
    print(f"\n{'='*76}")
    print("📊 按问题类型对比（MRR）")
    print(f"{'='*76}")
    kinds = ["语义", "术语", "混合"]
    print(f"{'策略':<24}" + "".join(f"{k:>12}" for k in kinds))
    print("─" * 76)
    for name, r in results.items():
        row = f"{name:<24}"
        for k in kinds:
            v = r["by_kind"].get(k, {})
            row += f"{v.get('mrr', 0):>12.3f}"
        print(row)

    # ---- 提升幅度 ----
    base = results["A. 仅向量检索（基线）"]
    best = results["D. 混合检索 + 重排"]
    print(f"\n{'='*76}")
    print("📈 相对基线的提升")
    print(f"{'='*76}")
    for metric, label in [("hit_rate", "Hit@K"), ("mrr", "MRR"), ("precision", "P@K")]:
        b, d = base[metric], best[metric]
        delta = d - b
        pct = (delta / b * 100) if b else 0
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  {label:<8} {b:.3f} → {d:.3f}  {arrow} {delta:+.3f} ({pct:+.1f}%)")

    # ---- 找出改善和退化的题目 ----
    print(f"\n{'='*76}")
    print("🔍 逐题变化（基线 → 混合+重排）")
    print(f"{'='*76}")
    improved, degraded = [], []
    for i, tc in enumerate(RETRIEVAL_TEST_SET):
        rr_base = base["details"][i]["rr"]
        rr_best = best["details"][i]["rr"]
        if rr_best > rr_base + 0.01:
            improved.append((tc["q"], rr_base, rr_best))
        elif rr_best < rr_base - 0.01:
            degraded.append((tc["q"], rr_base, rr_best))

    if improved:
        print(f"\n  ✅ 改善 {len(improved)} 题：")
        for q, a, b in improved:
            print(f"     {a:.2f} → {b:.2f}  {q}")
    if degraded:
        print(f"\n  ⚠️  退化 {len(degraded)} 题：")
        for q, a, b in degraded:
            print(f"     {a:.2f} → {b:.2f}  {q}")
    if not degraded:
        print(f"\n  ✅ 无退化题目")

    # ---- 保存结果 ----
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, "tests", "retrieval_experiment.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "top_k": top_k,
            "test_set_size": len(RETRIEVAL_TEST_SET),
            "summary": {
                name: {k: v for k, v in r.items() if k != "details"}
                for name, r in results.items()
            },
            "details": {name: r["details"] for name, r in results.items()},
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📄 详细结果已保存: tests/retrieval_experiment.json")
    print("=" * 76)


class HybridNoRerank:
    """混合检索但不重排（用于消融实验）。"""

    def __init__(self, retriever):
        self.r = retriever

    def retrieve(self, query: str, top_k: int):
        bm25_idx = self.r._search_bm25(query, 20)
        vector_docs = self.r._search_vector(query, 20)
        fused = self.r._rrf_fusion(bm25_idx, vector_docs)
        return [
            Document(page_content=t, metadata=dict(m))
            for t, m, _ in fused[:top_k]
        ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="检索策略对比实验")
    parser.add_argument("--k", type=int, default=5, help="Top-K")
    args = parser.parse_args()
    run_experiment(top_k=args.k)
