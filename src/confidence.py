"""检索置信度评估 —— 让系统知道「自己不知道」

====================================================================
为什么需要置信度？
====================================================================

RAG 系统有个危险特性：**无论问什么，它都会返回 Top-K 个片段**。

问「今天天气怎么样」，向量库照样返回 5 个医疗法规片段 ——
因为相似度检索只找「最近的」，不判断「够不够近」。

如果只靠 Prompt 里那句「找不到就说找不到」，模型面对
5 段看似相关的原文，很可能强行拼凑出一个答案。

在医疗政策场景，这个风险不可接受：
用户可能拿着错误的报销比例去医院，或者按错误的法条理解自己的权利。

所以系统需要在**检索阶段**就判断：这次检索到的东西，够不够回答？

====================================================================
三个置信度信号（基于实测数据选定）
====================================================================

对「知识库能答」和「知识库答不了」两组问题做对照实验：

    问题                        向量Top1距离   BM25Top1   两路重合
    ────────────────────────────────────────────────────────────
    【能答】
    护士执业注册需要什么条件？        0.373      17.53      60%
    医疗保障基金飞行检查的启动情形     0.394      23.17      90%
    什么是假药？                    0.400      28.04      30%
    无偿献血的年龄范围是多少？         0.563      10.08      20%
    【答不了】
    2025年医保报销比例是多少？        0.515      30.74      25%   ← 主题相关但无答案
    北京协和医院的挂号费是多少？        0.757       0.00       0%
    ChatGPT能用来做医疗诊断吗？       0.888       6.93       0%
    今天天气怎么样？                 1.235       0.00       0%
    怎么做红烧肉？                   1.188       0.00       0%

结论：
  · **向量 Top1 距离**是最强单一信号（能答的都 < 0.57）
  · **BM25 = 0 且两路重合 = 0** 是「完全跑题」的铁证
  · 「2025年医保报销比例」是个有趣的边界案例 —— 主题高度相关
    （BM25 甚至最高），但知识库里没有那个具体数字。
    这类问题**不该被置信度拦截**，而应该由拒答 Prompt 处理：
    检索是对的，只是原文里没有答案。

====================================================================
三档策略
====================================================================

    high    正常回答，不打扰用户
    medium  正常回答 + 顶部温和提示（"相关度一般，建议核对原文"）
    low     显著警告（"本次检索相关度很低，回答可能不准确"）

注意：置信度低 ≠ 直接拒答。
拒答的决定权仍在模型手里（它能看到原文，判断更准）。
置信度的作用是**给用户一个额外的信号**，而不是替模型做决定。
"""

from dataclasses import dataclass


# =====================================================
# 阈值（来自上述对照实验，留了安全余量）
# =====================================================

# 向量 Top1 距离（Chroma L2，归一化向量，越小越相似）
DIST_HIGH = 0.62      # 优于此值 → 高置信（实测能答的最差 0.563）
DIST_LOW = 0.85       # 差于此值 → 低置信（实测答不了的多在 0.75+）

# BM25 Top1 分数
BM25_WEAK = 1.0       # 低于此值视为关键词几乎没匹配上

# 两路检索重合度
OVERLAP_WEAK = 0.05   # 低于此值视为两路各说各话


@dataclass
class Confidence:
    """一次检索的置信度评估结果。"""
    level: str                 # "high" | "medium" | "low"
    vector_dist: float         # 向量 Top1 距离
    bm25_top: float            # BM25 Top1 分数
    overlap: float             # 两路重合度 0~1
    reason: str = ""           # 判定理由（面向用户）

    @property
    def is_low(self) -> bool:
        return self.level == "low"

    @property
    def should_warn(self) -> bool:
        return self.level in ("low", "medium")

    @property
    def score(self) -> float:
        """归一到 0~1 的综合置信分（仅用于展示进度条）。"""
        # 距离越小越好，映射到 0~1
        d = max(0.0, min(1.0, (1.3 - self.vector_dist) / (1.3 - 0.35)))
        # 重合度直接贡献
        o = min(1.0, self.overlap / 0.6)
        # BM25 有匹配即加分
        b = 1.0 if self.bm25_top >= 5 else (self.bm25_top / 5 if self.bm25_top > 0 else 0.0)
        return round(0.6 * d + 0.25 * o + 0.15 * b, 3)

    def user_message(self) -> str:
        """给用户看的提示文案。"""
        if self.level == "high":
            return ""
        if self.level == "medium":
            return ("本次检索的相关度一般，回答可能不够全面 —— "
                    "建议展开「查看检索片段」核对原文。")
        return ("本次检索未找到高度相关的法规条文，"
                "该问题可能超出知识库范围，回答仅供参考。")


def assess(retriever, query: str, rewritten_query: str = None) -> Confidence:
    """评估一次检索的置信度。

    Args:
        retriever: HybridRetriever 实例
        query: 用户原始问题
        rewritten_query: 术语归一化后的查询（没有则用原问题）

    Returns:
        Confidence
    """
    from hybrid_retriever import tokenize_zh

    q = rewritten_query or query

    # ---- 信号 1：向量 Top1 距离 ----
    try:
        hits = retriever.vectorstore.similarity_search_with_score(q, k=3)
        vector_dist = float(hits[0][1]) if hits else 99.0
    except Exception:
        vector_dist = 99.0

    # ---- 信号 2：BM25 Top1 分数 ----
    try:
        scores = retriever.bm25.get_scores(tokenize_zh(q))
        bm25_top = float(max(scores)) if len(scores) else 0.0
    except Exception:
        bm25_top = 0.0

    # ---- 信号 3：两路检索重合度 ----
    try:
        bm_idx = retriever._search_bm25(q, 20)
        bm_texts = {retriever.corpus_texts[i][:200] for i in bm_idx}
        vec_texts = {d.page_content[:200]
                     for d in retriever._search_vector(q, 20)}
        overlap = len(bm_texts & vec_texts) / 20 if bm_texts else 0.0
    except Exception:
        overlap = 0.0

    # ---- 综合判定 ----
    # 铁证级低置信：关键词完全没匹配 + 两路毫无重合 + 语义也远
    totally_off = (bm25_top < BM25_WEAK
                   and overlap < OVERLAP_WEAK
                   and vector_dist > DIST_HIGH)

    if totally_off or vector_dist > DIST_LOW:
        level = "low"
        if totally_off:
            reason = "关键词无匹配、两路检索无重合、语义距离远"
        else:
            reason = f"语义距离过大（{vector_dist:.2f}）"
    elif vector_dist <= DIST_HIGH:
        level = "high"
        reason = f"语义高度相关（距离 {vector_dist:.2f}）"
    else:
        level = "medium"
        reason = f"语义相关度一般（距离 {vector_dist:.2f}）"

    return Confidence(
        level=level,
        vector_dist=round(vector_dist, 3),
        bm25_top=round(bm25_top, 2),
        overlap=round(overlap, 3),
        reason=reason,
    )


# =====================================================
# 命令行自测
# =====================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from hybrid_retriever import get_retriever
    from query_rewriter import rewrite

    cases = [
        # (问题, 预期档位)
        ("无偿献血的年龄范围是多少？", "high"),
        ("什么是假药？", "high"),
        ("护士执业注册需要什么条件？", "high"),
        ("医疗保障基金飞行检查的启动情形", "high"),
        ("甲类传染病包括哪些？", "high"),
        ("2025年医保报销比例是多少？", "high/medium"),   # 主题相关，靠拒答处理
        ("北京协和医院的挂号费是多少？", "medium/low"),
        ("ChatGPT能用来做医疗诊断吗？", "low"),
        ("今天天气怎么样？", "low"),
        ("怎么做红烧肉？", "low"),
        ("如何用Python写一个爬虫？", "low"),
    ]

    print("=" * 82)
    print("🎯 检索置信度评估测试")
    print("=" * 82)
    print(f"{'问题':<30} {'判定':>7} {'预期':>12} {'距离':>7} {'BM25':>7} {'重合':>6} {'分数':>6}")
    print("─" * 82)

    r = get_retriever()
    for q, expect in cases:
        rq = rewrite(q)["rewritten"]
        c = assess(r, q, rq)
        ok = "✓" if c.level in expect else "✗"
        print(f"{ok} {q[:27]:<28} {c.level:>7} {expect:>12} "
              f"{c.vector_dist:>7.3f} {c.bm25_top:>7.2f} "
              f"{c.overlap:>6.0%} {c.score:>6.2f}")

    print("\n" + "─" * 82)
    print("提示文案示例：")
    for lvl in ("medium", "low"):
        demo = Confidence(level=lvl, vector_dist=0.9, bm25_top=0, overlap=0)
        print(f"  [{lvl}] {demo.user_message()}")
    print()
