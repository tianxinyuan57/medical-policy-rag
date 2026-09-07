"""混合检索 + 重排序（Hybrid Retrieval + Reranking）

====================================================================
为什么单纯的向量检索不够？
====================================================================

向量检索（Dense Retrieval）的优势与短板：
  ✅ 理解语义："禁止买卖器官" ≈ "不得交易人体器官"
  ❌ 精确术语弱："飞行检查""DIP 分值""第二十条" 这类专有名词，
     向量可能把它和语义相近但字面不同的内容混淆

BM25（Sparse Retrieval）的优势与短板：
  ✅ 精确匹配强：查"飞行检查"必然命中含这个词的文档
  ❌ 不懂同义："器官买卖" 查不到 "人体器官交易"

结论：两者互补 → 混合检索（Hybrid Search）

====================================================================
三阶段检索架构
====================================================================

  用户问题
     │
     ├──→ ① BM25 关键词检索      ──→ Top-20 候选
     │
     ├──→ ② 向量语义检索          ──→ Top-20 候选
     │
     ↓
  ③ RRF 融合（Reciprocal Rank Fusion）  ──→ 合并去重，Top-20
     │
     ↓
  ④ Cross-Encoder 重排（bge-reranker）  ──→ 精排 Top-5
     │
     ↓
  送入 LLM 生成

====================================================================
关键概念：RRF（倒数排名融合）
====================================================================

问题：BM25 的分数（0~30+）和向量相似度（0~1）量纲完全不同，
      不能直接加权求和。

RRF 的解法：不看分数，只看排名。

    RRF_score(d) = Σ  1 / (k + rank_i(d))
                  i∈检索器

其中 k=60（经验值，防止头部排名权重过大）。

例：某文档在 BM25 排第 1、在向量检索排第 5
    RRF = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

优点：无需归一化、无需调权重、对异常分数鲁棒。
这是 Elasticsearch、Azure AI Search 等生产系统的标准做法。

====================================================================
关键概念：Cross-Encoder 重排
====================================================================

检索用的 Bi-Encoder（双塔）：
    问题 → 向量A          }  分别编码，算余弦相似度
    文档 → 向量B          }  快，但问题和文档没有交互

重排用的 Cross-Encoder（交叉编码）：
    [问题 + 文档] → 一起进模型 → 相关性分数
    慢，但精度高得多（问题和文档的每个词都能互相 attention）

所以工程上的标准做法：
    Bi-Encoder 粗筛（快，2096 个片段 → 20 个）
    Cross-Encoder 精排（慢，但只处理 20 个）
"""

import os
import sys
import pickle
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import CHROMA_DIR, EMBED_MODEL, TOP_K


# =====================================================
# 参数
# =====================================================

RRF_K = 60              # RRF 平滑常数（经验值，业界标准）
CANDIDATE_K = 20        # 每路检索召回的候选数
RERANK_MODEL = "BAAI/bge-reranker-base"  # 中文重排模型

# 【重要的工程决策：默认关闭重排】
#
# 实验数据（20 题测试集，见 retrieval_experiment.py）：
#
#   策略                Hit@5    MRR      P@5     延迟
#   ─────────────────────────────────────────────────
#   仅向量（基线）        90%    0.825    0.700     19ms
#   仅 BM25             85%    0.785    0.570      3ms
#   混合（RRF）         100%    0.950    0.740     20ms  ← 最优
#   混合 + 重排         100%    0.938    0.690   6159ms  ← 负收益
#
# 结论：在本场景下重排是负收益 —— 指标略降，延迟涨 300 倍。
#
# 根因分析（以"个人健康信息被泄露了怎么办？"为例）：
#   重排模型被"怎么办"这个意图词带偏，把《举报处理办法》
#   《互联网诊疗管理办法》排到前面，正确答案《个人信息保护法》
#   从第 1 位被挤到第 4 位。Cross-Encoder 对意图词过度敏感，
#   反而稀释了核心实体"个人信息"的权重。
#
# 为什么 RRF 已经够好？
#   我们的 chunk 是条款级切分且带 [《法名》第X条] 标注，
#   语义边界本身就很干净，两路检索的排名信号已足够可靠，
#   重排能提供的额外信息有限。
#
# 保留 use_reranker 开关，便于换模型/换场景时重新验证。
DEFAULT_USE_RERANKER = False

# BM25 索引缓存路径（避免每次启动重建）
BM25_CACHE = "bm25_index.pkl"


# =====================================================
# 中文分词（BM25 的前置步骤）
# =====================================================
#
# 【为什么需要分词？】
# BM25 是词袋模型，需要把文本切成词。
# 英文天然用空格分词，中文没有空格，必须用分词器。
#
# jieba 是最常用的中文分词库。
# 例："医疗保障基金飞行检查" → ["医疗", "保障", "基金", "飞行", "检查"]

# 加入医疗政策领域的专有名词，提升分词准确率
DOMAIN_WORDS = [
    "飞行检查", "医疗保障", "医保基金", "定点医药机构", "异地就医",
    "跨省结算", "门诊统筹", "DRG付费", "DIP付费", "病种分值",
    "医保目录", "集中采购", "带量采购", "医师资格", "执业注册",
    "多点执业", "师承", "确有专长", "病历书写", "电子病历",
    "医疗纠纷", "医疗事故", "知情同意", "处方管理", "长期处方",
    "麻醉药品", "精神药品", "毒性药品", "放射性药品", "中药饮片",
    "医疗废物", "血液制品", "器官移植", "无偿献血", "免疫规划",
    "传染病防治", "职业病", "母婴保健", "计划生育", "智能监管",
    "骗保", "违规使用", "结算清单", "病案首页", "编码标准",
]

for word in DOMAIN_WORDS:
    jieba.add_word(word)


def tokenize_zh(text: str) -> list[str]:
    """中文分词，过滤停用词和标点。"""
    # 去掉常见的无意义符号
    tokens = jieba.lcut(text)
    return [
        t.strip() for t in tokens
        if t.strip() and len(t.strip()) > 1 or t.strip().isdigit()
    ]


# =====================================================
# 混合检索器
# =====================================================

class HybridRetriever:
    """BM25 + 向量检索 + RRF 融合 + Cross-Encoder 重排。

    用法：
        retriever = HybridRetriever(vectorstore)
        docs = retriever.retrieve("无偿献血年龄", top_k=5)
    """

    def __init__(self, vectorstore, use_reranker: bool = DEFAULT_USE_RERANKER):
        """
        Args:
            vectorstore: 已加载的 Chroma 向量库
            use_reranker: 是否启用 Cross-Encoder 重排。
                          默认 False —— 实验证明本场景下是负收益，
                          详见文件顶部的实验数据说明。
        """
        self.vectorstore = vectorstore
        self.use_reranker = use_reranker
        self._reranker = None  # 懒加载

        # 构建 BM25 索引
        self._build_bm25_index()

    # ---------- BM25 索引 ----------

    def _build_bm25_index(self):
        """从 Chroma 里取出全部文档，构建 BM25 索引。

        【面试点：为什么 BM25 索引要单独建？】
        Chroma 只存向量，不支持关键词检索。
        所以需要把文档全部取出来，用 rank_bm25 单独建一个倒排索引。
        文档量不大（2096 个片段）时，内存里建索引完全够用。
        数据量大了应该换成 Elasticsearch / OpenSearch。
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_path = os.path.join(project_root, BM25_CACHE)

        # 从 Chroma 取全部文档
        data = self.vectorstore.get(include=["documents", "metadatas"])
        self.corpus_texts = data["documents"]
        self.corpus_metas = data["metadatas"]

        # 尝试读缓存（分词很慢，2096 个片段约 3-5 秒）
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("count") == len(self.corpus_texts):
                    self.bm25 = cached["bm25"]
                    return
            except Exception:
                pass  # 缓存损坏，重建

        # 分词并建索引
        tokenized_corpus = [tokenize_zh(t) for t in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 写缓存
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"bm25": self.bm25, "count": len(self.corpus_texts)}, f)
        except Exception:
            pass

    # ---------- 重排模型（懒加载）----------

    @property
    def reranker(self):
        """懒加载 Cross-Encoder 重排模型（首次用到时才加载，约 1.1GB）。"""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cpu")
        return self._reranker

    # ---------- 三路检索 ----------

    def _search_bm25(self, query: str, k: int) -> list[int]:
        """BM25 关键词检索，返回文档索引列表（按相关性降序）。"""
        tokens = tokenize_zh(query)
        scores = self.bm25.get_scores(tokens)
        # 取分数最高的 k 个索引
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [i for i in top_idx if scores[i] > 0]  # 过滤掉零分

    def _search_vector(self, query: str, k: int) -> list[Document]:
        """向量语义检索。"""
        return self.vectorstore.similarity_search(query, k=k)

    # ---------- RRF 融合 ----------

    def _rrf_fusion(
        self,
        bm25_idx: list[int],
        vector_docs: list[Document],
    ) -> list[tuple[str, dict, float]]:
        """用 RRF 融合两路检索结果。

        Returns:
            [(文本, metadata, rrf_score), ...] 按 RRF 分数降序
        """
        # 用文本内容作为唯一标识（因为两路返回的对象不同）
        scores: dict[str, float] = {}
        contents: dict[str, tuple[str, dict]] = {}

        # BM25 路：rank 从 1 开始
        for rank, idx in enumerate(bm25_idx, start=1):
            text = self.corpus_texts[idx]
            meta = self.corpus_metas[idx] or {}
            key = text[:200]  # 用前200字做 key（完整文本可能很长）
            scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank)
            contents[key] = (text, meta)

        # 向量路
        for rank, doc in enumerate(vector_docs, start=1):
            text = doc.page_content
            key = text[:200]
            scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank)
            contents[key] = (text, doc.metadata or {})

        # 按融合分数排序
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(contents[k][0], contents[k][1], s) for k, s in ranked]

    # ---------- Cross-Encoder 重排 ----------

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[str, dict, float]],
        top_k: int,
    ) -> list[tuple[str, dict, float]]:
        """用 Cross-Encoder 对候选做精排。"""
        if not candidates:
            return []

        pairs = [(query, text) for text, _, _ in candidates]
        scores = self.reranker.predict(pairs)

        # 按重排分数排序
        scored = [
            (candidates[i][0], candidates[i][1], float(scores[i]))
            for i in range(len(candidates))
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    # ---------- 主入口 ----------

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        candidate_k: int = CANDIDATE_K,
        return_scores: bool = False,
    ) -> list[Document]:
        """完整的混合检索流程。

        Args:
            query: 用户问题
            top_k: 最终返回的片段数
            candidate_k: 每路检索的候选数（融合前）
            return_scores: metadata 里是否附带检索分数（调试用）

        Returns:
            list[Document]: 排序后的文档列表
        """
        # ① BM25 检索
        bm25_idx = self._search_bm25(query, candidate_k)

        # ② 向量检索
        vector_docs = self._search_vector(query, candidate_k)

        # ③ RRF 融合
        fused = self._rrf_fusion(bm25_idx, vector_docs)

        # ④ Cross-Encoder 重排
        if self.use_reranker and fused:
            final = self._rerank(query, fused[:candidate_k], top_k)
            score_name = "rerank_score"
        else:
            final = fused[:top_k]
            score_name = "rrf_score"

        # 转回 LangChain Document
        docs = []
        for text, meta, score in final:
            m = dict(meta)
            if return_scores:
                m[score_name] = round(score, 4)
            docs.append(Document(page_content=text, metadata=m))
        return docs

    # ---------- 对比用：单路检索 ----------

    def retrieve_vector_only(self, query: str, top_k: int = TOP_K) -> list[Document]:
        """仅向量检索（基线，用于 A/B 对比）。"""
        return self._search_vector(query, top_k)

    def retrieve_bm25_only(self, query: str, top_k: int = TOP_K) -> list[Document]:
        """仅 BM25 检索（用于 A/B 对比）。"""
        idx = self._search_bm25(query, top_k)
        return [
            Document(
                page_content=self.corpus_texts[i],
                metadata=self.corpus_metas[i] or {},
            )
            for i in idx
        ]


# =====================================================
# 便捷工厂函数
# =====================================================

_retriever_cache: Optional[HybridRetriever] = None


def get_retriever(vectorstore=None,
                  use_reranker: bool = DEFAULT_USE_RERANKER) -> HybridRetriever:
    """获取全局单例的混合检索器（避免重复建 BM25 索引）。"""
    global _retriever_cache
    if _retriever_cache is None:
        if vectorstore is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma
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
        _retriever_cache = HybridRetriever(vectorstore, use_reranker=use_reranker)
    return _retriever_cache


# =====================================================
# 命令行测试
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="混合检索测试")
    parser.add_argument("query", nargs="?", default="无偿献血的年龄范围",
                        help="测试查询")
    parser.add_argument("--k", type=int, default=5, help="返回片段数")
    parser.add_argument("--compare", action="store_true",
                        help="对比三种检索方式")
    args = parser.parse_args()

    print("=" * 70)
    print("🔍 混合检索测试")
    print("=" * 70)
    print(f"查询: {args.query}\n")

    print("正在加载向量库和构建 BM25 索引...")
    retriever = get_retriever()
    print(f"✅ 就绪（{len(retriever.corpus_texts)} 个片段）\n")

    def show(title: str, docs: list[Document]):
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print(f"{'─'*70}")
        for i, d in enumerate(docs, 1):
            src = os.path.basename(d.metadata.get("source", "")).replace(".txt", "")
            label = d.metadata.get("article_label", "")
            score_str = ""
            for sk in ("rerank_score", "rrf_score"):
                if sk in d.metadata:
                    score_str = f" | {sk}={d.metadata[sk]}"
            preview = d.page_content[:100].replace("\n", " ")
            print(f"  [{i}] 《{src}》{label}{score_str}")
            print(f"      {preview}...")

    if args.compare:
        show("① 仅向量检索（基线）", retriever.retrieve_vector_only(args.query, args.k))
        show("② 仅 BM25 检索", retriever.retrieve_bm25_only(args.query, args.k))
        show("③ 混合检索 + 重排", retriever.retrieve(args.query, args.k, return_scores=True))
    else:
        show("混合检索 + 重排结果", retriever.retrieve(args.query, args.k, return_scores=True))

    print()
