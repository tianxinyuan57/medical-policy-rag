"""RAGAS 式自动化评估 —— 用 LLM-as-judge 替代关键词匹配

====================================================================
为什么要换掉关键词匹配？
====================================================================

原来的 evaluate.py 用「回答里有没有出现某几个关键词」来判分。
这个方法便宜、快，但有明显缺陷：

    问题：护士执业注册需要什么条件？
    关键词：["民事行为能力", "护理", "资格考试"]

    回答 A：「需具有完全民事行为能力，完成护理专业学习，
             通过护士执业资格考试。」            → 命中 3/3 ✓
    回答 B：「民事行为能力、护理、资格考试。」    → 也命中 3/3 ✗

回答 B 什么都没说清楚，但关键词匹配给了满分。
反过来，同义改写（「护理学专业」vs「护理」）又会被误判为不命中。

业界标准是 RAGAS 的四个指标 + LLM-as-judge，它评的是**语义**而非字面。

====================================================================
本模块实现的六个指标
====================================================================

RAGAS 四大核心指标：

  1. Faithfulness（忠实度）★ 最重要
     把回答拆成若干原子陈述，逐条判断能否从检索原文推出。
     得分 = 有原文支撑的陈述数 / 总陈述数
     → 直接度量幻觉程度

  2. Answer Relevancy（答案相关性）
     回答是否切题、是否答非所问、有没有大量无关废话。

  3. Context Precision（上下文精确率）
     检索回来的 K 个片段里，有多少是真正相关的。
     → 度量检索的「准」

  4. Context Recall（上下文召回率）
     回答该有的要点，检索片段里是否都提供了依据。
     → 度量检索的「全」

本项目自研的两个指标：

  5. Citation Accuracy（引用准确率）★ RAGAS 没有
     回答里「《X法》第Y条」有多少是真的存在于检索片段中。
     这个指标 RAGAS 做不了 —— 它需要条款级 metadata，
     而我们的 smart_splitter 恰好提供了。

  6. Refusal Correctness（拒答正确性）
     该拒答的拒答了吗？不该拒答的有没有过度保守？

====================================================================
成本控制
====================================================================

每题的 LLM 调用次数：
    Faithfulness      2 次（拆陈述 + 批量判断）
    Answer Relevancy  1 次
    Context Precision 1 次（批量判断所有片段）
    Context Recall    1 次
    Citation Accuracy 0 次（纯程序，复用 citation_verifier）
    Refusal           0 次（纯程序，关键词判断）
                      ────
                      5 次 / 题

10 题约 50 次调用。DeepSeek 价格下这点成本可忽略。
支持 --metrics 参数只跑部分指标。
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import ask_llm_json


# =====================================================
# 指标 1：Faithfulness（忠实度）
# =====================================================

_CLAIM_SPLIT_SYS = """你是文本分析助手。把给定的回答拆解成若干条**原子陈述**。

要求：
1. 每条陈述表达一个独立的事实断言，不可再拆
2. 忽略纯格式性内容（如"以下是解读："、"📎 引用来源："）
3. 忽略主观建议（如"建议咨询当地部门"）
4. 最多拆 12 条

输出 JSON：{"claims": ["陈述1", "陈述2", ...]}"""

_FAITH_JUDGE_SYS = """你是严格的事实核查员。

给定【参考原文】和若干条【陈述】，逐条判断该陈述能否**直接从参考原文中推出**。

判断标准：
- 能从原文直接读出或合理推导 → supported: true
- 原文没提到、或与原文矛盾、或需要外部知识才能得出 → supported: false
- 同义改写、归纳概括算 supported（重点看语义，不看字面）

输出 JSON：
{"results": [{"index": 0, "supported": true, "reason": "简短理由"}, ...]}
results 的长度必须与陈述条数一致。"""


def eval_faithfulness(answer: str, context: str) -> dict:
    """忠实度：回答中有多少比例的陈述能被检索原文支撑。"""
    # 第 1 步：拆陈述
    r1 = ask_llm_json(_CLAIM_SPLIT_SYS, f"回答：\n{answer}")
    claims = r1.get("claims") or []
    if not claims:
        return {"score": None, "claims": 0, "supported": 0,
                "detail": [], "note": "未能拆出陈述"}

    # 第 2 步：批量判断
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    r2 = ask_llm_json(
        _FAITH_JUDGE_SYS,
        f"【参考原文】\n{context[:12000]}\n\n【陈述】\n{numbered}",
    )
    results = r2.get("results") or []

    supported = sum(1 for r in results if r.get("supported"))
    total = len(claims)

    detail = []
    for i, c in enumerate(claims):
        hit = next((r for r in results if r.get("index") == i), None)
        detail.append({
            "claim": c,
            "supported": bool(hit and hit.get("supported")),
            "reason": (hit or {}).get("reason", ""),
        })

    return {
        "score": round(supported / total, 3) if total else None,
        "claims": total,
        "supported": supported,
        "detail": detail,
    }


# =====================================================
# 指标 2：Answer Relevancy（答案相关性）
# =====================================================

_RELEVANCY_SYS = """你是回答质量评估员。判断【回答】是否切实回应了【问题】。

评分标准（0-1）：
1.0  直接、完整地回答了问题
0.7  回答了问题，但夹杂较多无关内容，或有遗漏
0.4  部分相关，但没答到核心
0.0  答非所问，或完全是套话

注意：
- 如果问题的答案确实不在知识范围内，回答明确说明「未找到相关规定」
  → 这是**正确行为**，应给 1.0
- 只是罗列关键词而没有实质说明 → 最多 0.4

输出 JSON：{"score": 0.0~1.0, "reason": "简短理由"}"""


def eval_answer_relevancy(question: str, answer: str) -> dict:
    r = ask_llm_json(
        _RELEVANCY_SYS,
        f"【问题】\n{question}\n\n【回答】\n{answer}",
    )
    s = r.get("score")
    try:
        s = round(float(s), 3)
    except (TypeError, ValueError):
        s = None
    return {"score": s, "reason": r.get("reason", "")}


# =====================================================
# 指标 3：Context Precision（上下文精确率）
# =====================================================

_PRECISION_SYS = """你是检索质量评估员。

给定一个【问题】和若干个【检索片段】，逐个判断该片段
对回答这个问题**是否有实质帮助**。

判断标准：
- 片段包含回答所需的信息（哪怕只是一部分）→ relevant: true
- 片段只是主题沾边、但不含有用信息 → relevant: false
- 片段完全无关 → relevant: false

输出 JSON：
{"results": [{"index": 0, "relevant": true}, ...]}
results 长度必须与片段数一致。"""


def eval_context_precision(question: str, chunks: list[str]) -> dict:
    """上下文精确率：检索片段中有多少是真正有用的。"""
    if not chunks:
        return {"score": None, "total": 0, "relevant": 0}

    listed = "\n\n".join(
        f"[片段{i}]\n{c[:900]}" for i, c in enumerate(chunks)
    )
    r = ask_llm_json(
        _PRECISION_SYS,
        f"【问题】\n{question}\n\n【检索片段】\n{listed}",
    )
    results = r.get("results") or []
    relevant = sum(1 for x in results if x.get("relevant"))
    total = len(chunks)

    return {
        "score": round(relevant / total, 3) if total else None,
        "total": total,
        "relevant": relevant,
        "flags": [bool(next((x.get("relevant") for x in results
                             if x.get("index") == i), False))
                  for i in range(total)],
    }


# =====================================================
# 指标 4：Context Recall（上下文召回率）
# =====================================================

_RECALL_SYS = """你是检索完整性评估员。

给定【问题】、【参考答案要点】和【检索片段】，
判断每个答案要点是否**能在检索片段中找到依据**。

输出 JSON：
{"results": [{"point": "要点原文", "found": true, "reason": "简短说明"}, ...]}"""


def eval_context_recall(question: str, context: str,
                        expected_points: list[str]) -> dict:
    """上下文召回率：该检索到的要点，检索片段里是否都有依据。

    Args:
        expected_points: 参考答案的关键要点（来自测试集）
    """
    if not expected_points:
        return {"score": None, "total": 0, "found": 0,
                "note": "该题无参考要点（如拒答题）"}

    pts = "\n".join(f"- {p}" for p in expected_points)
    r = ask_llm_json(
        _RECALL_SYS,
        f"【问题】\n{question}\n\n【参考答案要点】\n{pts}\n\n"
        f"【检索片段】\n{context[:12000]}",
    )
    results = r.get("results") or []
    found = sum(1 for x in results if x.get("found"))
    total = len(expected_points)

    return {
        "score": round(found / total, 3) if total else None,
        "total": total,
        "found": found,
        "detail": results,
    }


# =====================================================
# 指标 5：Citation Accuracy（自研，无需 LLM）
# =====================================================

def eval_citation_accuracy(answer: str, docs: list) -> dict:
    """引用准确率：回答里的法条引用有多少是真实存在于检索片段中的。

    这是本项目独有的指标 —— RAGAS 没有，因为它需要条款级 metadata。
    """
    from citation_verifier import verify

    res = verify(answer, docs)
    return {
        "score": round(res.accuracy, 3) if res.total else None,
        "total": res.total,
        "verified": res.verified_count,
        "abbrev": res.abbrev_count,
        "unverified": [f"《{c.law}》{c.article}（{c.note}）"
                       for c in res.unverified],
    }


# =====================================================
# 指标 6：Refusal Correctness（自研，无需 LLM）
# =====================================================

REFUSAL_MARKERS = ["未能找到", "未找到", "没有找到", "无法找到",
                   "未提及", "未涉及", "未包含", "不能直接"]

# 只看回答开头这么多字 —— 真正的拒答开门见山
_REFUSAL_HEAD = 160


def _is_refusal(answer: str) -> bool:
    """判断回答是否为拒答。

    【踩过的坑】最初用「全文包含拒答词」判断，结果大量误判：

        问：无偿献血的年龄范围是多少？
        答：根据《献血法》第二条，国家提倡十八周岁至五十五周岁……
            ⚠️ 提醒：可能存在特殊情形，但参考原文中**未涉及**此类特殊规定。

    末尾那句是在补充说明「某个细节原文没写」，不是拒答整个问题，
    却因为含「未涉及」被判成拒答。

    修正：拒答通常开门见山出现在**第一段**，
    而补充性说明出现在回答末尾。所以只检查开头一段。
    """
    if not answer:
        return False
    # 取第一段（首个空行前）或前 160 字，取更短的
    first_para = answer.split("\n\n", 1)[0]
    head = first_para[:_REFUSAL_HEAD]
    return any(m in head for m in REFUSAL_MARKERS)


def eval_refusal(answer: str, should_refuse: bool) -> dict:
    """拒答正确性：该拒答的拒答了吗？不该拒答的有没有过度保守？"""
    refused = _is_refusal(answer)
    correct = refused == should_refuse

    if should_refuse:
        note = "正确拒答" if refused else "⚠️ 应拒答却作答了（幻觉风险）"
    else:
        note = "正常作答" if not refused else "⚠️ 不该拒答却拒答了（过度保守）"

    return {"score": 1.0 if correct else 0.0,
            "refused": refused, "should_refuse": should_refuse, "note": note}


# =====================================================
# 单题综合评估
# =====================================================

ALL_METRICS = ["faithfulness", "answer_relevancy", "context_precision",
               "context_recall", "citation_accuracy", "refusal"]


@dataclass
class CaseResult:
    question: str
    type: str
    answer: str = ""
    metrics: dict = field(default_factory=dict)
    elapsed: float = 0.0


def evaluate_case(question: str, answer: str, docs: list,
                  expected_points: list[str] = None,
                  should_refuse: bool = False,
                  metrics: list[str] = None) -> dict:
    """对一道题跑指定的指标。

    Args:
        question: 问题
        answer:   系统生成的回答
        docs:     检索到的 Document 列表
        expected_points: 参考答案要点（用于 Context Recall）
        should_refuse:   这题是否应该拒答
        metrics:  要跑的指标名列表，None = 全部
    """
    metrics = metrics or ALL_METRICS
    chunks = [d.page_content for d in docs]
    context = "\n\n".join(chunks)
    out = {}

    if "faithfulness" in metrics:
        out["faithfulness"] = eval_faithfulness(answer, context)
    if "answer_relevancy" in metrics:
        out["answer_relevancy"] = eval_answer_relevancy(question, answer)
    if "context_precision" in metrics:
        out["context_precision"] = eval_context_precision(question, chunks)
    if "context_recall" in metrics:
        out["context_recall"] = eval_context_recall(
            question, context, expected_points or [])
    if "citation_accuracy" in metrics:
        out["citation_accuracy"] = eval_citation_accuracy(answer, docs)
    if "refusal" in metrics:
        out["refusal"] = eval_refusal(answer, should_refuse)

    return out


# =====================================================
# 全量评估运行器
# =====================================================

def run_full_evaluation(metrics: list[str] = None, limit: int = None,
                        verbose: bool = True, top_k: int = None,
                        graph_k: int = None, save: bool = True) -> dict:
    """在测试集上跑完整的 RAGAS 式评估。

    Args:
        metrics: 要跑的指标，None = 全部
        limit:   只跑前 N 题（调试用）
        verbose: 是否打印过程
        top_k:   检索片段数，None = 用 config 默认值（用于 K 值对照实验）
        graph_k: 图扩展法规数，None = 用 config 默认值
        save:    是否写入 tests/ragas_results.json

    Returns:
        汇总结果 dict
    """
    from evaluate import TEST_CASES
    from hybrid_retriever import get_retriever
    from rag import RAG_SYSTEM_PROMPT
    from llm import ask_llm
    from config import TOP_K, GRAPH_K

    metrics = metrics or ALL_METRICS
    cases = TEST_CASES[:limit] if limit else TEST_CASES
    retriever = get_retriever()
    TOP_K = TOP_K if top_k is None else top_k
    GRAPH_K = GRAPH_K if graph_k is None else graph_k

    if verbose:
        print("=" * 78)
        print("🧪 RAGAS 式自动化评估（LLM-as-judge）")
        print(f"   测试集 {len(cases)} 题 | Top-K={TOP_K} | Graph-K={GRAPH_K}")
        print(f"   指标: {', '.join(metrics)}")
        print("=" * 78)

    results = []
    t_start = time.time()

    for i, tc in enumerate(cases, 1):
        q = tc["question"]
        is_refusal = tc["type"] == "拒答测试"

        if verbose:
            print(f"\n[{i}/{len(cases)}] [{tc['type']}] {q}")
            print("  ⏳ 检索 + 生成 ...", end="", flush=True)

        t0 = time.time()

        # 检索 + 生成（走完整链路）
        docs, _ = retriever.retrieve_with_graph(q, top_k=TOP_K, graph_k=GRAPH_K)
        blocks = []
        for j, d in enumerate(docs):
            src = os.path.basename(d.metadata.get("source", "")).replace(".txt", "")
            blocks.append(f"[片段{j+1}｜来源：《{src}》]\n{d.page_content}")
        prompt = f"【参考原文】\n{chr(10).join(blocks)}\n\n【问题】\n{q}"
        answer = ask_llm(RAG_SYSTEM_PROMPT, prompt)

        if verbose:
            print(f"\r  ⏳ 评估中 ...          ", end="", flush=True)

        m = evaluate_case(
            question=q, answer=answer, docs=docs,
            expected_points=tc.get("expected_points"),
            should_refuse=is_refusal,
            metrics=metrics,
        )
        elapsed = time.time() - t0

        results.append({
            "question": q, "type": tc["type"],
            "answer": answer, "metrics": m, "elapsed": round(elapsed, 1),
        })

        if verbose:
            print("\r" + " " * 40 + "\r", end="")
            for name in metrics:
                v = m.get(name, {})
                s = v.get("score")
                if s is None:
                    print(f"    {_label(name):<18} —（{v.get('note', '不适用')}）")
                else:
                    bar = _bar(s)
                    print(f"    {_label(name):<18} {s:.2f} {bar} {_extra(name, v)}")
            print(f"    ⏱️  {elapsed:.1f}s")

    total_time = time.time() - t_start

    # ---- 汇总 ----
    summary = {}
    for name in metrics:
        vals = [r["metrics"][name]["score"] for r in results
                if r["metrics"].get(name, {}).get("score") is not None]
        summary[name] = round(sum(vals) / len(vals), 3) if vals else None

    # 按题型分组
    by_type = {}
    for r in results:
        t = r["type"]
        by_type.setdefault(t, {n: [] for n in metrics})
        for n in metrics:
            s = r["metrics"].get(n, {}).get("score")
            if s is not None:
                by_type[t][n].append(s)
    for t, d in by_type.items():
        by_type[t] = {n: (round(sum(v) / len(v), 3) if v else None)
                      for n, v in d.items()}

    if verbose:
        print(f"\n{'='*78}")
        print("📊 总体得分")
        print("=" * 78)
        for name in metrics:
            s = summary[name]
            if s is None:
                print(f"  {_label(name):<20} —")
            else:
                print(f"  {_label(name):<20} {s:.3f}  {_bar(s, 30)}")

        print(f"\n{'='*78}")
        print("📊 分题型得分")
        print("=" * 78)
        types = list(by_type.keys())
        print(f"  {'指标':<20}" + "".join(f"{t:>12}" for t in types))
        print("  " + "─" * 74)
        for name in metrics:
            row = f"  {_label(name):<20}"
            for t in types:
                v = by_type[t].get(name)
                row += f"{v if v is not None else '—':>12}"
            print(row)

        print(f"\n  总耗时 {total_time:.0f}s（{total_time/len(cases):.1f}s/题）")
        print("=" * 78)

    # ---- 保存 ----
    if not save:
        return {"summary": summary, "by_type": by_type, "cases": results}

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(project_root, "tests", "ragas_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary, "by_type": by_type,
            "metrics_used": metrics,
            "total_seconds": round(total_time, 1),
            "cases": results,
        }, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"📄 详细结果: tests/ragas_results.json\n")

    return {"summary": summary, "by_type": by_type, "cases": results}


# ---- 展示辅助 ----

_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "citation_accuracy": "Citation Accuracy",
    "refusal": "Refusal",
}


def _label(name: str) -> str:
    return _LABELS.get(name, name)


def _bar(score: float, width: int = 12) -> str:
    n = int(round(score * width))
    return "█" * n + "░" * (width - n)


def _extra(name: str, v: dict) -> str:
    if name == "faithfulness":
        return f"({v.get('supported')}/{v.get('claims')} 条陈述有支撑)"
    if name == "context_precision":
        return f"({v.get('relevant')}/{v.get('total')} 个片段相关)"
    if name == "context_recall":
        return f"({v.get('found')}/{v.get('total')} 个要点有依据)"
    if name == "citation_accuracy":
        return f"({v.get('verified')}/{v.get('total')} 处引用核实)"
    if name == "refusal":
        return v.get("note", "")
    return ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAGAS 式自动化评估（LLM-as-judge）")
    parser.add_argument("--metrics", type=str, default=None,
                        help=f"逗号分隔，可选: {','.join(ALL_METRICS)}")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 题")
    parser.add_argument("--top-k", type=int, default=None,
                        help="覆盖检索片段数（K 值对照实验用）")
    parser.add_argument("--graph-k", type=int, default=None,
                        help="覆盖图扩展法规数")
    args = parser.parse_args()

    mets = None
    if args.metrics:
        mets = [m.strip() for m in args.metrics.split(",") if m.strip()]
        bad = [m for m in mets if m not in ALL_METRICS]
        if bad:
            print(f"❌ 未知指标: {bad}\n可选: {ALL_METRICS}")
            sys.exit(1)

    run_full_evaluation(metrics=mets, limit=args.limit,
                        top_k=args.top_k, graph_k=args.graph_k)
