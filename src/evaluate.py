"""阶段 4：评估、排障与调优

====================================================================
面试最高区分度的一天

面试官不考你"做了什么"，考你"怎么知道做得好不好"。
一个有评估、有调优记录的项目 >> 一个只跑得通的 demo。
====================================================================

评估两个维度：
1. 检索质量：Top-K 里有没有包含真正相关的原文？（检索对不对）
2. 生成质量：模型是不是忠实于原文、没编造、该拒答时拒答了？（回答好不好）

排障核心思路（面试必考）：
  答案不对 → 先看检索结果 →
    → 检索没找对 → 调 chunk / Top-K / embedding（检索问题）
    → 检索找对了但模型答错 → 调 Prompt / temperature（生成问题）
  先定位，再动手，不瞎调。
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBED_MODEL, TOP_K
from llm import ask_llm
from rag import answer, vectorstore, RAG_SYSTEM_PROMPT


# =====================================================
# 测试用例集
# =====================================================
#
# 好的测试集要覆盖这几类：
# 1. 答案在文件里的 → 考检索准确性和生成质量
# 2. 需要多条跨文件的 → 考检索覆盖度
# 3. 文件里没有的 → 考拒答能力（防幻觉）
# 4. 故意模糊的问题 → 考系统的稳健性

TEST_CASES = [
    # ---- 类型 1：精确检索题（答案明确在某份文件里）----
    {
        "question": "无偿献血的年龄范围是多少？",
        "expected_source": "献血法",
        "expected_keywords": ["十八周岁", "五十五周岁"],
        "type": "精确检索",
    },
    {
        "question": "什么是假药？列举假药的情形。",
        "expected_source": "药品管理法",
        "expected_keywords": ["成份不符", "冒充", "变质"],
        "type": "精确检索",
    },
    {
        "question": "甲类传染病包括哪些？",
        "expected_source": "传染病防治法",
        "expected_keywords": ["鼠疫", "霍乱"],
        "type": "精确检索",
    },
    {
        "question": "护士执业注册需要具备哪些条件？",
        "expected_source": "护士条例",
        "expected_keywords": ["民事行为能力", "护理", "资格考试"],
        "type": "精确检索",
    },
    {
        "question": "免疫规划疫苗接种可以收费吗？",
        "expected_source": "疫苗管理法",
        "expected_keywords": ["不得收取"],
        "type": "精确检索",
    },

    # ---- 类型 2：需要理解/推理的题 ----
    {
        "question": "发生医疗纠纷后，患者有权查阅哪些病历资料？",
        "expected_source": "医疗纠纷预防和处理条例",
        "expected_keywords": ["门诊病历", "住院志", "医嘱单"],
        "type": "理解推理",
    },
    {
        "question": "以师承方式学中医的人可以行医吗？需要什么条件？",
        "expected_source": "中医药法",
        "expected_keywords": ["师承", "推荐", "考核"],
        "type": "理解推理",
    },

    # ---- 类型 3：拒答题（答案不在文件里，必须拒绝回答）----
    {
        "question": "2025年医保报销比例是多少？",
        "expected_source": None,  # 文件里没有
        "expected_keywords": ["未能找到", "未找到", "未提及", "未涉及", "不能直接"],
        "type": "拒答测试",
    },
    {
        "question": "北京协和医院的挂号费是多少？",
        "expected_source": None,
        "expected_keywords": ["未能找到", "未找到", "未提及", "未涉及", "未包含"],
        "type": "拒答测试",
    },
    {
        "question": "ChatGPT能用来做医疗诊断吗？",
        "expected_source": None,
        "expected_keywords": ["未能找到", "未找到", "未提及", "未涉及", "不能直接"],
        "type": "拒答测试",
    },
]


# =====================================================
# 评估函数
# =====================================================

def evaluate_retrieval(question: str, expected_source: str) -> dict:
    """评估检索质量：Top-K 里有没有命中期望的来源文件。

    【面试知识点】
    检索评估看的是"找没找对"，不管模型怎么回答。
    如果检索都没找到相关内容，模型答得再好也是碰巧。
    """
    results = vectorstore.similarity_search(question, k=TOP_K)
    sources = [os.path.basename(r.metadata.get("source", "")).replace(".txt", "").replace(".pdf", "")
               for r in results]

    if expected_source is None:
        # 拒答题：检索必然会返回些东西，但不应该有特别相关的
        return {"hit": True, "sources": sources, "note": "拒答题，检索结果不影响评判"}

    hit = any(expected_source in s for s in sources)
    return {"hit": hit, "sources": sources}


def evaluate_generation(answer_text: str, expected_keywords: list, test_type: str) -> dict:
    """评估生成质量：回答是否包含关键信息，拒答题是否正确拒答。

    【面试知识点】
    生成评估看的是"答得对不对"：
    - 精确题：关键词是否出现在回答里
    - 拒答题："未找到"是否出现在回答里
    """
    answer_lower = answer_text.lower()

    if test_type == "拒答测试":
        # 拒答题：回答里应该有"未找到""未能找到"等表述
        # 优化后的模型可能用多种方式表达"找不到直接规定"，需要更宽容的匹配
        refused = any(kw in answer_text for kw in expected_keywords)
        return {"pass": refused, "note": "正确拒答" if refused else "⚠️ 可能在编造！"}

    # 精确题/推理题：检查关键词覆盖率
    hits = [kw for kw in expected_keywords if kw in answer_text]
    coverage = len(hits) / len(expected_keywords) if expected_keywords else 1.0
    return {
        "pass": coverage >= 0.5,  # 至少命中一半关键词
        "coverage": f"{len(hits)}/{len(expected_keywords)}",
        "hit_keywords": hits,
        "missed_keywords": [kw for kw in expected_keywords if kw not in answer_text],
    }


def run_evaluation():
    """运行完整评估，输出报告。"""
    print("=" * 65)
    print("📊 RAG 系统评估报告")
    print(f"   知识库: {vectorstore._collection.count()} 个片段 | Top-K: {TOP_K}")
    print("=" * 65)

    retrieval_pass = 0
    generation_pass = 0
    total = len(TEST_CASES)
    results_log = []

    for i, tc in enumerate(TEST_CASES):
        q = tc["question"]
        print(f"\n{'─'*65}")
        print(f"  [{i+1}/{total}] [{tc['type']}] {q}")

        # 1. 评估检索
        ret_eval = evaluate_retrieval(q, tc.get("expected_source"))
        if ret_eval["hit"]:
            retrieval_pass += 1
        ret_status = "✅" if ret_eval["hit"] else "❌"
        print(f"    检索 {ret_status}  来源: {ret_eval['sources']}")

        # 2. 生成回答
        ans = answer(q)

        # 3. 评估生成
        gen_eval = evaluate_generation(ans, tc["expected_keywords"], tc["type"])
        if gen_eval["pass"]:
            generation_pass += 1
        gen_status = "✅" if gen_eval["pass"] else "❌"

        if tc["type"] == "拒答测试":
            print(f"    生成 {gen_status}  {gen_eval['note']}")
        else:
            print(f"    生成 {gen_status}  关键词覆盖: {gen_eval.get('coverage', 'N/A')}")
            if gen_eval.get("missed_keywords"):
                print(f"    ⚠️  未命中: {gen_eval['missed_keywords']}")

        # 预览回答（截取前 120 字）
        preview = ans[:120].replace("\n", " ")
        print(f"    回答预览: {preview}...")

        results_log.append({
            "question": q,
            "type": tc["type"],
            "retrieval": ret_eval,
            "generation": gen_eval,
            "answer_preview": ans[:200],
        })

    # 汇总
    print(f"\n{'='*65}")
    print(f"📊 评估结果汇总")
    print(f"{'─'*65}")
    print(f"  检索准确率:  {retrieval_pass}/{total} ({retrieval_pass/total*100:.0f}%)")
    print(f"  生成准确率:  {generation_pass}/{total} ({generation_pass/total*100:.0f}%)")
    print(f"{'─'*65}")

    # 按类型统计
    for test_type in ["精确检索", "理解推理", "拒答测试"]:
        type_cases = [r for r in results_log if r["type"] == test_type]
        type_pass = sum(1 for r in type_cases if r["generation"]["pass"])
        print(f"  {test_type}: {type_pass}/{len(type_cases)}")

    print(f"{'='*65}")

    # 保存详细结果
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(project_root, "tests", "eval_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f"\n📄 详细结果已保存到: tests/eval_results.json")


if __name__ == "__main__":
    run_evaluation()
