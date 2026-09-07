"""多跳推理引擎 —— 解除「输入必须是问题」的限制

====================================================================
问题：真实场景里，人往往不知道自己该问什么
====================================================================

现在的系统是「用户问，我答」。但这假设了用户已经知道要问什么。

真实工作中，人手里拿着的往往不是问题，而是**一份东西**：

    · 一段自己单位写的制度        「这么写合规吗？」
    · 一件打算开展的业务          「要办哪些手续？」
    · 一个刚发生的纠纷            「法律上怎么看？」
    · 一份转发来的通知            「这对我们意味着什么？」

这些输入没有一个是「问题」。硬塞进问答系统，检索会围着字面打转，
答出来的东西看似相关、实则没用。

====================================================================
解法：先看懂输入是什么，再决定怎么处理
====================================================================

                       用户输入
                          │
              ① 输入类型识别（LLM）
                          │
        ┌─────────┬───────┼───────┬─────────┐
     question   review  planning  case   interpret
      提问      核查     筹划     情境     解读
        │         │       │        │         │
        └─────────┴───────┼────────┴─────────┘
                          │
              ② 按类型拆解成多个「检索点」
                 （不同类型拆法不同）
                          │
              ③ 逐点走混合检索 + 图谱扩展
                 （这就是多跳 —— 一次输入触发 N 次检索）
                          │
              ④ 按类型用不同模板综合
                          │
              ⑤ 引用校验核实每条法条依据
                          │
                       结构化结论

关键设计：**输出形态由输入类型决定**，而不是统一成一种格式。
核查题给的是逐条比对结果，筹划题给的是条件清单，
情境题给的是法律关系分析 —— 强行统一成「风险列表」反而不好用。

====================================================================
与单跳问答的区别
====================================================================

    单跳：1 次检索 → 1 次生成
    多跳：1 次拆解 → N 次检索 → 1 次综合

N 次检索是**并列**而非串行的（每个检索点独立），
所以不需要 ReAct 那种「思考-行动」循环，延迟可控。
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import ask_llm, ask_llm_json


# =====================================================
# ① 输入类型识别
# =====================================================

INPUT_TYPES = {
    "question": {
        "cn": "提问",
        "icon": "❓",
        "desc": "想知道某个规定是什么",
    },
    "review": {
        "cn": "合规核查",
        "icon": "🔍",
        "desc": "给出了本单位的制度/做法，想知道合不合规",
    },
    "planning": {
        "cn": "开展筹划",
        "icon": "📋",
        "desc": "打算开展某项业务，想知道要满足什么条件",
    },
    "case": {
        "cn": "情境分析",
        "icon": "⚖️",
        "desc": "描述已发生的事件/纠纷，想知道法律上怎么看",
    },
    "interpret": {
        "cn": "政策解读",
        "icon": "📄",
        "desc": "给出政策原文，想要解读要点",
    },
}

_ROUTER_SYS = """你是医疗政策智能助手的输入分析器。判断用户输入属于哪种类型。

类型定义：
- question：在提问，想知道某个规定是什么
- review：给出了自己单位的制度/规定/做法/流程，想知道合不合规
- planning：打算开展某项业务或办某件事，想知道要满足什么条件、走什么流程
- case：描述一个已发生的事件、纠纷或争议，想知道法律上怎么看、怎么处理
- interpret：给出一段政策原文、通知或文件，想要解读其要点和影响

判断要点：
- 看用户**带来了什么**：只带问题 → question；带了自己的材料 → 其他类型
- "我们医院/我院/我科室"开头描述现状 → review
- "准备/打算/想要"开展某事 → planning
- 描述已经发生的事、有冲突或投诉 → case

输出 JSON：{"type": "...", "confidence": 0.0-1.0, "reason": "简短理由"}"""


def detect_input_type(text: str) -> dict:
    """识别用户输入的类型。

    Returns:
        {"type": ..., "confidence": ..., "reason": ...}
    """
    # 短输入且以问号结尾 —— 几乎必然是提问，省一次 LLM 调用
    t = text.strip()
    if len(t) < 40 and t.endswith(("?", "？")):
        return {"type": "question", "confidence": 1.0,
                "reason": "短问句", "_fast": True}

    r = ask_llm_json(_ROUTER_SYS, f"用户输入：\n{text[:2000]}")
    typ = r.get("type")
    if typ not in INPUT_TYPES:
        typ = "question"          # 兜底
    try:
        conf = round(float(r.get("confidence", 0.5)), 2)
    except (TypeError, ValueError):
        conf = 0.5
    return {"type": typ, "confidence": conf, "reason": r.get("reason", "")}


# =====================================================
# ② 按类型拆解成检索点
# =====================================================

_DECOMPOSE_SYS = {
    "review": """你是医疗合规分析师。用户给出了本单位的制度或做法。

把它拆解成若干个**独立的核查点**，每个核查点是一个需要对照法规验证的具体做法。

要求：
1. 每个核查点提取原文中的**具体做法**，不要笼统概括
2. 为每个核查点生成一个**检索查询**，用于在法规库中找到对照依据
   （检索查询要用法规术语，不要用口语）
3. 拆 3~8 个，覆盖输入中所有实质性内容

输出 JSON：
{"points": [{"claim": "原文中的做法", "query": "检索查询"}, ...]}""",

    "planning": """你是医疗政策顾问。用户打算开展某项业务。

拆解出**要办成这件事需要弄清楚的几个方面**，每个方面生成一个检索查询。

通常应覆盖：
- 准入条件/资质要求
- 审批或备案流程
- 人员要求
- 管理规范/操作要求
- 法律责任

输出 JSON：
{"points": [{"claim": "需要弄清的方面", "query": "检索查询"}, ...]}
拆 3~6 个。""",

    "case": """你是医疗法律分析师。用户描述了一个已发生的事件或纠纷。

拆解出**分析这个事件需要查明的法律要点**，每个要点生成一个检索查询。

通常应覆盖：
- 各方的权利与义务
- 相关行为的法定要求
- 违反的后果/责任
- 争议解决途径

输出 JSON：
{"points": [{"claim": "需查明的法律要点", "query": "检索查询"}, ...]}
拆 3~6 个。""",

    "interpret": """你是医疗政策解读专家。用户给出了一段政策原文或通知。

拆解出**解读这份文件需要关联查证的方面**，每个方面生成一个检索查询。

通常应覆盖：
- 该文件涉及的核心制度是什么
- 它的上位法依据
- 与既有规定的关系
- 对相关主体的具体要求

输出 JSON：
{"points": [{"claim": "需关联查证的方面", "query": "检索查询"}, ...]}
拆 3~5 个。""",
}


def decompose(text: str, input_type: str) -> list[dict]:
    """把输入拆解成若干检索点。

    Returns:
        [{"claim": ..., "query": ...}, ...]
    """
    sys_prompt = _DECOMPOSE_SYS.get(input_type)
    if not sys_prompt:
        # question 类型不需要拆解
        return [{"claim": text, "query": text}]

    r = ask_llm_json(sys_prompt, f"用户输入：\n{text[:4000]}")
    pts = r.get("points") or []

    out = []
    for p in pts:
        claim = (p.get("claim") or "").strip()
        query = (p.get("query") or claim).strip()
        if claim and query:
            out.append({"claim": claim, "query": query})

    # 兜底：拆解失败就整段当一个点
    return out or [{"claim": text[:200], "query": text[:200]}]


# =====================================================
# ③ 逐点检索（多跳的「跳」）
# =====================================================

def retrieve_for_points(points: list[dict], retriever,
                        top_k: int = 4, graph_k: int = 1) -> list[dict]:
    """为每个检索点做一次混合检索 + 图谱扩展。

    这是「多跳」的核心 —— 一次用户输入触发 N 次独立检索。
    每个点的 top_k 比单跳问答小（默认 4），避免总上下文爆炸。
    """
    enriched = []
    for p in points:
        docs, notes = retriever.retrieve_with_graph(
            p["query"], top_k=top_k, graph_k=graph_k)
        enriched.append({
            **p,
            "docs": docs,
            "graph_notes": notes,
            "sources": list(dict.fromkeys(
                os.path.basename(d.metadata.get("source", ""))
                .replace(".txt", "").replace(".pdf", "")
                for d in docs
            )),
        })
    return enriched


# =====================================================
# ④ 按类型综合
# =====================================================

_SYNTH_SYS = {
    "review": """你是资深医疗合规顾问。用户提交了本单位的制度或做法，
你已经为每个核查点检索到了对应的法规原文。

请逐点给出核查结论。

## 核心原则
1. **只依据参考原文**，不得使用外部知识
2. 每条结论必须引用具体法条（如「《处方管理办法》第十一条」）
3. 参考原文不足以判断时，明确说「现有法规原文不足以判断此项」，
   不要猜测

## 输出格式（Markdown）

开头用一句话给出总体结论。然后逐点：

### {风险标记} 核查点N：{简述}
- **现行做法**：（用户原文中的做法）
- **法规要求**：根据《XX》第X条，……
- **结论**：符合 / 不符合 / 需补充 / 无法判断
- **建议**：（不符合时给出具体修改建议）

风险标记用：🔴 高风险（明确违规）/ 🟡 待完善（缺失或不明确）/
🟢 符合 / ⚪ 无法判断

最后加一行免责说明：
> ⚠️ 本结论基于知识库中的公开法规自动生成，仅供初步自查参考，不替代专业法务审查。""",

    "planning": """你是资深医疗政策顾问。用户打算开展某项业务，
你已经为每个方面检索到了相关法规原文。

请给出一份「要办成这件事需要满足什么」的清单。

## 核心原则
1. **只依据参考原文**，不得使用外部知识
2. 每条要求必须引用具体法条
3. 原文没覆盖到的方面，明确说明「现有法规原文未涉及此项」

## 输出格式（Markdown）

开头一句话说明这件事的法规定位。然后：

### 一、准入条件
- 具体要求（依据《XX》第X条）

### 二、办理流程
1. 步骤（依据……）

### 三、持续管理要求
- ……

### ⚠️ 特别提醒
- 容易踩坑的地方、时限要求等

最后加免责说明。""",

    "case": """你是资深医疗法律顾问。用户描述了一个已发生的事件，
你已经为每个法律要点检索到了相关法规原文。

请给出法律分析。

## 核心原则
1. **只依据参考原文**，不得使用外部知识
2. 每条分析必须引用具体法条
3. 不要下「一定胜诉/一定败诉」这类绝对判断
4. 原文不足以支撑的判断，明确说明

## 输出格式（Markdown）

### 一、事件涉及的法律关系
（简述涉及哪些主体、什么性质的关系）

### 二、各方权利义务
- **{主体}**：根据《XX》第X条，……

### 三、关键判断点
（本案的争议焦点在哪，法规怎么规定）

### 四、建议的处理路径
1. ……

最后加免责说明：
> ⚠️ 本分析基于知识库中的公开法规自动生成，仅供参考，
> 不构成法律意见，具体处理请咨询专业法律人士。""",

    "interpret": """你是资深医疗政策解读专家。用户给出了一段政策文件，
你已经检索到了相关的关联法规。

请解读这份文件。

## 核心原则
1. **只依据参考原文**（包括用户给的文件和检索到的法规）
2. 关联到其他法规时必须引用具体法条
3. 不要编造文件中没有的内容

## 输出格式（Markdown）

### 一、这份文件是什么
（性质、层级、适用范围）

### 二、核心要点
1. ……

### 三、与既有规定的关系
（上位法依据、与哪些现行规定衔接）

### 四、对相关主体的具体影响
- **医疗机构**：……
- **医务人员**：……

最后加免责说明。""",
}


def build_synth_prompt(text: str, input_type: str,
                       enriched_points: list[dict]) -> tuple[str, str]:
    """构造综合阶段的 (system_prompt, user_prompt)。

    单独暴露出来，是为了让前端能自己做**流式输出** ——
    多跳要 10 秒以上，一次性吐出结果的体验很差。
    """
    sys_prompt = _SYNTH_SYS.get(input_type, "")
    if not sys_prompt:
        return "", ""

    # 拼装：每个检索点 + 它检索到的原文
    blocks = []
    for i, p in enumerate(enriched_points, 1):
        seg = [f"### 核查点 {i}：{p['claim']}"]
        for j, d in enumerate(p["docs"], 1):
            src = os.path.basename(d.metadata.get("source", "")) \
                    .replace(".txt", "").replace(".pdf", "")
            tag = "｜关联法规" if d.metadata.get("graph_expanded") else ""
            seg.append(f"[片段{i}.{j}｜来源：《{src}》{tag}]\n{d.page_content}")
        blocks.append("\n\n".join(seg))

    user_prompt = (
        f"【用户提交的内容】\n{text[:4000]}\n\n"
        f"【各核查点的参考原文】\n\n" + "\n\n---\n\n".join(blocks)
    )
    return sys_prompt, user_prompt


def synthesize(text: str, input_type: str,
               enriched_points: list[dict]) -> str:
    """把多路检索结果综合成最终结论（非流式）。"""
    sys_prompt, user_prompt = build_synth_prompt(text, input_type,
                                                 enriched_points)
    if not sys_prompt:
        return ""
    return ask_llm(sys_prompt, user_prompt)


# =====================================================
# 主入口
# =====================================================

@dataclass
class MultiHopResult:
    input_text: str
    input_type: str
    type_confidence: float
    type_reason: str = ""
    points: list = field(default_factory=list)     # 检索点（含 docs）
    answer: str = ""
    all_docs: list = field(default_factory=list)   # 去重后的全部片段
    elapsed: float = 0.0
    hop_count: int = 0

    @property
    def type_cn(self) -> str:
        return INPUT_TYPES.get(self.input_type, {}).get("cn", self.input_type)

    @property
    def type_icon(self) -> str:
        return INPUT_TYPES.get(self.input_type, {}).get("icon", "•")


def prepare(text: str, retriever, top_k: int = 4, graph_k: int = 1,
            force_type: str = None, progress=None) -> MultiHopResult:
    """只跑「识别 → 拆解 → 检索」，不生成结论。

    给前端用：拿到 result 后自己调 build_synth_prompt() 做流式输出。

    Returns:
        MultiHopResult，其中 answer 为空。
        若 input_type == "question"，points 为空（应转走单跳流程）。
    """
    t0 = time.time()

    def _p(stage, detail=""):
        if progress:
            progress(stage, detail)

    # ① 识别输入类型
    _p("detect", "正在识别输入类型")
    if force_type and force_type in INPUT_TYPES:
        info = {"type": force_type, "confidence": 1.0, "reason": "用户指定"}
    else:
        info = detect_input_type(text)
    itype = info["type"]

    base = MultiHopResult(
        input_text=text, input_type=itype,
        type_confidence=info["confidence"],
        type_reason=info.get("reason", ""),
    )

    # question 不进多跳，交回给单跳流程
    if itype == "question":
        base.elapsed = time.time() - t0
        return base

    # ② 拆解
    _p("decompose", f"识别为「{INPUT_TYPES[itype]['cn']}」，正在拆解要点")
    points = decompose(text, itype)

    # ③ 逐点检索（多跳）
    _p("retrieve", f"拆出 {len(points)} 个要点，正在逐点检索")
    enriched = retrieve_for_points(points, retriever,
                                   top_k=top_k, graph_k=graph_k)

    # 汇总去重的片段（供引用校验用）
    seen, all_docs = set(), []
    for p in enriched:
        for d in p["docs"]:
            key = d.page_content[:200]
            if key not in seen:
                seen.add(key)
                all_docs.append(d)

    base.points = enriched
    base.all_docs = all_docs
    base.hop_count = len(enriched)
    base.elapsed = time.time() - t0
    return base


def analyze(text: str, retriever, top_k: int = 4, graph_k: int = 1,
            force_type: str = None, progress=None) -> MultiHopResult:
    """多跳分析主入口。

    Args:
        text:       用户输入（可以是问题，也可以是制度/场景/案例/文件）
        retriever:  HybridRetriever 实例
        top_k:      每个检索点召回的片段数
        graph_k:    每个检索点的图扩展数
        force_type: 强制指定输入类型（跳过自动识别）
        progress:   进度回调 fn(stage: str, detail: str)

    Returns:
        MultiHopResult
    """
    t0 = time.time()

    def _p(stage, detail=""):
        if progress:
            progress(stage, detail)

    # ① 识别输入类型
    _p("detect", "正在识别输入类型")
    if force_type and force_type in INPUT_TYPES:
        info = {"type": force_type, "confidence": 1.0, "reason": "用户指定"}
    else:
        info = detect_input_type(text)
    itype = info["type"]

    # question 走原有单跳流程，不进多跳
    if itype == "question":
        return MultiHopResult(
            input_text=text, input_type="question",
            type_confidence=info["confidence"],
            type_reason=info.get("reason", ""),
            elapsed=time.time() - t0, hop_count=0,
        )

    # ② 拆解
    _p("decompose", f"识别为「{INPUT_TYPES[itype]['cn']}」，正在拆解要点")
    points = decompose(text, itype)

    # ③ 逐点检索（多跳）
    _p("retrieve", f"拆出 {len(points)} 个要点，正在逐点检索")
    enriched = retrieve_for_points(points, retriever,
                                   top_k=top_k, graph_k=graph_k)

    # 汇总去重的片段（供引用校验用）
    seen, all_docs = set(), []
    for p in enriched:
        for d in p["docs"]:
            key = d.page_content[:200]
            if key not in seen:
                seen.add(key)
                all_docs.append(d)

    # ④ 综合
    _p("synthesize", f"已检索 {len(all_docs)} 个法规片段，正在生成结论")
    answer = synthesize(text, itype, enriched)

    return MultiHopResult(
        input_text=text, input_type=itype,
        type_confidence=info["confidence"],
        type_reason=info.get("reason", ""),
        points=enriched, answer=answer, all_docs=all_docs,
        elapsed=time.time() - t0, hop_count=len(enriched),
    )


# =====================================================
# 命令行自测
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多跳推理测试")
    parser.add_argument("text", nargs="?", help="用户输入")
    parser.add_argument("--type", type=str, default=None,
                        choices=list(INPUT_TYPES), help="强制指定类型")
    parser.add_argument("--detect-only", action="store_true",
                        help="只做类型识别")
    args = parser.parse_args()

    SAMPLES = {
        "review": "我院麻醉药品管理规定：\n"
                  "第五条 麻醉药品处方由科室主任统一开具，医师需要时向科室主任申请。\n"
                  "第六条 麻醉药品处方保存一年后销毁。\n"
                  "第七条 麻醉药品由病区护士长负责保管，使用后在登记本上记录。",
        "planning": "我们医院准备开展互联网诊疗服务，需要满足什么条件？",
        "case": "患者要求复印全部病历，我们只给了住院志，他现在要投诉我们。",
    }

    text = args.text or SAMPLES["review"]

    print("=" * 78)
    print("🔗 多跳推理测试")
    print("=" * 78)
    print(f"输入：\n{text[:300]}\n")

    if args.detect_only:
        info = detect_input_type(text)
        t = info["type"]
        print(f"类型：{INPUT_TYPES[t]['icon']} {INPUT_TYPES[t]['cn']}"
              f"（{t}）  置信度 {info['confidence']}")
        print(f"理由：{info.get('reason','')}")
        sys.exit(0)

    from hybrid_retriever import get_retriever

    def show(stage, detail):
        print(f"  ⏳ [{stage}] {detail}")

    r = analyze(text, get_retriever(), force_type=args.type, progress=show)

    print(f"\n{'─'*78}")
    print(f"类型：{r.type_icon} {r.type_cn}（置信度 {r.type_confidence}）")
    print(f"跳数：{r.hop_count} 次检索 | 共 {len(r.all_docs)} 个片段 "
          f"| 耗时 {r.elapsed:.1f}s")

    if r.points:
        print(f"\n{'─'*78}\n拆解出的检索点：")
        for i, p in enumerate(r.points, 1):
            print(f"  {i}. {p['claim'][:56]}")
            print(f"     检索: {p['query'][:56]}")
            print(f"     命中: {', '.join(p['sources'][:3])}")

    if r.answer:
        print(f"\n{'─'*78}\n结论：\n")
        print(r.answer)
    print()
