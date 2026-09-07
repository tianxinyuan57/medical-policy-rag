"""引用校验（Citation Verification）—— 把「防幻觉」从口号变成可量化指标

====================================================================
为什么需要引用校验？
====================================================================

RAG 的标准防幻觉手段是在 Prompt 里写「只依据参考原文」。
但这只是**请求**模型别编，没有**验证**它有没有编。

模型仍然可能：
  · 张冠李戴：内容来自《护士条例》，却说成「根据《医师法》第 27 条」
  · 编造条号：参考原文里是第 67 条，回答写成第 76 条
  · 凭记忆作答：说出一条参考原文里根本没有的法条

本模块做的是**事后核验**：把回答里每一处「《X法》第Y条」抽出来，
回到检索片段里查证 —— 这条到底在不在？

这带来两个价值：
  1. 用户能看到哪些引用被核实过（可信度可视化）
  2. 系统能输出「引用准确率」这个硬指标

====================================================================
两个必须处理的陷阱（基于本项目 chunk 结构）
====================================================================

陷阱 1：metadata 的 article_label 只记录 chunk 的**起始条款**
    chunk 标签是「第六十一条」，正文实际含第 61~75 条。
    → 只比对 metadata 会漏判大量正确引用，必须扫描正文。

陷阱 2：片段正文里的条款号，可能是「引用别的法规」
    《药品管理法实施条例》第 61 条正文写着
    「……依照《药品管理法》第八十条的规定处罚」。
    这里的「第八十条」属于《药品管理法》，不属于本条例。
    → 必须区分「本文档的条款」和「转引其他文档的条款」。

判据：条款号前面紧跟「》」的，是转引；否则是本文档条款。
"""

import os
import re
from dataclasses import dataclass, field


# =====================================================
# 正则
# =====================================================

# 回答中的引用：《XX法》第X条 / 《XX法》的第X条 / 《XX法》第X条第Y款
RE_CITATION = re.compile(
    r'《([^》]{2,40})》\s*(?:的)?\s*第([一二三四五六七八九十百千零\d]+)条'
)

# 承接式引用：「《X法》第六十七条、第六十八条」中的后续条款
RE_FOLLOW_ARTICLE = re.compile(
    r'[、和及]\s*第([一二三四五六七八九十百千零\d]+)条'
)

# 只提法规不提条款：《XX法》（用于统计「模糊引用」）
RE_LAW_ONLY = re.compile(r'《([^》]{2,40})》')


def _normalize_law(name: str) -> str:
    """法规名归一化，与 citation_graph 保持一致。"""
    n = re.sub(r'\s+', '', name.strip())
    n = n.replace('中华人民共和国', '')
    n = re.sub(r'[（(][^）)]*[）)]', '', n)
    return n.strip()


# =====================================================
# 数据结构
# =====================================================

# 泛称：模型常用的简称，匹配时容易误判（任何"XX条例"都包含"条例"）
GENERIC_LAW_NAMES = {
    "条例", "规定", "办法", "细则", "实施细则", "规范", "通知", "意见",
    "本条例", "本规定", "本办法", "管理规定", "管理办法", "暂行办法",
    "暂行规定", "实施条例", "技术规范", "指导意见", "经办规程", "规程",
}


@dataclass
class Citation:
    """回答中的一处法条引用。"""
    law: str                    # 法规名（原文写法）
    article: str                # 条款号，如「第六十七条」
    verified: bool = False      # 是否在检索片段中核实到
    matched_source: str = ""    # 核实到的片段来源
    matched_text: str = ""      # 该条款的原文（供前端展开查看）
    note: str = ""              # 未核实时的说明
    is_abbrev: bool = False     # 是否用简称引用（如《条例》），置信度较低

    @property
    def confidence(self) -> str:
        """核实置信度：exact（精确匹配法规名）/ abbrev（简称匹配）/ none。"""
        if not self.verified:
            return "none"
        return "abbrev" if self.is_abbrev else "exact"


@dataclass
class VerifyResult:
    """一次回答的引用校验结果。"""
    citations: list[Citation] = field(default_factory=list)
    law_only_refs: list[str] = field(default_factory=list)  # 只提法规没提条款

    @property
    def total(self) -> int:
        return len(self.citations)

    @property
    def verified_count(self) -> int:
        return sum(1 for c in self.citations if c.verified)

    @property
    def accuracy(self) -> float:
        """引用准确率。无引用时返回 1.0（不算错）。"""
        return self.verified_count / self.total if self.total else 1.0

    @property
    def unverified(self) -> list[Citation]:
        return [c for c in self.citations if not c.verified]

    @property
    def abbrev_count(self) -> int:
        """通过简称（《条例》等泛称）匹配的数量，置信度较低。"""
        return sum(1 for c in self.citations if c.verified and c.is_abbrev)

    def summary(self) -> str:
        if self.total == 0:
            return "回答中未包含具体法条引用"
        if self.verified_count == self.total:
            s = f"✅ {self.total} 处法条引用全部核实无误"
            if self.abbrev_count:
                s += f"（其中 {self.abbrev_count} 处为简称引用）"
            return s
        return (f"⚠️ {self.verified_count}/{self.total} 处引用已核实，"
                f"{len(self.unverified)} 处未在参考原文中找到")


# =====================================================
# 核心：从片段中判断某条款是否存在
# =====================================================

def _article_in_chunk(article: str, chunk_text: str) -> bool:
    """判断「第X条」是否作为**本文档条款**出现在片段正文中。

    排除转引情况：「依照《药品管理法》第八十条」里的第八十条
    属于《药品管理法》，不是本片段所属法规的条款。

    判据：条款号前若紧跟「》」（允许中间有「的」「之」等虚词），
    视为转引，不计入。
    """
    for m in re.finditer(re.escape(article), chunk_text):
        # 往前看最多 4 个字符
        before = chunk_text[max(0, m.start() - 4):m.start()]
        # 去掉虚词后，若以「》」结尾 → 转引，跳过
        stripped = re.sub(r'[的之\s]', '', before)
        if stripped.endswith('》'):
            continue
        return True   # 找到一处非转引的出现
    return False


def _extract_article_text(article: str, chunk_text: str,
                          max_len: int = 700) -> str:
    """从片段中切出指定条款的原文。

    片段可能含多个条款（短条款被合并过），要精确定位到目标条款，
    截止到下一个条款开始处。

    Args:
        article: 目标条款，如「第六十七条」
        chunk_text: 片段全文
        max_len: 截断长度，避免超长条款撑爆界面
    """
    # 定位目标条款（跳过转引）
    start = -1
    for m in re.finditer(re.escape(article), chunk_text):
        before = chunk_text[max(0, m.start() - 4):m.start()]
        if re.sub(r'[的之\s]', '', before).endswith('》'):
            continue
        start = m.start()
        break

    if start < 0:
        return ""

    # 找下一个条款的起始位置作为结束点
    rest = chunk_text[start + len(article):]
    nxt = re.search(r'第[一二三四五六七八九十百千零\d]+条', rest)
    if nxt:
        # 同样要跳过转引的条款号
        offset = 0
        while nxt:
            abs_pos = start + len(article) + offset + nxt.start()
            before = chunk_text[max(0, abs_pos - 4):abs_pos]
            if not re.sub(r'[的之\s]', '', before).endswith('》'):
                end = abs_pos
                break
            offset += nxt.end()
            nxt = re.search(r'第[一二三四五六七八九十百千零\d]+条', rest[offset:])
        else:
            end = len(chunk_text)
        if nxt is None:
            end = len(chunk_text)
    else:
        end = len(chunk_text)

    text = chunk_text[start:end].strip()
    # 去掉 chunk 头部的 [《法名》第X条] 标注残留
    text = re.sub(r'^\[《[^》]+》[^\]]*\]\s*', '', text)

    if len(text) > max_len:
        text = text[:max_len].rstrip() + "……"
    return text


def _chunk_belongs_to_law(chunk_meta: dict, chunk_text: str,
                          law_norm: str) -> bool:
    """判断片段是否属于指定法规。

    优先看 metadata 的 source，兜底看 chunk 头部的 [《法名》…] 标注。
    """
    src = os.path.basename(chunk_meta.get("source", ""))
    src = src.replace(".txt", "").replace(".pdf", "")
    src_norm = _normalize_law(src)

    if not src_norm or not law_norm:
        return False

    # 双向包含（处理《医师法》vs《中华人民共和国医师法》等）
    if law_norm in src_norm or src_norm in law_norm:
        return True

    # 兜底：chunk 头部标注
    head = chunk_text[:60]
    m = re.match(r'\[《([^》]+)》', head)
    if m and _normalize_law(m.group(1)) == law_norm:
        return True

    return False


# =====================================================
# 主入口
# =====================================================

def extract_citations(answer: str) -> tuple[list[tuple[str, str]], list[str]]:
    """从回答中抽取所有法条引用。

    Returns:
        ([(法规名, 条款号), …], [只提到法规名的引用, …])
    """
    pairs: list[tuple[str, str]] = []
    seen = set()

    for m in RE_CITATION.finditer(answer):
        law = m.group(1)
        art = f"第{m.group(2)}条"
        key = (_normalize_law(law), art)
        if key not in seen:
            seen.add(key)
            pairs.append((law, art))

        # 处理承接式：「第六十七条、第六十八条」
        tail = answer[m.end():m.end() + 40]
        for fm in RE_FOLLOW_ARTICLE.finditer(tail):
            art2 = f"第{fm.group(1)}条"
            key2 = (_normalize_law(law), art2)
            if key2 not in seen:
                seen.add(key2)
                pairs.append((law, art2))

    # 只提法规、没提条款的引用
    cited_laws = {_normalize_law(l) for l, _ in pairs}
    law_only = []
    for m in RE_LAW_ONLY.finditer(answer):
        ln = _normalize_law(m.group(1))
        if ln and ln not in cited_laws and ln not in law_only and len(ln) > 3:
            law_only.append(ln)

    return pairs, law_only


def verify(answer: str, retrieved_docs: list) -> VerifyResult:
    """核验回答中的法条引用是否都能在检索片段中找到。

    Args:
        answer: 模型生成的回答
        retrieved_docs: 本次检索到的 LangChain Document 列表

    Returns:
        VerifyResult
    """
    pairs, law_only = extract_citations(answer)
    result = VerifyResult(law_only_refs=law_only)

    for law, article in pairs:
        law_norm = _normalize_law(law)
        cit = Citation(law=law, article=article,
                       is_abbrev=law_norm in GENERIC_LAW_NAMES)

        # 在所有检索片段里找：来源匹配 且 条款存在于正文
        for doc in retrieved_docs:
            meta = doc.metadata or {}
            text = doc.page_content

            if not _chunk_belongs_to_law(meta, text, law_norm):
                continue
            if _article_in_chunk(article, text):
                cit.verified = True
                src = os.path.basename(meta.get("source", ""))
                cit.matched_source = src.replace(".txt", "").replace(".pdf", "")
                cit.matched_text = _extract_article_text(article, text)
                break

        if not cit.verified:
            # 区分两种失败：法规没检索到 vs 法规检索到了但没这一条
            law_present = any(
                _chunk_belongs_to_law(d.metadata or {}, d.page_content, law_norm)
                for d in retrieved_docs
            )
            cit.note = ("该法规在参考原文中，但未找到此条款"
                        if law_present else "该法规不在本次参考原文中")

        result.citations.append(cit)

    return result


# =====================================================
# 命令行自测
# =====================================================

if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from hybrid_retriever import get_retriever
    from llm import ask_llm
    from rag import RAG_SYSTEM_PROMPT
    from config import TOP_K, GRAPH_K

    questions = [
        "什么是假药？列举假药的情形。",
        "护士执业注册需要具备哪些条件？",
        "医疗机构使用麻醉药品有什么特殊要求？",
        "无偿献血的年龄范围是多少？",
    ]

    print("=" * 74)
    print("🔍 引用校验测试")
    print("=" * 74)

    retriever = get_retriever()
    total_c = 0
    total_v = 0

    for q in questions:
        docs, _ = retriever.retrieve_with_graph(q, top_k=TOP_K, graph_k=GRAPH_K)

        blocks = []
        for i, d in enumerate(docs):
            src = os.path.basename(d.metadata.get("source", "")).replace(".txt", "")
            blocks.append(f"[片段{i+1}｜来源：《{src}》]\n{d.page_content}")
        prompt = f"【参考原文】\n{chr(10).join(blocks)}\n\n【问题】\n{q}"

        answer = ask_llm(RAG_SYSTEM_PROMPT, prompt)
        res = verify(answer, docs)

        total_c += res.total
        total_v += res.verified_count

        print(f"\n{'─'*74}")
        print(f"❓ {q}")
        print(f"   {res.summary()}")
        for c in res.citations:
            mark = "✅" if c.verified else "❌"
            extra = f" ← {c.matched_source}" if c.verified else f" ({c.note})"
            print(f"   {mark} 《{c.law}》{c.article}{extra}")
        if res.law_only_refs:
            print(f"   ℹ️  只提法规未提条款: {res.law_only_refs[:4]}")

    print(f"\n{'='*74}")
    rate = total_v / total_c if total_c else 1.0
    print(f"📊 总体引用准确率: {total_v}/{total_c} = {rate:.1%}")
    print("=" * 74)
