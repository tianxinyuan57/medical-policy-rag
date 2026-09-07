"""智能切分器：按法律条款/文章结构切分，而非固定字数

====================================================================
面试重点：为什么不用固定 500 字切分？
====================================================================

法律文档有天然的语义单元——"第X条"。固定字数切分会：
1. 把一个条款切成两半 → 检索到残缺法条
2. 把不相关的条款混在一个 chunk → 向量语义模糊
3. 忽略文档自身的结构信号

我们的 Smart Splitter 策略：
- A 类文档（有"第X条"结构）→ 按条款切分
- B 类文档（有"一、二、三"结构）→ 按项切分
- C 类文档（无明显结构）→ fallback 到段落/固定字数切分

切完后还有两道后处理：
- 太短的条款（< 100 字）→ 和下一条合并，避免语义过弱
- 太长的条款（> 800 字）→ 按句号二次切分，避免超 embedding 窗口

每个 chunk 头部注入元信息，例如：
  [《药品管理法》第二十条]
  开展药物临床试验，应当按照……

这样检索命中后，LLM 能精确引用"根据《药品管理法》第二十条"。

====================================================================
"""

import re
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =====================================================
# 正则模式
# =====================================================

# "第一条" ~ "第九百九十九条"，以及 "第1条" ~ "第999条"
RE_ARTICLE = re.compile(
    r'^(第[一二三四五六七八九十百千零\d]+条)\s*',
    re.MULTILINE,
)

# "一、" "二、" "三、" … 中文数字序号
RE_NUMCN = re.compile(
    r'^([一二三四五六七八九十]+、)\s*',
    re.MULTILINE,
)

# "第一章" "第二章" …
RE_CHAPTER = re.compile(
    r'^(第[一二三四五六七八九十]+章)\s*',
    re.MULTILINE,
)

# 合并阈值：太短的 chunk 和下一个合并
# 设为 150，因为中位条款仅 91 字，合并后 chunk 语义更丰富
MERGE_THRESHOLD = 150

# 二次切分阈值：超过这个长度的 chunk 需要拆开
SPLIT_THRESHOLD = 800

# 二次切分的目标大小
SECONDARY_CHUNK_SIZE = 500
SECONDARY_CHUNK_OVERLAP = 80


# =====================================================
# 核心函数
# =====================================================

def _extract_doc_title(text: str) -> str:
    """从文档全文中提取标题（第一行非空文本）。

    例如：
      "中华人民共和国药品管理法\n（2019年修订）\n第一章…"
      → "药品管理法"
    """
    for line in text.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            # 去掉 "中华人民共和国" 前缀
            title = line.replace('中华人民共和国', '').strip()
            # 去掉括号里的修订年份
            title = re.sub(r'[（(].*?[）)]', '', title).strip()
            return title
    return "未知文档"


def _detect_doc_type(text: str) -> str:
    """检测文档的结构类型。

    Returns:
        'article': 有"第X条"结构（法律法规类）
        'numbered': 有"一、二、三"结构（通知/意见类）
        'plain':    无明显结构
    """
    lines = text.split('\n')
    article_count = sum(1 for l in lines if RE_ARTICLE.match(l.strip()))
    numcn_count = sum(1 for l in lines if RE_NUMCN.match(l.strip()))

    if article_count >= 3:
        return 'article'
    elif numcn_count >= 3:
        return 'numbered'
    else:
        return 'plain'


def _split_by_pattern(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """按正则模式切分文本。

    返回 [(标记, 内容), …]。
    第一个元素的标记是 "" （前言部分，如果有的话）。

    例如按 RE_ARTICLE 切分后：
      [("", "中华人民共和国药品管理法\n第一章 总则\n"),
       ("第一条", "为了加强药品管理……"),
       ("第二条", "在中华人民共和国境内……"),
       ...]
    """
    # 找到所有匹配的位置
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]

    result = []

    # 前言部分（第一个匹配之前的内容）
    preamble = text[:matches[0].start()].strip()
    if preamble:
        result.append(("", preamble))

    # 逐个条款
    for i, m in enumerate(matches):
        label = m.group(1)  # 如 "第一条" 或 "一、"
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        result.append((label, content))

    return result


def _merge_short_chunks(
    chunks: list[tuple[str, str]],
    threshold: int = MERGE_THRESHOLD,
) -> list[tuple[str, str]]:
    """把太短的 chunk 和后一个合并。

    面试点：为什么要合并？
    - "第五条 本法自公布之日起施行。" 只有30字
    - 这种 chunk 的向量几乎没有区分度（信息量太少）
    - 合并后 "第四条+第五条" 一起，语义更丰富
    """
    if not chunks:
        return chunks

    merged = []
    buffer_label = ""
    buffer_text = ""

    for label, text in chunks:
        if buffer_text:
            # 上一个还在 buffer 里（太短），合并
            buffer_text = buffer_text + "\n" + text
            # label 保留第一个的
        else:
            buffer_label = label
            buffer_text = text

        # 如果 buffer 够长了，输出
        # 注意：前言也参与合并——短前言（如法律名称+日期）
        # 会和第一条合并，提供文档上下文
        if len(buffer_text) >= threshold:
            merged.append((buffer_label, buffer_text))
            buffer_label = ""
            buffer_text = ""

    # 最后还有残留
    if buffer_text:
        if merged:
            # 合并到最后一个
            last_label, last_text = merged[-1]
            merged[-1] = (last_label, last_text + "\n" + buffer_text)
        else:
            merged.append((buffer_label, buffer_text))

    return merged


def _secondary_split(text: str) -> list[str]:
    """对超长 chunk 做二次切分（用句号、分号等自然断点）。

    面试点：为什么需要二次切分？
    - 某些条款非常长（附表、细则等），可能上千字
    - bge-small-zh 的 max_seq_length = 512 token ≈ 600-700 汉字
    - 超过这个长度，embedding 会截断，后半部分的语义丢失
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SECONDARY_CHUNK_SIZE,
        chunk_overlap=SECONDARY_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    # split_text 接受纯字符串
    return splitter.split_text(text)


def smart_split_document(doc: Document) -> list[Document]:
    """对单个 LangChain Document 做智能切分。

    这是 Smart Splitter 的入口函数。

    Args:
        doc: LangChain Document 对象（.page_content + .metadata）

    Returns:
        list[Document]: 切分后的 chunk 列表，每个带有增强的 metadata
    """
    text = doc.page_content
    source = doc.metadata.get("source", "")
    source_name = os.path.basename(source).replace('.txt', '').replace('.pdf', '')

    # 提取文档标题（用于 chunk 标注）
    doc_title = _extract_doc_title(text)

    # 检测文档类型
    doc_type = _detect_doc_type(text)

    # ========== 第一步：按结构切分 ==========
    if doc_type == 'article':
        raw_chunks = _split_by_pattern(text, RE_ARTICLE)
    elif doc_type == 'numbered':
        raw_chunks = _split_by_pattern(text, RE_NUMCN)
    else:
        # 无结构，直接用 RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=SECONDARY_CHUNK_SIZE,
            chunk_overlap=SECONDARY_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        fallback_chunks = splitter.split_documents([doc])
        return fallback_chunks

    # ========== 第二步：合并太短的 chunk ==========
    merged_chunks = _merge_short_chunks(raw_chunks)

    # ========== 第三步：二次切分太长的 chunk ==========
    final_docs = []

    for label, content in merged_chunks:
        # 构建头部标注
        if label:
            header = f"[《{doc_title}》{label}]"
        else:
            header = f"[《{doc_title}》前言/总则]"

        if len(content) > SPLIT_THRESHOLD:
            # 太长，需要二次切分
            sub_chunks = _secondary_split(content)
            for j, sub in enumerate(sub_chunks):
                chunk_text = f"{header}\n{sub}"
                final_docs.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "doc_type": doc_type,
                        "article_label": label,
                        "chunk_method": "smart_article_split",
                    },
                ))
        else:
            # 正常长度，直接输出
            chunk_text = f"{header}\n{content}"
            final_docs.append(Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "doc_type": doc_type,
                    "article_label": label,
                    "chunk_method": "smart_article_split",
                },
            ))

    return final_docs


def smart_split_documents(docs: list[Document]) -> list[Document]:
    """批量智能切分，替代原来的 split_documents()。

    用法：
        # 原来的写法
        chunks = split_documents(docs)

        # 现在的写法
        chunks = smart_split_documents(docs)
    """
    all_chunks = []
    stats = {"article": 0, "numbered": 0, "plain": 0}

    for doc in docs:
        doc_type = _detect_doc_type(doc.page_content)
        stats[doc_type] += 1
        chunks = smart_split_document(doc)
        all_chunks.extend(chunks)

    print(f"  📋 文档类型统计: "
          f"按条款切={stats['article']}, "
          f"按序号切={stats['numbered']}, "
          f"段落切={stats['plain']}")
    print(f"  📊 Smart Split 结果: {len(docs)} 个文档 → {len(all_chunks)} 个片段")

    return all_chunks
