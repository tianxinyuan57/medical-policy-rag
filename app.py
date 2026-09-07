"""医疗政策 RAG 问答系统 —— Streamlit Web 界面"""

import sys
import os
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

import streamlit as st

# ── 页面配置 ──────────────────────────────────────────

st.set_page_config(
    page_title="医疗政策 RAG 问答",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义样式 ────────────────────────────────────────

st.markdown("""
<style>
/* ── 全局 ── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

[data-testid="stAppViewContainer"] {
    font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── 侧边栏 ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px;
    border-radius: 8px;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.08);
}

/* ── 主内容区顶部 hero ── */
.hero-section {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
}
.hero-section h1 {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-section p {
    color: #64748b;
    font-size: 0.95rem;
}

/* ── 示例问题卡片 ── */
.example-btn button {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    font-size: 0.85rem !important;
    text-align: left !important;
    transition: all 0.2s !important;
    background: white !important;
    color: #334155 !important;
}
.example-btn button:hover {
    border-color: #3b82f6 !important;
    background: #eff6ff !important;
    color: #1d4ed8 !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(59,130,246,0.15) !important;
}
/* 暗色模式适配 */
@media (prefers-color-scheme: dark) {
    .example-btn button {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
    }
    .example-btn button:hover {
        background: #1e3a5f !important;
        border-color: #3b82f6 !important;
        color: #93c5fd !important;
    }
    .file-card {
        background: #1e293b;
        border-color: #334155;
    }
    .file-card:hover {
        background: #1e3a5f;
        border-color: #3b82f6;
    }
    .file-card .name { color: #e2e8f0; }
    .file-card .meta { color: #64748b; }
    .retrieval-card {
        background: #1e293b;
    }
    .retrieval-card .source-tag {
        background: #1e3a5f;
        color: #93c5fd;
    }
    [data-testid="stMetric"] {
        background: #1e293b;
        border-color: #334155;
    }
}
/* Streamlit 自带 dark 主题也适配 */
[data-theme="dark"] .example-btn button {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border-color: #334155 !important;
}
[data-theme="dark"] .example-btn button:hover {
    background: #1e3a5f !important;
    border-color: #3b82f6 !important;
    color: #93c5fd !important;
}

/* ── 聊天气泡 ── */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    margin-bottom: 1rem;
    padding: 1rem 1.25rem;
}

/* ── 指标卡片 ── */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
}

/* ── 知识库文件网格 ── */
.file-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    transition: all 0.15s;
}
.file-card:hover {
    border-color: #93c5fd;
    background: #eff6ff;
}
.file-card .name {
    font-weight: 500;
    color: #1e293b;
    font-size: 0.9rem;
}
.file-card .meta {
    color: #94a3b8;
    font-size: 0.75rem;
    margin-top: 2px;
}

/* ── 检索片段卡片 ── */
.retrieval-card {
    background: #f1f5f9;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.85rem;
}
.retrieval-card .source-tag {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
}

/* ── Expander 样式 ── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
}

/* ── 隐藏默认页脚 ── */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── 缓存加载 ──────────────────────────────────────────

@st.cache_resource(show_spinner="正在加载向量库和 Embedding 模型...")
def load_rag_components():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from config import CHROMA_DIR, EMBED_MODEL
    chroma_path = os.path.join(PROJECT_ROOT, CHROMA_DIR)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vs = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    return vs, embeddings


@st.cache_resource(show_spinner="正在构建 BM25 索引...")
def load_retriever():
    """加载混合检索器（BM25 + 向量 + RRF 融合）。"""
    from hybrid_retriever import HybridRetriever
    vs, _ = load_rag_components()
    return HybridRetriever(vs)


@st.cache_resource(show_spinner="正在构建引用图谱...")
def load_citation_graph():
    """加载法条引用图谱。"""
    from citation_graph import CitationGraph
    return CitationGraph()


@st.cache_resource
def load_llm_client():
    from openai import OpenAI
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def ask_llm_stream(system_prompt: str, user_prompt: str):
    """调用 LLM，返回流式迭代器。"""
    from config import CHAT_MODEL
    client = load_llm_client()
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        stream=True,
    )


# ── RAG 核心 ──────────────────────────────────────────

RAG_SYSTEM_PROMPT = """你是一位资深医疗政策顾问，既熟悉法律条文，又善于向非专业人士解释政策要点。

## 核心原则
1. **只依据参考原文**：所有结论必须基于下方【参考原文】，不得使用外部知识。
2. **拒答兜底**：参考原文中找不到相关信息时，明确回答"根据现有政策文件，未能找到相关规定"，不要猜测或编造。

## 回答风格
- **先给结论**：开头用一两句话直接回答问题核心，让读者一眼抓住要点。
- **再展开解读**：分条列出关键细节，引用具体法条（如"根据《XX法》第X条"），适当做归纳整理而非逐字搬运原文。
- **加实用提醒**（可选）：如果原文中有相关的注意事项、例外条款或时间节点，在末尾补充提醒，用"⚠️"标注。
- **语气专业亲和**：像给同事做政策解读一样说人话，避免干巴巴地堆砌条文。可以用加粗、列表等格式让回答更易读。

## 格式要求
- 使用 Markdown 格式（加粗、列表、分段）提高可读性。
- 回答末尾用"📎 引用来源："标注参考了哪些文件片段。
"""


def rag_retrieve(question: str, top_k: int = 5, graph_k: int = 2):
    """混合检索 + 图谱扩展，返回 (context_str, retrieval_info)。"""
    retriever = load_retriever()
    results, graph_notes = retriever.retrieve_with_graph(
        question, top_k=top_k, graph_k=graph_k
    )

    context_blocks = []
    retrieval_info = []
    for i, doc in enumerate(results):
        src = os.path.basename(doc.metadata.get("source", "未知"))
        src_name = src.replace(".txt", "").replace(".pdf", "")
        is_graph = bool(doc.metadata.get("graph_expanded"))

        tag = (f"[片段{i+1}｜来源：《{src_name}》｜关联法规]" if is_graph
               else f"[片段{i+1}｜来源：《{src_name}》]")
        context_blocks.append(f"{tag}\n{doc.page_content}")

        retrieval_info.append({
            "index": i + 1,
            "source": src_name,
            "content": doc.page_content,
            "label": doc.metadata.get("article_label", ""),
            "graph_expanded": is_graph,
            "graph_reason": doc.metadata.get("graph_reason", ""),
        })

    context = "\n\n".join(context_blocks)
    return context, retrieval_info


def render_retrieval_cards(items: list[dict]):
    """渲染检索片段卡片，图扩展的片段有独立配色和来源说明。"""
    for item in items:
        is_g = item.get("graph_expanded")
        border = "#a78bfa" if is_g else "#3b82f6"
        tag_bg = "#3730a3" if is_g else "#dbeafe"
        tag_fg = "#c7d2fe" if is_g else "#1d4ed8"

        label_html = ""
        if item.get("label"):
            label_html = (f' · <span class="source-tag" '
                          f'style="background:{tag_bg};color:{tag_fg};">'
                          f'{item["label"]}</span>')

        badge = ""
        if is_g:
            badge = ('<span class="source-tag" style="background:#4c1d95;'
                     'color:#ddd6fe;">🕸️ 图谱关联</span> ')

        reason_html = ""
        if is_g and item.get("graph_reason"):
            reason_html = (f'<div style="margin-top:6px;font-size:0.72rem;'
                           f'color:#a5b4fc;opacity:.85;">'
                           f'↳ {item["graph_reason"]}</div>')

        content = item["content"]
        preview = content[:300] + ("..." if len(content) > 300 else "")

        st.markdown(f"""<div class="retrieval-card"
            style="border-left-color:{border};">
            {badge}<span class="source-tag"
                style="background:{tag_bg};color:{tag_fg};">
                📄 {item['source']}</span>{label_html}
            <div style="margin-top:6px;color:#475569;line-height:1.6;">
                {preview}</div>
            {reason_html}
        </div>""", unsafe_allow_html=True)


# ── 侧边栏 ───────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏥 医疗政策问答")
    st.caption("基于 94 部公开法规 · RAG 检索增强生成")

    st.divider()

    page = st.radio(
        "导航",
        ["💬 对话", "🕸️ 引用图谱", "📊 评估", "📚 知识库", "⚙️ 系统"],
        label_visibility="collapsed",
    )

    st.divider()

    # 系统状态
    try:
        vs, _ = load_rag_components()
        chunk_count = vs._collection.count()
        st.markdown(f"**状态** &nbsp; 🟢 就绪")
        st.caption(f"{chunk_count} 个片段已索引")
    except Exception as e:
        st.markdown(f"**状态** &nbsp; 🔴 异常")
        st.caption(str(e)[:60])
        chunk_count = 0

    # 参数调节
    st.markdown("**检索参数**")
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=5,
                       help="检索返回的片段数", label_visibility="collapsed")
    st.caption(f"Top-K = {top_k}")

    use_graph = st.toggle("🕸️ 引用图谱扩展", value=True,
                          help="沿法条引用关系补充召回关联法规")
    graph_k = st.slider("扩展法规数", 1, 4, 2,
                        label_visibility="collapsed") if use_graph else 0
    if use_graph:
        st.caption(f"额外召回 {graph_k} 部关联法规")

    st.divider()
    st.caption("⚠️ 仅使用公开法规文本，不构成法律建议")


# ── 页面：对话 ────────────────────────────────────────

if page == "💬 对话":

    # 初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 空状态：显示 hero + 示例
    if not st.session_state.messages:
        st.markdown("""
        <div class="hero-section">
            <h1>🏥 医疗政策智能问答</h1>
            <p>基于 94 部公开法规及政策文件 · 条款级检索 · 引用可溯源</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 💡 试试这些问题")

        examples = [
            ("🩸", "无偿献血的年龄范围是多少？"),
            ("💊", "什么是假药？列举假药的情形。"),
            ("🩺", "医师和护士在紧急情况下的责任有什么不同？"),
            ("🌿", "以师承方式学中医可以行医吗？需要什么条件？"),
            ("📋", "发生医疗纠纷后，患者有权查阅哪些病历资料？"),
            ("🏥", "甲类传染病包括哪些？"),
        ]

        cols = st.columns(2)
        for i, (icon, q) in enumerate(examples):
            with cols[i % 2]:
                st.markdown('<div class="example-btn">', unsafe_allow_html=True)
                if st.button(f"{icon}  {q}", key=f"ex_{i}", use_container_width=True):
                    st.session_state["pending_q"] = q
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        # 渲染历史对话
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

                # 助手消息的元信息
                if msg["role"] == "assistant" and "elapsed" in msg:
                    st.caption(f"⏱️ {msg['elapsed']:.1f}s · Top-K={msg.get('top_k', 5)} · {len(msg.get('retrieval', []))} 个片段")

            # 检索详情
            if msg["role"] == "assistant" and msg.get("retrieval"):
                n_graph = sum(1 for it in msg["retrieval"]
                              if it.get("graph_expanded"))
                title = f"🔍 查看检索片段（{len(msg['retrieval'])} 个"
                title += f"，含 {n_graph} 个图谱关联）" if n_graph else "）"
                with st.expander(title, expanded=False):
                    render_retrieval_cards(msg["retrieval"])

    # 输入框
    question = st.chat_input("输入医疗政策相关问题...")
    if question is None and "pending_q" in st.session_state:
        question = st.session_state.pop("pending_q")

    if question:
        # 用户消息
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        # 助手回答（流式）
        with st.chat_message("assistant", avatar="🤖"):
            t0 = time.time()

            # 检索
            context, retrieval_info = rag_retrieve(
                question, top_k=top_k, graph_k=graph_k)
            user_prompt = f"【参考原文】\n{context}\n\n【问题】\n{question}"

            # 流式生成
            stream = ask_llm_stream(RAG_SYSTEM_PROMPT, user_prompt)
            full_answer = st.write_stream(
                (chunk.choices[0].delta.content or ""
                 for chunk in stream
                 if chunk.choices[0].delta.content is not None)
            )

            elapsed = time.time() - t0
            st.caption(f"⏱️ {elapsed:.1f}s · Top-K={top_k} · {len(retrieval_info)} 个片段")

        # 保存历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "retrieval": retrieval_info,
            "elapsed": elapsed,
            "top_k": top_k,
        })

        # 检索详情
        n_graph = sum(1 for it in retrieval_info if it.get("graph_expanded"))
        title = f"🔍 查看检索片段（{len(retrieval_info)} 个"
        title += f"，含 {n_graph} 个图谱关联）" if n_graph else "）"
        with st.expander(title, expanded=False):
            render_retrieval_cards(retrieval_info)

    # 底部操作栏
    if st.session_state.messages:
        st.markdown("---")
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            st.download_button(
                "📥 导出对话",
                data=json.dumps(
                    [{"role": m["role"], "content": m["content"]}
                     for m in st.session_state.messages],
                    ensure_ascii=False, indent=2,
                ),
                file_name="chat_export.json",
                mime="application/json",
                use_container_width=True,
            )


# ── 页面：引用图谱 ────────────────────────────────────

elif page == "🕸️ 引用图谱":
    import streamlit.components.v1 as components
    from graph_viz import build_graph_html, build_ego_graph_html

    g = load_citation_graph()
    gs = g.stats()

    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.6rem;">
        <h2 style="margin:0 0 0.3rem;font-size:1.7rem;
            background:linear-gradient(135deg,#60a5fa,#a78bfa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🕸️ 法条引用图谱
        </h2>
        <p style="color:#64748b;font-size:0.9rem;margin:0;">
            法规不是孤立文本，而是相互引用的网络 —— 用结构化关系补充向量检索的盲区
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 指标卡
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("有引用关系", f"{gs['connected_count']} 部",
              f"共 {gs['node_count']} 部")
    m2.metric("引用边", f"{gs['edge_count']} 条")
    m3.metric("引用总数", f"{gs['total_citations']} 处")
    based = gs["type_distribution"].get("based_on", 0)
    m4.metric("依据关系", f"{based} 条", "下位法→上位法")

    st.markdown("")

    # ── 主图 ──
    min_w = st.select_slider(
        "边过滤（引用次数下限）",
        options=[1, 2, 3, 5],
        value=1,
        help="调高可以过滤弱关联，只看强引用关系",
    )

    vis_data = g.to_vis_json(min_weight=min_w)
    if vis_data["nodes"]:
        components.html(
            build_graph_html(vis_data, height=660),
            height=680,
            scrolling=False,
        )
    else:
        st.info("当前过滤条件下没有边，请调低阈值")

    # ── 图谱洞察 ──
    st.markdown("### 📈 图谱洞察")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**📥 被引用最多** — 法律位阶高、基础性强")
        for n, d in gs["most_cited"][:8]:
            pct = d / gs["most_cited"][0][1] if gs["most_cited"] else 0
            st.markdown(f"""<div style="display:flex;align-items:center;
                gap:10px;margin-bottom:6px;font-size:0.83rem;">
                <div style="width:26px;text-align:right;color:#60a5fa;
                     font-weight:600;">{d}</div>
                <div style="flex:1;background:rgba(96,165,250,.12);
                     border-radius:4px;height:22px;position:relative;">
                    <div style="width:{pct*100:.0f}%;height:100%;
                         background:linear-gradient(90deg,#3b82f6,#60a5fa);
                         border-radius:4px;"></div>
                    <div style="position:absolute;left:8px;top:2px;
                         font-size:0.78rem;">{n[:22]}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("**📤 引用他人最多** — 综合性、依赖性强")
        for n, d in gs["most_citing"][:8]:
            pct = d / gs["most_citing"][0][1] if gs["most_citing"] else 0
            st.markdown(f"""<div style="display:flex;align-items:center;
                gap:10px;margin-bottom:6px;font-size:0.83rem;">
                <div style="width:26px;text-align:right;color:#34d399;
                     font-weight:600;">{d}</div>
                <div style="flex:1;background:rgba(52,211,153,.12);
                     border-radius:4px;height:22px;position:relative;">
                    <div style="width:{pct*100:.0f}%;height:100%;
                         background:linear-gradient(90deg,#10b981,#34d399);
                         border-radius:4px;"></div>
                    <div style="position:absolute;left:8px;top:2px;
                         font-size:0.78rem;">{n[:22]}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.caption("💡 图谱自动发现了法律位阶结构 —— "
               "被引用最多的正是《医疗机构管理条例》《药品管理法》《医师法》等基础法规")

    # ── 单点探查 ──
    st.divider()
    st.markdown("### 🔎 探查单部法规的引用网络")

    connected_nodes = sorted(
        n for n in g.nodes
        if g.out_edges.get(n) or g.in_edges.get(n)
    )
    picked = st.selectbox("选择法规", connected_nodes,
                          index=connected_nodes.index("药品管理法")
                          if "药品管理法" in connected_nodes else 0)

    if picked:
        out_e = g.neighbors(picked, "out")
        in_e = g.neighbors(picked, "in")

        ego_html = build_ego_graph_html(g, picked, height=360)
        if ego_html:
            components.html(ego_html, height=375, scrolling=False)

        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"**↗ 《{picked}》引用了 {len(out_e)} 部**")
            for e in sorted(out_e, key=lambda x: -x["weight"]):
                arts = f" · {', '.join(e['articles'][:2])}" if e["articles"] else ""
                type_cn = {"based_on": "依据", "refer_to": "参照",
                           "mention": "提及"}.get(e["type"], e["type"])
                st.markdown(f"""<div class="retrieval-card" style="padding:8px 12px;">
                    <span class="source-tag">{type_cn} ×{e['weight']}</span>
                    <div style="margin-top:4px;font-size:0.85rem;">
                        《{e['target']}》{arts}</div>
                </div>""", unsafe_allow_html=True)
            if not out_e:
                st.caption("无对外引用")

        with d2:
            st.markdown(f"**↙ {len(in_e)} 部引用了《{picked}》**")
            for e in sorted(in_e, key=lambda x: -x["weight"]):
                arts = f" · {', '.join(e['articles'][:2])}" if e["articles"] else ""
                type_cn = {"based_on": "依据", "refer_to": "参照",
                           "mention": "提及"}.get(e["type"], e["type"])
                st.markdown(f"""<div class="retrieval-card" style="padding:8px 12px;
                    border-left-color:#34d399;">
                    <span class="source-tag" style="background:#064e3b;color:#6ee7b7;">
                        {type_cn} ×{e['weight']}</span>
                    <div style="margin-top:4px;font-size:0.85rem;">
                        《{e['source']}》{arts}</div>
                </div>""", unsafe_allow_html=True)
            if not in_e:
                st.caption("无被引用")

        # 图扩展演示
        st.markdown("**🌐 图扩展效果**（检索命中此法规时，会额外召回）")
        expanded = g.expand([picked], max_add=4)
        if expanded:
            for name, reason, w in expanded:
                st.markdown(f"- **《{name}》** — {reason}")
        else:
            st.caption("无可扩展的关联法规")


# ── 页面：评估 ────────────────────────────────────────

elif page == "📊 评估":
    st.markdown("## 📊 系统评估")
    st.caption("10 道测试题覆盖精确检索、理解推理、拒答测试三类场景")

    eval_path = os.path.join(PROJECT_ROOT, "tests", "eval_results.json")
    has_saved = os.path.exists(eval_path)

    col1, col2 = st.columns(2)
    with col1:
        run_eval = st.button("▶ 运行评估", type="primary", use_container_width=True)
    with col2:
        load_saved = st.button("📂 加载上次结果", use_container_width=True, disabled=not has_saved)

    # 运行评估
    if run_eval:
        from evaluate import TEST_CASES, evaluate_retrieval, evaluate_generation

        progress = st.progress(0, text="准备中...")
        results_log = []
        retrieval_pass = 0
        generation_pass = 0
        total = len(TEST_CASES)

        for i, tc in enumerate(TEST_CASES):
            q = tc["question"]
            progress.progress((i + 1) / total, text=f"[{i+1}/{total}] {q}")

            ret_eval = evaluate_retrieval(q, tc.get("expected_source"))
            if ret_eval["hit"]:
                retrieval_pass += 1

            context, _ = rag_retrieve(q, top_k=top_k, graph_k=graph_k)
            user_prompt = f"【参考原文】\n{context}\n\n【问题】\n{q}"
            ans_text = ""
            stream = ask_llm_stream(RAG_SYSTEM_PROMPT, user_prompt)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    ans_text += chunk.choices[0].delta.content

            gen_eval = evaluate_generation(ans_text, tc["expected_keywords"], tc["type"])
            if gen_eval["pass"]:
                generation_pass += 1

            results_log.append({
                "question": q,
                "type": tc["type"],
                "retrieval": ret_eval,
                "generation": gen_eval,
                "answer_preview": ans_text[:300],
            })

        progress.empty()

        os.makedirs(os.path.join(PROJECT_ROOT, "tests"), exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(results_log, f, ensure_ascii=False, indent=2)

        st.session_state["eval_results"] = results_log
        st.session_state["eval_r_pass"] = retrieval_pass
        st.session_state["eval_g_pass"] = generation_pass

    # 加载已有结果
    if (has_saved and load_saved) or (has_saved and "eval_results" not in st.session_state):
        with open(eval_path, "r", encoding="utf-8") as f:
            results_log = json.load(f)
        st.session_state["eval_results"] = results_log
        st.session_state["eval_r_pass"] = sum(1 for r in results_log if r["retrieval"].get("hit"))
        st.session_state["eval_g_pass"] = sum(1 for r in results_log if r["generation"].get("pass"))

    # 展示结果
    if "eval_results" in st.session_state:
        results_log = st.session_state["eval_results"]
        r_pass = st.session_state["eval_r_pass"]
        g_pass = st.session_state["eval_g_pass"]
        total = len(results_log)

        st.divider()

        m1, m2, m3 = st.columns(3)
        m1.metric("检索准确率", f"{r_pass}/{total}", f"{r_pass/total*100:.0f}%")
        m2.metric("生成准确率", f"{g_pass}/{total}", f"{g_pass/total*100:.0f}%")

        type_stats = {}
        for r in results_log:
            t = r["type"]
            if t not in type_stats:
                type_stats[t] = {"total": 0, "pass": 0}
            type_stats[t]["total"] += 1
            if r["generation"].get("pass"):
                type_stats[t]["pass"] += 1
        m3.metric("分类",
                  " · ".join(f"{t} {s['pass']}/{s['total']}" for t, s in type_stats.items()))

        st.divider()

        for r in results_log:
            ret_ok = r["retrieval"].get("hit", False)
            gen_ok = r["generation"].get("pass", False)
            icon = "✅" if (ret_ok and gen_ok) else "❌"

            with st.expander(f"{icon} [{r['type']}] {r['question']}", expanded=not gen_ok):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**检索** {'✅' if ret_ok else '❌'}")
                    sources = r["retrieval"].get("sources", [])
                    if sources:
                        st.caption(f"来源：{', '.join(sources[:3])}")
                with c2:
                    st.markdown(f"**生成** {'✅' if gen_ok else '❌'}")
                    if r["type"] == "拒答测试":
                        st.caption(r["generation"].get("note", ""))
                    else:
                        st.caption(f"关键词：{r['generation'].get('coverage', 'N/A')}")
                        missed = r["generation"].get("missed_keywords", [])
                        if missed:
                            st.warning(f"未命中：{missed}")

                st.info(r.get("answer_preview", "")[:300])


# ── 页面：知识库 ──────────────────────────────────────

elif page == "📚 知识库":
    st.markdown("## 📚 知识库")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    files = sorted([f for f in os.listdir(data_dir) if f.endswith((".txt", ".pdf"))]) if os.path.isdir(data_dir) else []

    # 概览
    m1, m2, m3 = st.columns(3)
    m1.metric("政策文件", f"{len(files)} 部")
    m2.metric("向量片段", f"{chunk_count}")
    m3.metric("切分方式", "条款级智能切分")

    st.divider()

    # 分类展示
    categories = {
        "法律": [], "行政法规": [], "部门规章": [],
        "规范性文件": [], "草案": [], "司法解释": [], "其他": [],
    }

    # 简单分类（按文件名特征）
    for f in files:
        name = f.replace(".txt", "").replace(".pdf", "")
        if "条例" in name and "草案" not in name:
            categories["行政法规"].append(name)
        elif "办法" in name or "规定" in name or "规范" in name or "规程" in name:
            categories["部门规章"].append(name)
        elif "关于" in name or "意见" in name or "通知" in name or "指导" in name:
            categories["规范性文件"].append(name)
        elif "草案" in name:
            categories["草案"].append(name)
        elif "骗保刑事案件" in name:
            categories["司法解释"].append(name)
        elif "法" in name:
            categories["法律"].append(name)
        else:
            categories["其他"].append(name)

    for cat, items in categories.items():
        if not items:
            continue
        with st.expander(f"**{cat}**（{len(items)} 部）", expanded=(cat == "法律")):
            for name in items:
                fpath = os.path.join(data_dir, name + ".txt")
                if os.path.exists(fpath):
                    size_kb = os.path.getsize(fpath) / 1024
                    st.markdown(f"""<div class="file-card">
                        <div class="name">📄 {name}</div>
                        <div class="meta">{size_kb:.1f} KB</div>
                    </div>""", unsafe_allow_html=True)

    # 语义搜索
    st.divider()
    st.markdown("#### 🔍 混合检索测试")
    st.caption("BM25 关键词检索 + 向量语义检索，RRF 融合排序（不经过 LLM）")
    search_q = st.text_input("输入查询", placeholder="例如：飞行检查的启动情形")
    if search_q:
        retriever = load_retriever()
        results = retriever.retrieve(search_q, top_k=top_k)
        for i, doc in enumerate(results):
            src = os.path.basename(doc.metadata.get("source", "")).replace(".txt", "")
            label = doc.metadata.get("article_label", "")
            st.markdown(f"""<div class="retrieval-card">
                <span class="source-tag">#{i+1} {src}</span>
                {' · <span class="source-tag">' + label + '</span>' if label else ''}
                <div style="margin-top:6px; color:#475569; line-height:1.6;">{doc.page_content[:400]}</div>
            </div>""", unsafe_allow_html=True)


# ── 页面：系统 ────────────────────────────────────────

elif page == "⚙️ 系统":
    st.markdown("## ⚙️ 系统信息")

    from config import (DEEPSEEK_BASE_URL, CHAT_MODEL, EMBED_MODEL,
                        CHUNK_SIZE, CHUNK_OVERLAP, TOP_K as DEFAULT_TOP_K)

    st.markdown("#### 技术栈")

    tech_data = [
        ("LLM", CHAT_MODEL, "DeepSeek API（OpenAI 兼容接口）"),
        ("Embedding", EMBED_MODEL, "本地中文模型，512 维"),
        ("向量库", "Chroma 0.5.x", "轻量持久化，支持增量更新"),
        ("切分", "ArticleAwareSplitter", "按条款结构切分，非固定字数"),
        ("检索", "Hybrid (BM25 + Vector + RRF)", "关键词+语义双路召回，倒数排名融合"),
        ("框架", "LangChain + Streamlit", "检索框架 + Web 界面"),
    ]

    for name, value, desc in tech_data:
        st.markdown(f"**{name}** &nbsp; `{value}`")
        st.caption(desc)

    st.divider()
    st.markdown("#### 架构")
    st.code("""
┌──────────────────────────────────────────────┐
│  离线建库                                      │
│  data/*.txt                                    │
│    → 加载                                      │
│    → Smart Split（按条款/序号/段落自适应切分）    │
│    → bge-small-zh 向量化                       │
│    → Chroma 存储                               │
└──────────────────────────────────────────────┘
                        ↕
┌──────────────────────────────────────────────┐
│  在线问答（混合检索）                            │
│  用户问题                                      │
│    ├─→ BM25 关键词检索 ──→ Top-20              │
│    └─→ 向量语义检索   ──→ Top-20              │
│         ↓                                      │
│      RRF 倒数排名融合 ──→ Top-K                │
│         ↓                                      │
│    拼入 Prompt + 反幻觉约束                     │
│    → DeepSeek API 流式生成                     │
│    → 带法条引用的结构化回答                      │
└──────────────────────────────────────────────┘
""", language=None)

    st.divider()
    st.markdown("#### 检索策略对比实验")
    st.caption("20 题测试集，4 种策略的量化对比")

    exp_path = os.path.join(PROJECT_ROOT, "tests", "retrieval_experiment.json")
    if os.path.exists(exp_path):
        with open(exp_path, encoding="utf-8") as f:
            exp = json.load(f)

        import pandas as pd
        rows = []
        for name, s in exp["summary"].items():
            rows.append({
                "策略": name,
                "Hit@5": f"{s['hit_rate']:.0%}",
                "MRR": f"{s['mrr']:.3f}",
                "P@5": f"{s['precision']:.3f}",
                "延迟": f"{s['avg_latency']*1000:.0f}ms",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("结论：RRF 混合检索为最优解。Cross-Encoder 重排在本场景下负收益"
                   "（指标略降，延迟涨 300 倍），故默认关闭。")
    else:
        st.caption("运行 `python src/retrieval_experiment.py` 生成对比数据")

    st.divider()
    st.markdown("#### 参数")
    params = {
        "Fallback Chunk Size": f"{CHUNK_SIZE} 字符",
        "Chunk Overlap": f"{CHUNK_OVERLAP} 字符",
        "默认 Top-K": DEFAULT_TOP_K,
        "合并阈值": "150 字符（短条款合并）",
        "二次切分阈值": "800 字符（长条款拆分）",
        "Temperature": "0.2",
    }
    for k, v in params.items():
        col1, col2 = st.columns([1, 2])
        col1.markdown(f"**{k}**")
        col2.code(str(v), language=None)
