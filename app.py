"""🏥 医疗政策 RAG 问答系统 —— Streamlit Web 界面

启动命令：
  streamlit run app.py

功能：
  1. 智能问答：输入问题 → 检索原文 → 生成带引用的回答
  2. 检索透视：可视化展示每个检索片段的来源和内容
  3. 评估面板：一键运行 10 道测试题，查看检索/生成质量
  4. 知识库管理：查看已入库的政策文件和片段统计
"""

import sys
import os
import json
import time

# ---- 路径设置：确保能 import src/ 下的模块 ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

import streamlit as st

# ---- 页面基础配置 ----
st.set_page_config(
    page_title="医疗政策 RAG 问答系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# 缓存加载：只在首次运行时加载模型和向量库
# =====================================================

@st.cache_resource(show_spinner="正在加载向量库和 Embedding 模型...")
def load_rag_components():
    """加载 RAG 组件（只执行一次，后续复用缓存）。"""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from config import CHROMA_DIR, EMBED_MODEL

    chroma_path = os.path.join(PROJECT_ROOT, CHROMA_DIR)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vs = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
    )
    return vs, embeddings


@st.cache_resource
def load_llm_client():
    """加载 LLM 客户端。"""
    from openai import OpenAI
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def ask_llm(system_prompt: str, user_prompt: str) -> str:
    """调用 LLM。"""
    from config import CHAT_MODEL
    client = load_llm_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# =====================================================
# RAG 核心：检索 + 生成（返回更多信息给前端展示）
# =====================================================

RAG_SYSTEM_PROMPT = """你是医疗政策问答助手。请严格遵守以下规则：

1. 只依据下面提供的【参考原文】回答，不得使用参考原文之外的任何知识。
2. 如果参考原文中没有相关信息，明确回答"根据现有政策文件，未能找到相关规定"，不要猜测。
3. 回答时引用具体条款（如"根据《XX法》第X条"），让答案可追溯。
4. 回答末尾用"📎 引用来源"标注你参考了哪些文件片段。
5. 语言简洁准确，像政策解读而非学术论文。
"""


def rag_answer(question: str, top_k: int = 5):
    """RAG 问答，返回 (回答文本, 检索结果列表, 耗时)。"""
    vs, _ = load_rag_components()

    t0 = time.time()

    # 检索
    results = vs.similarity_search(question, k=top_k)

    # 构建上下文
    context_blocks = []
    retrieval_info = []
    for i, doc in enumerate(results):
        src = os.path.basename(doc.metadata.get("source", "未知"))
        src_name = src.replace(".txt", "").replace(".pdf", "")
        context_blocks.append(
            f"[片段{i+1}｜来源：《{src_name}》]\n{doc.page_content}"
        )
        retrieval_info.append({
            "index": i + 1,
            "source": src_name,
            "content": doc.page_content,
        })

    context = "\n\n".join(context_blocks)
    user_prompt = f"【参考原文】\n{context}\n\n【问题】\n{question}"

    # 生成
    answer_text = ask_llm(RAG_SYSTEM_PROMPT, user_prompt)
    elapsed = time.time() - t0

    return answer_text, retrieval_info, elapsed


# =====================================================
# 侧边栏
# =====================================================

with st.sidebar:
    st.markdown("## 🏥 医疗政策 RAG")
    st.caption("基于公开政策文件 · 答案可溯源 · 拒绝编造")

    page = st.radio(
        "功能导航",
        ["💬 智能问答", "📊 评估面板", "📚 知识库", "⚙️ 系统信息"],
        label_visibility="collapsed",
    )

    st.divider()

    # 系统状态
    try:
        vs, _ = load_rag_components()
        chunk_count = vs._collection.count()
        st.success(f"✅ 向量库已加载：{chunk_count} 个片段")
    except Exception as e:
        st.error(f"❌ 向量库加载失败：{e}")
        chunk_count = 0

    # 参数调节
    st.markdown("### 🎛️ 参数调节")
    top_k = st.slider("Top-K（检索片段数）", min_value=1, max_value=10, value=5,
                       help="K 越大覆盖越全，但可能引入噪音")


# =====================================================
# 页面 1：智能问答
# =====================================================

if page == "💬 智能问答":
    st.markdown("# 💬 智能问答")
    st.markdown("输入关于医疗政策法规的问题，系统将从 12 部公开法规中检索相关条款并生成回答。")

    # 初始化聊天历史
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 示例问题（快速体验）
    example_questions = [
        "无偿献血的年龄范围是多少？",
        "什么是假药？列举假药的情形。",
        "医师和护士在紧急情况下的责任有什么不同？",
        "以师承方式学中医的人可以行医吗？",
        "发生医疗纠纷后，患者有权查阅哪些病历资料？",
        "2025年医保报销比例是多少？",
    ]

    st.markdown("**快速试试：**")
    cols = st.columns(3)
    for i, eq in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(eq, key=f"ex_{i}", use_container_width=True):
                st.session_state["pending_question"] = eq

    # 渲染历史消息
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "elapsed" in msg:
                st.caption(f"⏱️ 用时 {msg['elapsed']:.1f} 秒 | Top-K = {msg.get('top_k', 5)}")
        # 历史消息的检索透视
        if msg["role"] == "assistant" and "retrieval" in msg:
            with st.expander(f"🔍 检索透视 — {len(msg['retrieval'])} 个片段", expanded=False):
                for item in msg["retrieval"]:
                    st.markdown(f"**片段 {item['index']}** — 来源：《{item['source']}》")
                    st.code(item["content"], language=None)

    # 获取新问题：来自 chat_input 或示例按钮
    question = st.chat_input("输入你的问题...")
    if question is None and "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        # 添加用户消息到历史
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在检索并生成回答..."):
                answer_text, retrieval_info, elapsed = rag_answer(question, top_k=top_k)

            st.markdown(answer_text)
            st.caption(f"⏱️ 用时 {elapsed:.1f} 秒 | Top-K = {top_k}")

        # 添加助手消息到历史
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
            "retrieval": retrieval_info,
            "elapsed": elapsed,
            "top_k": top_k,
        })

        # 检索透视
        with st.expander(f"🔍 检索透视 — 共检索到 {len(retrieval_info)} 个片段", expanded=False):
            st.markdown("""
            > **面试知识点**：排障时先看检索结果——如果检索就没找对段落，那是检索问题；
            > 如果检索到了对的段落但模型答错，那是生成问题。
            """)
            for item in retrieval_info:
                st.markdown(f"**片段 {item['index']}** — 来源：《{item['source']}》")
                st.code(item["content"], language=None)

    # 清空历史按钮
    if st.session_state.chat_history:
        if st.button("🗑️ 清空对话", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# =====================================================
# 页面 2：评估面板
# =====================================================

elif page == "📊 评估面板":
    st.markdown("# 📊 评估面板")
    st.markdown("一键运行 10 道测试题，评估检索准确率和生成质量。")

    # 先尝试加载上次的评估结果
    eval_path = os.path.join(PROJECT_ROOT, "tests", "eval_results.json")
    has_saved = os.path.exists(eval_path)

    col1, col2 = st.columns(2)

    with col1:
        run_eval = st.button("🚀 运行完整评估", type="primary", use_container_width=True)
    with col2:
        if has_saved:
            load_saved = st.button("📂 查看上次结果", use_container_width=True)
        else:
            load_saved = False

    # ---- 运行评估 ----
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

            # 检索评估
            ret_eval = evaluate_retrieval(q, tc.get("expected_source"))
            if ret_eval["hit"]:
                retrieval_pass += 1

            # 生成回答
            ans_text, _, _ = rag_answer(q, top_k=top_k)

            # 生成评估
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

        # 保存结果
        os.makedirs(os.path.join(PROJECT_ROOT, "tests"), exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(results_log, f, ensure_ascii=False, indent=2)

        st.session_state["eval_results"] = results_log
        st.session_state["eval_retrieval_pass"] = retrieval_pass
        st.session_state["eval_generation_pass"] = generation_pass
        st.success("评估完成！")

    # ---- 加载已有结果 ----
    if has_saved and (load_saved or "eval_results" not in st.session_state):
        if load_saved or "eval_results" not in st.session_state:
            with open(eval_path, "r", encoding="utf-8") as f:
                results_log = json.load(f)
            r_pass = sum(1 for r in results_log if r["retrieval"].get("hit", False))
            g_pass = sum(1 for r in results_log if r["generation"].get("pass", False))
            st.session_state["eval_results"] = results_log
            st.session_state["eval_retrieval_pass"] = r_pass
            st.session_state["eval_generation_pass"] = g_pass

    # ---- 展示结果 ----
    if "eval_results" in st.session_state:
        results_log = st.session_state["eval_results"]
        retrieval_pass = st.session_state["eval_retrieval_pass"]
        generation_pass = st.session_state["eval_generation_pass"]
        total = len(results_log)

        # 总分卡片
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("检索准确率", f"{retrieval_pass}/{total}", f"{retrieval_pass/total*100:.0f}%")
        m2.metric("生成准确率", f"{generation_pass}/{total}", f"{generation_pass/total*100:.0f}%")

        # 按类型统计
        type_stats = {}
        for r in results_log:
            t = r["type"]
            if t not in type_stats:
                type_stats[t] = {"total": 0, "pass": 0}
            type_stats[t]["total"] += 1
            if r["generation"].get("pass", False):
                type_stats[t]["pass"] += 1

        type_str = " | ".join(
            f"{t} {s['pass']}/{s['total']}" for t, s in type_stats.items()
        )
        m3.metric("分类通过率", type_str)

        # 详细列表
        st.divider()
        for i, r in enumerate(results_log):
            ret_ok = r["retrieval"].get("hit", False)
            gen_ok = r["generation"].get("pass", False)
            status = "✅" if (ret_ok and gen_ok) else "❌"

            with st.expander(f"{status} [{r['type']}] {r['question']}", expanded=not gen_ok):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**检索** {'✅' if ret_ok else '❌'}")
                    sources = r["retrieval"].get("sources", [])
                    if sources:
                        st.markdown(f"来源：{', '.join(sources)}")
                with c2:
                    st.markdown(f"**生成** {'✅' if gen_ok else '❌'}")
                    if r["type"] == "拒答测试":
                        st.markdown(r["generation"].get("note", ""))
                    else:
                        st.markdown(f"关键词覆盖：{r['generation'].get('coverage', 'N/A')}")
                        missed = r["generation"].get("missed_keywords", [])
                        if missed:
                            st.warning(f"未命中：{missed}")

                st.markdown("**回答预览：**")
                st.info(r.get("answer_preview", "")[:300])


# =====================================================
# 页面 3：知识库管理
# =====================================================

elif page == "📚 知识库":
    st.markdown("# 📚 知识库管理")

    data_dir = os.path.join(PROJECT_ROOT, "data")

    # 文件列表
    if os.path.isdir(data_dir):
        files = sorted([f for f in os.listdir(data_dir) if f.endswith((".txt", ".pdf"))])
    else:
        files = []

    col1, col2, col3 = st.columns(3)
    col1.metric("政策文件数", len(files))
    col2.metric("向量片段数", chunk_count)
    col3.metric("Top-K 设置", top_k)

    st.divider()

    # 文件列表
    st.markdown("### 📄 已入库文件")
    for i, f in enumerate(files):
        fpath = os.path.join(data_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        name = f.replace(".txt", "").replace(".pdf", "")

        with st.expander(f"📑 {name}（{size_kb:.1f} KB）"):
            with open(fpath, "r", encoding="utf-8") as fp:
                content = fp.read()
            st.text_area(
                "文件内容",
                value=content[:2000] + ("..." if len(content) > 2000 else ""),
                height=200,
                disabled=True,
                key=f"file_{i}",
                label_visibility="collapsed",
            )
            st.caption(f"全文 {len(content)} 字符")

    # 语义搜索测试
    st.divider()
    st.markdown("### 🔍 语义搜索测试")
    st.markdown("直接测试向量检索（不经过 LLM），看检索出哪些片段。")

    search_q = st.text_input("输入搜索词", placeholder="例如：献血年龄")
    if search_q:
        vs, _ = load_rag_components()
        results = vs.similarity_search(search_q, k=top_k)

        for i, doc in enumerate(results):
            src = os.path.basename(doc.metadata.get("source", "未知")).replace(".txt", "")
            st.markdown(f"**片段 {i+1}** — 来源：《{src}》")
            st.code(doc.page_content, language=None)


# =====================================================
# 页面 4：系统信息
# =====================================================

elif page == "⚙️ 系统信息":
    st.markdown("# ⚙️ 系统信息")

    from config import (DEEPSEEK_BASE_URL, CHAT_MODEL, EMBED_MODEL,
                        CHUNK_SIZE, CHUNK_OVERLAP, TOP_K as DEFAULT_TOP_K)

    # 配置表
    st.markdown("### 当前配置")
    config_data = {
        "LLM 模型": CHAT_MODEL,
        "LLM API": DEEPSEEK_BASE_URL,
        "Embedding 模型": EMBED_MODEL,
        "Chunk 大小": f"{CHUNK_SIZE} 字符",
        "Chunk 重叠": f"{CHUNK_OVERLAP} 字符",
        "默认 Top-K": DEFAULT_TOP_K,
        "向量维度": "512 维",
        "向量库": "Chroma 0.5.x",
    }
    for k, v in config_data.items():
        st.markdown(f"- **{k}**：`{v}`")

    # 架构图
    st.divider()
    st.markdown("### 🏗️ RAG 架构")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────┐
    │  离线建库                                        │
    │  data/*.txt → 加载 → 切分 → 向量化 → Chroma 存储 │
    └─────────────────────────────────────────────────┘
                            ↕
    ┌─────────────────────────────────────────────────┐
    │  在线问答                                        │
    │  用户问题 → 向量化 → 检索 Top-K → 拼 Prompt      │
    │          → DeepSeek API → 带引用的回答            │
    └─────────────────────────────────────────────────┘
    ```
    """)

    # 面试要点
    st.divider()
    st.markdown("### 🎯 面试要点速查")
    points = {
        "RAG 七步链路": "加载 → 切分 → 向量化 → 存储（离线）→ 问题向量化 → 检索 Top-K → 拼 Prompt 生成（在线）",
        "防幻觉三板斧": "Prompt 约束'只用原文' + 给拒答退路 + 低 temperature（0.2）",
        "引用溯源": "chunk 带 metadata → Prompt 里标注来源 → 要求模型引用",
        "排障思路": "答案不对 → 先看检索 → 检索没找对改检索，检索找对了改 Prompt",
        "调优记录": "Top-K 3→5 解决跨文件对比检索覆盖不足，回归测试 10/10 无副作用",
    }
    for title, desc in points.items():
        st.markdown(f"**{title}**：{desc}")
