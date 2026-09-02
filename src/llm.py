"""阶段 0-1：LLM 调用与 Prompt 工程

知识点：
- messages 里 system 定角色、user 是具体请求，是对话式 API 的标准结构
- temperature 控制随机性：越低越稳定；政策场景要低温度（0.2）
- DeepSeek 使用 OpenAI 兼容接口，换 base_url 即可
"""

import sys
import os

# 把项目根目录加入 path，这样无论从哪里运行都能找到 config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CHAT_MODEL


client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def ask_llm(system_prompt: str, user_prompt: str) -> str:
    """调用大模型，返回回答文本。

    Args:
        system_prompt: 系统提示词，定义角色和行为约束
        user_prompt:   用户输入，具体的请求内容
    Returns:
        模型生成的文本
    """
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # 政策场景要稳定、少发挥，温度调低
    )
    return resp.choices[0].message.content


# ---- 阶段 0 验证：读文件 → 调模型 → 输出摘要 ----
if __name__ == "__main__":
    # 先找一个能读的文件
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    test_files = [f for f in os.listdir(data_dir) if f.endswith((".txt", ".pdf"))]

    if not test_files:
        print("⚠️  data/ 目录下还没有文件！")
        print("   请把一份公开政策文件（PDF 或 TXT）放进 data/ 目录，再重新运行。")
        sys.exit(1)

    # 优先用 txt，简单直接
    txt_files = [f for f in test_files if f.endswith(".txt")]
    target = txt_files[0] if txt_files else test_files[0]
    filepath = os.path.join(data_dir, target)

    print(f"📄 正在读取：{target}")

    if target.endswith(".txt"):
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
    elif target.endswith(".pdf"):
        # 简单读 PDF（后面阶段 2 会用更专业的 loader）
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            print("⚠️  读 PDF 需要 pypdf，请运行: pip install pypdf")
            sys.exit(1)

    # 只取前 3000 字符——长文档塞不进上下文，这正是后面需要 RAG 的原因
    text_snippet = text[:3000]
    print(f"   文本长度：{len(text)} 字符，截取前 3000 字符发给模型\n")

    print("🤖 正在调用 DeepSeek...")
    summary = ask_llm(
        "你是一个严谨的政策分析助手。",
        f"请用 200 字概括以下政策文件的核心内容：\n\n{text_snippet}",
    )
    print("\n📋 模型摘要：")
    print("-" * 50)
    print(summary)
    print("-" * 50)
    print("\n✅ 阶段 0 完成！LLM 调用链路跑通。")
