"""阶段 0-1：LLM 调用与 Prompt 工程

知识点（面试必背）：
- messages 里 system 定角色、user 是具体请求，是对话式 API 的标准结构
- temperature 控制随机性：越低越稳定；政策场景要低温度（0.2）
- DeepSeek 使用 OpenAI 兼容接口，换 base_url 即可

Prompt 工程三板斧：
1. 角色设定（system prompt）—— 告诉模型"你是谁"，锚定行为基调
2. 明确约束 —— 告诉模型"不许做什么"，比如不许编造
3. 固定输出格式 —— 告诉模型"按什么结构回答"，保证输出可预期
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


def ask_llm_json(system_prompt: str, user_prompt: str,
                 temperature: float = 0.0) -> dict:
    """调用大模型并要求返回 JSON。

    用于 LLM-as-judge 等需要机器解析结果的场景。
    DeepSeek 支持 OpenAI 的 response_format JSON 模式。

    Args:
        system_prompt: 系统提示词（必须在其中说明输出 JSON 及其结构）
        user_prompt:   用户输入
        temperature:   评判场景用 0.0，保证可复现

    Returns:
        解析后的 dict；解析失败返回 {"_error": "...", "_raw": "..."}
    """
    import json as _json

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
    except Exception as e:
        return {"_error": f"API 调用失败: {e}"}

    try:
        return _json.loads(raw)
    except Exception:
        # 兜底：模型偶尔会在 JSON 外包一层 ```json ```
        import re as _re
        m = _re.search(r'\{.*\}', raw or "", _re.S)
        if m:
            try:
                return _json.loads(m.group(0))
            except Exception:
                pass
        return {"_error": "JSON 解析失败", "_raw": (raw or "")[:300]}


# =====================================================
# 阶段 1：提示词工程 —— 结构化输出 + 反幻觉
# =====================================================

# 【技巧 1：角色设定】
#   "你是严谨的医疗政策分析助手" —— 一句话锚定模型的行为基调
#   不同的角色会让同一个模型的风格完全不同
#   面试时说：system prompt 里的角色设定决定了模型的"人格"

# 【技巧 2：明确约束（反幻觉核心）】
#   "只依据用户提供的原文回答，绝不编造" —— 堵死模型瞎编的路
#   "找不到就说未找到" —— 给模型一个安全的退路，比让它硬猜好得多
#   面试时说：反幻觉的关键是给模型明确的"不知道"选项

# 【技巧 3：固定输出格式】
#   列出具体的字段（发文机关、发文字号、核心条款……）
#   模型会按这个结构填充，输出稳定、可解析
#   面试时说：结构化输出让下游程序可以直接解析，而不是靠正则去猜

POLICY_SYSTEM_PROMPT = """你是严谨的医疗政策分析助手。请严格遵守以下规则：

1. 只依据用户提供的原文回答，绝不编造原文中没有的信息。
2. 如果原文中找不到某项信息，该项明确写"未在原文中找到"，不要猜测。
3. 按以下结构输出：
   - 法规名称：
   - 发文机关：
   - 发文字号/通过日期：
   - 适用范围：
   - 核心条款（分条列出）：
   - 生效时间：
   - 特别注意事项：
"""


def extract_policy_info(policy_text: str) -> str:
    """用结构化 Prompt 提取政策文件的关键信息。

    这个函数展示了 Prompt 工程最核心的三件事：
    1. 角色设定 → POLICY_SYSTEM_PROMPT 开头
    2. 行为约束 → "只依据原文""找不到说未找到"
    3. 输出格式 → 固定的字段列表

    Args:
        policy_text: 政策文件的原文文本
    Returns:
        结构化的政策信息提取结果
    """
    return ask_llm(POLICY_SYSTEM_PROMPT, f"请分析以下政策文件：\n\n{policy_text}")


# ---- 读取政策文件的工具函数 ----
def _load_text_file(filepath: str) -> str:
    """读取 txt 或 pdf 文件，返回文本内容。"""
    if filepath.endswith(".txt"):
        with open(filepath, encoding="utf-8") as f:
            return f.read()
    elif filepath.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM 调用测试")
    parser.add_argument("--stage", type=int, default=1, choices=[0, 1],
                        help="运行哪个阶段：0=简单摘要, 1=结构化提取（默认）")
    parser.add_argument("--file", type=str, default=None,
                        help="指定 data/ 下的文件名（默认自动选第一个）")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

    if args.file:
        target = args.file
    else:
        test_files = [f for f in os.listdir(data_dir) if f.endswith((".txt", ".pdf"))]
        if not test_files:
            print("⚠️  data/ 目录下还没有文件！")
            sys.exit(1)
        target = test_files[0]

    filepath = os.path.join(data_dir, target)
    text = _load_text_file(filepath)
    print(f"📄 文件：{target}（{len(text)} 字符）\n")

    if args.stage == 0:
        # --- 阶段 0：简单摘要 ---
        print("🤖 [阶段 0] 简单摘要模式...")
        result = ask_llm(
            "你是一个严谨的政策分析助手。",
            f"请用 200 字概括以下政策文件的核心内容：\n\n{text[:3000]}",
        )
    else:
        # --- 阶段 1：结构化提取 ---
        print("🤖 [阶段 1] 结构化提取模式...")
        print("   使用 Prompt 三板斧：角色设定 + 行为约束 + 固定格式\n")
        result = extract_policy_info(text[:3000])

    print("📋 输出结果：")
    print("=" * 55)
    print(result)
    print("=" * 55)
    print(f"\n✅ 阶段 {args.stage} 完成！")
