"""测试 DeepSeek API 是否真正被调用 —— 打印完整调试信息"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CHAT_MODEL

print("=" * 60)
print("DeepSeek API 连接测试")
print("=" * 60)

# 显示配置（key 只显示前几位和后几位）
key_display = DEEPSEEK_API_KEY[:8] + "..." + DEEPSEEK_API_KEY[-4:] if DEEPSEEK_API_KEY else "未设置！"
print(f"API Key:   {key_display}")
print(f"Base URL:  {DEEPSEEK_BASE_URL}")
print(f"Model:     {CHAT_MODEL}")
print()

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 测试 1：最简单的调用
print("📡 测试 1：发送一个简单请求...")
try:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "请用一句话回答：1+1等于几？"}],
        temperature=0,
    )
    print(f"   ✅ 返回成功！")
    print(f"   模型回答: {resp.choices[0].message.content}")
    print(f"   模型 ID:  {resp.model}")
    print(f"   用量 - prompt tokens:     {resp.usage.prompt_tokens}")
    print(f"   用量 - completion tokens: {resp.usage.completion_tokens}")
    print(f"   用量 - total tokens:      {resp.usage.total_tokens}")
    print(f"   响应 ID: {resp.id}")
except Exception as e:
    print(f"   ❌ 调用失败: {e}")
    sys.exit(1)

print()

# 测试 2：用新的政策文件测试
print("📡 测试 2：用《基本医疗卫生与健康促进法》测试...")
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
filepath = os.path.join(data_dir, "基本医疗卫生与健康促进法.txt")

with open(filepath, encoding="utf-8") as f:
    text = f.read()

try:
    resp2 = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个严谨的政策分析助手。"},
            {"role": "user", "content": f"请用150字概括以下法律的核心要点：\n\n{text[:2000]}"},
        ],
        temperature=0.2,
    )
    print(f"   ✅ 返回成功！")
    print(f"   响应 ID: {resp2.id}")
    print(f"   用量 - total tokens: {resp2.usage.total_tokens}")
    print()
    print("📋 模型摘要：")
    print("-" * 50)
    print(resp2.choices[0].message.content)
    print("-" * 50)
except Exception as e:
    print(f"   ❌ 调用失败: {e}")

print()
print("💡 如果以上两个测试都返回了内容，说明 API 调用正常。")
print("   请去 DeepSeek 开放平台刷新页面查看用量是否更新。")
print("   （用量统计有时会有几分钟延迟）")
