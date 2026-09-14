# policy_app.py —— 招生政策问答助手（RAG 检索 + DeepSeek 生成）网页版
# 先运行 build_kb.py 建库，再: python policy_app.py
import os
import json
import numpy as np
import gradio as gr
from openai import OpenAI
from sentence_transformers import SentenceTransformer

API_KEY = os.environ["DEEPSEEK_API_KEY"]     # 从环境变量读取
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
TOP_K = 4          # 每次检索的条文数
THRESHOLD = 0.35   # 相似度门槛，低于视为"政策里没有"

print("加载知识库...")
kb = json.load(open("kb.json", encoding="utf-8"))
kb_emb = np.load("kb_emb.npy")
emb_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print(f"知识库 {len(kb)} 块条文就绪")

SYSTEM = """你是学校招生政策问答助手。请严格根据提供的政策条文回答问题：
1. 只依据条文内容作答，不得编造政策里没有的信息
2. 回答末尾注明依据的条文原文关键句
3. 如果条文中找不到答案，明确说"政策文件中未找到相关规定，建议咨询招生办"
4. 用通俗易懂的语言解释，但涉及数字、日期、条件必须与原文完全一致"""

def answer(message, history):
    q = emb_model.encode([message], normalize_embeddings=True)
    scores = (kb_emb @ q.T).ravel()
    idx = scores.argsort()[::-1][:TOP_K]
    hits = [(kb[i], float(scores[i])) for i in idx if scores[i] >= THRESHOLD]

    if not hits:
        return "政策文件中未找到与此问题相关的条文，建议直接咨询招生办公室。"

    context = "\n\n".join(f"【条文{n+1}】{x['text']}"
                          for n, (x, s) in enumerate(hits))
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user",
                   "content": f"政策条文如下：\n{context}\n\n用户问题：{message}"}],
        temperature=0.2)                     # 低温度,政策问答要稳不要浪
    return r.choices[0].message.content

gr.ChatInterface(
    answer,
    title="招生政策问答助手",
    description="基于官方招生政策文件回答，条文原文可溯源。",
).launch(inbrowser=True)
