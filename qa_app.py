# qa_app.py —— LangChain版多地区招生政策问答Agent(带地区过滤)网页版
# 先运行 ingest.py 建库，再: python qa_app.py
import os
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------- 1. 连接向量数据库 ----------
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",
                            encode_kwargs={"normalize_embeddings": True})
db = Chroma(persist_directory="chroma_db", embedding_function=emb)
regions = sorted({m["region"] for m in db.get()["metadatas"]})
print(f"数据库就绪，覆盖地区: {regions}")

# ---------- 2. 大模型(DeepSeek) ----------
llm = ChatOpenAI(model="deepseek-chat",
                 api_key=os.environ["DEEPSEEK_API_KEY"],
                 base_url="https://api.deepseek.com",
                 temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是{region}招生政策问答助手。严格根据提供的政策条文回答：
1. 只依据条文作答，不得编造；数字、日期、条件必须与原文一致
2. 回答末尾注明依据的条文关键句和出处文件名
3. 条文中找不到答案时，明确说：该地区政策文件中未找到相关规定，建议咨询当地招生办"""),
    ("user", "政策条文：\n{context}\n\n问题：{question}"),
])
chain = prompt | llm   # LangChain 管道: 提示词模板 → 大模型

# ---------- 3. 问答: 按所选地区过滤检索 ----------
def answer(message, history, region):
    retriever = db.as_retriever(
        search_kwargs={"k": 4, "filter": {"region": region}})  # 关键:地区过滤
    hits = retriever.invoke(message)
    if not hits:
        return "该地区政策文件中未找到相关条文，建议咨询当地招生办公室。"
    context = "\n\n".join(
        f"【{d.metadata['source']}】{d.page_content}" for d in hits)
    r = chain.invoke({"region": region, "context": context, "question": message})
    return r.content

gr.ChatInterface(
    answer,
    additional_inputs=[gr.Dropdown(regions, value=regions[0] if regions else None,
                                   label="选择地区")],
    title="多地区招生政策问答助手",
    description="先在下方选择地区，回答严格基于该地区官方政策文件。",
).launch(inbrowser=True)
