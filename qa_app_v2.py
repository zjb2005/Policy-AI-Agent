# qa_app_v2.py —— 支持上传照片/文档的多地区招生政策问答助手
# 新增: 聊天框可上传 docx/pdf/图片，助手结合上传内容+政策库回答
# 需要: export DEEPSEEK_API_KEY=...  和  export DASHSCOPE_API_KEY=...(读图用,阿里云DashScope开通)
import os
import base64
import gradio as gr
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------- 向量库 ----------
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",
                            encode_kwargs={"normalize_embeddings": True})
db = Chroma(persist_directory="chroma_db", embedding_function=emb)
regions = sorted({m["region"] for m in db.get()["metadatas"]})
print(f"数据库就绪，覆盖地区: {regions}")

# ---------- 文本大模型(DeepSeek) ----------
llm = ChatOpenAI(model="deepseek-chat",
                 api_key=os.environ["DEEPSEEK_API_KEY"],
                 base_url="https://api.deepseek.com", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是{region}招生政策问答助手。严格根据提供的政策条文和用户上传的材料回答：
1. 只依据条文和材料作答，不得编造；数字、日期、条件必须与原文一致
2. 回答末尾注明依据的条文关键句和出处
3. 找不到答案时明确说：未找到相关规定，建议咨询当地招生办"""),
    ("user", "政策条文：\n{context}\n\n用户上传材料：\n{uploads}\n\n问题：{question}"),
])
chain = prompt | llm

# ---------- 视觉模型(qwen-vl, 用于读图) ----------
vl_client = OpenAI(api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
                   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def read_image(path):
    """把图片交给视觉模型，转写图中文字和关键信息"""
    if not os.environ.get("DASHSCOPE_API_KEY"):
        return "[未配置DASHSCOPE_API_KEY，无法识别图片]"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = path.rsplit(".", 1)[-1].lower()
    r = vl_client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
            {"type": "text",
             "text": "请完整转写图片中的所有文字，并简述图片内容。"}]}])
    return r.choices[0].message.content

def read_file(path):
    """按类型解析上传文件，统一返回文本"""
    p = path.lower()
    if p.endswith(".docx"):
        import docx2txt
        return docx2txt.process(path)
    if p.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(pg.extract_text() or "" for pg in PdfReader(path).pages)
    if p.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return read_image(path)
    if p.endswith((".xlsx", ".xls", ".csv")):
        import pandas as pd
        sheets = pd.read_excel(path, sheet_name=None) if not p.endswith(".csv") \
                 else {"Sheet1": pd.read_csv(path)}
        out = []
        for name, df in sheets.items():
            out.append(f"工作表【{name}】共{len(df)}行:\n{df.to_markdown(index=False)}")
        return "\n\n".join(out)
    if p.endswith(".txt"):
        return open(path, encoding="utf-8", errors="ignore").read()
    return f"[暂不支持的文件类型: {os.path.basename(path)}]"

# ---------- 问答主逻辑 ----------
def answer(message, history, region):
    question = message["text"]
    files = message.get("files", [])

    uploads = "（无）"
    if files:
        parts = [f"【{os.path.basename(f)}】\n{read_file(f)[:3000]}" for f in files]
        uploads = "\n\n".join(parts)

    retriever = db.as_retriever(
        search_kwargs={"k": 4, "filter": {"region": region}})
    hits = retriever.invoke(question if question else uploads[:500])
    context = "\n\n".join(
        f"【{d.metadata['source']}】{d.page_content}" for d in hits) or "（无匹配条文）"

    r = chain.invoke({"region": region, "context": context,
                      "uploads": uploads, "question": question or "请分析上传的材料"})
    return r.content

gr.ChatInterface(
    answer,
    multimodal=True,          # 关键: 输入框出现附件按钮，可传图片/文档
    additional_inputs=[gr.Dropdown(regions, value=regions[0] if regions else None,
                                   label="选择地区")],
    title="招生政策问答助手（支持上传照片/文档）",
    description="可直接提问，也可上传通知截图、政策文件等材料一起询问。",
).launch(inbrowser=True)
