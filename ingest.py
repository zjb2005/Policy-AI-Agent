# ingest.py —— LangChain版建库: 批量读取多地区政策文档 → 切块 → 存入Chroma向量数据库
# 文档目录结构(地区名做文件夹名,自动成为元数据标签):
#   docs/北京/招生政策2026.docx
#   docs/上海/普高招生办法.pdf
#   docs/广东/...
# 用法: python ingest.py
import os
import glob
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "docs"
DB_DIR = "chroma_db"

# ---------- 1. 批量加载: 文件夹名 = 地区标签 ----------
docs = []
for region in sorted(os.listdir(DOCS_DIR)):
    rdir = os.path.join(DOCS_DIR, region)
    if not os.path.isdir(rdir):
        continue
    for path in glob.glob(os.path.join(rdir, "*")):
        if path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            continue
        for d in loader.load():
            d.metadata = {"region": region, "source": os.path.basename(path)}
            docs.append(d)
    print(f"[{region}] 已加载")

# ---------- 2. 切块: 优先按"第X条/章"切,兜底按段落/句子 ----------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=50,
    separators=[r"(?=第[一二三四五六七八九十百\d]+[条章节])", "\n\n", "\n", "。"],
    is_separator_regex=True)
chunks = splitter.split_documents(docs)
print(f"共 {len(docs)} 页文档，切出 {len(chunks)} 块")

# ---------- 3. 向量化并持久化到 Chroma(真正的向量数据库,存磁盘) ----------
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",
                            encode_kwargs={"normalize_embeddings": True})
Chroma.from_documents(chunks, emb, persist_directory=DB_DIR)
print(f"完成！向量数据库已存入 ./{DB_DIR}，运行 python qa_app.py 启动问答")
