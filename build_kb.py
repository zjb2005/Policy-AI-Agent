# build_kb.py —— 把招生政策 Word/PDF 解析、切块、向量化，建成知识库
# 用法: python build_kb.py 政策文件.docx   (或 .pdf，可一次传多个文件)
# 产物: kb.json(条文文本) + kb_emb.npy(对应向量)，供 policy_app.py 使用
import sys
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- 1. 读取文件文本 ----------
def read_docx(path):
    from docx import Document
    return "\n".join(p.text for p in Document(path).paragraphs if p.text.strip())

def read_pdf(path):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

# ---------- 2. 切块：优先按"第X条/第X章"切，否则按段落合并 ----------
def split_chunks(text, max_len=400):
    parts = re.split(r"(?=第[一二三四五六七八九十百\d]+[条章节])", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 3:                      # 文档没有条款结构 → 按段落合并
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        parts, buf = [], ""
        for p in paras:
            if len(buf) + len(p) > max_len:
                parts.append(buf)
                buf = p
            else:
                buf += "\n" + p
        if buf:
            parts.append(buf)
    chunks = []
    for p in parts:                          # 超长条文再二次切分
        while len(p) > max_len:
            chunks.append(p[:max_len])
            p = p[max_len - 50:]             # 保留50字重叠，防止语义断裂
        chunks.append(p)
    return [c for c in chunks if len(c) > 20]

# ---------- 3. 主流程 ----------
if len(sys.argv) < 2:
    sys.exit("用法: python build_kb.py 文件1.docx [文件2.pdf ...]")

all_chunks = []
for path in sys.argv[1:]:
    print(f"解析 {path} ...")
    text = read_docx(path) if path.endswith(".docx") else read_pdf(path)
    cs = split_chunks(text)
    all_chunks += [{"source": path, "text": c} for c in cs]
    print(f"  切出 {len(cs)} 块")

print("向量化中...")
emb_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
emb = emb_model.encode([c["text"] for c in all_chunks],
                       normalize_embeddings=True, show_progress_bar=True)

json.dump(all_chunks, open("kb.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
np.save("kb_emb.npy", emb)
print(f"完成！知识库共 {len(all_chunks)} 块，已保存 kb.json + kb_emb.npy")
