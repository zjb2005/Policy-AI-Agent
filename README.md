# Policy-AI-Agent
Policy AI Agent made for answer questions about policy
为什么做这个项目

每年招生季，家长们都在重复同一件事：翻十几页的政策文件，找"随迁子女怎么报名""学区怎么划分"这几行字。政策原文写得严谨，但没人有耐心从第一条读到最后一条；打招生办电话，占线；问群里的其他家长，得到三个互相矛盾的答案。

更麻烦的是，招生政策一县一策。新干县的规定搬到浮梁县就不适用，网上搜到的"经验帖"往往张冠李戴，误导比不知道更糟。

这个项目把各地招生政策文件变成一个可以直接提问的助手：家长用一句大白话提问，它在所选地区的官方文件里找到对应条文，用平实的语言解释清楚，并附上条文原文出处。查不到的，它会明确说"文件里没有这条，请咨询招生办"，而不是编一个听起来合理的答案。

核心功能
多地区隔离检索：每份文件按地区打标签，提问前先选地区，北京的政策绝不会混进上海的回答里。这是同类工具最常见的翻车点，也是本项目的设计重点。
按条款切分：政策文档按"第X条 / 第X章"的天然结构切块，保证每条规定完整入库，不会被拦腰截断。
回答可溯源：每个回答末尾标注依据的条文关键句和来源文件名，方便家长核对原文。
拒绝编造：检索不到相关条文时直接说明，并引导用户咨询当地招生办。涉及日期、分数、名额的数字要求与原文严格一致。
网页界面：基于 Gradio 的聊天页面，带地区下拉框，手机浏览器也能正常使用。
知识库热更新：政策换新版？替换文件重跑一条命令即可，无需重新训练任何模型。
技术架构
Word/PDF 政策文件
     │  ingest.py: 解析 → 按条款切块 → 打地区标签
     ▼
Chroma 向量数据库（bge-small-zh-v1.5 向量化，本地持久化）
     │  按地区过滤，检索 Top-4 相关条文
     ▼
DeepSeek API（低温度生成，强约束提示词）
     │
     ▼
Gradio 网页界面

技术选型的考量：向量化在本地完成，政策文件不出机器；生成走 API，保证回答质量的同时省去 GPU 成本；LangChain 负责文档加载和检索管道，地区过滤用 Chroma 的元数据机制实现。

快速开始
bash
# 1. 安装依赖
pip install langchain langchain-community langchain-huggingface \
    langchain-chroma langchain-openai docx2txt pypdf \
    sentence-transformers gradio

# 2. 按地区放置政策文件（文件夹名 = 地区名）
docs/
├── 新干县/2026年考试招生工作实施意见.docx
├── 浮梁县/2026年城区义务教育学校招生工作方案.docx
└── 月湖区/义务教育学校招生入学工作的通知.docx

# 3. 建库
python ingest.py

# 4. 配置 API Key 并启动
export DEEPSEEK_API_KEY="sk-xxxx"
python qa_app.py

浏览器会自动打开问答页面。新增地区时，在 docs/ 下建对应文件夹放入文件，重跑 ingest.py 即可。

项目结构
文件	作用
ingest.py	建库脚本：批量解析文档、切块、向量化、写入 Chroma
qa_app.py	问答服务：地区过滤检索 + 大模型生成 + 网页界面
docs/	政策原始文件，按地区分文件夹
chroma_db/	向量数据库（运行 ingest.py 后生成）
已知边界
回答质量受政策原文清晰度限制，原文本身含糊的条款，助手会照实呈现而非替官方解释
本工具输出仅供参考，正式报名事项请以当地招生办答复为准
扫描版 PDF（图片型）暂不支持，需先做 OCR
路线图
 重排序（bge-reranker）提升检索精度
 BM25 + 向量混合检索，强化对"分数线""摇号"等术语的精确匹配
 跨地区政策对比问答
 增量更新知识库，免全量重建
English Documentation
Why This Project Exists

Every admissions season, parents repeat the same ritual: digging through a dozen pages of policy documents for the few lines that answer "how do migrant children register" or "which school district do we belong to." The official documents are rigorous but nobody reads them cover to cover. The admissions office phone line is busy. Parent group chats offer three contradictory answers to the same question.

It gets worse: admissions policies differ county by county. What holds in Xingan County does not apply in Fuliang County, and "experience posts" found online routinely mix regions up. Wrong information is worse than no information.

This project turns regional admissions policy documents into an assistant you can simply ask. A parent types a question in plain language; the assistant retrieves the relevant clauses from the official documents of the selected region, explains them in accessible terms, and cites the source text. When the documents contain no answer, it says so explicitly and directs users to the local admissions office, instead of fabricating something plausible.

Features
Region-isolated retrieval. Every document carries a region tag, and users pick a region before asking. Beijing's policy never leaks into an answer about Shanghai. Cross-region contamination is the most common failure mode for tools of this kind, and it is the central design concern here.
Clause-aware chunking. Documents are split along their natural "Article X / Chapter X" structure, so no regulation gets cut in half.
Traceable answers. Each answer ends with the key sentences of the supporting clauses and the source filename, so parents can verify against the original.
No fabrication. If retrieval finds nothing relevant, the assistant says so. Dates, scores, and quotas must match the original text exactly.
Web interface. A Gradio chat page with a region dropdown, usable from a phone browser.
Hot-swappable knowledge base. Policy updated? Replace the file, rerun one command. No model retraining involved.
Architecture
Word/PDF policy documents
     │  ingest.py: parse → clause-aware chunking → region tagging
     ▼
Chroma vector store (embedded with bge-small-zh-v1.5, persisted locally)
     │  region-filtered top-4 retrieval
     ▼
DeepSeek API (low temperature, strictly constrained prompt)
     │
     ▼
Gradio web UI

Design rationale: embedding runs locally so policy files never leave the machine; generation goes through an API for answer quality without GPU costs; LangChain handles document loading and the retrieval pipeline, with region filtering built on Chroma's metadata mechanism.

Quick Start
bash
# 1. Install dependencies
pip install langchain langchain-community langchain-huggingface \
    langchain-chroma langchain-openai docx2txt pypdf \
    sentence-transformers gradio

# 2. Organize documents by region (folder name = region tag)
docs/
├── Xingan/admissions-plan-2026.docx
├── Fuliang/admissions-plan-2026.docx
└── Yuehu/admissions-notice-2026.docx

# 3. Build the knowledge base
python ingest.py

# 4. Set the API key and launch
export DEEPSEEK_API_KEY="sk-xxxx"
python qa_app.py

The chat page opens automatically in your browser. To add a region, create its folder under docs/, drop the files in, and rerun ingest.py.

Project Layout
File	Purpose
ingest.py	Ingestion: batch parsing, chunking, embedding, writing to Chroma
qa_app.py	QA service: region-filtered retrieval + LLM generation + web UI
docs/	Source policy files, one folder per region
chroma_db/	Vector store (generated by ingest.py)
Known Limitations
Answer quality is bounded by the clarity of the source text; where the policy itself is ambiguous, the assistant presents it as-is rather than interpreting on behalf of officials
Output is for reference only; final decisions rest with the local admissions office
Scanned (image-based) PDFs require OCR preprocessing and are not yet supported
Roadmap
 Reranking with bge-reranker for higher retrieval precision
 Hybrid BM25 + vector retrieval for exact-term matching ("cutoff score", "lottery")
 Cross-region policy comparison
 Incremental knowledge base updates

 If this project helps you, a star would be appreciated.
