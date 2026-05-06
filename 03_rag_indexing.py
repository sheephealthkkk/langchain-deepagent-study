"""RAG 完整流程 — 第一步：索引（Indexing）
工作流程：
  网页 → WebBaseLoader 加载 → RecursiveCharacterTextSplitter 切分
  → HuggingFaceEmbeddings 向量化 → Chroma 向量库存储
"""
import sys
import os
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# 设置 HuggingFace 镜像（国内加速），必须在导入 embeddings 之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ============================================================
# 1. 初始化大模型（temperature=0.8）
# ============================================================
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.8,  # 控制随机性：0=确定，1=最大随机
)

# ============================================================
# 2. 加载网页内容（WebBaseLoader + BS4）
# ============================================================
from langchain_community.document_loaders import WebBaseLoader

# 要抓取的网页
WEB_URL = "https://docs.langchain.com/oss/python/langchain/overview"

loader = WebBaseLoader(
    web_path=WEB_URL,
    # --- BS4 解析参数 ---
    default_parser="lxml",                # 用 lxml 解析 HTML，速度比 html.parser 快
    bs_get_text_kwargs={
        "separator": "\n",                # 元素间用换行分隔
        "strip": True,                    # 去除首尾空白
    },
    # --- 请求参数 ---
    header_template={
        "User-Agent": "Mozilla/5.0 (compatible; LangChain-Bot/1.0)",
    },
    requests_per_second=2,                # 礼貌爬取，每秒最多2次
    continue_on_failure=False,            # 失败立即报错
)

print(f"⏳ 正在加载网页: {WEB_URL}")
docs = loader.load()
print(f"✅ 加载完成，共 {len(docs)} 篇文档")
print(f"   前100字预览:\n{docs[0].page_content[:100]}...\n")

# ============================================================
# 3. 文档切分（RecursiveCharacterTextSplitter）
# ============================================================
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # 每块最大 500 字符
    chunk_overlap=100,      # 相邻块重叠 100 字符，保证语义连贯
    separators=[            # 按优先级依次尝试切分
        "\n\n",             # 优先按段落（双换行）
        "\n",               # 其次按单行
        "。",               # 中文句号
        ".",                # 英文句号
        " ",
        "",
    ],
)

chunks = text_splitter.split_documents(docs)
print(f"📦 切分完成，共 {len(chunks)} 个文本块")
print(f"   第一块长度: {len(chunks[0].page_content)} 字符")
print(f"   第二块长度: {len(chunks[1].page_content)} 字符\n")

# ============================================================
# 4. 向量化 + 存入向量库（Embedding & Vector Store）
# ============================================================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 嵌入模型（把文本变成向量）
# 使用本地 HuggingFace 模型，无需外部 API，完全离线运行
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",  # BGE 中文小模型
    model_kwargs={"device": "cpu"},       # 用 CPU 运行
    encode_kwargs={"normalize_embeddings": True},  # 归一化，提升检索精度
)

# 存放向量库的本地路径
CHROMA_PATH = "./chroma_db"

print("⏳ 正在生成向量并存入 Chroma...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH,  # 持久化到磁盘
    collection_name="langchain_docs",
)
print(f"✅ 向量库已存储到 '{CHROMA_PATH}'")
print(f"   集合名称: langchain_docs")
print(f"   文档数量: {vectorstore._collection.count()}\n")

# ============================================================
# 5. 验证：做一个简单的相似度检索
# ============================================================
query = "What is LangChain?"
print(f"🔍 检索测试: \"{query}\"")
retrieved = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(retrieved):
    print(f"\n--- 结果 {i+1} (source: {doc.metadata.get('source', 'N/A')}) ---")
    print(doc.page_content[:200])
