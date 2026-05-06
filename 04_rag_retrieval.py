"""RAG 完整流程 — 第二步：检索 + 增强生成（Retrieval & Generation）
工作流程：
  用户提问 → 向量化问题 → Chroma 相似度检索 → 拼接上下文
  → 填充 Prompt 模板 → LLM 生成回答
"""
import sys
import os
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ============================================================
# 1. 初始化大模型
# ============================================================
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.8,
)

# ============================================================
# 2. 加载已有的向量库
# ============================================================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "./chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name="langchain_docs",
)

print(f"✅ 已加载向量库，共 {vectorstore._collection.count()} 条记录\n")

# ============================================================
# 3. 创建 Retriever（检索器）
# ============================================================
# Retriever 是 LangChain 的检索抽象，统一了各种检索方式的接口
# as_retriever() 把 Chroma 包装成标准的 Retriever 对象
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 相似度检索
    search_kwargs={"k": 4},    # 返回最相似的 4 个文档块
)

# ============================================================
# 4. 构建增强提示词模板
# ============================================================
from langchain_core.prompts import ChatPromptTemplate

# {context} 位置会被检索结果自动填充
# {question} 位置会被用户问题自动填充
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一个知识渊博的 AI 助手。请严格根据以下「参考资料」回答用户的问题。\n\n"
        "## 规则\n"
        "1. 如果参考资料中包含答案，请基于资料内容回答，并注明引用来源。\n"
        "2. 如果参考资料不包含答案，请明确告知用户「当前知识库中没有相关信息」。\n"
        "3. 回答要简洁、准确，用中文回复。\n\n"
        "## 参考资料\n"
        "{context}"
    )),
    ("user", "{question}"),
])

print("📋 提示词模板（system 部分）：")
print(RAG_PROMPT.messages[0].prompt.template[:200], "...\n")

# ============================================================
# 5. 构建完整 RAG Chain（LCEL）
# ============================================================
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document


# --- 辅助函数：将检索到的 Document 列表拼接为单一上下文字符串 ---
def format_docs(docs: list[Document]) -> str:
    """把检索到的文档块格式化为带序号的上下文文本。"""
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[来源{i+1}] {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# --- LCEL Chain ---
#  用户输入 question
#       │
#       ├──→ retriever  ──→  format_docs  ──→ 填入 {context}
#       │
#       └──→ RunnablePassthrough  ──────────→ 填入 {question}
#                            │
#                       RAG_PROMPT
#                            │
#                          llm
#                            │
#                     StrOutputParser
#                            │
#                       最终回答
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

print("=" * 60)

# ============================================================
# 6. 交互式问答
# ============================================================
print("\n🤖 RAG 问答就绪！输入问题开始（输入 quit 退出）\n")

questions = [
    "What is LangChain?",
    "LangChain 和 LangGraph 有什么区别？",
]

for question in questions:
    print(f"👤 用户: {question}")
    print("⏳ 检索中...")

    # invoke 触发整条 Chain：
    #   question → retriever 检索 → format_docs 格式化
    #   → 填入 RAG_PROMPT → llm 生成 → StrOutputParser 输出
    answer = rag_chain.invoke(question)

    print(f"🤖 助手: {answer}")
    print("\n" + "-" * 60 + "\n")
