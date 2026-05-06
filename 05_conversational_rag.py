"""RAG 完整流程 — 第三步：带记忆的对话式 RAG（Conversational RAG）

架构（链中链）：
┌─ RunnableWithMessageHistory（最外层，管理会话历史）──────────────┐
│  ┌─ 主 RAG Chain ─────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  ┌─ 子链1: 历史感知检索器 ──────────────────────────────┐  │ │
│  │  │  chat_history + 当前问题                                │  │ │
│  │  │       ↓                                                │  │ │
│  │  │  contextualize_prompt (含 MessagesPlaceholder)         │  │ │
│  │  │       ↓                                                │  │ │
│  │  │  LLM 改写为独立问题 → retriever 检索                  │  │ │
│  │  └───────────────────────────────────────────────────────┘  │ │
│  │                         ↓                                   │ │
│  │              检索到的相关文档                                │ │
│  │                         ↓                                   │ │
│  │  ┌─ 子链2: QA 链（create_stuff_documents_chain）──┐       │ │
│  │  │  qa_prompt (含 context + chat_history + input)         │ │
│  │  │       ↓                                                │ │
│  │  │  LLM 基于文档 + 历史 → 生成回答                       │ │
│  │  └───────────────────────────────────────────────────────┘  │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
"""
import sys
import os
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

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
# 2. 加载向量库 + Retriever
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

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

print(f"✅ 向量库已加载，共 {vectorstore._collection.count()} 条记录\n")

# ============================================================
# 3. 子链1：历史感知检索器（History-Aware Retriever）
# ============================================================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever

# --- 上下文改写 Prompt ---
# MessagesPlaceholder("chat_history") 是关键：
# - 有历史时：展开为之前的 HumanMessage / AIMessage 对
# - 无历史时：展开为空，什么都不插入
# 这样同一个模板就能处理两种场景
contextualize_system = (
    "Given the chat history and the latest user question, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed. If the question is already "
    "standalone, return it as is. Respond in the SAME LANGUAGE "
    "as the original question."
)

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_system),
    MessagesPlaceholder("chat_history"),   # ← 历史消息占位符
    ("human", "{input}"),                  # ← 当前用户问题
])

print("📋 上下文改写 Prompt（system 部分）：")
print(contextualize_system[:150], "...\n")

# create_history_aware_retriever 内部流程：
#   1. 将 chat_history + input 填入 contextualize_prompt
#   2. 用 llm 生成一个独立的、去上下文的检索查询
#   3. 用这个查询去调用 retriever
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_prompt,
)

# ============================================================
# 4. 子链2：QA 链（Stuff Documents Chain）
# ============================================================
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- QA Prompt ---
# {context} 由 create_stuff_documents_chain 自动填充（检索到的文档）
# MessagesPlaceholder("chat_history") 同上，有历史展开，无历史为空
# {input} 是用户的原始问题
qa_system = (
    "You are a knowledgeable AI assistant. Answer the user's question "
    "based STRICTLY on the following retrieved context.\n\n"
    "Rules:\n"
    "1. If the context contains the answer, answer based on it and cite sources.\n"
    "2. If the context does NOT contain the answer, say: '当前知识库中没有相关信息。'\n"
    "3. Keep answers concise and accurate.\n"
    "4. Respond in the SAME LANGUAGE as the user's question.\n\n"
    "## Retrieved Context\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system),
    MessagesPlaceholder("chat_history"),   # ← 历史消息占位符
    ("human", "{input}"),                  # ← 当前用户问题
])

print("📋 QA Prompt（system 部分）：")
print(qa_system[:150], "...\n")

# create_stuff_documents_chain 做三件事：
#   1. 把检索到的 Document 列表拼接为字符串 → 填入 {context}
#   2. 把 chat_history + input 填入模板
#   3. 调用 llm 生成回答
qa_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt,
    document_variable_name="context",   # 模板中上下文变量名
)

# ============================================================
# 5. 组装主 RAG Chain
# ============================================================
from langchain_classic.chains import create_retrieval_chain

# create_retrieval_chain 把检索器和 QA 链串起来：
#   input → history_aware_retriever → qa_chain → output
rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=qa_chain,
)

# ============================================================
# 6. 包装会话记忆（RunnableWithMessageHistory）
# ============================================================
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 用字典管理多个会话的历史（key = session_id）
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """根据 session_id 获取或创建对应的聊天历史。

    这个函数被 RunnableWithMessageHistory 在每次调用时触发：
    - 调用前：从 store 取出历史，注入到 chain 的 chat_history 占位符
    - 调用后：将本轮对话（用户问题 + AI 回答）追加到历史
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_session_history,
    input_messages_key="input",           # 用户输入从哪个 key 取
    history_messages_key="chat_history",  # 历史消息注入到模板的哪个 key
    output_messages_key="answer",         # 输出回答的 key（用于追加到历史）
)

print("=" * 60)
print("🤖 对话式 RAG 就绪！支持多轮对话，输入 quit 退出\n")

# ============================================================
# 7. 多轮对话演示
# ============================================================
demo_conversations = [
    # 第一轮：无历史，直接问答
    ("session_1", "What is LangChain?"),
    # 第二轮：有历史，代词 "it" 需要结合上文理解
    ("session_1", "How is it different from LangGraph?"),
    # 第三轮：验证记忆是否持续
    ("session_1", "Summarize what we just discussed about."),
]

for session_id, question in demo_conversations:
    print(f"👤 [{session_id}] 用户: {question}")
    print("⏳ 思考中...")

    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )

    # response 是 dict：{"input": ..., "answer": ..., "context": [...]}
    print(f"🤖 助手: {response['answer']}")
    print(f"   📎 检索到 {len(response.get('context', []))} 个文档块")
    print("\n" + "-" * 60 + "\n")
