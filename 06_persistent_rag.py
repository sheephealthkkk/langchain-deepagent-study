"""RAG 完整流程 — 第四步：SQLite 持久化对话式 RAG

与 05 的区别：
  05: RunnableWithMessageHistory + InMemoryChatMessageHistory（内存，重启丢失）
  06: SQLAlchemy + SQLite（持久化，重启保留）

架构：
┌─ 持久化存储层（SQLAlchemy ORM）──────────────────────┐
│                                                      │
│  SQLite 数据库  ←→  SessionModel（会话表）             │
│                        MessageModel（消息表，外键关联） │
│                                                      │
├─ 自定义 ChatHistory 适配层 ──────────────────────────┤
│                                                      │
│  SQLiteChatMessageHistory(BaseChatMessageHistory)    │
│     ┌─ add_message()   → 写入 SQLite                  │
│     ├─ add_messages()  → 批量写入                     │
│     ├─ clear()         → 删除该会话所有消息             │
│     └─ messages 属性   → 从 SQLite 读取               │
│                                                      │
├─ RAG Chain 层（与 05 相同）───────────────────────────┤
│                                                      │
│  history_aware_retriever → qa_chain → rag_chain      │
│                                                      │
└──────────────────────────────────────────────────────┘
"""
import sys
import os
from datetime import datetime, timezone
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
# 3. SQLAlchemy ORM 模型定义
# ============================================================
from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, ForeignKey, Integer,
)
from sqlalchemy.orm import (
    declarative_base, Session as DBSession, relationship,
)

SQLITE_PATH = "sqlite:///./chat_history.db"
engine = create_engine(SQLITE_PATH, echo=False)  # echo=True 可看 SQL 日志

# Base = 所有 ORM 模型的基类
Base = declarative_base()

# --- 会话表 ---
class SessionModel(Base):
    """会话表：一个 session_id 对应一个会话记录。

    只存元数据（创建/更新时间、标题），具体消息在 MessageModel 中。
    """
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(256), default="New Chat")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # relationship：一个 Session 对应多条 Message
    messages = relationship(
        "MessageModel",
        back_populates="session",
        cascade="all, delete-orphan",  # 删会话时级联删所有消息
    )

    def __repr__(self):
        return f"<Session(session_id='{self.session_id}', title='{self.title}')>"


# --- 消息表 ---
class MessageModel(Base):
    """消息表：一条消息对应一次 Human 或 AI 的发言。

    通过 session_id 外键关联到 SessionModel。
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(64),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(16), nullable=False)   # "human" 或 "ai"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # relationship：每条 Message 属于一个 Session
    session = relationship("SessionModel", back_populates="messages")

    def __repr__(self):
        preview = self.content[:40] if self.content else ""
        return f"<Message(role='{self.role}', content='{preview}...')>"


# 创建所有表（如果不存在）
Base.metadata.create_all(engine)

print("✅ SQLite 数据库表已就绪\n")


# ============================================================
# 4. 自定义 ChatHistory（基于 SQLite）
# ============================================================
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """基于 SQLite 的聊天历史存储。

    实现 BaseChatMessageHistory 接口，可以被 RunnableWithMessageHistory 使用。
    每次调用都会从 SQLite 读/写，实现真正的持久化。
    """

    def __init__(self, session_id: str, db_session: DBSession):
        self._session_id = session_id
        self._db = db_session

        # 确保会话记录存在
        existing = self._db.query(SessionModel).filter_by(
            session_id=session_id
        ).first()
        if not existing:
            self._db.add(SessionModel(session_id=session_id))
            self._db.commit()

    @property
    def messages(self) -> list[BaseMessage]:
        """从 SQLite 读取该会话的所有消息，按时间排序。"""
        rows = (
            self._db.query(MessageModel)
            .filter_by(session_id=self._session_id)
            .order_by(MessageModel.created_at.asc())
            .all()
        )
        result: list[BaseMessage] = []
        for row in rows:
            if row.role == "human":
                result.append(HumanMessage(content=row.content))
            elif row.role == "ai":
                result.append(AIMessage(content=row.content))
        return result

    def add_message(self, message: BaseMessage) -> None:
        """将一条消息写入 SQLite。"""
        role = "human" if isinstance(message, HumanMessage) else "ai"
        db_msg = MessageModel(
            session_id=self._session_id,
            role=role,
            content=message.content,
        )
        self._db.add(db_msg)
        # 同时更新会话的 updated_at
        (
            self._db.query(SessionModel)
            .filter_by(session_id=self._session_id)
            .update({"updated_at": datetime.now(timezone.utc)})
        )
        self._db.commit()

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """批量写入消息。"""
        for msg in messages:
            self.add_message(msg)

    def clear(self) -> None:
        """清空该会话的所有消息。"""
        (
            self._db.query(MessageModel)
            .filter_by(session_id=self._session_id)
            .delete()
        )
        self._db.commit()


print("✅ SQLiteChatMessageHistory 已就绪\n")

# ============================================================
# 5. 会话历史工厂（每次调用创建一个新的 DB session）
# ============================================================
def get_session_history(session_id: str) -> SQLiteChatMessageHistory:
    """RunnableWithMessageHistory 的回调：根据 session_id 返回历史对象。

    每次都创建新的 DB session，保证线程安全。
    """
    db = DBSession(engine)
    return SQLiteChatMessageHistory(session_id=session_id, db_session=db)


# ============================================================
# 6. 子链1：历史感知检索器（与 05 相同）
# ============================================================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given the chat history and the latest user question, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed. If the question is already "
        "standalone, return it as is. Respond in the SAME LANGUAGE "
        "as the original question."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=contextualize_prompt,
)

# ============================================================
# 7. 子链2：QA 链（与 05 相同）
# ============================================================
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledgeable AI assistant. Answer the user's question "
        "based STRICTLY on the following retrieved context.\n\n"
        "Rules:\n"
        "1. If the context contains the answer, answer based on it and cite sources.\n"
        "2. If it does NOT, say: '当前知识库中没有相关信息。'\n"
        "3. Keep answers concise. Respond in the SAME LANGUAGE as the user.\n\n"
        "## Retrieved Context\n"
        "{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(
    llm=llm, prompt=qa_prompt, document_variable_name="context",
)

# ============================================================
# 8. 组装主 RAG Chain + 包装历史管理（与 05 相同）
# ============================================================
from langchain_classic.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=qa_chain,
)

conversational_rag_chain = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ============================================================
# 9. 多轮对话演示 + 会话管理 API
# ============================================================
print("=" * 60)
print("🤖 SQLite 持久化 RAG 就绪！\n")

# --- 9a. 列出已有会话 ---
def list_sessions():
    """列出 SQLite 中所有会话。"""
    db = DBSession(engine)
    sessions = db.query(SessionModel).order_by(SessionModel.updated_at.desc()).all()
    db.close()
    return sessions


print("📋 当前已有会话：")
sessions = list_sessions()
for s in sessions:
    print(f"   session_id={s.session_id}  title={s.title}  updated={s.updated_at}")
if not sessions:
    print("   (空)")

# --- 9b. 多轮对话 ---
from langchain_core.messages import HumanMessage, AIMessage

demo = [
    ("session_2", "What is LangChain?"),
    ("session_2", "How is it different from LangGraph?"),
    ("session_2", "Summarize what we just discussed."),
    # 新会话，验证隔离
    ("session_3", "What is RAG?"),
]

for session_id, question in demo:
    print(f"\n👤 [{session_id}] 用户: {question}")
    print("⏳ 思考中...")

    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )

    print(f"🤖 助手: {response['answer'][:200]}")
    print(f"   📎 检索到 {len(response.get('context', []))} 个文档块")

# --- 9c. 验证持久化 ---
print(f"\n{'=' * 60}")
print("🔍 验证持久化：重新查询 SQLite 中的会话和消息")

db = DBSession(engine)
all_sessions = db.query(SessionModel).order_by(SessionModel.updated_at.desc()).all()
for s in all_sessions:
    msg_count = (
        db.query(MessageModel).filter_by(session_id=s.session_id).count()
    )
    print(f"\n📁 会话: {s.session_id}")
    print(f"   标题: {s.title}")
    print(f"   消息数: {msg_count}")
    print(f"   创建时间: {s.created_at}")
    print(f"   更新时间: {s.updated_at}")

    # 显示最后 2 条消息
    msgs = (
        db.query(MessageModel)
        .filter_by(session_id=s.session_id)
        .order_by(MessageModel.created_at.asc())
        .limit(2)
        .all()
    )
    for m in msgs:
        print(f"     [{m.role}] {m.content[:80]}...")
db.close()

print(f"\n✅ 持久化验证完成！数据库文件: chat_history.db")
print("   重启程序后会话历史仍然保留。")
