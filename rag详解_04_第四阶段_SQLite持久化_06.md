# 第四阶段：SQLite 持久化对话式 RAG（06_persistent_rag.py）

## 为什么需要持久化？

`05_conversational_rag.py` 使用 `InMemoryChatMessageHistory` 存对话历史：

```python
store: dict[str, InMemoryChatMessageHistory] = {}  # 就是个字典
```

问题：**程序退出 → 所有对话消失 → 重启后从零开始**。

`06` 改用 SQLite：**重启程序后，之前的对话历史仍然保留**。

---

## 架构对比

```
05 (内存模式)：                        06 (持久化模式)：

store = {}                            chat_history.db (SQLite 文件)
  ├─ session_1: [msg,msg,...]            ├─ sessions 表
  └─ session_2: [msg,msg,...]            │   ├─ session_2
       ↑                                 │   └─ session_3
       │                                 └─ messages 表
    重启丢失                                  ├─ (session_2, human, "What is...")
                                             ├─ (session_2, ai, "LangChain is...")
                                             ├─ (session_3, human, "What is RAG?")
                                             └─ ...
                                                  ↑
                                               重启保留
```

---

## 逐块详解

### 1. SQLAlchemy ORM 模型 — 数据库表设计

```python
from sqlalchemy.orm import declarative_base
Base = declarative_base()
```

#### `Base` 是什么？

`Base` 是所有 ORM 模型的**注册表**。任何继承 `Base` 的类会自动映射为数据库表。调用 `Base.metadata.create_all(engine)` 时，SQLAlchemy 遍历所有继承 `Base` 的类，在数据库中创建对应的表。

#### SessionModel — 会话表

```python
class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(256), default="New Chat")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=..., onupdate=...)

    messages = relationship("MessageModel", back_populates="session",
                            cascade="all, delete-orphan")
```

| 列 | 类型 | 作用 |
|---|---|---|
| `id` | Integer, PK | 自增主键，数据库内部用 |
| `session_id` | String, unique | 业务层的会话标识（如 "session_2"），带索引加速查询 |
| `title` | String | 会话标题，可用于 UI 展示 |
| `created_at` | DateTime | 创建时间 |
| `updated_at` | DateTime | 最后更新时间，`onupdate` 使得每次更新自动刷新 |

**`relationship` 的作用**：声明 `SessionModel` 和 `MessageModel` 的一对多关系。`cascade="all, delete-orphan"` 意味着删除会话时，其所有消息也级联删除。

#### MessageModel — 消息表

```python
class MessageModel(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64),
                        ForeignKey("sessions.session_id", ondelete="CASCADE"),
                        nullable=False, index=True)
    role = Column(String(16), nullable=False)    # "human" 或 "ai"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=...)

    session = relationship("SessionModel", back_populates="messages")
```

| 列 | 类型 | 作用 |
|---|---|---|
| `id` | Integer, PK | 自增主键 |
| `session_id` | String, FK | 外键，指向 `sessions.session_id` |
| `role` | String | `"human"` 或 `"ai"`，区分消息来源 |
| `content` | Text | 消息正文（无长度限制） |
| `created_at` | DateTime | 消息创建时间，用于排序 |

**`ForeignKey("sessions.session_id")`**：确保每条消息必须关联一个存在的会话。`ondelete="CASCADE"` 在数据库层面也做级联删除。

#### 两张表的 ER 关系

```
┌──────────────────────┐          ┌──────────────────────────┐
│      sessions        │          │        messages          │
├──────────────────────┤          ├──────────────────────────┤
│ id (PK)              │──┐       │ id (PK)                  │
│ session_id (UNIQUE)  │  │───→   │ session_id (FK)          │
│ title                │  │       │ role ("human" / "ai")    │
│ created_at           │  │       │ content (TEXT)           │
│ updated_at           │  │       │ created_at               │
└──────────────────────┘  │       └──────────────────────────┘
                          │
                    1 : N (一个会话有多条消息)
                    CASCADE DELETE (删会话 → 删所有消息)
```

---

### 2. SQLiteChatMessageHistory — 关键适配层

这是整个持久化方案的**核心桥梁**：继承了 `BaseChatMessageHistory`，但所有数据读写走 SQLite。

```python
class SQLiteChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, db_session: DBSession):
        ...
        # 确保会话在 sessions 表中存在
        existing = self._db.query(SessionModel).filter_by(
            session_id=session_id
        ).first()
        if not existing:
            self._db.add(SessionModel(session_id=session_id))
            self._db.commit()
```

#### `messages` 属性（读）

```python
@property
def messages(self) -> list[BaseMessage]:
    rows = (
        self._db.query(MessageModel)
        .filter_by(session_id=self._session_id)
        .order_by(MessageModel.created_at.asc())
        .all()
    )
    result = []
    for row in rows:
        if row.role == "human":
            result.append(HumanMessage(content=row.content))
        elif row.role == "ai":
            result.append(AIMessage(content=row.content))
    return result
```

**流程**：`SELECT * FROM messages WHERE session_id=? ORDER BY created_at` → 按时间排序 → 每条 `MessageModel` 转为 `HumanMessage` 或 `AIMessage`。

**调用时机**：每次 `RunnableWithMessageHistory` 注入历史时触发（即每轮对话前）。

#### `add_message`（写）

```python
def add_message(self, message: BaseMessage) -> None:
    role = "human" if isinstance(message, HumanMessage) else "ai"
    db_msg = MessageModel(
        session_id=self._session_id,
        role=role,
        content=message.content,
    )
    self._db.add(db_msg)
    self._db.query(SessionModel).filter_by(
        session_id=self._session_id
    ).update({"updated_at": datetime.now(timezone.utc)})
    self._db.commit()
```

**流程**：`INSERT INTO messages (...)` + `UPDATE sessions SET updated_at=...` → `COMMIT`。

**调用时机**：`RunnableWithMessageHistory` 在链执行完后，把本轮的用户输入和 AI 回答追加到历史时触发。

#### `clear`（删）

```python
def clear(self) -> None:
    self._db.query(MessageModel).filter_by(
        session_id=self._session_id
    ).delete()
    self._db.commit()
```

---

### 3. 与 05 的关键差异对比

#### 差异一：历史存储

| | `05` | `06` |
|---|---|---|
| 存储方式 | `dict[str, InMemoryChatMessageHistory]` | SQLite (`chat_history.db`) |
| 数据结构 | Python 字典 + 内存列表 | `sessions` 表 + `messages` 表 |
| 生命周期 | 随进程，重启丢失 | 持久化文件，重启保留 |
| 多进程共享 | 不可 | 可（SQLite 支持并发读） |
| 查询能力 | 只能按 session_id 取 | 支持任意 SQL 查询（时间范围、关键词搜索） |

#### 差异二：会话管理

`05` 只有隐形的一维结构：`session_id → [messages]`，没有会话元数据。

`06` 有显式的 `sessions` 表，可以存 `title`、`created_at`、`updated_at`。这意味着可以：
- 列出所有会话及最后活跃时间
- 删除指定会话
- 统计每个会话的消息数

#### 差异三：get_session_history 返回值不同

```python
# 05 — 返回内存对象
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 06 — 返回数据库适配对象（每次创建新的 DB session）
def get_session_history(session_id):
    db = DBSession(engine)  # 新的 DB 连接
    return SQLiteChatMessageHistory(session_id=session_id, db_session=db)
```

**为什么 06 每次创建新的 `DBSession`？**

SQLAlchemy 的 `Session` 不是线程安全的。`RunnableWithMessageHistory` 可能在异步环境下调用，每次创建新 Session 保证隔离性。

#### 差异四：重启后的行为

```
05 重启：
  store = {}  ← 空字典，历史全部丢失
  用户问 "Summarize what we discussed" → 理解不了 "we discussed"

06 重启：
  打开 chat_history.db → 消息都在
  用户问 "Summarize what we discussed" → 正常理解上下文
```

---

### 4. 实际运行验证

运行结果展示了两个会话的隔离和持久化：

```
📁 会话: session_3
   消息数: 2                                         ← 独立的会话
     [human] What is RAG?
     [ai] 当前知识库中没有相关信息。

📁 会话: session_2
   消息数: 6                                         ← 包含完整 3 轮对话
     [human] What is LangChain?
     [ai] 根据提供的上下文，LangChain 是一个...
     [human] How is it different from LangGraph?      ← 历史感知检索器正常
     [ai] LangChain 是一个高级框架...
     [human] Summarize what we just discussed.
     [ai] 我们刚才讨论了...                             ← 记忆正常
```

---

## 关键概念速查（新增）

| 概念 | 一句话解释 |
|---|---|
| `Base` (declarative_base) | ORM 模型的注册表，继承它即可映射为数据库表 |
| `SessionModel` | 会话表的 ORM 映射，存 session_id 和元数据 |
| `MessageModel` | 消息表的 ORM 映射，通过 FK 关联到 SessionModel |
| `ForeignKey` | 数据库外键约束，保证消息引用的会话必须存在 |
| `relationship` | ORM 层面的关联声明，让 `session.messages` 可以直接访问关联消息 |
| `cascade="all, delete-orphan"` | 级联删除：删会话 → 自动删所有关联消息 |
| `SQLiteChatMessageHistory` | 自定义历史存储，实现 `BaseChatMessageHistory` 接口，底层读写 SQLite |
| `DBSession` (SQLAlchemy) | 数据库会话，封装事务操作，`add/query/delete/commit` |
| `engine` | 数据库连接引擎，管理连接池和底层通信 |

---

## 从 03 到 06 的演进路线

```
03_rag_indexing.py          ← 索引：网页 → 切分 → 向量化 → 存储
        ↓
04_rag_retrieval.py         ← 检索+生成：问题 → 检索 → Prompt → 回答
        ↓
05_conversational_rag.py    ← 加记忆：RunnableWithMessageHistory + 内存存储
        ↓
06_persistent_rag.py        ← 加持久化：SQLite + SQLAlchemy，重启不丢失
```
