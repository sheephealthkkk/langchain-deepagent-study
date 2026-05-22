# Messages：消息体系的角色与执行顺序

## 一、Messages 是什么——一句话

**Messages 是 LangChain 对话系统的"原子单位"——所有组件间的信息传递都靠它，没有 Messages 就没有对话。**

## 二、Messages 的三大作用

### 作用 1：结构化对话——角色分离

没有 Messages，对话就是一个大字符串：

```python
# 没有 Messages — 角色混在一起，模型无法区分
"你是助手。用户：天气怎么样？助手：我查一下。工具结果：25°C。助手：今天25°C。"
```

有了 Messages，每个发言的角色、内容、元数据各自独立：

```python
[
    SystemMessage("你是助手，用中文回答。"),     # ← 系统知道自己是助手
    HumanMessage("北京天气怎么样？"),            # ← 模型知道这是用户说的
    AIMessage("我来查一下。", tool_calls=[...]), # ← 模型知道这是自己说的 + 想调工具
    ToolMessage("25°C, 晴天", id="call_1"),     # ← 模型知道这是工具返回的
    AIMessage("北京今天25°C，晴天！"),            # ← 模型继续回复
]
```

### 作用 2：统一传输协议——组件间的共同语言

LLM 吃 `List[BaseMessage]`，ChatPromptTemplate 产出 `List[BaseMessage]`，ChatHistory 存储 `List[BaseMessage]`，Tool 返回 `ToolMessage`。所有组件通过同一种数据类型沟通，用 `|` 串联时不需要做类型转换。

```
ChatPromptTemplate  ──List[BaseMessage]──→  ChatOpenAI  ──AIMessage──→  StrOutputParser
                                                       │
                                                   tool_calls
                                                       │
                                                       ▼
                                                  Tool.invoke()
                                                       │
                                                  ToolMessage
```

### 作用 3：承载元数据——不只是文本

```python
ai_msg = AIMessage(
    content="北京今天25°C，晴天！",     # 给人看的文本
    tool_calls=[                        # 工具调用请求
        {"name": "get_weather", "args": {"city": "北京"}, "id": "call_1"}
    ],
    response_metadata={                 # API 返回的原始信息
        "token_usage": {"prompt_tokens": 120, "completion_tokens": 30},
        "finish_reason": "stop",
        "model_name": "deepseek-v4-pro",
    },
    id="chatcmpl-xxx",                 # 唯一消息 ID
)
```

一条 AIMessage 同时携带：文本回复 + 工具调用意图 + Token 用量 + 模型信息，后续组件可以根据需要取不同的字段。

---

## 三、RunnableWithMessageHistory 用法详解

### 3.1 它解决什么问题

没有 `RunnableWithMessageHistory` 时，每次调用 Chain 都需要手动管理聊天历史：

```python
# ❌ 手动管理历史 — 繁琐且容易出错
history = []

# 第 1 轮
r1 = chain.invoke({"input": "What is LangChain?", "history": history})
history.append(HumanMessage("What is LangChain?"))
history.append(AIMessage(r1))

# 第 2 轮
r2 = chain.invoke({"input": "How is it different?", "history": history})
history.append(HumanMessage("How is it different?"))
history.append(AIMessage(r2))

# 每次都要记得传 history、记得追加、记得构造 HumanMessage/AIMessage
# 换一个 session_id 还要手动切换不同的 history 列表
```

`RunnableWithMessageHistory` 把这一切自动化：**调用前自动注入历史、调用后自动追加新消息、按 session_id 自动隔离不同会话**。

### 3.2 六个参数逐个讲解

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

runnable_with_history = RunnableWithMessageHistory(
    runnable,                     # ① 底层真正干活的链
    get_session_history,          # ② 根据 session_id 返回 BaseChatMessageHistory 对象
    input_messages_key="input",   # ③ 用户输入在 dict 中的 key 名（入口）
    history_messages_key="history", # ④ 历史消息注入到 dict 的哪个 key（桥梁）
    output_messages_key="output", # ⑤ 模型回复在输出 dict 中的 key 名（出口）
)
```

#### 参数 ①：`runnable` — 被包装的链

这是真正干活的 Runnable。可以是任何 Chain、Agent、LLM。`RunnableWithMessageHistory` 不关心它在做什么，只负责在调用它**之前**把历史注入 dict，在它返回**之后**从输出提取回答追加到历史。

```python
# runnable 可以是简单 LLM
runnable = prompt | llm | StrOutputParser()

# 也可以是复杂 RAG 链
runnable = create_retrieval_chain(retriever, qa_chain)

# 也可以是 Agent
runnable = create_agent(llm, tools)
```

#### 参数 ②：`get_session_history` — 历史存取的回调函数

**这是整个机制的核心**。它是一个函数，接收 `session_id: str`，返回一个实现了 `BaseChatMessageHistory` 接口的对象。

```python
# === 内存实现（开发用）===
from langchain_core.chat_history import InMemoryChatMessageHistory

store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """根据 session_id 返回对应的聊天历史。不存在则自动创建。"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# === SQLite 实现（生产用）===
# 见 06_persistent_rag.py 的 SQLiteChatMessageHistory
# 重启后历史仍然保留
```

**为什么是回调函数而不是直接传对象？** 因为 `RunnableWithMessageHistory` 每次调用都要根据 `config` 中的 `session_id` 动态获取对应会话的历史。传一个函数让它自己调，比传一个固定的对象更灵活——同一个链实例可以同时服务多个 `session_id`。

#### 参数 ③：`input_messages_key="input"` — 入口 Key

指定用户传入 dict 中的**哪个字段**是用户当前的输入。框架从 `config` 中读出来的 `session_id` → 调 `get_session_history(session_id)` → 拿到历史 → **把历史和用户输入拼成一个新的 dict**：

```python
# 用户调用时：
runnable_with_history.invoke(
    {"input": "How is it different?"},  # ← "input" 在用户 dict 中
    config={"configurable": {"session_id": "session_1"}},
)

# 框架内部自动拼装：
# {
#     "input": "How is it different?",          ← 用户传的，原样保留
#     "history": [HumanMessage(...), AIMessage(...)]  ← 框架自动注入！
# }
# ↑ "history" 就是 history_messages_key 指定的 key
```

**为什么用户只传 `input` 而不传 `history`？** 因为历史应该由框架管理，不应该让用户手动传——用户只需要关心当前要问什么。

#### 参数 ④：`history_messages_key="history"` — 桥梁 Key

指定**历史消息列表**在 dict 中的 key 名。框架将 `get_session_history` 返回的 `messages` 列表注入到 dict 的这个 key 下。

```python
# 这个 key 必须和 Chain 的 Prompt 模板中的 MessagesPlaceholder 名称对应！

# RunnableWithMessageHistory 侧：
history_messages_key = "history"               # ← 框架拼 dict 时用的 key

# Prompt 模板侧：
MessagesPlaceholder("history")                 # ← 模板中占位符的名字

# 两者必须一致！否则历史消息注入不了模板。
```

**为什么叫"桥梁 Key"？** 它是 RunnableWithMessageHistory（历史管理）和 Prompt 模板（消息构造）之间的**唯一耦合点**。改了这个 key，两头都要改。

#### 参数 ⑤：`output_messages_key="output"` — 出口 Key

指定链的输出 dict 中**哪个字段是 AI 的最终回答**。框架从输出中提取这个字段的值，追加到聊天历史。

```python
# rag_chain 的输出（由 create_retrieval_chain 构造）：
{
    "input": "How is it different from LangGraph?",
    "chat_history": [...],
    "context": [Document, ...],
    "answer": "LangChain 是一个高级框架..."  ← output_messages_key 指向这里
}

# 框架提取：output["answer"] → AIMessage("LangChain 是一个高级框架...")
# 然后追加到 history.add_message(AIMessage(...))
```

**如果链的输出是纯字符串（不是 dict）怎么办？** 那就不需要这个参数。框架直接把整个字符串当作回答追加到历史：

```python
# 链的输出是纯字符串
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "Hi"})
# → "Hello! How can I help you?"（纯字符串）

# RunnableWithMessageHistory 会直接把字符串包装为 AIMessage 追加
```

### 3.3 完整使用示例（从创建到调用）

```python
# ================================================================
# RunnableWithMessageHistory 完整用法
# ================================================================
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ---- 第 1 步：创建底层链 ----
# 这个链需要有 MessagesPlaceholder 来接收历史消息
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好助手。"),
    MessagesPlaceholder("history"),   # ← 历史消息注入位置
    ("human", "{input}"),             # ← 用户当前输入
])

llm = ChatOpenAI(model="deepseek-v4-pro")
chain = prompt | llm | StrOutputParser()
# 注意：此时 chain 还没有记忆功能。如果直接调 chain.invoke({"input": "Hi"})
# 会因为缺少 "history" 字段而报错——MessagesPlaceholder 需要这个字段。

# ---- 第 2 步：定义历史存取回调 ----
store = {}

def get_session_history(session_id: str):
    """根据 session_id 获取对应的聊天历史。"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ---- 第 3 步：包装为有记忆的链 ----
chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,                      # 底层链
    get_session_history=get_session_history,
    input_messages_key="input",          # 用户传 {"input": "..."}
    history_messages_key="history",      # 历史注入到 "history" 字段
    # output_messages_key 不指定——因为 chain 输出是纯字符串 (StrOutputParser)
)

# ---- 第 4 步：多轮对话 ----
config = {"configurable": {"session_id": "chat_1"}}

# 第 1 轮
r1 = chain_with_memory.invoke(
    {"input": "我叫 Alice，我喜欢 Python"},
    config=config,
)
print(f"AI: {r1}")  # → "你好 Alice！Python 很棒的..."

# 第 2 轮 — 不需要手动传历史！
r2 = chain_with_memory.invoke(
    {"input": "我叫什么？我喜欢什么语言？"},
    config=config,
)
print(f"AI: {r2}")  # → "你叫 Alice，你喜欢 Python！"
# RunnableWithMessageHistory 自动把第 1 轮的对话注入到了 Prompt 中

# 第 3 轮 — 换一个 session_id → 全新记忆
config_new = {"configurable": {"session_id": "chat_2"}}
r3 = chain_with_memory.invoke(
    {"input": "我叫什么？"},
    config=config_new,
)
print(f"AI: {r3}")  # → "我不知道你的名字，这是我们第一次对话。"
```

### 3.4 参数对应关系图

```
用户调用                      RunnableWithMessageHistory                底层 Chain
────────                    ──────────────────────────                ──────────

invoke(
  {"input": "Hi"},  ─────────→  提取 "input" 字段                      
  config={                       ↓                                      
    "session_id": "s1"         调 get_session_history("s1")            
  }                              ↓                                      
)                              拼接:                                   
                                {                                      
                                  "input": "Hi",       ← input_messages_key
                                  "history": [          ← history_messages_key
                                    HumanMsg("..."),                 
                                    AIMsg("..."),                    
                                  ],                                  
                                }                                      
                                  │                                    
                                  ▼                                    
                              chain.invoke(上面的dict) ────────────→  prompt 填模板
                                                                       │ {input}→"Hi"
                                                                       │ MessagesPlaceholder("history")
                                                                       │  ← 展开为 HumanMsg+AIMsg
                                                                       ▼
                                                                     LLM 生成回答
                                                                       │
                                                                       ▼
                              ←────────── "Hello Alice!" ──────────  返回值(纯字符串)
                                │
                                ▼
                              追加到历史:
                                HumanMsg("Hi")
                                AIMsg("Hello Alice!")
                                │
                                ▼
返回给用户: "Hello Alice!"
```

### 3.5 常见问题

**Q: 什么时候需要 `output_messages_key`？**

当底层链的输出是 **dict 格式**时。例如 `create_retrieval_chain` 返回 `{"input":..., "chat_history":..., "context":..., "answer":...}`。框架需要知道取哪个字段作为 AI 回答。

当底层链的输出是**纯字符串**时（用了 `StrOutputParser`），不需要指定——框架自动把整个字符串当作回答。

**Q: `input_messages_key` 和 `history_messages_key` 可以取任意名字吗？**

可以，但必须和 Prompt 模板中的对应字段名一致。常用的命名约定：
- `input_messages_key="input"` → 模板中 `{input}`
- `history_messages_key="history"` → 模板中 `MessagesPlaceholder("history")`
- 或者 `history_messages_key="chat_history"` → 模板中 `MessagesPlaceholder("chat_history")`

**Q: 如果底层链同时需要 `{input}` 和 `{context}`（如 RAG），怎么传？**

`RunnableWithMessageHistory` 只注入 `history_messages_key` 这一项。`{context}` 由 `create_retrieval_chain` 内部从 retriever 的结果注入，不经过 `RunnableWithMessageHistory`。

### 3.6 一句话概括

**`RunnableWithMessageHistory` 本质上是一个"自动追加聊天记录的代理"**：你在外层调用 `chain_with_memory.invoke({"input": "新消息"})`，它内部替你做了三件事——(1) 根据 `session_id` 取出历史消息列表，(2) 把历史消息通过 `MessagesPlaceholder` 注入底层 Runnable，(3) 底层 Runnable 返回后，把本轮对话自动追加回历史。底层那个 Runnable（`prompt | llm | parser`）通过 `MessagesPlaceholder` 接收历史，每次 `invoke` 看到的是"历史消息 + 当前输入"拼好的完整 Prompt，所以它自然就有了上下文记忆。

---

