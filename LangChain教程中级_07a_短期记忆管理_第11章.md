## 第十一章：记忆管理（Memory Management）

### 11.1 三个核心要素

LangGraph 的记忆体系建立在三个概念上：

```
记忆 = State（状态） + Checkpointer（检查点） + Thread ID（线程标识）
```

| 要素 | 是什么 | 类比 | 存储内容 |
|---|---|---|---|
| **State** | Agent 的"大脑状态"快照 | 游戏的存档文件 | 消息列表 + 自定义字段 |
| **Checkpointer** | 状态持久化器 | 存档管理器 | 每次 State 变更的时间点记录 |
| **Thread ID** | 会话唯一标识 | 游戏角色 ID | 不同角色 → 不同存档 |

**三者协作流程**：

```
第1轮: 用户问 "What is LangChain?"
  Agent 推理 → 回答 → State 变更了
    │
    ▼
  Checkpointer 把 State 保存在 Thread("session_1") 下

第2轮: 用户问 "How is it different from LangGraph?"
  同一个 Thread("session_1")
    │
    ▼
  Checkpointer 读出上次的 State → Agent 有记忆 → 理解"it"=LangChain
    │
    ▼
  Agent 推理 → 回答 → State 变更 → Checkpointer 保存新的 State
```

### 11.2 短期记忆 vs 长期记忆

**最常见的误区：短期记忆 = 存在内存里，长期记忆 = 存在数据库里。这是错的。**

| 维度 | 短期记忆 | 长期记忆 |
|---|---|---|
| **时间范围** | 当前会话内 | 跨会话 |
| **能否跨会话** | 不能——新会话 = 新 thread_id = 空白状态 | 能——同一用户多会话共享 |
| **典型存储** | 内存 / SQLite / PostgreSQL 都可以 | 向量库 + 检索 |
| **LangChain 实现** | LangGraph Checkpointer | 向量库 + RAG + 会话管理 |
| **查询方式** | 拿整个 State | 按语义检索相关记忆 |
| **你之前写的 06** | **短期记忆**——SQLite 存了但只按 session_id 读取，不跨会话 | — |

**判断标准只有一个：记忆是否跟随同一个 thread_id（会话），还是跨 thread_id（用户级）共享。**

```python
# 短期记忆：同一 thread_id 内累积
config_1 = {"configurable": {"thread_id": "session_A"}}
agent.invoke({"messages": [HumanMessage("我是 Alice")]}, config_1)
agent.invoke({"messages": [HumanMessage("我叫什么？")]}, config_1)
# → "你叫 Alice。" ← 同 thread_id，有记忆

# 新 thread_id → 无记忆
config_2 = {"configurable": {"thread_id": "session_B"}}
agent.invoke({"messages": [HumanMessage("我叫什么？")]}, config_2)
# → "我不知道你的名字。" ← 新线程，无历史

# 长期记忆：跨 thread_id 共享
# 需要用一个"用户记忆库"（向量库），存 Alice 的所有偏好
# 每个新会话都从记忆库检索相关信息 → 注入 Prompt
```

**为什么 SQLite/PostgreSQL 存储仍然叫"短期"？** 因为它还是跟 thread_id 绑定。除非你手动用同一个 thread_id 隔天再次调用——但通常每个对话窗口就是一个新 thread_id。真正的长期记忆需要用户级记忆库，不依赖 thread_id。

### 11.3 底层原理：AgentState + Checkpointer

LangGraph 的 `create_react_agent` 内部使用 `StateGraph`，其中 State 的默认结构：

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Agent 的默认 State 结构。"""
    messages: Annotated[list[BaseMessage], add_messages]
    # add_messages = 追加而非覆写——新消息自动拼到旧消息后面
```

**`add_messages` 是关键**：定义了 messages 字段的"归并"方式——不是覆盖，而是追加。这样每轮对话只需传新消息，历史自动拼接。

### 11.4 Checkpointer 机制

**Checkpointer = Agent 状态的版本控制系统。**

```
不加 Checkpointer：
  agent.invoke(input) → response  ← State 只在内存中，调用结束后丢弃
  下次调用 → 全新 State → 无历史 → 无法理解上下文

加了 Checkpointer：
  agent.invoke(input, config) → response
    │
    ├── 调用前：从 Thread("session_1") 读出上次的 State → 拼上历史
    ├── 调用中：Agent 推理，可能多次修改 State
    └── 调用后：State → 保存到 Thread("session_1") 下

  下次同一 config 调用 → State 从上次停止的地方继续
```

**每个 checkpoint 记录以下内容**：

```python
Checkpoint:
  {
    "v": 3,                           # 版本号
    "id": "1ef...",                   # checkpoint ID
    "ts": "2026-05-10T10:00:00Z",    # 时间戳
    "channel_values": {               # 当前所有字段的值
      "messages": [SystemMessage(...), HumanMessage(...), AIMessage(...)],
      # ... 其他自定义字段
    },
    "channel_versions": {             # 每个字段的版本
      "messages": 3,
    },
    "versions_seen": {                # 父 checkpoint 的版本
      "__start__": {"messages": 2},
    },
  }
```

**效果对比**：

| | 不加 Checkpointer | 加了 Checkpointer |
|---|---|---|
| 多轮对话 | 不支持（每次全新 State） | 支持 |
| 暂停/恢复 | 不支持 | 支持（`interrupt` 暂停，下次继续） |
| 时间回溯 | 不支持 | 支持（回退到历史 checkpoint） |
| 断点调试 | 不支持 | 支持 |
| 跨进程 | 不支持 | 支持（数据库存储的话） |

### 11.5 Thread ID — 会话隔离

```python
# thread_id = session 隔离
config_a = {"configurable": {"thread_id": "user_A_chat_1"}}
config_b = {"configurable": {"thread_id": "user_B_chat_1"}}

# 两个线程独立互不影响
agent.invoke({"messages": [HumanMessage("我叫 Alice")]}, config_a)
agent.invoke({"messages": [HumanMessage("我叫 Bob")]}, config_b)

agent.invoke({"messages": [HumanMessage("我叫什么？")]}, config_a)  # → "Alice"
agent.invoke({"messages": [HumanMessage("我叫什么？")]}, config_b)  # → "Bob"
```

**thread_id 设计策略**：

| 场景 | thread_id 设计 | 示例 |
|---|---|---|
| 一对一聊天 | 每个对话窗口一个 thread_id | `f"chat_{user_id}_{conversation_id}"` |
| 客服系统 | 每个工单一个 thread_id | `f"ticket_{ticket_id}"` |
| 多 Agent 协作 | 子 Agent 用不同 thread_id | `f"orchestrator_{task_id}"` + 子 Agent 各自 |
| 临时工具调用 | 每次调用用 UUID | `uuid4()` |

### 11.6 InMemorySaver — 内存版入门

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# 1. 创建 Checkpointer
checkpointer = InMemorySaver()

# 2. 创建 Agent，指定 checkpointer
agent = create_react_agent(
    model=llm,
    tools=[get_weather, search_web],
    checkpointer=checkpointer,      # ← 启用记忆
)

# 3. 每次调用带 thread_id
config = {"configurable": {"thread_id": "chat_001"}}

# 第1轮
r1 = agent.invoke(
    {"messages": [HumanMessage("北京天气？")]},
    config=config,                 # ← 必须传 config
)
print(r1["messages"][-1].content)  # → "北京今天晴，25°C"

# 第2轮 — 自动有上下文
r2 = agent.invoke(
    {"messages": [HumanMessage("适合户外运动吗？")]},
    config=config,                 # ← 同一个 thread_id
)
# Agent 记得第1轮问了北京天气 → "25°C 晴天，非常适合户外运动！"

# 查看 checkpoint 版本
state = agent.get_state(config)
print(f"当前 checkpoint: {state.values['messages']}")
```

**InMemorySaver 的限制**：进程重启 → 所有记忆消失。适合开发调试，不适合生产。

### 11.7 PostgresSaver — 数据库存储

**为什么 Postgres 存储的仍然是"短期记忆"？**

因为记忆还是跟 `thread_id` 绑定——换一个 `thread_id` 就没有历史。只是存储介质从内存变成了 PostgreSQL 文件。**短期/长期区分的是"会话级/用户级"，不是"内存/磁盘"。**

```python
# pip install langgraph-checkpoint-postgres psycopg

from langgraph.checkpoint.postgres import PostgresSaver

# 1. 创建 Postgres 连接
DB_URI = "postgresql://user:pass@localhost:5432/agent_db"
checkpointer = PostgresSaver.from_conn_string(DB_URI)

# 2. 初始化表（首次运行）
await checkpointer.setup()

# 3. 创建 Agent
agent = create_react_agent(
    model=llm,
    tools=[...],
    checkpointer=checkpointer,
)

# 4. 使用 — 同 InMemorySaver 完全一样的 API
config = {"configurable": {"thread_id": "chat_001"}}
r1 = agent.invoke({"messages": [HumanMessage("北京天气？")]}, config)
r2 = agent.invoke({"messages": [HumanMessage("适合运动吗？")]}, config)

# 5. 重启进程后 — 记忆还在！
# agent.invoke(..., config) → 仍然记得之前的对话

# 6. 查看所有会话
all_threads = await checkpointer.alist()
for t in all_threads:
    print(f"thread={t['thread_id']}, checkpoint_count={len(t['checkpoints'])}")
```

**Postgres 存储结构**：

```sql
-- LangGraph 自动创建的表
checkpoints (
    thread_id TEXT,
    checkpoint_id TEXT,
    parent_id TEXT,           -- 上一个 checkpoint
    checkpoint BYTEA,         -- JSON 序列化后的 State
    metadata JSONB,           -- 时间戳、step 等
    PRIMARY KEY (thread_id, checkpoint_id)
);

-- 查询某个会话的所有 checkpoint（可以时间回溯！）
SELECT * FROM checkpoints
WHERE thread_id = 'chat_001'
ORDER BY metadata->>'ts' ASC;
```

**PostgresSaver vs InMemorySaver**：

| | InMemorySaver | PostgresSaver |
|---|---|---|
| 持久化 | 否（重启丢失） | 是 |
| 可查所有会话 | 否 | 是 |
| 时间回溯 | 否 | 是（所有历史 checkpoint） |
| 多进程共享 | 否 | 是 |
| 安装复杂度 | 零 | 需 PostgreSQL |
| 适用 | 本地开发 | 生产环境 |

### 11.8 上下文裁剪 — trim_messages

**问题**：长期对话后消息列表太长 → 超过模型上下文窗口 → 报错或截断。

```python
from langchain_core.messages import trim_messages
import tiktoken

# === 策略1：按 Token 数量裁剪（保留最近 N token）===
encoding = tiktoken.get_encoding("cl100k_base")

trimmed = trim_messages(
    messages=history,
    max_tokens=4000,                     # 保留最近 4000 token
    token_counter=encoding,              # 用什么 tokenizer 计数
    strategy="last",                     # 保留最后（最近）的消息
    include_system=True,                 # 始终保留 SystemMessage
    allow_partial=True,                  # 允许部分裁剪
    start_on="human",                    # 从 HumanMessage 开始（不以 AI 开头）
)

# === 策略2：按消息数量裁剪 ===
trimmed = trim_messages(
    messages=history,
    max_tokens=20,                       # 保留最近 20 条
    token_counter=lambda msgs: len(msgs), # 用消息数代替 token
    strategy="last",
)

# === 策略3：按轮次裁剪 ===
def count_rounds(messages):
    """一轮 = 一次 Human + AI 对"""
    return sum(1 for m in messages if isinstance(m, HumanMessage))

trimmed = trim_messages(
    messages=history,
    max_tokens=5,                        # 保留最近 5 轮
    token_counter=count_rounds,
    strategy="last",
)

# === 在实践中使用 ===
from langchain_core.runnables import RunnableLambda

def trim_history(state):
    """每次调用前自动裁剪历史。"""
    state["messages"] = trim_messages(
        messages=state["messages"],
        max_tokens=4000,
        token_counter=tiktoken.get_encoding("cl100k_base"),
        strategy="last",
        include_system=True,
        start_on="human",
    )
    return state

# 把这个步骤嵌入 Graph 的入口
workflow.add_node("trim", trim_history)
workflow.add_edge("trim", "agent")
```

**四种常用裁剪策略**：

| 策略 | 做法 | 适用 |
|---|---|---|
| `strategy="last"` | 保留最后 N token/条 | **最常用**，对话场景 |
| `strategy="first"` | 保留前 N token/条 | 系统指令优先 |
| `Summarization` | 裁剪前先把旧消息摘要化 | 需要保留旧上下文但节省空间 |
| 滑动窗口 | 保留最近 K 轮 + 摘要旧轮 | 长对话平衡 |

**`start_on="human"` 的作用**：确保裁剪后的消息列表从 HumanMessage 开始，而不是从 AIMessage 开始——避免 LLM 困惑"怎么一开始就是 AI 在说话"。

### 11.9 自定义 State — 扩展字段

基础 `AgentState` 只有 `messages`，但实际场景需要更多：

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from operator import add

class ExtendedAgentState(TypedDict):
    """扩展 State：messages + 业务字段。"""
    # 原生字段（消息列表，追加式归并）
    messages: Annotated[list[BaseMessage], add_messages]

    # 自定义字段
    user_id: str                            # 固定值（每次覆盖）
    user_name: str                          # 同上
    conversation_turn: Annotated[int, add]  # 自增计数器（+1 累加）
    extracted_entities: list[str]           # 对话中提取的实体
    tool_call_count: Annotated[int, add]    # 工具调用总次数
    summary: str                            # 对话摘要（每次更新覆盖）
```

**`Annotated[type, reducer]` 的含义**：

```python
# add_messages → 追加（消息列表）
messages: Annotated[list, add_messages]  # 新消息拼到旧消息后

# add → 累加（数值）
turn: Annotated[int, add]                # 新值 + 旧值

# 无 Annotated → 覆盖（默认行为）
summary: str                             # 新值直接覆盖旧值
```

**用途**：

| 字段 | 用途 |
|---|---|
| `user_id` / `user_name` | 个性化 System Prompt 注入 |
| `conversation_turn` | 限制轮次（如客服对话限 10 轮后转人工） |
| `extracted_entities` | 累积提取的关键信息，做长期记忆 |
| `tool_call_count` | 监控工具调用次数，异常检测 |
| `summary` | 旧对话摘要，裁剪前保存关键信息 |

### 11.10 带状态的 Tool — ToolRuntime

#### 从 Java 程序员视角看问题

假设你是一个 Java 程序员，正在写一个多步骤的 Workflow：

**Step 1** — Tool A 从用户消息中提取了实体：
```java
// Tool A 提取了 ["Alice", "Beijing", "Python"]
List<String> entities = extractEntities(userMessage);
```

**Step 2** — Tool B 需要使用这些实体来做后续处理。在 LLM Agent 架构中，如果你没有 ToolRuntime，数据流是这样的：

```
Tool A 提取了 ["Alice", "Beijing", "Python"]
    ↓ 必须返回给 LLM（作为字符串！）
LLM 看到："已提取实体：Alice, Beijing, Python"
    ↓ LLM 决定调用 Tool B，手动把实体列表作为参数传入
Tool B 收到 LLM 传的参数
    ↓ 但如果 LLM 记错了、漏了、或自己编了？
Tool B 拿到了 ["Alice", "Beijing"] ← "Python" 丢了！LLM 幻觉！
```

**这就像在 Spring 中，如果没有依赖注入**：

```java
// 没有 Spring DI 的世界：每个 Controller 自己 new Service
// 还要手动把 HttpRequest、UserContext、DB Connection 到处传参

@RestController
public class OrderController {
    // 没有 @Autowired！只能自己创建
    private OrderService orderService = new OrderService(
        new PaymentService(new DatabaseConnection()),
        new NotificationService(new EmailSender())
    );

    @PostMapping("/order")
    public Order createOrder(@RequestBody OrderRequest req, HttpServletRequest httpReq) {
        // 你需要手动把 Session 里的 user_id 传给 Service
        String userId = httpReq.getSession().getAttribute("userId");
        // 如果忘了传，或者传错了 → Bug
        return orderService.create(req, userId, httpReq.getRemoteAddr(), ...);
    }
}
```

**ToolRuntime 就是 LLM Agent 世界的 `@Autowired`**：工具不需要从 LLM 那里接收框架级信息，框架自动注入。

---

#### 没有 ToolRuntime 时的三大痛点

**痛点 1：数据必须通过 LLM 来回传递**

```
A 工具提取了实体 → 返回给 LLM（转成自然语言）
  → LLM 理解后 → 调用 B 工具 → 把实体列表作为参数传给 B
     ↑
  问题：LLM 可能漏掉、改错、或自己编造实体
```

**痛点 2：无法访问框架级上下文**

工具的签名里只能有"业务参数"（LLM 决定传什么），但工具还需要：
- `user_id`（当前是谁在对话）→ 不是 LLM 决定的，是系统已知的
- `conversation_turn`（第几轮）→ LLM 不需要知道这个
- `permissions`（用户权限）→ 安全相关的，不能让 LLM 传递（可篡改）

没有 ToolRuntime → 你要么让 LLM 传（不可靠），要么写在全局变量里（线程不安全）。

**痛点 3：工具之间完全隔离**

```
Tool A 和 Tool B 是两个独立黑盒
   → 没有共享内存
   → 没有 Session 级别的变量
   → 数据交换的唯一通道是 LLM 这个"不可靠的中间人"
```

这就像在 Java 中每个 Service 都是独立的，没有 DI 容器来管理单例和共享 Bean。

---

#### ToolRuntime 怎么解决：类比 Spring 依赖注入

```
Java Spring:                   LangChain ToolRuntime:
─────────────                  ─────────────────────
@Autowired                     ToolRuntime[State]
private UserContext ctx;       runtime.state.get("user_id")
                               ↓
Spring 容器自动注入             LangGraph 框架自动注入
Controller 只管业务参数         工具只管业务参数
```

**对比**：

```java
// === Java Spring：框架自动注入 UserContext ===
@RestController
public class OrderController {
    @Autowired
    private UserContext userContext;  // ← 框架注入，方法签名里不需要传

    @PostMapping("/order")
    public Order createOrder(@RequestBody OrderRequest req) {
        // userContext 已就绪，直接使用
        String userId = userContext.getUserId();
        return orderService.create(req, userId);
    }
}
```

```python
# === LangChain ToolRuntime：框架自动注入 State ===
@tool
def create_order(
    product: str,                                # LLM 只决定业务参数
    quantity: int,
    runtime: ToolRuntime[ExtendedAgentState],    # ← 框架自动注入 State
) -> str:
    """创建订单。LLM 不需要知道 user_id，框架自动提供。"""
    # 从 State 直接读取框架级上下文（不经过 LLM！）
    user_id = runtime.state["user_id"]           # 系统已知，不依赖 LLM 传值
    permissions = runtime.state.get("permissions", [])
    turn = runtime.state.get("turn_count", 0)

    # 权限校验：不让 LLM 传 user_id 是安全设计
    # 如果 user_id 由 LLM 传 → 任何用户都可以通过 Prompt 伪装成别人
    if "order:create" not in permissions:
        return "权限不足：你无权创建订单。"

    order_id = db.orders.insert(user_id, product, quantity)
    return f"订单 {order_id} 已创建：{product} × {quantity}（用户：{user_id}，第 {turn} 轮）"
```

---

#### 完整实战：提取实体 + 多个工具共享 State

这个例子展示了 Tool A 提取实体 → Tool B 直接使用实体，数据不通过 LLM 中转：

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain.tools import tool, ToolRuntime

# ---- State 定义：工具间的共享内存 ----
class WorkflowState(TypedDict):
    """
    对话的全局状态，类似于 Java 中一个 Bean 的 Scope="session"。
    所有工具都可以通过 ToolRuntime 读写这个对象。
    """
    # 基础字段：消息列表（追加式归并，不是覆盖）
    messages: Annotated[list[BaseMessage], add_messages]

    # 业务字段：下面这些字段 LLM 不需要知道，但工具需要
    user_id: str                         # 当前用户 ID（从 config 注入，不经过 LLM）
    extracted_entities: list[str]        # 工具之间共享：Tool A 写，Tool B 读
    entity_extraction_count: int         # 计数器：总共提取了多少次实体
    last_extraction_time: str            # 最后一次提取的时间戳


# ---- Tool A：写入共享状态 ----
@tool
def extract_and_remember_entities(
    user_message: str,
    runtime: ToolRuntime[WorkflowState],   # ← 类似 @Autowired WorkflowState
) -> str:
    """
    从用户消息中提取关键实体（人名、地名、技术名），并记住它们。
    提取后的实体存入 runtime.state，不需要 LLM 记住或转述。

    参数说明：
      user_message: 用户原始消息（由 LLM 决定传入）
      runtime: 框架自动注入的 State 访问器（LLM 不参与，不经过 LLM）
    """
    # 1. 用 NLP 提取实体（这里用简单规则演示）
    words = user_message.split()
    entities = [w.strip(",.") for w in words if w[0].isupper()]
    # 输入 "Alice works at Google using Python" → ["Alice", "Google", "Python"]

    # 2. 直接写入 State，不返回给 LLM（LLM 不需要看到完整实体列表）
    old_entities: list[str] = runtime.state.get("extracted_entities", [])
    old_count: int = runtime.state.get("entity_extraction_count", 0)

    # 去重合并：类似 Java 中的 Set.addAll()
    merged = list(set(old_entities + entities))
    runtime.state["extracted_entities"] = merged           # 写入共享内存
    runtime.state["entity_extraction_count"] = old_count + 1  # 计数器 +1

    # 3. 返回给 LLM 的只是简洁的确认信息（LLM 不需要知道细节）
    new_entities = [e for e in entities if e not in old_entities]
    return f"新增实体: {', '.join(new_entities)}（累计 {len(merged)} 个实体）"


# ---- Tool B：读取共享状态 ----
@tool
def lookup_entity_context(
    entity_name: str,
    runtime: ToolRuntime[WorkflowState],   # ← 同一个 State 实例，跨工具共享
) -> str:
    """
    查询某个实体是否在当前对话中出现过。
    数据来源是 runtime.state（Tool A 写入的），不是 LLM 传的。

    参数说明：
      entity_name: LLM 想知道"Google 出现过吗？"时传入
      runtime: 框架自动注入，包含 Tool A 写入的全部实体
    """
    # 直接从 State 读取（不经过 LLM，不会丢失，不会被 LLM 篡改）
    entities: list[str] = runtime.state.get("extracted_entities", [])
    extract_count: int = runtime.state.get("entity_extraction_count", 0)

    if entity_name in entities:
        return f"'{entity_name}' 在当前对话出现过的 {len(entities)} 个实体中。已提取 {extract_count} 次。"
    else:
        return f"'{entity_name}' 未在当前对话中出现过。"


# ---- Tool C：不使用 ToolRuntime 的传统工具（对比）----
@tool
def old_style_search(query: str) -> str:
    """
    传统工具：只能拿到 LLM 传的参数，无法访问 State。
    如果这个工具需要知道 user_id → 只能靠 LLM 传 → 不可靠。
    """
    # 这里拿不到 user_id、拿不到 extracted_entities
    # 因为方法签名里没有 ToolRuntime
    return f"搜索结果：关于 '{query}' 的 3 条记录"
```

---

#### 数据流对比图

```
❌ 没有 ToolRuntime — 数据必须经过 LLM，每步都丢失/篡改风险：

  Tool A(提取实体)
      │ 返回: "提取了 Alice, Google, Python"  ← 转成了自然语言！
      ▼
  LLM 看到自然语言 → 决定调用 Tool B → 手动传参
      │ 传入: entity_name="Alice"  ← LLM 自己拼的，可能漏了 Google
      ▼
  Tool B(查询实体)  ← 只拿到 LLM 传的参数，完整实体列表在 LLM 那层"丢失"了


✅ 有 ToolRuntime — 数据走 State 总线，不经过 LLM，零丢失：

  Tool A(提取实体)
      │ runtime.state["extracted_entities"] = ["Alice", "Google", "Python"]
      │ 返回: "新增 3 个实体"  ← LLM 拿到简洁确认
      ▼
  State 总线:  entities = ["Alice", "Google", "Python"]  ← 框架持有
      │
      ▼
  LLM 决定调用 Tool B → 传 entity_name="Google"
      │
      ▼
  Tool B(查询实体)
      │ runtime.state["extracted_entities"]  ← 直接从 State 读，完整准确
      │ → "Google" in entities → True
```

---

#### Java 程序员记忆口诀

```
LangChain ToolRuntime     ≈    Java Spring 的 @Autowired + Session Scope Bean

State (TypedDict)         ≈    Session-scoped Bean
ToolRuntime[State]        ≈    @Autowired State state
runtime.state.get("key")  ≈    state.getKey()
工具只管业务参数            ≈    Controller 只管 @RequestBody
框架上下文由框架注入        ≈    UserContext 由 Spring Security 注入
```

---

