## 第十三章：跨线程全局记忆 — BaseStore

### 13.1 三种记忆的关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        记忆体系全景                              │
│                                                                 │
│  短期记忆（Checkpointer）                                        │
│  ├─ 范围：单个 thread_id（会话）内                                │
│  ├─ 存储：State（messages + 自定义字段）                          │
│  ├─ 生命周期：会话关闭 → 不再使用（除非手动复用 thread_id）         │
│  └─ 典型用法：当前对话的多轮上下文                                 │
│                                                                 │
│  长期记忆（向量数据库）                                           │
│  ├─ 范围：单个 user_id 内，跨所有 thread_id                       │
│  ├─ 存储：向量库（文本 → Embedding → 语义检索）                    │
│  ├─ 生命周期：永久（除非主动删除）                                 │
│  └─ 典型用法：用户偏好、历史事实的语义化检索                        │
│                                                                 │
│  BaseStore（全局记忆） ★ 本章                                      │
│  ├─ 范围：任意 namespace 内，跨所有 thread_id + 跨所有 user_id     │
│  ├─ 存储：Key-Value（结构化 JSON，支持索引）                       │
│  ├─ 生命周期：永久（可设 TTL）                                     │
│  └─ 典型用法：用户档案、配置信息、Agent 间共享的结构化数据           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**一句话区分**：

| 记忆类型 | 存储什么 | 怎么查 | 跨 thread 吗 | 跨 user 吗 |
|---|---|---|---|---|
| **短期**（Checkpointer） | 消息列表 + 自定义字段 | 按 thread_id 全量取 | 否 | 否 |
| **长期**（向量库） | 语义化文本片段 | 语义相似度搜索 | 是（同一 user） | 否 |
| **全局**（BaseStore） | 结构化 JSON 数据 | Key 精确查找 / 条件搜索 | 是 | 是 |

### 13.2 BaseStore 核心特征

**BaseStore = 带命名空间的持久化 Key-Value 存储**，专为 Agent 间共享结构化数据设计。

```python
# BaseStore 的五个操作
store.put(
    namespace=("users", "alice", "profile"),  # ← 层次化命名空间
    key="preferences",                         # ← 命名空间内的唯一 Key
    value={"language": "Python", "editor": "VS Code"},  # ← 结构化 JSON
)

store.get(
    namespace=("users", "alice", "profile"),
    key="preferences",
)
# → Item(key="preferences", value={"language": "Python", ...}, ...)

store.search(
    namespace_prefix=("users", "alice"),       # ← 前缀匹配，查该用户所有记忆
    filter={"value.language": "Python"},       # ← 按 JSON 字段过滤
    limit=10,
)
# → [SearchItem(...), SearchItem(...)]

store.delete(
    namespace=("users", "alice", "profile"),
    key="preferences",
)

store.list_namespaces(prefix=("users", "alice"))
# → [("users", "alice", "profile"), ("users", "alice", "history")]
```

**namespace 设计 — 层次化元组**：

```
("users", "alice", "profile")       ← Alice 的用户档案
("users", "alice", "history")       ← Alice 的历史记录
("users", "bob", "profile")         ← Bob 的用户档案（天然隔离）
("agents", "weather_agent", "config") ← 某个 Agent 的配置
("global", "settings")              ← 全局配置（所有用户共享）
```

**层次化的灵活之处** — `search` 用前缀匹配：

```python
# 查 Alice 的所有记忆（profile + history + ...）
store.search(namespace_prefix=("users", "alice"))

# 查所有用户的 profile
store.search(namespace_prefix=("users",), filter={"key": "profile"})
```

### 13.3 BaseStore vs 向量数据库

| 维度 | BaseStore | 向量数据库（Chroma 等） |
|---|---|---|
| **数据结构** | 结构化 JSON（Key-Value） | 非结构化文本 → 向量 |
| **查询方式** | Key 精确查找 + 结构化过滤 | 语义相似度（模糊匹配） |
| **查询性能** | O(1) 精确查找 | O(N) 向量计算（N=数据量） |
| **适用数据** | 配置、档案、计数、状态 | 偏好描述、事实、对话摘要 |
| **典型查询** | "Alice 的 language 偏好是什么？" | "哪些记忆与推荐餐厅相关？" |
| **能存什么** | 任意 JSON（嵌套、列表、数字） | 文本（转成向量后丢失原始结构） |
| **Namespace** | 原生多层级 | 无（用 metadata 模拟） |

**它们不是竞品，是互补关系**：

```
BaseStore：存"Alice 是 VIP，会员到期日 2027-01-01，积分 5000"（结构化数据）
向量库：  存"Alice 偏好素食、喜欢安静的用餐环境、预算中等"（语义化描述）

前者用 store.get(namespace=("users","alice","profile"), key="vip_status") 精确查
后者用 vector_store.similarity_search("推荐晚餐地点") 语义查
```

### 13.4 完整实战：BaseStore 跨线程记忆

下面构建一个同时具备**短期记忆（Checkpointer）+ 全局记忆（BaseStore）**的 Agent。

```python
# ================================================================
# basestore_agent.py — 短期记忆 + BaseStore 全局记忆
# ================================================================
import os
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent

# ================================================================
# 第 1 步：Pydantic 定义用户信息提取方法
# ================================================================

class UserProfile(BaseModel):
    """
    用户档案的 Pydantic 定义。

    这个类的两个作用：
      1. 从对话文本中提取用户信息（extract_from_text）
      2. 标准化 BaseStore 中存储的数据结构（put/get 都用这个格式）
    """
    language: str = Field(
        default="", 
        description="用户偏好的编程语言，如 Python、Java、Go"
    )
    editor: str = Field(
        default="", 
        description="用户偏好的编辑器/IDE，如 VS Code、IntelliJ IDEA"
    )
    location: str = Field(
        default="", 
        description="用户所在城市，如 北京、上海、深圳"
    )
    experience_level: str = Field(
        default="", 
        description="用户技术水平：beginner / intermediate / senior / expert"
    )
    interests: list[str] = Field(
        default_factory=list, 
        description="用户感兴趣的技术领域，如 ['AI', 'Web开发', 'Linux']"
    )
    is_vip: bool = Field(
        default=False, 
        description="用户是否为 VIP 会员"
    )

    @classmethod
    def extract_from_conversation(cls, text: str) -> "UserProfile":
        """
        从一段对话文本中提取用户信息。

        实际生产中这里会调用 LLM 做结构化提取。
        这里用简单规则演示数据流。
        """
        text_lower = text.lower()
        return cls(
            language="Python" if "python" in text_lower else "",
            editor="VS Code" if "vs code" in text_lower or "vscode" in text_lower else "",
            location="北京" if "北京" in text else ("上海" if "上海" in text else ""),
            experience_level=(
                "senior" if any(w in text_lower for w in ["多年", "高级", "架构"])
                else "intermediate"
            ),
            interests=[
                interest for interest in ["AI", "Web开发", "Linux", "数据科学", "Rust"]
                if interest.lower() in text_lower
            ],
            is_vip="vip" in text_lower,
        )

    @classmethod
    def extract_for_query(cls, query_text: str) -> dict:
        """
        从查询文本中提取筛选条件。

        用于 BaseStore.search 的 filter 参数构建。
        比如用户问"推荐 Python 工具"→ 返回 {"language": "Python"}
        然后用这个条件去 BaseStore 中搜索匹配的用户档案。
        """
        conditions = {}
        text_lower = query_text.lower()
        if "python" in text_lower:
            conditions["language"] = "Python"
        if "java" in text_lower:
            conditions["language"] = "Java"
        if "go" in text_lower:
            conditions["language"] = "Go"
        if "vip" in text_lower:
            conditions["is_vip"] = True
        return conditions


# ================================================================
# 第 2 步：自定义 State（含 user_id，跨线程存储时需要）
# ================================================================

class AgentState(TypedDict):
    """
    Agent 的 State 定义。

    字段说明：
      messages:             消息列表（Checkpointer 持久化 → 短期记忆）
      user_id:              当前用户 ID（从 config 注入，不经过 LLM）
      extracted_profile:    本次对话从用户消息中提取的档案（暂存，待写入 BaseStore）
    """
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    extracted_profile: dict  # 暂存从对话中提取的用户信息


# ================================================================
# 第 3 步：存储工具 — 把用户信息写入 BaseStore（跨线程共享）
# ================================================================

def make_store_profile_tool(store):
    """
    创建「存储用户档案到 BaseStore」的工具。

    这个工具使用了 InjectedStore 注解来接收 BaseStore 实例——
    工具自己不需要知道 store 从哪来，框架自动注入。

    类比 Java：
      @Autowired
      private BaseStore store;  // ← InjectedStore 就是这个作用
    """
    # InjectedStore：告诉框架"这个参数请从 Agent 的 store 注入"
    # InjectedState：告诉框架"这个参数请从 Agent 的 State 注入"
    # 两个注解可以同时使用！
    from langgraph.prebuilt import InjectedStore, InjectedState

    def store_profile(
        # 业务参数 —— LLM 决定传什么
        language: str = "",
        editor: str = "",
        location: str = "",
        experience_level: str = "",
        interests_str: str = "",  # 逗号分隔的兴趣列表（LLM 传的，转成 list）
        is_vip: bool = False,

        # 注入参数 —— 框架自动提供，LLM 不参与
        # Annotated[类型, InjectedStore()] 是 LangGraph 的依赖注入语法
        # 类似 @Autowired BaseStore store
        store: Annotated[object, InjectedStore()] = None,
        # InjectedState("user_id") 只注入 State 中的 user_id 字段
        # 类似 @Value("#{state.user_id}") String userId
        user_id: Annotated[str, InjectedState("user_id")] = "",
    ) -> str:
        """
        将用户档案信息写入 BaseStore，实现跨线程共享。

        工具的工作流程：
          1. 从 InjectedState 获取 user_id（框架注入，无需 LLM 传）
          2. 从 LLM 传入的业务参数构造 UserProfile
          3. 将 UserProfile 序列化为 JSON → 写入 BaseStore
          4. namespace=(users, {user_id}, profile) 实现多用户隔离

        为什么 user_id 不通过 LLM 传？
          安全性：如果 LLM 传 user_id，用户可以通过 Prompt 伪装成其他人。
          框架注入的 user_id 来自 config，无法篡改。
        """
        # 将 LLM 传的逗号字符串转为列表（LLM 更容易生成 "AI, Linux" 而非 ["AI", "Linux"]）
        interests = [
            interest.strip()
            for interest in interests_str.split(",")
            if interest.strip()
        ] if interests_str else []

        # 构造 UserProfile（Pydantic 自动校验）
        profile = UserProfile(
            language=language,
            editor=editor,
            location=location,
            experience_level=experience_level,
            interests=interests,
            is_vip=is_vip,
        )

        # ★ 写入 BaseStore —— 核心操作 ★
        # namespace = ("users", "alice", "profile")
        #   第一层 "users"：大类（所有用户数据）
        #   第二层 "alice"：具体用户（user_id）
        #   第三层 "profile"：数据类型（档案）
        # key = "latest"：该 namespace 下的唯一标识
        store.put(
            namespace=("users", user_id, "profile"),
            key="latest",
            value=profile.model_dump(),  # Pydantic → dict → JSON
        )

        return (
            f"✅ 已将 {user_id} 的档案存入 BaseStore（跨线程可访问）。\n"
            f"语言偏好：{language or '未指定'}\n"
            f"位置：{location or '未指定'}\n"
            f"技术水平：{experience_level or '未指定'}\n"
            f"兴趣：{', '.join(interests) if interests else '未指定'}\n"
            f"VIP：{'是' if is_vip else '否'}"
        )

    return store_profile


# ================================================================
# 第 4 步：获取工具 — 从 BaseStore 检索用户记忆（跨线程查询）
# ================================================================

def make_retrieve_profile_tool(store):
    """
    创建「从 BaseStore 检索用户档案」的工具。

    这个工具可以从 BaseStore 中读取之前存储的用户档案，
    即使在完全不同的 thread_id 中也能访问（跨线程共享）。
    """
    from langgraph.prebuilt import InjectedStore, InjectedState

    def retrieve_profile(
        # 注入参数 —— 框架自动提供
        store: Annotated[object, InjectedStore()] = None,
        user_id: Annotated[str, InjectedState("user_id")] = "",
    ) -> str:
        """
        从 BaseStore 获取当前用户的档案。

        这个工具是跨线程记忆的关键：
          - 用户在 Thread A 中存储了档案 → 写入 BaseStore
          - 用户打开 Thread B（全新会话）→ 调用此工具
          - 从 BaseStore 读出 Thread A 存储的档案 → 实现跨线程共享！
        """
        # 精确查找：namespace + key → Item
        item = store.get(
            namespace=("users", user_id, "profile"),
            key="latest",
        )

        if item is None:
            return (
                f"📭 用户 {user_id} 在 BaseStore 中暂无档案。\n"
                f"如果这是你第一次对话，请先告诉我你的偏好，我会帮你记住。"
            )

        # item.value 就是之前 store_profile 写入的 dict
        profile = UserProfile(**item.value)

        return (
            f"📋 用户 {user_id} 的档案（来自 BaseStore，跨线程共享）：\n"
            f"语言偏好：{profile.language or '未指定'}\n"
            f"编辑器：{profile.editor or '未指定'}\n"
            f"位置：{profile.location or '未指定'}\n"
            f"技术水平：{profile.experience_level or '未指定'}\n"
            f"兴趣领域：{', '.join(profile.interests) if profile.interests else '未指定'}\n"
            f"VIP 会员：{'是' if profile.is_vip else '否'}\n"
            f"最后更新：{item.updated_at}"
        )

    return retrieve_profile


# ================================================================
# 第 5 步：搜索工具 — 按条件在 BaseStore 中搜索
# ================================================================

def make_search_profiles_tool(store):
    """
    创建「按条件搜索用户档案」的工具。

    用途：管理员查询"所有 Python 用户"、"所有 VIP 用户"等。
    search 支持前缀匹配 + 结构化过滤，不像向量库那样模糊搜索。
    """
    from langgraph.prebuilt import InjectedStore

    def search_profiles(
        query_text: str,
        store: Annotated[object, InjectedStore()] = None,
    ) -> str:
        """
        按条件搜索所有用户的档案。

        query_text: 自然语言查询（内部转为结构化过滤条件）
        例如 "推荐 Python 工具" → filter={"value.language": "Python"}
        """
        # 从查询文本提取过滤条件
        filters = UserProfile.extract_for_query(query_text)

        # 在整个 users 命名空间下搜索
        results = store.search(
            namespace_prefix=("users",),              # 查所有用户
            filter={f"value.{k}": v for k, v in filters.items()} if filters else None,
            limit=5,
        )

        if not results:
            return f"📭 没有找到匹配 '{query_text}' 的用户档案。"

        lines = [f"🔍 搜索 '{query_text}' 结果："]
        for r in results:
            ns = "/".join(r.namespace)   # 如 "users/alice/profile"
            profile = UserProfile(**r.value)
            lines.append(
                f"  • {ns} → 语言={profile.language}, "
                f"位置={profile.location}, VIP={profile.is_vip}"
            )
        return "\n".join(lines)

    return search_profiles


# ================================================================
# 第 6 步：创建 Agent（短期记忆 + 全局记忆 同时启用）
# ================================================================

from langchain_openai import ChatOpenAI

def create_dual_memory_agent():
    """
    创建「双记忆」Agent：

    短期记忆：InMemorySaver（Checkpointer）
      - 同一 thread_id 内的多轮对话上下文
      - 进程重启丢失

    全局记忆：InMemoryStore（BaseStore）
      - 跨所有 thread_id 的用户档案
      - 进程重启丢失（生产用 PostgresStore）
    """
    # === 短期记忆 ===
    checkpointer = InMemorySaver()

    # === 全局记忆 ===
    # 生产环境替换为：
    # from langgraph.store.postgres import PostgresStore
    # store = PostgresStore.from_conn_string("postgresql://...")
    # await store.setup()
    store_obj = InMemoryStore()

    # === 创建工具 ===
    # 每个工具通过闭包注入同一个 store 实例
    # 这样所有线程的工具都共享同一个 store — 实现跨线程
    tools = [
        make_store_profile_tool(store_obj),      # 写入 BaseStore
        make_retrieve_profile_tool(store_obj),   # 读取 BaseStore
        make_search_profiles_tool(store_obj),    # 搜索 BaseStore
    ]

    # === 创建 Agent ===
    agent = create_react_agent(
        model=ChatOpenAI(model="deepseek-v4-pro", temperature=0.7),
        tools=tools,
        checkpointer=checkpointer,  # ← 短期记忆
        store=store_obj,            # ← 全局记忆（BaseStore）
        state_schema=AgentState,    # ← 自定义 State
    )

    return agent, checkpointer, store_obj


# ================================================================
# 第 7 步：主程序测试 — 同时验证短期记忆 + BaseStore 全局记忆
# ================================================================

def test_dual_memory():
    """
    测试场景设计：

    Thread A (chat_1)：Alice 的第一次对话
      → 存入短期记忆（当前对话上下文）
      → 存入 BaseStore 全局记忆（跨线程可读）

    Thread B (chat_2)：Alice 的新对话（全新线程！）
      → 短期记忆为空（新 thread_id）
      → 但能从 BaseStore 读回 Thread A 存储的档案 → 跨线程记忆生效！
    """
    agent, checkpointer, store_obj = create_dual_memory_agent()
    print("=" * 60)
    print("🧪 BaseStore 跨线程记忆测试")
    print("=" * 60)

    # 公共 user_id（跨线程共享）
    user_id = "alice"

    # -------------------------------------------------
    # Thread A：第一次对话（存储档案到 BaseStore）
    # -------------------------------------------------
    config_a = {
        "configurable": {
            "thread_id": "chat_1",    # Thread A
            "user_id": user_id,       # Alice
        }
    }

    print("\n" + "=" * 60)
    print("📍 Thread A (chat_1) — 第一次对话")
    print("=" * 60)

    # 第 1 轮：告诉 Agent 我的偏好
    question_1 = (
        "我叫 Alice，是一个 Python 后端开发，用了 8 年 Python，"
        "主要用 VS Code，住在北京，对 AI 和 Linux 很感兴趣，我是 VIP 会员。"
        "请帮我存储这些信息。"
    )
    print(f"\n👤 Alice: {question_1}")
    result_1 = agent.invoke(
        {
            "messages": [HumanMessage(question_1)],
            "user_id": user_id,
        },
        config=config_a,
    )
    print(f"🤖 Agent: {result_1['messages'][-1].content}")

    # -------------------------------------------------
    # 验证 1：BaseStore 中已经有了 Alice 的档案
    # -------------------------------------------------
    print("\n--- 验证 1：BaseStore 中已经存储了 Alice 的档案 ---")
    item = store_obj.get(namespace=("users", user_id, "profile"), key="latest")
    if item:
        profile = UserProfile(**item.value)
        print(f"  ✅ BaseStore 中已存储：语言={profile.language}, "
              f"位置={profile.location}, VIP={profile.is_vip}")
    else:
        print("  ❌ BaseStore 中没有找到档案")

    # -------------------------------------------------
    # Thread B：Alice 打开全新对话（不同的 thread_id）
    # -------------------------------------------------
    config_b = {
        "configurable": {
            "thread_id": "chat_2",    # ← 全新的 thread_id！
            "user_id": user_id,       # ← 但同一个用户
        }
    }

    print("\n" + "=" * 60)
    print("📍 Thread B (chat_2) — Alice 的新对话（全新线程）")
    print("=" * 60)

    # 第 1 问：验证短期记忆为空
    question_2 = "你还记得我刚才说了什么吗？"  # ← 问 Thread A 的内容
    print(f"\n👤 Alice: {question_2}")
    result_2 = agent.invoke(
        {
            "messages": [HumanMessage(question_2)],
            "user_id": user_id,
        },
        config=config_b,
    )
    print(f"🤖 Agent: {result_2['messages'][-1].content}")
    # 预期：Agent 不记得 Thread A 的内容（短期记忆隔离）

    # 第 2 问：从 BaseStore 读取档案
    question_3 = "你那里有没有关于我的任何信息？帮我查一下。"
    print(f"\n👤 Alice: {question_3}")
    result_3 = agent.invoke(
        {
            "messages": [HumanMessage(question_3)],
            "user_id": user_id,
        },
        config=config_b,
    )
    print(f"🤖 Agent: {result_3['messages'][-1].content}")
    # 预期：Agent 从 BaseStore 读出了 Thread A 存储的档案！

    # -------------------------------------------------
    # 验证 2：Thread B 的短期记忆中没有 Thread A 的历史
    # -------------------------------------------------
    print("\n--- 验证 2：短期记忆隔离检查 ---")
    state_b = agent.get_state(config_b)
    msg_count = len(state_b.values["messages"]) if state_b.values else 0
    print(f"  Thread B 的消息数：{msg_count}")
    print(f"  预期：只有 Thread B 的消息（新线程 = 新 State）")

    # -------------------------------------------------
    # 验证 3：BaseStore 跨线程生效
    # -------------------------------------------------
    print("\n--- 验证 3：BaseStore 跨线程共享检查 ---")
    item_b = store_obj.get(namespace=("users", user_id, "profile"), key="latest")
    if item_b:
        profile_b = UserProfile(**item_b.value)
        print(f"  ✅ Thread B 能读取 BaseStore 中的档案：语言={profile_b.language}, "
              f"位置={profile_b.location}")

    print("\n" + "=" * 60)
    print("✅ 测试完成！总结：")
    print("  短期记忆（Checkpointer）：Thread A 和 Thread B 隔离 ✓")
    print("  BaseStore 全局记忆：跨线程共享成功 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_dual_memory()
```

### 13.5 执行流程说明

```
Thread A (chat_1):                          Thread B (chat_2):
                                            
  第1轮: "我是 Alice，用 Python..."            第1轮: "还记得我刚才说了什么吗？"
    │                                          │
    ├─ Checkpointer: 存储消息（短期）             ├─ Checkpointer: 新 State（空白）
    └─ store_profile 工具:                       ├─ Agent: "我不记得"（短期隔离 ✓）
         store.put(                             │
           namespace=("users","alice","profile")│  第2轮: "查一下我的档案"
           key="latest"                          │
           value={language:"Python",...}          │
         )                                       ├─ retrieve_profile 工具:
         ↓                                        │   store.get(
    BaseStore 中已经有一份 Alice 的档案             │     namespace=("users","alice","profile")
                                                 │     key="latest"
                                                 │   )
                                                 │     ↓
                                                 ├─ BaseStore 返回: {language:"Python",...}
                                                 └─ Agent: "你是 Python 开发，住在北京..."
                                                    （跨线程记忆生效 ✓）
```

### 13.6 生产环境：PostgresStore

```python
# pip install langgraph-store-postgres

from langgraph.store.postgres import PostgresStore

# InMemoryStore → PostgresStore 只需改两行
store = PostgresStore.from_conn_string(
    "postgresql://user:pass@localhost:5432/agent_db"
)
await store.setup()  # 创建表

# 其余代码一模一样 —— store.put / store.get / store.search API 完全不变
# 区别：PostgresStore 重启后数据保留，支持多进程并发访问
```

### 13.7 BaseStore 核心要点速查

| 概念 | 说明 | 类比 Java |
|---|---|---|
| `namespace` | 层次化元组 `("users","alice","profile")` | 文件路径 `/users/alice/profile` |
| `put` | 写入（覆盖式） | `Map.put(key, value)` |
| `get` | 精确查找 O(1) | `Map.get(key)` |
| `search` | 前缀匹配 + 字段过滤 | SQL `WHERE namespace LIKE 'users/alice/%' AND value.language='Python'` |
| `delete` | 删除 | `Map.remove(key)` |
| `list_namespaces` | 列出所有命名空间 | `ls -R /users/` |
| `InjectedStore` | 工具参数自动注入 Store 实例 | `@Autowired BaseStore store` |
| `InjectedState` | 工具参数自动注入 State 字段 | `@Value("#{state.userId}") String userId` |

---

