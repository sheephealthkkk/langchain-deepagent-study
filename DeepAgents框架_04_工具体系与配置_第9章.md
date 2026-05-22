## 第九章：深度上手 — 工具体系与配置

### 9.1 `create_deep_agent()` 完整参数

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    # ===== 模型配置 =====
    model="deepseek-v4-pro",               # 主 Agent 模型（字符串或 BaseChatModel 实例）
    # model=ChatOpenAI(model="deepseek-v4-pro", temperature=0.7),

    # ===== Prompt 配置 =====
    system_prompt="你是 AI 研究助手。",     # System Prompt（覆盖默认）

    # ===== 工具配置（详见 9.2~9.4）=====
    tools=[                                # 自定义工具列表（LangChain BaseTool）
        get_weather,
        search_web,
    ],

    # ===== 子 Agent 配置（详见 9.5）=====
    subagents=[
        SubAgent(
            name="researcher",
            description="搜索和整理信息",
            system_prompt="你是研究专家...",
            tools=[search_web, search_arxiv],  # 子 Agent 独有工具
        ),
    ],

    # ===== 中间件配置（详见 9.6）=====
    middleware=[                            # 额外的自定义中间件
        # DeepAgents 已内置的中间件（自动加载）：
        #   FilesystemMiddleware  — 文件系统操作
        #   MemoryMiddleware      — 持久记忆
        #   SubAgentMiddleware    — 子 Agent 委派
        #   SummarizationMiddleware — 上下文压缩
        # 用户自定义的中间件会追加到默认列表后面
        MyCustomMiddleware(),
    ],

    # ===== 记忆配置 =====
    memory=["user_preferences", "project_knowledge"],  # 启用的记忆源

    # ===== 文件系统配置 =====
    permissions=[                           # 文件操作权限
        FilesystemPermission.READ,          # 允许读
        FilesystemPermission.WRITE,         # 允许写
        FilesystemPermission.EDIT,          # 允许编辑
    ],
    backend=FilesystemBackend(root_dir="./my_workspace"),  # 自定义文件后端

    # ===== 安全配置 =====
    interrupt_on={                          # 哪些操作需要人工审批
        "delete_file": True,
        "execute_code": True,
        "send_email": {"allowed_decisions": ["approve", "reject"]},
    },

    # ===== 结构化输出 =====
    response_format=MyOutputSchema,         # 强制结构化输出

    # ===== 持久化配置 =====
    checkpointer=InMemorySaver(),           # 短期记忆（生产用 PostgresSaver）
    store=InMemoryStore(),                  # 长期记忆（生产用 PostgresStore）
    # 如果设为 None，DeepAgents 自动创建 InMemory 版本

    # ===== 其他配置 =====
    name="my-research-agent",              # Agent 名称
    debug=True,                             # 调试模式（详细日志）
    cache=InMemoryCache(),                  # LLM 响应缓存
)
```

### 9.2 Agent 加载的三类工具

```
DeepAgents Agent 的工具来源（自动聚合）：

┌──────────────────────────────────────────────────────────────┐
│                     Agent 最终工具列表                        │
│                                                              │
│  ┌────────────────────┐                                      │
│  │  系统工具（内置）    │ ← DeepAgents 自动注入，无需手动配置   │
│  │                    │                                      │
│  │  • ls              │  目录列表                             │
│  │  • read            │  读文件                               │
│  │  • write           │  写文件                               │
│  │  • edit            │  编辑文件（精确替换）                   │
│  │  • glob            │  文件模式匹配                          │
│  │  • grep            │  文件内容搜索                          │
│  │  • execute         │  执行代码（沙箱）                      │
│  │  • task            │  子任务描述和跟踪                      │
│  └────────────────────┘                                      │
│                         +                                     │
│  ┌────────────────────┐                                      │
│  │  记忆工具（内置）    │ ← MemoryMiddleware 注入              │
│  │                    │                                      │
│  │  • remember        │  记住信息                             │
│  │  • recall          │  回忆信息                             │
│  │  • search_memory   │  搜索记忆                             │
│  └────────────────────┘                                      │
│                         +                                     │
│  ┌────────────────────┐                                      │
│  │  自定义工具（用户）  │ ← 用户通过 tools= 参数传入            │
│  │                    │                                      │
│  │  • get_weather     │  @tool 装饰的函数                     │
│  │  • search_web      │  @tool 装饰的函数                     │
│  │  • query_database  │  继承 BaseTool 的类                   │
│  │  • ...             │  StructuredTool                      │
│  └────────────────────┘                                      │
│                         +                                     │
│  ┌────────────────────┐                                      │
│  │  子 Agent（用户）   │ ← 用户通过 subagents= 参数传入        │
│  │                    │                                      │
│  │  • researcher      │  作为工具出现（可被主 Agent 调用）     │
│  │  • code-reviewer   │  主 Agent 看到的是工具描述             │
│  │  • writer          │  调用子 Agent = 调用工具              │
│  └────────────────────┘                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 9.3 系统工具详解（文件系统工具）

DeepAgents 的文件系统工具通过 `FilesystemMiddleware` 注入，无需手动定义：

```python
# 这些工具自动可用，LLM 能直接调用（你不需要写任何代码）

# === ls — 列出目录内容 ===
# LLM 调用示例: ls(path="/workspace/notes")
# 返回: ["search_1.md", "search_2.md", "draft.md"]

# === read — 读取文件内容 ===
# LLM 调用示例: read(path="/workspace/notes/search_1.md")
# 返回: 文件全文（如果文件太大，只返回前 N 行 + 提示使用 offset/limit）

# === write — 写入文件 ===
# LLM 调用示例: write(path="/workspace/report.md", content="# RAG 对比报告\n\n...")
# 返回: "文件已写入: /workspace/report.md (1234 字节)"

# === edit — 精确编辑文件（替换指定行范围）===
# LLM 调用示例: edit(path="/workspace/report.md", old_string="## 旧标题", new_string="## 新标题")
# 返回: "文件已编辑: /workspace/report.md（1 处替换）"
# 类似 sed 's/old/new/' 但带文件感知

# === glob — 文件模式匹配 ===
# LLM 调用示例: glob(pattern="/workspace/**/*.md")
# 返回: ["/workspace/report.md", "/workspace/notes/search_1.md", ...]

# === grep — 文件内容搜索 ===
# LLM 调用示例: grep(pattern="RAG", path="/workspace/")
# 返回: [
#   "/workspace/notes/search_1.md:12: RAG (Retrieval-Augmented Generation)",
#   "/workspace/notes/search_2.md:5: RAG 的三种范式...",
# ]
```

**与 LangChain 的 `@tool` 定义对比**：

```python
# LangChain 方式：你要手动定义每个文件操作工具
@tool
def read_file(path: str) -> str:
    """读取文件"""
    with open(path) as f:
        return f.read()

@tool  
def write_file(path: str, content: str) -> str:
    """写入文件"""
    with open(path, "w") as f:
        f.write(content)
    return f"已写入 {path}"

# ... ls, edit, glob, grep 等每个都需要自己写

# DeepAgents 方式：0 行代码，全部内置
agent = create_deep_agent(model="deepseek-v4-pro")
# Agent 自动拥有 read/write/ls/edit/glob/grep 全部文件系统工具
```

### 9.4 自定义工具 — 与 LangChain 完全兼容

DeepAgents 的 `tools=` 参数接受任何 LangChain 工具：

```python
from langchain.tools import tool
from langchain_core.tools import StructuredTool, BaseTool
from pydantic import BaseModel, Field

# === 方式 1：@tool 装饰器（最常用，与 LangChain 完全相同）===
@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气。"""
    return f"{city}：晴，25°C"

# === 方式 2：StructuredTool（包装已有 Runnable）===
from langchain_core.tools import StructuredTool

async def search_api(query: str, limit: int = 5) -> str:
    """调用搜索 API"""
    results = await external_search(query, limit)
    return format_results(results)

search_tool = StructuredTool.from_function(
    coroutine=search_api,
    name="search_web",
    description="搜索互联网获取最新信息",
)

# === 方式 3：继承 BaseTool（复杂工具）===
class DatabaseQueryTool(BaseTool):
    name: str = "query_database"
    description: str = "查询公司内部数据库"
    args_schema: type[BaseModel] = DatabaseQueryInput

    query_count: int = 0  # 内部状态

    def _run(self, sql: str, limit: int = 100) -> str:
        self.query_count += 1
        # ... 真实数据库查询
        return results

# === 传入 DeepAgents ===
agent = create_deep_agent(
    model="deepseek-v4-pro",
    tools=[get_weather, search_tool, DatabaseQueryTool()],
    # ↑ 与 LangChain 的工具定义 100% 兼容
)
```

### 9.5 工具与中间件的对应关系

每种工具背后都有对应的中间件在管理：

```
工具                          背后中间件                   负责什么
────                          ──────────                   ────────
ls / read / write / edit     FilesystemMiddleware         文件操作 + 权限校验
  / glob / grep / execute
remember / recall /           MemoryMiddleware            记忆存取 + 过期管理
  search_memory
子 Agent 调用                  SubAgentMiddleware          子 Agent 生成/执行/通信
  (researcher / writer)
普通 @tool 函数               ToolNode (LangGraph)        标准工具执行 + 结果返回
上下文裁剪                    SummarizationMiddleware      Token 超限时触发压缩
危险操作审批                  HumanInTheLoopMiddleware     Interrupt → 等待人工
PII 脱敏                      PIIMiddleware               敏感信息检测和替换
Shell/代码执行安全            ShellToolMiddleware          沙箱策略管理
```

**关键理解**：你在 `create_deep_agent()` 里看到的简洁参数——`tools`、`subagents`、`memory`、`permissions`——背后都是这些中间件在协作。DeepAgents 的"约定大于配置"体现在：**你声明"我要记忆功能"（`memory=[...]`），框架自动装配对应的 MemoryMiddleware 全套工具和逻辑。**

### 9.6 中间件扩展 — 在默认基础上追加

```python
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
)

agent = create_deep_agent(
    model="deepseek-v4-pro",
    tools=[...],
    subagents=[...],

    # 自定义中间件会追加到 DeepAgents 默认中间件后面
    middleware=[
        # 默认中间件执行顺序：Filesystem → Memory → SubAgent → Summary
        # 你的中间件追加在最后 → 洋葱最内层

        ModelRetryMiddleware(max_retries=3),       # LLM 失败自动重试
        PIIMiddleware("email", strategy="redact"),  # PII 脱敏
    ],

    # 不需要重复配置的：
    # ✗ 不需要手动添加 FilesystemMiddleware
    # ✗ 不需要手动添加 MemoryMiddleware
    # ✗ 不需要手动添加 SubAgentMiddleware
    # ✗ 不需要手动添加 SummarizationMiddleware
    # DeepAgents 自动处理！
)
```

### 9.7 与 LangChain 的关键差异总结

| | LangChain 方式 | DeepAgents 方式 |
|---|---|---|
| **创建 Agent** | `create_agent(llm, tools, middleware=[...])` | `create_deep_agent(model, tools, subagents, memory, ...)` |
| **文件系统** | 需手写 `@tool` 实现每个文件操作 | 自动注入 `ls/read/write/edit/glob/grep` |
| **子 Agent** | 需手写 LangGraph 子图 + Send API | `SubAgent(name, description, prompt, tools)` 一行 |
| **持久记忆** | 需手动集成向量库 + RAG | `memory=["user_prefs", "project"]` 声明即启用 |
| **上下文管理** | 需手动 trim 或添加 SummarizationMiddleware | 自动压缩 + ContextEditing |
| **中间件** | 需手动列举所有中间件 | 默认中间件自动装配 + 用户追加 |
| **默认值** | 几乎无默认值 | System Prompt、工具列表、权限、后端 全有默认值 |
| **配置量** | 基础 Agent 需 20+ 行 | 基础 Agent 需 1 行 |

### 9.8 完整开发者工作流

```
第 1 步：一行起步
  agent = create_deep_agent(model="deepseek-v4-pro")
  → 测试基本功能

第 2 步：加系统 Prompt
  agent = create_deep_agent(model=..., system_prompt="你是...")
  → 定制 Agent 角色

第 3 步：加自定义工具
  agent = create_deep_agent(model=..., tools=[my_tool])
  → Agent 拥有文件系统 + 你的工具

第 4 步：加子 Agent
  agent = create_deep_agent(model=..., subagents=[researcher])
  → Agent 能委派复杂任务

第 5 步：加记忆
  agent = create_deep_agent(model=..., memory=["prefs"])
  → Agent 跨对话保持上下文

第 6 步：调权限 + 安全
  agent = create_deep_agent(model=..., permissions=[...], interrupt_on={...})
  → 生产级安全配置

第 7 步：换后端 + 持久化
  agent = create_deep_agent(model=..., backend=FilesystemBackend("./prod"),
                            checkpointer=PostgresSaver(...))
  → 生产环境部署
```

---

