## 第十三章：文件系统集成 — Agent 的外挂大脑

### 13.1 FilesystemMiddleware = Agent 的"操作系统"

在传统 Agent 中，工具返回结果全部堆积在对话历史里。DeepAgents 引入了 `FilesystemMiddleware`——**让 Agent 拥有类 Unix 的文件操作能力**，把文件系统当作"外挂大脑"。

```
传统 Agent 的记忆层次：            DeepAgents Agent 的记忆层次：
                                  
  上下文窗口（唯一记忆）              上下文窗口（工作记忆）
  ├─ System Prompt                 ├─ System Prompt
  ├─ 对话历史                      ├─ 最近对话（精简）  
  ├─ 工具返回 × N (噪声！)          └─ 当前任务上下文
  └─ ...（越堆越多）                   │
                                      │ 外挂
                                      ▼
                                   文件系统（外部持久记忆）
                                   ├─ /workspace/TODO.md
                                   ├─ /workspace/notes/
                                   ├─ /workspace/reports/
                                   └─ ...（容量无限）
```

**为什么是"外挂大脑"**：

| 大脑特性 | 上下文窗口 | 文件系统 |
|---|---|---|
| 容量 | 128K tokens（有限） | 无限（磁盘上限） |
| 持久化 | 对话结束 = 丢失 | 永久保留 |
| 结构化 | 线性消息列表 | 目录树 + 文件内容 |
| 检索 | 只能顺序浏览 | grep / glob 精确定位 |
| 共享 | 不可跨对话 | 跨对话、跨 Agent 共享 |

### 13.2 核心工具集

`FilesystemMiddleware` 自动注入 7 个文件操作工具，Agent 无需手动定义：

```
┌────────────────────────────────────────────────────────────┐
│                FilesystemMiddleware 工具集                  │
│                                                            │
│  📖 读取类                                                 │
│  ├─ ls(path)          → 列出目录内容                       │
│  ├─ read(path,offset,limit) → 分页读取文件（带行号）        │
│  └─ glob(pattern)     → 通配符匹配文件                      │
│                                                            │
│  ✏️ 写入类                                                 │
│  ├─ write(path,content) → 创建新文件                       │
│  └─ edit(path,old,new)  → 精确字符串替换（需先 read）       │
│                                                            │
│  🔍 搜索类                                                 │
│  └─ grep(pattern,path) → 文件内容搜索                       │
│                                                            │
│  ⚡ 执行类                                                 │
│  └─ execute(command)   → 沙箱运行 Shell 命令               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**每个工具的使用场景**：

```python
# Agent 在完成任务时的典型工具调用序列：

# 1. 探索——看看工作区有什么
ls("/workspace/")
# → ["TODO.md", "notes/", "reports/"]

# 2. 读取——了解当前进度
read("/workspace/TODO.md")
# → 返回带行号的 TODO 内容

# 3. 搜索——精确找到需要的信息
grep("LangChain", path="/workspace/notes/")
# → ["/workspace/notes/1.md:12: LangChain 的 Agent 模块..."]

# 4. 创建——产生新内容
write("/workspace/reports/final.md", content="...")

# 5. 修改——更新已有内容
edit("/workspace/TODO.md", "- [ ] 3. 写报告", "- [x] 3. 写报告")

# 6. 匹配——批量查找文件
glob("/workspace/notes/*.md")
# → ["/workspace/notes/langchain.md", "/workspace/notes/llamaindex.md"]
```

### 13.3 大结果自动转存机制

`FilesystemMiddleware` 最巧妙的设计：**工具返回结果太大时，自动写入文件，上下文只保留一句提示**。

**传统方式** — 工具返回全部堆积在对话历史：

```python
# 传统 Agent 搜索返回了 5000 字的搜索结果
[Human] 帮我研究 RAG
[AI] tool_calls: search("RAG")
[Tool] RAG (Retrieval-Augmented Generation) 是一种检索增强生成技术...
       (以下 5000 字搜索结果全部堆在消息历史中)
       ... 结论：RAG 是当前最有效的 LLM 知识增强方案之一。
```
→ 这 5000 字永远在上下文中，每次 LLM 调用都要处理一遍。

**DeepAgents 方式** — 超过阈值自动转存文件：

```python
# DeepAgents: 搜索返回 > 20000 tokens → 自动写入文件
[Human] 帮我研究 RAG
[AI] tool_calls: search("RAG")
[Tool] 工具返回内容过长（5000 字），已自动转存到 /workspace/.artifacts/search_abc123.md
       前 200 字预览：RAG (Retrieval-Augmented Generation) 是一种...
       使用 read("/workspace/.artifacts/search_abc123.md") 查看完整内容。
```
→ 上下文中只保留 200 字预览 + 文件路径。需要时 `read` 即可。

**自动转存的触发机制**：

```
每次 ToolMessage 返回后，FilesystemMiddleware 检查：
  1. 计算 ToolMessage.content 的 Token 估算值
  2. content_tokens > tool_token_limit_before_evict (默认 20000)？
     ├─ 否 → 保留在上下文中（正常）
     └─ 是 → 自动转存：
          ├─ 将完整内容写入 /workspace/.artifacts/{tool}_{id}.md
          ├─ 替换 ToolMessage.content 为简短提示：
          │   "内容过长（估计 N tokens），已转存到 [文件路径]。
          │    前 200 字预览：..."
          └─ 后续 LLM 调用不再看到这 5000 字，除非主动 read 文件

同样，HumanMessage 超过 human_message_token_limit_before_evict (默认 50000) 
也会自动转存，防止用户上传的超长文档撑爆上下文。
```

**为什么这样设计**：

| 好处 | 说明 |
|---|---|
| **上下文永远精简** | 只有最近/最重要的信息在上下文中，旧的自动"归档" |
| **Agent 自主选择** | Agent 决定哪些信息需要回顾（主动 read），哪些可以忽略 |
| **可恢复** | 信息没有丢失——只是从上下文移到了文件系统，随时可读 |
| **Token 成本降** | 每次 LLM 调用不重复处理 5000 字的旧结果 |
| **类似人脑** | 短期记忆有限 → 记到纸上 → 需要时翻看 |

### 13.4 Backend 体系 — 文件系统的底层实现

`FilesystemMiddleware` 通过统一的 `BackendProtocol` 接口操作文件。不同 Backend 实现这套接口，提供不同的存储后端。

#### 所有可用 Backend

```python
from deepagents.backends import (
    FilesystemBackend,    # 本地文件系统（默认）
    StateBackend,         # AgentState 内存存储
    StoreBackend,         # BaseStore 持久化存储
    CompositeBackend,     # 多后端混合路由
    LangSmithSandbox,     # LangSmith 云端沙箱
    LocalShellBackend,    # 本地 Shell 执行
)
```

#### 各 Backend 详解

**1. FilesystemBackend — 本地文件系统**

```python
from deepagents.backends import FilesystemBackend

backend = FilesystemBackend(
    root_dir="./agent_workspace",    # 文件存储根目录
    # virtual_mode=False,            # 默认：真实文件系统
    # virtual_mode=True,             # 虚拟模式：不落盘（测试用）
    max_file_size_mb=10,             # 单文件最大 10MB
)

agent = create_deep_agent(model="deepseek-v4-pro", backend=backend)
# Agent 的所有文件操作都在 ./agent_workspace/ 目录下
```

**2. StateBackend — 内存文件系统**

```python
from deepagents.backends import StateBackend

backend = StateBackend()
# 所有文件存在 AgentState 中（内存），不落盘
# 优点：快速、零配置、随 Checkpointer 持久化
# 缺点：进程重启丢失（除非用 PostgresSaver）、大文件占内存

agent = create_deep_agent(model="deepseek-v4-pro", backend=backend)
```

**3. LangSmithSandbox — 云端隔离执行**

```python
from deepagents.backends import LangSmithSandbox
from langsmith import Client

sandbox = Client().create_sandbox()
backend = LangSmithSandbox(sandbox=sandbox)
# 所有代码执行在 LangSmith 云端沙箱中
# 优点：完全隔离、预装常用库、大内存/CPU
# 缺点：需网络、有延迟、可能有费用
```

**4. LocalShellBackend — 本地 Shell + 超时控制**

```python
from deepagents.backends import LocalShellBackend

backend = LocalShellBackend(
    timeout=120,              # 命令超时 120 秒
    max_output_bytes=100000,  # 输出上限 100KB
    env={"PATH": "/usr/bin"}, # 环境变量
    inherit_env=False,        # 不继承当前进程环境（更安全）
)

# Docker 后端（通过 LocalShellBackend 实现）
docker_backend = LocalShellBackend(
    root_dir="/workspace",
    timeout=300,
    # 所有命令通过 docker exec 在容器中执行
)
```

**5. StoreBackend — BaseStore 持久化**

```python
from deepagents.backends import StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
backend = StoreBackend(store=store)
# 文件内容存入 BaseStore → 跨线程共享
# 优点：天然跨 Agent/跨 Thread 共享文件
# 生产：换成 PostgresStore → 文件持久化到 PostgreSQL
```

#### Backend 选择指南

| Backend | 持久化 | 速度 | 隔离 | 适用 |
|---|---|---|---|---|
| `FilesystemBackend` | 是 | 快 | 本地 OS | 开发/单机生产 |
| `StateBackend` | 随 Checkpointer | 最快 | 无 | 测试/轻量任务 |
| `LocalShellBackend` | 是 | 快 | Docker/子进程 | 代码执行 + 安全隔离 |
| `LangSmithSandbox` | 是（云端） | 慢（网络） | 完全 | 非信任务 + 大计算 |
| `StoreBackend` | 随 Store | 中 | BaseStore 级别 | 跨 Agent 共享 |

### 13.5 Backend vs Checkpointer vs Store 三维对比

这三个组件容易混淆——它们都涉及"存储"，但定位完全不同：

| 维度 | Backend | Checkpointer | Store (BaseStore) |
|---|---|---|---|
| **定位** | Agent 的工具文件操作的后端 | Agent 状态（State）的持久化 | 跨线程的结构化键值存储 |
| **负责什么** | ls/read/write/edit/grep 的实际实现 | 每次 State 变更的快照和恢复 | 跨 Thread 的任意 JSON 数据共享 |
| **数据类型** | 文件系统（文本/二进制） | `AgentState`（messages + 自定义字段） | 结构化 `dict`（任意 JSON） |
| **典型实现** | FilesystemBackend / StateBackend / CompositeBackend | InMemorySaver / PostgresSaver / SqliteSaver | InMemoryStore / PostgresStore |
| **生命周期** | 文件持久化（永久/虚拟） | Thread 生命周期 + Checkpoint 链 | 全局（namespace 隔离） |
| **API 风格** | 类 Unix 文件操作 | get/put/list | get/put/search/delete |
| **类比 Java** | `java.nio.file.FileSystem` | 数据库 WAL（预写日志） | Redis / Memcached |
| **谁在用** | FilesystemMiddleware | Agent 执行引擎 | MemoryMiddleware、跨 Agent 共享 |

**它们的关系**：

```
用户问你 "上次的研究报告还在吗？"
  │
  ├─ Checkpointer: 把当前对话的 State 恢复到上次的 checkpoint
  │    → messages 列表恢复了，包含 "研究完成" 那条 AIMessage
  │
  ├─ Backend: read("/workspace/reports/final.md")
  │    → 文件系统中有，返回报告内容
  │    → 报告是通过 FilesystemMiddleware 的 write 工具写入的
  │
  └─ Store: store.get(namespace=("users","alice","memory"), key="last_report")
       → 跨 Thread 的元信息：报告的创建时间、主题标签
       → 即使换了一个 thread_id，也知道 Alice 上次写过报告
```

### 13.6 CompositeBackend — 混合云文件系统

**核心思想**：不同路径路由到不同 Backend，像 Nginx 的 location 路由。

```python
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend

composite = CompositeBackend(
    # 默认后端（未匹配路由时使用）
    default=FilesystemBackend(root_dir="./workspace"),

    # 路由规则——最长前缀匹配
    routes={
        "/memories/": StoreBackend(store=InMemoryStore()),
        # ↑ /memories/ 下的文件存到 BaseStore（跨线程共享）
        
        "/cache/": StateBackend(),
        # ↑ /cache/ 下的文件存到 State（进程内，速度快）
        
        "/sandbox/": LangSmithSandbox(sandbox=my_sandbox),
        # ↑ /sandbox/ 下的代码执行在云端沙箱
    },
)
```

**CompositeBackend 的四大优势**：

**优势 1：性能强 — 热数据走内存，冷数据走磁盘**

```python
# 频繁读写的文件走 StateBackend（内存），不落盘 = 极快
# 大文件/归档走 FilesystemBackend（磁盘），不占内存
composite = CompositeBackend(
    default=FilesystemBackend("./workspace"),
    routes={
        "/temp/": StateBackend(),           # 临时文件 → 内存
        "/.artifacts/": FilesystemBackend("./archive"),  # 归档 → 磁盘
    },
)
```

**为什么比单 Docker Backend 快**：

| 操作 | 全 Docker Backend | Composite (State + Local) |
|---|---|---|
| ls("/temp/") | docker exec ls → 100ms+ | 内存读取 → <1ms |
| write(小文件) | docker exec → 200ms+ | 内存写入 → <1ms |
| execute(脚本) | docker exec → 300ms | 本地 Shell → 50ms |

热数据走内存（StateBackend）= 读写延迟从百毫秒降到微秒级。

**优势 2：计算与存储隔离**

```
CompositeBackend 的路由将"执行环境"和"存储位置"解耦：

  /workspace/  → FilesystemBackend    ← 代码和数据存本地磁盘
  /sandbox/    → LangSmithSandbox     ← 执行在云端隔离环境
  /memories/   → StoreBackend         ← 持久记忆存数据库

Agent 不需要知道后端差异——它只操作文件路径。
框架根据路径自动路由到正确的后端。
```

**优势 3：混合云 — 让 Agent 像人一样选择工作环境**

```python
# 类比人类：
#   - 本地草稿 → /workspace/
#   - 共享文档 → /memories/（团队成员都能看）
#   - 危险操作 → /sandbox/（隔离执行，不影响本地）

composite = CompositeBackend(
    default=FilesystemBackend("./my_workspace"),
    routes={
        "/memories/":  StoreBackend(store=shared_store),    # 团队知识库
        "/sandbox/":   LangSmithSandbox(sandbox=sb),        # 隔离执行
        "/artifacts/": FilesystemBackend("./.artifacts"),    # 本地归档
    },
)
```

**优势 4：解决的核心问题**

| 问题 | 单 Backend 的痛点 | CompositeBackend 的解法 |
|---|---|---|
| **跨 Agent 共享** | 本地文件不能被其他 Agent 读取 | `/memories/` → StoreBackend，天然共享 |
| **安全隔离** | 本地执行有风险 | `/sandbox/` → 云端沙箱，隔离执行 |
| **内存爆炸** | 大文件全放 State | `/artifacts/` → 磁盘，State 只放热数据 |
| **性能瓶颈** | 所有 I/O 走同一慢后端 | 热路径走内存，冷路径走磁盘 |

**CompositeBackend 完整示例**：

```python
from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend, FilesystemBackend, StateBackend, StoreBackend,
)
from langgraph.store.memory import InMemoryStore

# 共享存储（跨 Agent）
shared_store = InMemoryStore()
shared_backend = StoreBackend(store=shared_store)

# 本地工作区
local_backend = FilesystemBackend(root_dir="./agent_workspace")

# 快速缓存
cache_backend = StateBackend()

composite = CompositeBackend(
    default=local_backend,  # 默认：普通文件存本地
    routes={
        "/shared/": shared_backend,    # 共享文件 → BaseStore
        "/cache/":  cache_backend,     # 临时缓存 → State
    },
)

agent = create_deep_agent(
    model="deepseek-v4-pro",
    backend=composite,
    system_prompt="你是企业助手。共享文件存 /shared/，临时文件存 /cache/。",
)

# Agent 使用时完全无感知：
# write("/shared/report.md") → 自动路由到 StoreBackend
# write("/workspace/draft.md") → 自动路由到 FilesystemBackend
# write("/cache/temp.json") → 自动路由到 StateBackend
```

---

