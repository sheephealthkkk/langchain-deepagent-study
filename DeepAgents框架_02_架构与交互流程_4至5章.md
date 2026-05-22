## 第四章：DeepAgents 架构详解

### 4.1 整体架构

```
create_deep_agent()
      │
      ├──→ 组装 Middleware
      │      ├─ FilesystemMiddleware    (文件系统操作)
      │      ├─ MemoryMiddleware        (持久记忆)
      │      ├─ SubAgentMiddleware      (子 Agent 委派)
      │      ├─ SummarizationMiddleware (上下文压缩)
      │      └─ 用户自定义中间件
      │
      ├──→ 注册 Backend
      │      ├─ FilesystemBackend / StateBackend / CompositeBackend
      │      └─ 提供文件系统操作的真实实现
      │
      ├──→ 构建 LangGraph StateGraph
      │      ├─ 节点: agent → tools → agent (ReAct 循环)
      │      ├─ 节点: planning → agent (Task Planning)
      │      └─ 节点: subagent_spawn → subagent_execute
      │
      ├──→ 绑定 Profiles
      │      └─ 根据模型自动选择最优配置
      │
      └──→ 返回 CompiledStateGraph (可 invoke/stream)
```

### 4.2 核心组件

| 组件 | 位置 | 作用 |
|---|---|---|
| `create_deep_agent()` | `deepagents` | 工厂函数，一行创建完整 Agent |
| `FilesystemMiddleware` | `deepagents.middleware` | 提供 ls/read/write/edit/glob/grep 工具 |
| `MemoryMiddleware` | `deepagents.middleware` | 提供持久记忆的存储和检索 |
| `SubAgentMiddleware` | `deepagents.middleware` | 子 Agent 生成和委派 |
| `SubAgent` / `AsyncSubAgent` | `deepagents` | 定义子 Agent 的 name/description/prompt/tools |
| `SummarizationMiddleware` | `deepagents.middleware` | 自动压缩上下文 |
| `BackendProtocol` | `deepagents.backends` | 文件系统后端的抽象协议 |
| `CompositeBackend` | `deepagents.backends` | 多后端路由（如 /workspace → 本地，/sandbox → 远程） |
| `HarnessProfile` | `deepagents.profiles` | 针对不同模型（Claude/GPT/DeepSeek）的优化配置 |

### 4.3 底层 Graph 结构

```python
# DeepAgents 内部构建的 LangGraph 图（简化）
StateGraph(AgentState)
    │
    ├─ START → agent_node
    │           │
    │           ├─ LLM 输出 content（无 tool_calls）
    │           │   └─ → END
    │           │
    │           ├─ LLM 输出 tool_calls
    │           │   ├─ 是文件系统工具？→ FilesystemMiddleware 处理
    │           │   ├─ 是子 Agent 调用？→ SubAgentMiddleware 处理
    │           │   ├─ 是记忆操作？→ MemoryMiddleware 处理
    │           │   └─ 是普通工具？→ ToolNode 处理
    │           │
    │           └─ → agent_node（循环）
    │
    └─ Checkpointer + Store（持久化）
```

---

## 第五章：完整交互流程图

### 5.1 一次请求的完整旅程

```
用户请求: "帮我研究 RAG 的最新进展，写一份对比报告"
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   Orchestrator (主 Agent)                    │
│                                                             │
│  [1] Context Engineer (上下文工程)                            │
│      ├─ MemoryMiddleware: 检索相关记忆                        │
│      │   "用户 Alice 偏好技术细节、讨厌营销话术"                │
│      ├─ Filesystem: 读取上次未完成的计划                       │
│      │   /workspace/plan.md                                  │
│      └─ 构建 System Prompt (动态注入用户偏好 + 工作计划)       │
│                                                             │
│  [2] Task Planning (任务规划)                                 │
│      ├─ LLM 分析目标 "RAG 对比报告"                           │
│      ├─ 拆解为子任务:                                         │
│      │   1. 搜索 RAG 论文 (→ research-agent)                  │
│      │   2. 搜索 LangChain RAG 文档 (→ research-agent)        │
│      │   3. 搜索 LlamaIndex 对比 (→ research-agent)           │
│      │   4. 整理关键发现 (→ writer-agent)                     │
│      │   5. 验证报告准确性 (→ reviewer-agent)                 │
│      └─ 写入 /workspace/plan.md                               │
│                                                             │
│  [3] SubAgent Spawning (并行委派)                              │
│      ├─ spawn("research-agent", "搜索 RAG 论文")              │
│      ├─ spawn("research-agent", "搜索 LangChain RAG")         │
│      └─ spawn("research-agent", "搜索 LlamaIndex")            │
│                                                             │
│  [4] SubAgent Execution (子 Agent 独立执行)                    │
│      ┌──────────────────────────────┐                        │
│      │  research-agent              │                        │
│      │  搜索 → 过滤 → 保存到文件     │                        │
│      │  /workspace/notes/rag_1.md   │                        │
│      │  /workspace/notes/rag_2.md   │                        │
│      │  /workspace/notes/rag_3.md   │                        │
│      └──────────────────────────────┘                        │
│                                                             │
│  [5] Aggregation (汇总)                                       │
│      ├─ 读 /workspace/notes/rag_*.md                         │
│      └─ spawn("writer-agent", "基于 3 个文件写对比报告")       │
│                                                             │
│  [6] Writer SubAgent                                          │
│      ┌──────────────────────────────┐                        │
│      │  writer-agent                │                        │
│      │  读文件 → 整理 → 写报告        │                        │
│      │  /workspace/report.md        │                        │
│      └──────────────────────────────┘                        │
│                                                             │
│  [7] Reviewer SubAgent                                        │
│      ┌──────────────────────────────┐                        │
│      │  reviewer-agent              │                        │
│      │  读报告 → 检查引用 → 验证逻辑  │                        │
│      │  修改 /workspace/report.md   │                        │
│      └──────────────────────────────┘                        │
│                                                             │
│  [8] Persistent Memory (持久记忆)                             │
│      ├─ MemoryMiddleware: 保存关键发现到长期记忆               │
│      │   "Alice 的 RAG 对比报告已完成，结论: LangChain for     │
│      │    快速原型，LlamaIndex for 数据管道"                   │
│      └─ 下次 Alice 再问 RAG 相关 → 直接从记忆调取              │
│                                                             │
│  [9] 返回结果                                                 │
│      └─ "报告已生成在 /workspace/report.md，摘要如下: ..."     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

