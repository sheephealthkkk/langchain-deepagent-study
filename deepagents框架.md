# DeepAgents 框架教学

## 第一章：DeepAgents 是什么

### 1.1 定位

**DeepAgents = LangChain + LangGraph 团队打造的"电池全装"型 Agent 框架。** 它不是替代品，而是在前两者之上构建的高级抽象层。

```
┌─────────────────────────────────────────────┐
│              DeepAgents                      │
│         (高级 Agent 框架，开箱即用)            │
├─────────────────────────────────────────────┤
│              LangGraph                       │
│         (有状态编排引擎，图执行)               │
├─────────────────────────────────────────────┤
│              LangChain                      │
│    (Runnable 协议、工具、Prompt、中间件)       │
└─────────────────────────────────────────────┘
```

**一句话定位**：如果 LangChain 提供零件（Runnable、Tool、Prompt），LangGraph 提供引擎（Graph、State、Checkpoint），那 DeepAgents 提供**整机**——文件系统、子 Agent 生成、持久记忆、上下文工程全部内置。

### 1.2 适用任务

| 任务类型 | 示例 | 为什么适合 DeepAgents |
|---|---|---|
| **复杂研究** | "调查 LangChain vs LlamaIndex 差异，写对比报告" | 需文件系统存储中间结果、子 Agent 并行搜索 |
| **代码生成+调试** | "写一个 Flask API，包含用户认证、数据库、测试" | 需沙箱执行、文件编辑、多文件协调 |
| **数据分析** | "分析这份 CSV，找趋势，生成可视化报告" | 需 Python 执行、文件读写、图表生成 |
| **多步工作流** | "帮我规划并执行一个产品发布流程" | 需 Task Planning、分步执行、上下文跟踪 |
| **知识工作** | "阅读这篇论文并总结关键发现" | 需长文档处理、上下文压缩、归档 |

---

## 第二章：ReAct Agent 的"浅层陷阱"与 DeepAgents 的解法

### 2.1 传统 Agent 的五大缺陷

#### 缺陷 1：浅层陷阱（Shallow Trap）

传统 ReAct Agent 是"即兴反应"——想到一步做一步，没有全局规划。

```
用户: "帮我研究 RAG 的最新进展并写一份报告"

传统 ReAct:
  Thought: 先搜索 RAG → Action: search("RAG 2024")
  Thought: 再搜索 RAG 最新论文 → Action: search("RAG paper 2024")
  Thought: 还要看看 LangChain RAG → Action: search("LangChain RAG")
  Thought: 信息够了，写报告 → Action: write_report([所有搜索结果])

问题：
  - 没有规划阶段 —— 搜索了 3 次才觉得"够了"（可能不够，也可能多了）
  - 没有中间结果保存 —— 3 次搜索结果都在 context 里堆积
  - 没有验证 —— 写完报告不检查
```

```
DeepAgents:
  [Planner] 分析目标 → 生成任务计划:
    1. 搜索 RAG 最新论文 (assign to ResearchAgent)
    2. 搜索 LangChain RAG 文档 (assign to ResearchAgent)
    3. 搜索 LlamaIndex 对比 (assign to ResearchAgent)
    4. 整理关键发现 → 写报告 (assign to WriterAgent)
    5. 检查报告准确性 (assign to ReviewerAgent)

  [Executor] 并行执行 1~3 → 文件系统存储中间结果
  [Executor] 汇总 4 → 写报告 → 文件系统
  [Executor] 验证 5 → 检查引用 → 修改 → 完成

优势：
  ✓ 先规划后执行 → 步骤完整
  ✓ 文件系统中转 → 中间结果不丢
  ✓ 子 Agent 并行 → 效率高
```

#### 缺陷 2：规划能力丢失

传统 Agent 在长对话中逐渐"忘记"最初的目标。第 5 轮时，System Prompt 已被历史消息淹没。

DeepAgents 的 Task Planning 把计划写入**文件系统中的 TODO 文件**，每次推理前从中读取，不依赖对话历史：

```
文件系统充当"外部硬盘"：
  /workspace/plan.md     ← 整体计划
  /workspace/notes/      ← 搜索结果暂存
  /workspace/report.md   ← 正在写的报告
  /workspace/checklist.md ← 验证检查项
```

#### 缺陷 3：上下文污染

传统 Agent 的所有工具返回都堆积在对话历史中（context window），噪声越来越大：

```
对话历史（25 轮后）：
  [Human] 帮我研究 RAG
  [AI] 先查资料。tool_calls: search("RAG")
  [Tool] 搜索返回 5000 字结果...          ← 噪声
  [AI] 再查。tool_calls: search("RAG papers")
  [Tool] 搜索返回 3800 字结果...          ← 噪声
  ... (8 次搜索，每次返回几千字)
  [AI] 写报告。tool_calls: write(...)
  [Tool] 写入成功                         ← 有用的
  ...
  上下文已达 18000 tokens，其中 80% 是搜索噪声
```

DeepAgents 的文件系统中转方案——工具结果写文件，LLM 只看到"写入成功"：

```
对话历史（简洁）：
  [Human] 帮我研究 RAG
  [AI] tool_calls: search_and_save("RAG")  ← 搜索结果写到 /workspace/notes/
  [Tool] 已保存 3 条搜索结果到 /workspace/notes/search_1.md
  [AI] tool_calls: read("/workspace/notes/search_1.md") ← 按需读取
  [Tool] 文件内容：...（只读需要的）
  
  上下文始终保持精炼，不受搜索噪声污染
```

#### 缺陷 4：环境交互困难

传统 Agent 没有真正的文件系统——只能靠工具函数模拟。每个文件操作都是一个新的 tool_call → 慢且容易出错。

DeepAgents 内置 `FilesystemMiddleware` + 多种 Backend：

| Backend | 说明 | 适用 |
|---|---|---|
| `FilesystemBackend` | 本地文件系统 | 开发/单机 |
| `StateBackend` | 内存文件系统 | 测试/临时 |
| `CompositeBackend` | 多后端组合 | 混合场景 |
| `StoreBackend` | BaseStore 注入 | 跨线程共享 |
| `LangSmithSandbox` | 云端沙箱 | 安全隔离 |
| `LocalShellBackend` | Shell 执行 | 需要运行脚本 |

Agent 可以像人一样：`ls` 查看目录 → `read` 读文件 → `write` 写报告 → `edit` 修改 → `grep` 搜索。

#### 缺陷 5：协作编排复杂

传统 Agent 是单线程的。要让多个 Agent 协作 → 需要手写 LangGraph 图（节点 + 边 + 条件路由）。

DeepAgents 的 `SubAgent` 机制——创建子 Agent 就像调用工具一样简单：

```python
# 子 Agent 就是工具
subagents = [
    SubAgent(
        name="code-reviewer",
        description="审查代码质量、安全性和最佳实践",
        system_prompt="你是资深代码审查员...",
        tools=[read_file, search_web],
    ),
    SubAgent(
        name="test-writer",
        description="为给定代码编写单元测试",
        system_prompt="你是测试工程师...",
        tools=[read_file, execute_python],
    ),
]

agent = create_deep_agent(
    model="deepseek-v4-pro",
    subagents=subagents,
    ...
)
# 主 Agent 可以像调工具一样调用 code-reviewer 和 test-writer
```

---

## 第三章：类人工作流——DeepAgents 的核心设计

### 3.1 类人工作流模型

```
人类处理复杂任务的方式          DeepAgents 的对应设计
──────────────────────          ──────────────────────

1. 先规划，后执行              Task Planning + TODO 文件
   "先想清楚要做什么"              /workspace/plan.md

2. 外包给专家                   SubAgent 机制
   "这个我不懂，找专业的人做"       code-reviewer / researcher

3. 用纸笔记录中间结果            Filesystem Middleware
   "记下来，怕忘"                  /workspace/notes/*.md

4. 查参考资料                    Memory Middleware
   "你昨天说的那个..."            持久记忆检索

5. 回头检查                      Reviewer SubAgent
   "写完再读一遍"                  自动验证步骤
```

### 3.2 四大支柱组件

```
┌─────────────────────────────────────────────────────────────┐
│                   DeepAgents 四大支柱                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Task Planning│  │  Sub-Agents  │                        │
│  │              │  │              │                        │
│  │ 目标→计划     │  │ 主Agent→子   │                        │
│  │ 计划→步骤     │  │ Agent委派    │                        │
│  │ 步骤→TODO    │  │ 子Agent返回  │                        │
│  └──────┬───────┘  └──────┬───────┘                        │
│         │                 │                                 │
│         └────────┬────────┘                                 │
│                  ▼                                          │
│  ┌──────────────────────────────┐                          │
│  │        Orchestrator          │                          │
│  │    (LangGraph StateGraph)    │                          │
│  └──────────┬──────────┬────────┘                          │
│             │          │                                    │
│  ┌──────────┴──┐ ┌─────┴──────────┐                        │
│  │  Filesystem  │ │   Memory       │                        │
│  │  Middleware  │ │   Middleware   │                        │
│  │              │ │               │                        │
│  │ 文件系统操作  │ │  持久记忆管理  │                        │
│  │ ls/read/write│ │  检索/存储/过期│                        │
│  │ edit/grep    │ │               │                        │
│  └──────────────┘ └───────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

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

## 第六章：快速上手

### 6.1 最小示例

```python
from deepagents import create_deep_agent

# 一行创建完整 Agent（文件系统 + 子Agent + 记忆 全部内置）
agent = create_deep_agent(
    model="deepseek-v4-pro",       # 或 "claude-sonnet-4-6" / "gpt-4o"
    system_prompt="你是 AI 研究助手。使用文件系统管理中间结果。",
)

# 使用
result = agent.invoke(
    {"messages": [HumanMessage("帮我研究 RAG 的最新进展，结果保存为 report.md")]},
    config={"configurable": {"thread_id": "research_1"}},
)
```

### 6.2 带子 Agent + 文件系统

```python
from deepagents import create_deep_agent, SubAgent
from langchain_openai import ChatOpenAI

# 定义子 Agent
researcher = SubAgent(
    name="researcher",
    description="搜索和整理信息。适合做文献调研、资料收集。",
    system_prompt="你是研究专家。搜索时注重权威性和时效性。搜索结果保存到 /workspace/notes/。",
)

writer = SubAgent(
    name="writer",
    description="基于研究材料撰写报告。",
    system_prompt="你是技术写手。写报告时严格基于提供的研究材料，注明引用来源。",
)

agent = create_deep_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0.7),
    subagents=[researcher, writer],
    system_prompt=(
        "你是项目经理。分配研究任务给 researcher，汇总后交给 writer 写报告。"
        "所有中间结果保存到 /workspace/。"
    ),
)

result = agent.invoke({
    "messages": [HumanMessage("调查 GPT-5 发布后的市场反应，生成分析报告。")],
})
```

### 6.3 配置持久记忆

```python
agent = create_deep_agent(
    model="deepseek-v4-pro",
    memory=["user_preferences", "project_context"],  # ← 启用记忆
    # Agent 会在每次对话中自动：
    #   1. 检索相关记忆（"用户偏好什么风格？上次项目进展到哪？"）
    #   2. 回答后保存新信息（"用户表示偏好简洁的回复"）
)
```

---

## 第七章：核心要点总结

```
DeepAgents 解决了什么：

  传统 Agent 的缺陷              DeepAgents 的解法
  ──────────────────            ──────────────────────────
  浅层陷阱                       Task Planning + TODO 文件
  规划丢失                       文件系统充当外部工作记忆
  上下文污染                     工具结果写入文件，按需读取
  环境交互困难                   内置 Filesystem Middleware + 多后端
  协作编排复杂                   SubAgent 一键委派

架构层次：
  create_deep_agent() ──→ LangGraph StateGraph ──→ LangChain Runnable
        │                           │
    装配中间件                    图执行引擎
  (Filesystem / Memory           (节点 / 边 / 条件)
   / SubAgent / Summary)

核心对象：
  Orchestrator : 主 Agent，负责任务分解和委派
  SubAgent     : 子 Agent，执行具体子任务
  Filesystem   : 文件系统后端，充当外部工作记忆
  Memory       : 持久记忆，跨对话保持上下文
  Backend      : 文件系统底层实现（本地 / 沙箱 / 远程）

  一句话：
  DeepAgents 让 Agent 像人一样工作——
  先规划、外包给专家、用纸笔记录、完成后检查、记住经验教训。

---

## 第八章：Chain vs Graph vs DeepAgents 三维对比

### 8.1 三层抽象架构

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  抽象层级：                                                      │
│                                                                 │
│  DeepAgents  ████████████████████████████  最高抽象              │
│  (约定 > 配置，开箱即用，Spring Boot 级)                          │
│       │                                                         │
│       │ 构建于                                                   │
│       ▼                                                         │
│  LangGraph   ██████████████████           中等抽象               │
│  (有状态图编排，自定义流程，Spring Framework 级)                   │
│       │                                                         │
│       │ 构建于                                                   │
│       ▼                                                         │
│  LangChain   ██████████                   最低抽象               │
│  (Runnable 协议 + 零件，纯 JDK 级)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 全维度对比

| 维度 | LangChain (Chain) | LangGraph (Graph) | DeepAgents |
|---|---|---|---|
| **定位** | LLM 应用的基础零件库 | 有状态多步编排引擎 | 电池全装的 Agent 框架 |
| **核心抽象** | `Runnable` — 一切皆管道 | `StateGraph` — 一切皆有向图 | `create_deep_agent()` — 一切皆内置 |
| **编程模型** | 声明式 (`A \| B \| C`) | 图构建 (`add_node` + `add_edge`) | 工厂函数 (`create_deep_agent(...)`) |
| **流程控制** | 线性（输入→输出） | 图（循环、条件、分支、并行） | 图 + 内置规划（Task Planning） |
| **状态管理** | 无状态（需要手动管理） | 有状态（Checkpointer 持久化） | 有状态 + 文件系统 + 记忆 |
| **灵活性** | ★★★★★ 完全自由组合 | ★★★★ 可自定义图结构 | ★★★ 通过中间件 + SubAgent 扩展 |
| **上手难度** | ★★ 需理解 Runnable 协议 | ★★★★ 需理解图状态机 | ★ 一行代码出 Agent |
| **文件系统** | 无（需手动实现工具） | 无（需手动实现工具） | ★ 内置（ls/read/write/edit/grep/glob） |
| **子 Agent** | 无 | 需手写子图节点 | ★ 内置 SubAgent 委派 |
| **持久记忆** | 无（需手动集成向量库） | 部分（Checkpointer 是短期记忆） | ★ 内置 MemoryMiddleware |
| **上下文管理** | 需手动 trim | 需手动 trim 或中间件 | ★ 内置 Summarization + ContextEditing |
| **并行执行** | `batch()` 有限支持 | `Send()` API 支持 | ★ 子 Agent 天然并行 |
| **适用场景** | 简单链式调用、RAG、Prompt→LLM→Parser | 复杂 Agent、多步工作流、人机协作 | 复杂研究、代码生成、知识工作 |
| **类比 Java** | JDK 标准库（`List`, `Stream`） | Spring Framework（`@Bean`, `ApplicationContext`） | Spring Boot（`@SpringBootApplication` 一行启动） |

### 8.3 选择决策

```
你的任务特征                         → 选什么
─────────────────────────────────────────────────────
简单的 Prompt → LLM → 输出           → LangChain Chain
  "翻译这段文字"
  "总结这篇文章"

有状态的单 Agent + 自定义流程         → LangGraph
  "我需要精确控制 Agent 的每一步"
  "需要在特定步骤暂停等人工审批"
  "需要构建复杂的条件分支逻辑"

需要文件系统 + 子 Agent + 记忆         → DeepAgents
  "帮我研究一个课题，写报告"
  "分析代码仓库，找出安全问题"
  "多步知识工作任务"

混合使用（推荐）：
  DeepAgents 做顶层编排（任务分解 + 文件系统 + 记忆）
  内部子 Agent 用 LangGraph 做精细流程控制
  底层工具用 LangChain 的 @tool 定义
```

### 8.4 约定大于配置 — Spring 风格设计

DeepAgents 从 Spring Boot 借鉴了核心设计哲学：**给出合理默认值，用户只需要声明差异**。

```java
// Spring Boot 风格：一行注解启动整个应用
@SpringBootApplication
public class MyApp {
    public static void main(String[] args) {
        SpringApplication.run(MyApp.class, args);
    }
}

// DeepAgents 风格：一行工厂函数创建完整 Agent
agent = create_deep_agent(model="deepseek-v4-pro")
```

**对比传统 LangChain 方式**（配置 > 约定）：

```python
# LangChain 方式：每个零件都要手动组装（配置 > 约定）
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI(model="deepseek-v4-pro")
prompt = ChatPromptTemplate.from_messages([...])
parser = StrOutputParser()
chain = prompt | llm | parser
# 还需要手动：管理历史、裁剪上下文、处理工具调用...
```

**DeepAgents 方式**（约定 > 配置）：

```python
# DeepAgents 方式：声明差异即可，其余默认（约定 > 配置）
agent = create_deep_agent(model="deepseek-v4-pro")

# 这就够了！内部自动配置了：
# ✓ FilesystemMiddleware（文件系统操作）
# ✓ MemoryMiddleware（持久记忆）
# ✓ SummarizationMiddleware（上下文压缩）
# ✓ SubAgentMiddleware（子 Agent 能力）
# ✓ Checkpointer（短期记忆持久化）
# ✓ Context Editing（上下文清理）
# ✓ 默认 System Prompt
# ✓ 默认 Tool 列表
```

**DeepAgents 的约定清单**（你不需要显式配置的）：

| 约定 | 默认值 | 可覆盖？ |
|---|---|---|
| 文件系统后端 | `FilesystemBackend`（本地 `./workspace/`） | 是（`backend=...`） |
| 上下文压缩策略 | 超过 4000 tokens 触发 Summarization | 是（中间件参数） |
| 子 Agent 模型 | 与主 Agent 相同 | 是（`SubAgent(model=...)`） |
| 记忆存储 | 内存（与 Checkpointer 共享） | 是（`memory=[...]` 指定源） |
| 文件权限 | 默认允许读写 `/workspace/` | 是（`permissions=[...]`） |
| System Prompt | 通用助手 Prompt | 是（`system_prompt="..."`） |
| 工具列表 | 文件系统工具 + 自定义工具 | 是（`tools=[...]`） |

---

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

## 第十章：System Prompt — DeepAgents 的"宪法"与三大中间件契约

### 10.1 为什么 System Prompt 是 DeepAgents 的核心

在 LangChain 中，System Prompt 是你自己写的几行文字。在 DeepAgents 中，System Prompt 不是一段静态文本——它是 **FilesystemMiddleware、MemoryMiddleware、SubAgentMiddleware 三大中间件的协作契约**。

```
┌─────────────────────────────────────────────────────────────┐
│              DeepAgents System Prompt 分层架构               │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  BASE_AGENT_PROMPT（基础行为层）                       │ │
│  │  "你是 deep agent，用工具帮助用户完成任务"               │ │
│  │  定义：角色本质、行为准则、任务执行模型                   │ │
│  └──────────────────────────┬────────────────────────────┘ │
│                             │ append_to_system_message()    │
│  ┌──────────────────────────┴────────────────────────────┐ │
│  │  _FILESYSTEM_SYSTEM_PROMPT_TEMPLATE（文件系统层）      │ │
│  │  "你有文件系统工具：ls/read/write/edit/glob/grep"      │ │
│  │  定义：文件操作规范、命名约定、错误处理                   │ │
│  └──────────────────────────┬────────────────────────────┘ │
│                             │ append_to_system_message()    │
│  ┌──────────────────────────┴────────────────────────────┐ │
│  │  EXECUTION_SYSTEM_PROMPT（代码执行层）                  │ │
│  │  "你有 execute 工具，在沙箱中运行 shell 命令"            │ │
│  │  定义：沙箱安全边界、超时限制、输出格式                   │ │
│  └──────────────────────────┬────────────────────────────┘ │
│                             │                               │
│  ┌──────────────────────────┴────────────────────────────┐ │
│  │  MemoryMiddleware 注入（记忆层）                        │ │
│  │  "你可以记住和回忆信息，跨对话保持上下文"                 │ │
│  └──────────────────────────┬────────────────────────────┘ │
│                             │                               │
│  ┌──────────────────────────┴────────────────────────────┐ │
│  │  SubAgentMiddleware 注入（委派层）                      │ │
│  │  "你可以调用子 Agent 处理复杂子任务"                     │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  最终 System Prompt = BASE + Filesystem + Execute +         │
│                        Memory + SubAgent + 用户自定义         │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 BASE_AGENT_PROMPT — 角色本质定义

这是 DeepAgents 的"出厂默认人格"，通过 `dynamic_prompt` 钩子在每次 LLM 调用前注入。与 LangChain 手写 System Prompt 的区别是——它精确约束了 Agent 的行为模式，不是"建议"而是"规则"。

**完整内容与逐段解读**：

```
You are a deep agent, an AI assistant that helps users
accomplish tasks using tools.
```

> **角色定位**："deep agent" = 能深入完成任务的 Agent，不是简单问答 ChatBot。"using tools" = 主动使用工具的 Agent，不是被动等待指令。

```
## Core Behavior

- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble ("Sure!", "Great question!",
  "I'll now...").
- Don't say "I'll now do X" — just do it.
- If the request is underspecified, ask only the minimum
  followup needed to take the next useful action.
- If asked how to approach something, explain first, then act.
```

> **行为准则**：禁止废话前缀（"好的，我来帮你..."）——这是 Agent 而不是客服。先理解再行动。不明确时只问最少的问题。

```
## Doing Tasks

When the user asks you to do something:

1. **Understand first** — read relevant files, check existing
   patterns. Quick but thorough.
2. **Act** — implement the solution. Work quickly but accurately.
3. **Verify** — check your work against what was asked, not
   against your own output.

Keep working until the task is fully complete. Don't stop
partway and explain what you would do — just do it.
```

> **任务模型**：三阶段循环（Understand → Act → Verify）。强调"做到完成为止，不要半路停下来解释"——这是 Agent 和 Chain 的本质区别。

```
**When things go wrong:**
- If something fails repeatedly, stop and analyze *why*
  — don't keep retrying the same approach.
- If you're blocked, tell the user what's wrong and ask.
```

> **失败处理**：不要无脑重试（传统 Agent 的最大问题之一）。分析原因 → 向用户求助。

```
## Professional Objectivity

- Prioritize accuracy over validating the user's beliefs
- Disagree respectfully when the user is incorrect
- Avoid unnecessary superlatives, praise, or emotional validation
```

> **职业客观性**：不迎合用户、不用过度赞美。准确 > 情绪价值。

**BASE_AGENT_PROMPT 的核心设计思想**：

| 传统 LLM System Prompt | BASE_AGENT_PROMPT |
|---|---|
| "你是一个友好的助手，用中文回复" | "你是 deep agent，用工具帮助用户完成任务" |
| 角色描述（模糊） | 行为规则（精确） |
| 被动等待指令 | 主动使用工具、自主执行到完成 |
| 不控制输出风格 | 禁止废话前缀、先做后说 |
| 无失败处理 | 分析原因 → 求助（不无脑重试） |

### 10.3 提示词 = 三大中间件的协作契约

DeepAgents 的三个核心中间件（Filesystem、Memory、SubAgent）不是通过代码耦合协同的——它们通过 **System Prompt 注入各自的"使用说明"** 来协同。每个中间件在 `before_model` 阶段把自己的指令追加到 SystemMessage 末尾。

```python
# FilesystemMiddleware 的 before_model 钩子（简化）
def before_model(self, state, runtime):
    # 构建文件系统提示词
    prompt = _FILESYSTEM_SYSTEM_PROMPT_TEMPLATE.format(
        available_tools=tool_list,    # 动态注入当前可用的工具名
    )
    # 如果有 execute 工具 → 追加执行提示词
    if has_execute_tool:
        prompt += "\n\n" + EXECUTION_SYSTEM_PROMPT
    
    # 追加到已有的 SystemMessage（不覆盖！）
    return {"messages": [SystemMessage(content=prompt)]}

# MemoryMiddleware 和 SubAgentMiddleware 同理——
# 各自追加自己的"使用说明"，最终构成完整契约。
```

**为什么用 Prompt 而不是代码耦合？**

| 代码耦合方式 | Prompt 契约方式 |
|---|---|
| 中间件 A 直接调用中间件 B 的方法 | 中间件 A 注入"如何使用 B 的说明" → LLM 自己决定 |
| B 的接口改了 → A 的代码崩溃 | B 的注入改了 → LLM 自动适应新的说明 |
| 加新中间件 C → 需要修改 A 和 B | 加新中间件 C → C 注入自己的说明，A 和 B 不受影响 |
| 中间件的组合方式是硬编码的 | LLM 根据聚合的 Prompt 动态决定调用策略 |

### 10.4 文件系统提示词详解

文件系统提示词分为两部分：**通用文件操作规范** + **每个工具的使用说明**。

**通用规范**（`_FILESYSTEM_SYSTEM_PROMPT_TEMPLATE`）：

```
## Following Conventions

- Read files before editing — understand existing content
  before making changes
- Mimic existing style, naming conventions, and patterns

## Filesystem Tools `ls`, `read_file`, `write_file`,
  `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with
using these tools.
All file paths must start with a /. Follow the tool docs
for the available tools, and use pagination (offset/limit)
when reading large files.
```

> **解决什么问题**：
> - "先读再改"规则 → 防止 Agent 盲目覆盖文件
> - "模仿已有风格" → 保证生成代码/文档的一致性
> - "分页读大文件" → 防止 Agent 一次读入 10000 行文本撑爆上下文

**七个文件系统工具描述**：

| 工具 | 描述要求 | 解决的问题 |
|---|---|---|
| `ls` | "探索文件系统的正确方式。几乎总是应该先 ls 再 read/edit" | 防止 Agent 直接猜文件路径 |
| `read_file` | "假设这个工具能读任何文件。读不存在的文件会返回错误" | 让 Agent 放心尝试，相信错误提示 |
| `write_file` | "创建新文件。优先用 edit_file 修改已有文件" | 防止无谓的新文件创建 |
| `edit_file` | "精确字符串替换。必须先用 read_file 读过才能 edit" | 防止盲目编辑、强制先读 |
| `glob` | "通配符模式匹配。`**` 匹配任意层级目录" | 防止 Agent 手动遍历目录树 |
| `grep` | "按文本模式搜索文件，非正则" | 防止 Agent 读每个文件后手动搜索 |
| `execute` | "沙箱中运行 shell 命令，返回输出和退出码" | 定义安全边界、超时和输出格式 |

### 10.5 执行工具提示词 — 沙箱安全契约

```
## Execute Tool `execute`

You have access to an `execute` tool for running shell commands
in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and
other shell operations.

- execute: run a shell command in the sandbox
  (returns output and exit code)

Before executing, please consider:
1. Is the command safe?
2. Will it complete within a reasonable time?
3. Is the working directory correct?
```

> **解决什么问题**：
> - "沙箱环境"提示 → Agent 知道命令不会影响真实系统（敢执行）
> - "返回输出和退出码" → Agent 知道怎么判断成功/失败
> - 三个检查 → 防止危险命令、防止超时等待、防止路径错误

### 10.6 子 Agent 调用协议

SubAgentMiddleware 注入的不是固定的 prompt，而是**动态生成的工具描述**——每个 SubAgent 在主 Agent 眼中就是一个"能处理某类任务的工具"：

```python
# SubAgentMiddleware 内部（简化）
# 用户定义的 SubAgent：
#   SubAgent(name="researcher", description="搜索和整理信息...")
# 注入到主 Agent 的 System Prompt：
#   "你可以调用 researcher 工具来搜索和整理信息..."
#   "researcher 是这个子 Agent 的描述..."
```

**协议的具体内容**（LLM 视角）：

```
## Sub-Agent Tools

You can delegate complex sub-tasks to specialized sub-agents.
Each sub-agent appears as a tool you can call.

- researcher: Search and compile information. Best for
  literature review and data collection.
- code-reviewer: Review code quality, security, and best
  practices.
- writer: Write reports based on research materials.

To use a sub-agent, call its tool with the task description.
The sub-agent will execute independently and return results.
```

> **这个协议达成什么效果**：
> - 主 Agent 不需要知道子 Agent 内部如何实现（黑盒调用）
> - 子 Agent 失败了 → 主 Agent 看到的是工具返回错误（和解法一样）
> - 加新子 Agent → 只是工具列表多了一项，主 Agent 逻辑不动

---

### 10.7 System Prompt 动态注入流程

```
每次 LLM 调用前（before_model 钩子）：

[1] BASE_AGENT_PROMPT（来自 create_deep_agent 或 HarnessProfile）
      │
      ▼ SystemMessage(content=BASE_AGENT_PROMPT)
      │
[2] FilesystemMiddleware.before_model()
      │ 检测可用工具 → 拼装 _FILESYSTEM_SYSTEM_PROMPT_TEMPLATE
      │ 如果有 execute → 追加 EXECUTION_SYSTEM_PROMPT
      │ append_to_system_message() 追加到已有 SystemMessage
      ▼
[3] MemoryMiddleware.before_model()
      │ 如果启用了 memory → 追加记忆相关提示词
      │ append_to_system_message() 追加
      ▼
[4] SubAgentMiddleware.before_model()
      │ 如果有子 Agent → 为每个 SubAgent 生成工具描述
      │ append_to_system_message() 追加
      ▼
[5] 用户自定义 middleware
      │ 如果有自定义中间件的 before_model → 继续追加
      ▼
[6] 最终 SystemMessage 发送给 LLM
```

**与传统 LangChain Prompt 的差异**：

| | LangChain | DeepAgents |
|---|---|---|
| System Prompt 来源 | 手动写一段文本 | 多层动态拼装 |
| 修改方式 | 改字符串 → 重启 | 加中间件 → 自动注入 |
| 不同环境差异 | 手写多套 Prompt | Backend + 中间件自动适配 |
| 工具描述 | 手动写 description | 中间件自动生成 + 注入 |
| Prompt 可观测性 | 启动时固定，运行时不可见 | 每次 LLM 调用前动态生成 |

### 10.8 LangChain vs DeepAgents System Prompt 对比

```python
# ===== LangChain 方式：手动写一切 =====
system_prompt = """
你是 AI 研究助手。

规则：
- 先读文件再修改
- 错误时分析原因，不要重试
- 保持简洁

工具：
- ls: 列出目录
- read_file: 读取文件
- write_file: 写入文件
- search_web: 搜索网络
"""

agent = create_agent(
    llm=llm,
    tools=[ls_tool, read_tool, write_tool, search_tool],
    system_prompt=system_prompt,
)
# 问题：
# 1. 工具描述是你手写的 — 和实际工具可能不同步
# 2. 规则是你手写的 — 忘了加"先读再写"就是忘了
# 3. 加新工具需要改 Prompt — 忘了改就不同步

# ===== DeepAgents 方式：中间件自动注入 =====
agent = create_deep_agent(
    model="deepseek-v4-pro",
    tools=[search_web],  # 不需要手动写文件系统工具
    system_prompt="你是 AI 研究助手。",  # 只写业务相关的
)
# 内部自动注入：
# ✓ BASE_AGENT_PROMPT（角色 + 行为规则）
# ✓ Filesystem 提示词（7 个工具 + 使用规范）
# ✓ Execute 提示词（沙箱安全）
# ✓ Memory 提示词（如果启用）
# ✓ SubAgent 提示词（如果配置）
# → 最终 System Prompt 可能是你写的那行的 10 倍长，
#   但每一段都是对应中间件自动生成的，保证一致性。
```

---

### 10.9 要点总结

```
System Prompt 在 DeepAgents 中的角色演变：

LangChain 时代：System Prompt = 一段固定的文字
DeepAgents 时代：System Prompt = 中间件的协作契约

每个中间件通过 before_model 注入自己的 "使用说明"：
  FilesystemMiddleware → 文件操作规范 + 7 个工具描述
  MemoryMiddleware    → 记忆存取指令
  SubAgentMiddleware  → 子 Agent 调用协议

好处：
  ✓ 中间件解耦 — 通过 Prompt 协同而非代码耦合
  ✓ 自动同步 — 工具列表变了，Prompt 自动更新
  ✓ 动态适配 — 不同 Backend 有不同工具，Prompt 自动适配
  ✓ 可观测 — 每次 LLM 调用的完整 Prompt 可在 LangSmith 中查看
```

---

## 第十一章：规划工具 — 工作流思维链与 TODO.md

### 11.1 核心：工作流思维链是什么

DeepAgents 的"规划"不是写在某个 Python 模块里的硬编码逻辑——它是 **Agent 利用文件系统工具自己写给自己的 TODO 列表**。

```
BASE_AGENT_PROMPT 里的"理解→行动→验证"三阶段循环
        +
文件系统工具（write_file / edit_file / read_file）
        +
Agent 的自主推理能力
        =
Agent 自己写出 plan.md → 按计划执行 → 完成后划掉
```

**这就像你（Claude Code）的 TodoWrite 工具**——我每接到一个复杂任务，先列出步骤清单，然后逐个执行，完成一个划掉一个。DeepAgents 用完全相同的方式工作，只是它的"TodoWrite"是 `write_file("/workspace/plan.md", ...)`。

### 11.2 对比：传统 Agent vs DeepAgents 规划

```
传统 ReAct Agent:
  用户: "帮我写一个 Flask API，包含认证、数据库、测试"
  
  Thought: 先创建 app.py
  Action: write_file("app.py", "from flask import Flask...")
  Thought: 再加认证
  Action: edit_file("app.py", ...)
  Thought: 还要数据库
  Action: write_file("models.py", "...")
  Thought: 好像忘了什么...测试！
  Action: write_file("test_app.py", "...")
  
  问题：
    ✗ 没有整体规划 → 想到哪做到哪
    ✗ 没有进度跟踪 → 做到一半不知道还剩什么
    ✗ 容易漏步骤 → 最后才想起测试
    ✗ 无法恢复 → 中断后不知道从哪继续


DeepAgents (文件系统规划):
  用户: "帮我写一个 Flask API，包含认证、数据库、测试"
  
  [1] Understand: 读项目现有文件
       ls("/workspace/") → []
       "这是一个新项目，从零开始"
  
  [2] Plan: 写 TODO 文件
       write_file("/workspace/TODO.md",
         "# Flask API 项目计划\n"
         "- [ ] 1. 创建 app.py 主文件\n"
         "- [ ] 2. 实现 JWT 用户认证\n"
         "- [ ] 3. 创建数据库模型 models.py\n"
         "- [ ] 4. 实现 CRUD API 端点\n"
         "- [ ] 5. 编写单元测试 test_app.py\n"
         "- [ ] 6. 添加 requirements.txt\n"
         "- [ ] 7. 最终检查：启动 + 测试"
       )
  
  [3] Execute: 逐个完成
       edit_file("/workspace/TODO.md", 
         "- [ ] 1. 创建 app.py", 
         "- [x] 1. 创建 app.py")  ← 划掉已完成的
       ...逐个执行...
  
  [4] Verify: 对照 TODO 检查
       read_file("/workspace/TODO.md")
       "全部 [x] → 任务完成！"
  
  优势：
    ✓ 先规划后执行 → 不遗漏
    ✓ 进度可见 → TODO.md 就是进度条
    ✓ 可恢复 → 中断后读 TODO.md 继续
    ✓ 可审计 → TODO.md 记录了完整执行过程
```

### 11.3 TODO.md — Agent 的"外部大脑"

TODO.md 不只是记录——它是 Agent 的 **外部工作记忆**。Agent 的上下文窗口有限（~128K tokens），但文件系统无限。把计划写进文件 = 把大脑"外挂"到磁盘。

```
Agent 的两种记忆：

  内部记忆（上下文窗口）              外部记忆（文件系统）
  ─────────────────────               ──────────────────
  对话历史（messages 列表）            /workspace/TODO.md
  容量: 128K tokens                   容量: 无限（磁盘有多大就能多大）
  生命周期: 对话结束 = 丢失             生命周期: 持久化（除非主动删除）
  用途: 当前推理                      用途: 长期规划、跨对话保持
  
  ★ DeepAgents 的创新：用文件系统弥补上下文窗口的容量限制
```

**TODO.md 的典型结构**（Agent 自己写的，不是模板）：

```markdown
# 项目: 构建 RAG 对比报告

## 进度: 3/5 完成

- [x] 1. 搜索 LangChain RAG 最新文档
      → 结果保存在 /workspace/notes/langchain_rag.md
- [x] 2. 搜索 LlamaIndex RAG 最新文档  
      → 结果保存在 /workspace/notes/llamaindex_rag.md
- [x] 3. 搜索 RAG 学术论文
      → 结果保存在 /workspace/notes/papers.md
- [ ] 4. 整理对比分析 → 写报告
      → 待写: /workspace/report.md
- [ ] 5. 验证报告引用 → 修改 → 完成

## 发现
- LangChain 侧重快速原型，LlamaIndex 侧重数据管道
- 2024 RAG 趋势: Agentic RAG, Graph RAG, Multimodal RAG

## 阻塞
- (无)
```

**TODO.md 的几个关键特性**：

| 特性 | 实现方式 | 为什么重要 |
|---|---|---|
| 进度追踪 | `[ ]` → `[x]` 标记 | Agent 一眼看到进展 |
| 中间结果锚定 | 记录文件路径 | 不会"搜了但忘了结果在哪" |
| 发现记录 | `## 发现` 章节 | 关键结论不丢，不被后续消息冲掉 |
| 阻塞标记 | `## 阻塞` 章节 | 中断后恢复，知道为什么停 |
| 跨对话 | 文件持久化 | 下次对话读 TODO.md 继续 |

### 11.4 企业级规划模式

#### 模式 1：层级 TODO（大任务拆小任务）

```markdown
# 项目: 微服务迁移

- [x] 1. 评估现有架构 → /workspace/evaluation.md
- [ ] 2. 用户服务迁移
      - [x] 2.1 写 Dockerfile
      - [ ] 2.2 写 K8s 部署配置
      - [ ] 2.3 集成测试
- [ ] 3. 订单服务迁移
- [ ] 4. 数据迁移
- [ ] 5. 监控和告警配置
```

#### 模式 2：SOP 驱动（标准操作流程模板化）

企业可以将常见任务的 SOP 写成模板文件，Agent 读取模板后按步骤执行：

```markdown
# SOP: 新服务上线路由检查清单
# (保存在 /workspace/templates/deploy_sop.md)

- [ ] 1. 检查 Docker 镜像构建状态
- [ ] 2. 验证 K8s manifest 文件语法
- [ ] 3. 检查依赖服务健康状态
- [ ] 4. 执行滚动更新
- [ ] 5. 验证 /health 端点
- [ ] 6. 检查日志无异常
- [ ] 7. 更新变更管理工单
```

Agent 读取这个 SOP → 逐条执行 → 逐条标记完成 → 生成执行报告。

#### 模式 3：多 Agent 并行规划

```markdown
# 项目: 年度技术报告

## 分配给 research-agent
- [x] 搜索 AI 趋势
- [x] 搜索云原生趋势
- [ ] 搜索安全趋势

## 分配给 writer-agent  
- [ ] 写 AI 章节（等待 research-agent 完成）

## 分配给 reviewer-agent
- [ ] 初审报告
- [ ] 事实核查
```

### 11.5 规划工具在企业中的实际应用

**场景 1：代码审查自动化**

```
Agent 收到 PR → 
  写 TODO.md:
    - [ ] 读 PR 描述
    - [ ] 读变更的文件列表
    - [ ] 逐个文件审查
    - [ ] 运行测试
    - [ ] 生成审查报告
  → 执行 → 报告写入 /workspace/review_report.md
```

**场景 2：故障排查 Runbook**

```
Agent 收到 "服务响应 5xx" →
  写 TODO.md:
    - [ ] 查服务状态
    - [ ] 查最近日志（grep ERROR）
    - [ ] 查系统资源（CPU/内存/磁盘）
    - [ ] 查依赖服务
    - [ ] 定位根因
    - [ ] 写故障报告 + 建议处置
  → 执行 → 报告 + 告警
```

**场景 3：数据管道监控**

```
Agent 定时触发 →
  写 TODO.md:
    - [ ] 检查昨天的数据导入是否完成
    - [ ] 验证数据质量（行数、空值率）
    - [ ] 检查下游依赖任务状态
    - [ ] 如有异常 → 生成告警 + 写原因分析
  → 执行 → 日报 + 异常告警
```

### 11.6 要点总结

```
规划工具的核心思想：

  传统方式：Prompt 中写 "Let's think step by step"
            → LLM 在脑子里想，不记录 → 容易忘、不可恢复

  DeepAgents：文件系统 = 外部大脑
            → write_file("TODO.md") = 记住计划
            → edit_file("TODO.md") = 更新进度  
            → 任何时候 read_file("TODO.md") = 知道做到哪了

类比：
  你（Claude Code）的 TodoWrite 工具 ≈ DeepAgents 的 write_file("TODO.md")
  都是 —— 把计划写下来 → 按计划执行 → 完成标记 → 不遗漏、可恢复

企业级应用：
  SOP 模板 — 标准化操作流程
  层级 TODO — 大任务拆小任务
  多 Agent 并行 — 各子 Agent 独立 TODO
  故障 Runbook — 自动化故障排查
  数据管道 — 定时巡检日报
```

---

## 第十二章：Sub-Agent — 任务隔离与并行委派

### 12.1 核心概念

SubAgent 是 DeepAgents 最独特的特性之一。**主 Agent 不是自己完成所有子任务，而是在需要时"召唤"一个临时的子 Agent 去独立完成，然后只接收结果摘要。**

```
传统 Agent（单线程）：                  DeepAgents（主 Agent + 子 Agent）：

  主 Agent                              主 Agent
    │                                     │
    搜索 RAG ──→ 搜索 Llama ──→ 写报告      ├─ spawn researcher ──→ 搜索 RAG ──→ 返回摘要
    (3步串行)                              ├─ spawn researcher ──→ 搜索 Llama ──→ 返回摘要
                                           │                    (2个并行！)
                                           └─ 汇总 → spawn writer ──→ 写报告 ──→ 返回报告
```

### 12.2 四大核心机制

#### 机制 1：任务隔离 — 独立上下文

每个子 Agent 拥有**全新的消息历史**。它看不到主 Agent 的对话上下文，主 Agent 也看不到子 Agent 的内部推理过程。

```python
# SubAgentMiddleware 内部（简化）—— 子 Agent 启动时的上下文
subagent_initial_state = {
    "messages": [
        SystemMessage(subagent_system_prompt),     # 子 Agent 专属 Prompt
        HumanMessage(task_description),            # 只有任务描述，没有主 Agent 的历史
    ],
    # 继承主 Agent 的配置
    "user_id": parent_state["user_id"],
    # 继承文件系统权限（可覆盖）
    "permissions": subagent_permissions or parent_permissions,
}
# ★ 关键：主 Agent 的对话历史不传给子 Agent
# 子 Agent 从零开始，只专注于分配的任务
```

**为什么这样设计**：

| 隔离的好处 | 说明 |
|---|---|
| **上下文干净** | 子 Agent 只看到任务描述，不被主 Agent 历史干扰 |
| **Token 节省** | 主 Agent 的 20000 token 历史不需要复制给每个子 Agent |
| **并行安全** | 多个子 Agent 同时运行，各自独立上下文，不会互相覆盖 |
| **专注力** | 子 Agent 只做一件事，不会被"之前我们讨论过..."分心 |
| **可恢复** | 子 Agent 崩溃只丢子任务，主 Agent 不受影响 |

#### 机制 2：并行执行

主 Agent 可以同时 `spawn` 多个子 Agent——它们并行运行，主 Agent 等待全部完成后汇总。

```python
# 主 Agent 在单次 LLM 调用中输出多个 tool_calls
# 每个 tool_call 触发一个子 Agent：
[
    task(name="researcher", task="搜索 RAG 最新论文"),
    task(name="researcher", task="搜索 LlamaIndex 最新文档"),
    task(name="code-reviewer", task="审查 app.py 的安全性"),
]
# ↑ 这三个子 Agent 并行启动，各自独立执行
# 都完成后 → 结果返回主 Agent → LLM 汇总
```

**并行 vs 串行对比**：

```
串行（一个 Agent 自己做）：
  搜索 RAG (3s) → 搜索 LlamaIndex (3s) → 写报告 (5s) = 11s

并行（子 Agent）：
  researcher-task1 ─┐
  researcher-task2 ─┤ 同时进行 (3s) → 汇总 → write-report (5s) = 8s
  code-reviewer    ─┘
```

#### 机制 3：智能聚合 — 只返回摘要

子 Agent 执行完成后，**中间步骤全部丢弃**，只把最终结果返回给主 Agent。

```
子 Agent 内部（完整的 ReAct 循环）：
  Thought: 搜索 RAG → Action: search("RAG")
  Thought: 不够精确 → Action: search("RAG 2024 paper")
  Thought: 整理结果 → 写总结
  → 这 3 步的过程主 Agent 看不到

主 Agent 收到的（只返回摘要）：
  ToolMessage(
    content="研究完成。RAG 2024 关键发现: 1. Agentic RAG 成为主流...",
    tool_call_id="call_001"
  )
  → 只有最终结论，没有中间过程
```

**为什么只返回摘要**：

| 如果返回全部过程 | 如果只返回摘要 |
|---|---|
| 主 Agent 上下文被 3 个子 Agent 的中间步骤填满 | 只多 1 条 ToolMessage |
| Token 用量 = 主 Agent + 所有子 Agent | Token 用量 = 主 Agent + 摘要 |
| 噪声大：子 Agent 的 "嗯...让我想一想" 也传回来 | 干净：只传结论 |

#### 机制 4：环境继承 — 必要的配置传递

子 Agent 不是完全从零开始的——它继承主 Agent 的关键配置，但可以选择性覆盖：

```python
SubAgent(
    name="researcher",
    description="搜索和整理信息",
    system_prompt="你是研究专家...",     # ★ 必填：子 Agent 专属 Prompt
    tools=[search_web, search_arxiv],  # 可覆盖：子 Agent 独有的工具
    model="openai:gpt-4o",            # 可覆盖：子 Agent 用不同模型
    permissions=[...],                 # 可覆盖：子 Agent 的文件权限
    middleware=[...],                  # 可覆盖：子 Agent 的中间件
)
```

**继承 vs 覆盖的默认规则**：

| 配置项 | 默认继承自主 Agent | 可被子 Agent 覆盖 |
|---|---|---|
| `model` | 是（用主 Agent 的模型） | 是 |
| `tools` | 是（用主 Agent 的工具） | 是 |
| `permissions` | 是 | 是 |
| `middleware` | DeepAgents 默认栈 | 追加自定义 |
| `system_prompt` | 否（必须指定） | — |
| `checkpointer` | 是（持久化子 Agent 状态） | 否（自动继承） |
| **对话历史** | **否（全新上下文）** | **不可继承** |
| **文件系统** | 是（同一个 workspace） | 否（共享文件系统） |

### 12.3 实现原理

**SubAgentMiddleware 的 `wrap_model_call`** —— 把子 Agent 包装成一个 LangChain Tool：

```python
# SubAgentMiddleware 内部（框架代码，简化流程）

# 第 1 步：__init__ 阶段——把每个 SubAgent 编译为 Runnable
for subagent_spec in user_subagents:
    # 用 create_deep_agent() 为每个子 Agent 创建独立的 Agent 实例
    # 但内部注入了一个 "StructuredResponseMiddleware"
    # → 强制子 Agent 返回结构化输出（包含所有 messages）
    compiled_runnable = create_deep_agent(
        model=subagent_spec.model or parent_model,
        tools=subagent_spec.tools or default_tools,
        system_prompt=subagent_spec.system_prompt,
        # ... 其他配置
    )

# 第 2 步：生成一个 task() 工具——主 Agent 调用它来 spawn 子 Agent
task_tool = _build_task_tool(subagent_specs)
# task_tool 的参数：
#   - name: str       → 选择哪个子 Agent 类型（如 "researcher"）
#   - task: str       → 任务的文字描述（子 Agent 的 HumanMessage）
#   - subagent_type: str → 子 Agent 类型（name 的别名）

# 第 3 步：注入到主 Agent
self.tools = [task_tool]

# 第 4 步：before_model 注入子 Agent 的描述到 System Prompt
# "Available subagent types:\n- researcher: 搜索和整理信息\n- writer: 写报告"
```

**执行时的完整调用链**：

```
主 Agent LLM 调用
  → 输出 tool_calls: [{"name": "task", "args": {"subagent_type": "researcher", "task": "搜索 RAG 论文"}}]

      ↓ ToolNode 执行 task 工具

  task_tool.invoke()
    → 1. 根据 subagent_type="researcher" 找到对应的 compiled_runnable
    → 2. 构造子 Agent 初始状态：
        {"messages": [SystemMessage(researcher_prompt), HumanMessage("搜索 RAG 论文")]}
    → 3. 调用 compiled_runnable.invoke(initial_state)
        → 子 Agent 内部执行完整的 ReAct 循环（搜索→读取→过滤→总结）
    → 4. 子 Agent 返回完整 State（包含所有 messages）
    → 5. 提取最后一条 AIMessage 的内容 = 摘要
    → 6. 返回 ToolMessage(content=摘要) 给主 Agent

      ↓ 主 Agent 收到 ToolMessage
      
  主 Agent LLM 看到：ToolMessage("研究完成。RAG 2024 关键发现: ...")
  → 继续推理（汇总所有子 Agent 结果）
```

### 12.4 完整使用示例

```python
# ================================================================
# subagent_full_demo.py — SubAgent 完整演示
# ================================================================
from deepagents import create_deep_agent, SubAgent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# ================================================================
# 第 1 步：定义主 Agent 的工具
# ================================================================
@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息。"""
    return f"搜索 '{query}': 找到 3 条相关结果..."

# ================================================================
# 第 2 步：定义子 Agent —— 每个都有独立的角色和 Prompt
# ================================================================

# 子 Agent 1：研究员
researcher = SubAgent(
    name="researcher",
    description="搜索和整理信息。适合文献调研、资料收集、竞品分析。"
                "调用时提供搜索主题，返回结构化的研究摘要。",
    system_prompt=(
        "你是研究专家。\n\n"
        "## 工作流程\n"
        "1. 用 search_web 搜索相关主题\n"
        "2. 整理搜索结果，提取关键信息\n"
        "3. 用 write_file 将研究摘要保存到 /workspace/research/\n"
        "4. 返回结构化的研究摘要（包含：关键发现、来源、置信度）\n\n"
        "## 输出格式\n"
        "研究主题: [主题]\n"
        "关键发现: \n- 发现1\n- 发现2\n"
        "来源: [URL 列表]\n"
        "置信度: [高/中/低]"
    ),
    tools=[search_web],      # 子 Agent 独有的工具
)

# 子 Agent 2：代码审查员
code_reviewer = SubAgent(
    name="code-reviewer",
    description="审查代码质量、安全性和最佳实践。",
    system_prompt=(
        "你是资深代码审查员。\n"
        "审查范围：安全性（SQL注入/XSS/认证）、性能（N+1/内存泄漏）、最佳实践（命名/结构/测试覆盖率）。\n"
        "输出格式：按严重程度（Critical/High/Medium/Low）列出问题，给出修复建议。"
    ),
    # 不指定 tools → 继承主 Agent 的文件系统工具（ls/read_file/grep）
)

# 子 Agent 3：技术写手
writer = SubAgent(
    name="writer",
    description="基于研究材料撰写技术报告。",
    system_prompt=(
        "你是技术写手。\n"
        "基于提供的研究材料撰写报告。严格基于材料，不编造。注明引用来源。\n"
        "报告结构：摘要 → 背景 → 对比分析 → 结论 → 参考文献。"
    ),
    model="openai:gpt-4o-mini",  # 子 Agent 用便宜模型（写报告不需要强推理）
)

# ================================================================
# 第 3 步：创建主 Agent，带上子 Agent 列表
# ================================================================
agent = create_deep_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0.7),
    tools=[search_web],
    subagents=[researcher, code_reviewer, writer],
    system_prompt=(
        "你是项目经理。分配研究任务给 researcher，"
        "审查任务给 code-reviewer，写作任务给 writer。"
        "如果任务可以分解为独立的子任务，使用 task 工具并行委派。"
    ),
)

# ================================================================
# 第 4 步：调用 —— 主 Agent 自动决策什么时候 spawn 子 Agent
# ================================================================
config = {"configurable": {"thread_id": "subagent_demo"}}

result = agent.invoke({
    "messages": [HumanMessage(
        "请调查 LLM Agent 框架的最新趋势（LangChain vs LlamaIndex vs CrewAI），"
        "然后写一份对比报告。报告写完后审查其完整性和准确性。"
    )],
}, config=config)

# ================================================================
# 内部执行流程（自动发生的）：
# ================================================================
#
# 主 Agent LLM 分析任务：
#   "这个任务可以拆为：1. 研究三个框架 2. 写报告 3. 审查"
#
# 第 1 轮 tool_calls（并行）：
#   task(subagent_type="researcher", task="调查 LangChain Agent 最新趋势")
#   task(subagent_type="researcher", task="调查 LlamaIndex Agent 最新趋势")
#   task(subagent_type="researcher", task="调查 CrewAI Agent 最新趋势")
#   → 3 个子 Agent 并行执行，各自搜索并返回研究摘要
#
# 第 2 轮 tool_calls：
#   task(subagent_type="writer", task="基于 3 份研究摘要写对比报告")
#   → writer 子 Agent 读文件 → 写报告 → 返回报告摘要
#
# 第 3 轮 tool_calls：
#   task(subagent_type="code-reviewer", task="审查对比报告")
#   → 读报告文件 → 检查完整性 → 返回审查意见
#
# 第 4 轮：主 Agent 汇总 → 返回最终结果

print(result["messages"][-1].content)
```

### 12.5 企业级扩展

#### 场景 1：多模型策略 — 省钱且保证质量

```python
# 不同子 Agent 用不同模型
# 简单任务用便宜模型，复杂推理用贵模型

agent = create_deep_agent(
    model="claude-sonnet-4-6",  # 主 Agent：中等模型
    subagents=[
        SubAgent(
            name="deep-researcher",
            description="深度研究（需要强推理）",
            system_prompt="你是研究专家...",
            model="claude-opus-4-7",  # ★ 子 Agent 用最强模型
            tools=[search_web, search_arxiv],
        ),
        SubAgent(
            name="data-formatter",
            description="格式化数据",
            system_prompt="你是数据格式化专家...",
            model="openai:gpt-4o-mini",  # ★ 子 Agent 用最便宜模型
        ),
    ],
)
```

#### 场景 2：安全隔离 — 危险操作只给子 Agent

```python
# 执行代码的权限只给 code-executor 子 Agent
# 主 Agent 没有 execute 权限 → 不能直接运行代码 → 安全

agent = create_deep_agent(
    model="deepseek-v4-pro",
    tools=[search_web, read_file, write_file],  # 主 Agent：安全工具
    permissions=[FilesystemPermission.READ, FilesystemPermission.WRITE],
    subagents=[
        SubAgent(
            name="code-executor",
            description="在沙箱中执行代码",
            system_prompt="你是代码执行专家...",
            permissions=[FilesystemPermission.READ],  # 子 Agent：只能读
            # execute 工具需要 ShellToolMiddleware 或沙箱 Backend
        ),
    ],
)
```

#### 场景 3：审批流 — 子 Agent 的输出需要人工审核

```python
agent = create_deep_agent(
    model="deepseek-v4-pro",
    subagents=[
        SubAgent(
            name="deployment-planner",
            description="制定部署计划",
            system_prompt="你是部署专家...",
        ),
    ],
    interrupt_on={
        "task": {
            "allowed_decisions": ["approve", "edit", "reject"],
            # 子 Agent 的任务描述需要人工审核后才能执行
        },
    },
)
```

### 12.6 要点总结

```
SubAgent 的四个核心机制：

  1. 任务隔离    子 Agent = 全新上下文，不继承主 Agent 历史
                → 干净、专注、Token 省、并行安全

  2. 并行执行    多个子 Agent 同时运行
                → 3 倍加速（搜索 A/B/C 同时进行）

  3. 智能聚合    中间步骤丢弃，只返回摘要
                → 主 Agent 上下文干净，不堆噪声

  4. 环境继承    文件系统共享、配置可覆盖
                → 子 Agent 能访问 workspace 但不污染配置

  为什么这样设计：
  ┌─────────────────────────────────────────────┐
  │ 类比：人类项目经理的工作方式                 │
  │                                             │
  │ 项目经理（主 Agent）                         │
  │   ├─ 不自己做所有事                          │
  │   ├─ 把子任务委派给专家（子 Agent）           │
  │   ├─ 专家独立工作（独立上下文）               │
  │   ├─ 可以同时进行（并行）                     │
  │   ├─ 只关心最终结果（摘要）                   │
  │   └─ 汇总专家的成果（聚合）                   │
  └─────────────────────────────────────────────┘
```

---

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

## 第十四章：Human-in-the-Loop — 人工干预

### 14.1 什么场景需要人工干预

Agent 不是万能的。某些操作必须经过人类确认才能执行——这不是技术限制，而是业务规则和安全要求。

```
需要人工干预的典型场景：

┌─────────────────────────────────────────────────────────────┐
│ 场景                      示例                               │
├─────────────────────────────────────────────────────────────┤
│ 涉及金钱/资源              删除生产数据库记录                  │
│                           退款/转账操作                       │
│                           修改计费配置                        │
│                                                             │
│ 影响线上服务               重启生产服务器                      │
│                           修改 K8s Deployment                 │
│                           切换 DNS / 负载均衡                 │
│                                                             │
│ 不可逆操作                 删除用户数据（GDPR）                │
│                           强制合并/覆盖 Git 分支               │
│                           发送全员邮件通知                     │
│                                                             │
│ 合规要求                   修改审计日志保留策略                │
│                           变更安全组/防火墙规则               │
│                           导出敏感数据                        │
│                                                             │
│ 高风险代码执行             eval/exec 用户提供的代码             │
│                           数据库 Migration                    │
│                           SQL 写操作（UPDATE/DELETE）          │
└─────────────────────────────────────────────────────────────┘
```

### 14.2 底层机制 — LangGraph 的 `interrupt()` + `Command.resume`

DeepAgents 的 HITL 直接使用 LangGraph 的中断/恢复机制：

```python
# LangGraph 的中断机制（DeepAgents 内部调用的底层 API）

# 第 1 步：Agent 执行到需要审批的节点
# HumanInTheLoopMiddleware 在 before_model 钩子中调用：
from langgraph.types import interrupt

# interrupt() 会：
#   1. 暂停当前 Graph 执行
#   2. 将当前 State 通过 Checkpointer 持久化
#   3. 把 prompt 信息返回给调用方（人类审批者）
#   4. 等待外部 resume 信号
interrupt_value = interrupt({
    "action": "delete_record",
    "args": {"table": "users", "id": "123"},
    "message": "确认删除用户 123？此操作不可逆！",
})

# 第 2 步：人类审批后，外部程序调用：
from langgraph.types import Command
agent.invoke(
    Command(resume={"decision": "approve", "reason": "已核实"}),
    config=config,  # 同一个 thread_id
)

# 第 3 步：Graph 从暂停点恢复执行
# interrupt() 返回人类输入的值
# print(interrupt_value)  → {"decision": "approve", "reason": "已核实"}
# Agent 继续执行后续逻辑
```

**核心流转图**：

```
Agent 执行                    人类审批者
──────────                    ──────────

  正常推理
    │
    ▼
  决定执行工具
    │
    ▼
  HumanInTheLoopMiddleware
  检查：这个工具需要审批吗？
    ├─ 不需要 → 直接执行
    └─ 需要   → interrupt()
                  │
                  │  State 持久化到 Checkpointer
                  │  "等待人工决策"
                  │
                  ▼ (暂停) ──────────────→ 👤 收到审批通知
                                               │
                                          approve / edit / reject
                                               │
                  ◄────────────────────────────┘
                  │  Command(resume={...})
                  │  State 从 Checkpointer 恢复
                  ▼
  根据审批结果继续：
    approve → 执行工具 → 返回结果
    edit    → 修改参数 → 执行工具
    reject  → 跳过工具 → 返回拒绝说明
    respond → 人工直接回复 → 返回人工回复
```

### 14.3 `interrupt_on` — HITL 的配置开关

`interrupt_on` 是 DeepAgents 中把 HITL 插入执行流程的唯一入口。它是一个字典，key 是工具名，value 是配置。

```python
from deepagents import create_deep_agent
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig

agent = create_deep_agent(
    model="deepseek-v4-pro",
    tools=[...],
    # ★ interrupt_on: HITL 的开关
    interrupt_on={
        # === 用法 1：bool 值（简单开关）===
        "delete_record": True,
        # ↑ True = 启用审批，允许所有决策类型（approve/edit/reject/respond）

        "get_server_status": False,
        # ↑ False = 自动通过（和没配置一样，显式声明而已）

        # === 用法 2：InterruptOnConfig（精细控制）===
        "restart_service": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            # ↑ 只允许同意或拒绝，不允许编辑参数（防止改错服务名）
            description="⚠️ 重启服务 {tool_name}？参数: {tool_args}",
            # ↑ 审批时显示的描述信息，支持模板变量
        ),

        "send_email": InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"],
            # ↑ 允许编辑（修改邮件内容/收件人后再发送）
        ),

        "write_database": InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject", "respond"],
            # ↑ 全部允许，包括人工直接回复（respond = 不执行工具，人工写回答）
            description=lambda tool_call, state, runtime: (
                f"数据写入操作:\n"
                f"  表: {tool_call['args'].get('table', 'unknown')}\n"
                f"  操作: {tool_call['args'].get('operation', 'unknown')}\n"
                f"  操作人: {state.get('user_id', 'unknown')}\n"
                f"  时间: {datetime.now().isoformat()}"
            ),
        ),
    },
)
```

### 14.4 `interrupt_on` 的四种决策类型

| 决策类型 | 含义 | Agent 收到后的行为 |
|---|---|---|
| `approve` | 同意执行 | 正常执行工具 → 返回结果给 LLM |
| `edit` | 修改参数后执行 | 使用人类修改的参数执行工具 → 返回结果 |
| `reject` | 拒绝执行 | 跳过工具 → 返回 "操作被拒绝" 给 LLM |
| `respond` | 人工直接回复 | 不执行工具 → 把人类的回复直接作为工具结果 |

### 14.5 完整示例

```python
# ================================================================
# hitl_demo.py — Human-in-the-Loop 完整演示
# ================================================================
import asyncio
from datetime import datetime
from deepagents import create_deep_agent
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage

# ================================================================
# 第 1 步：创建带 HITL 的 Agent
# ================================================================

agent = create_deep_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0),

    # ★ 关键配置：哪些工具需要人工审批 ★
    interrupt_on={
        # delete_record: 只需要同意/拒绝（不可编辑——防止改错 ID）
        "delete_record": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            description=(
                "⚠️⚠️ 不可逆操作警告 ⚠️⚠️\n"
                "工具: delete_record\n"
                "此操作将永久删除数据，无法恢复。\n"
                "请仔细确认后决定。"
            ),
        ),

        # restart_service: 需要同意或拒绝
        "restart_service": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
        ),

        # send_email: 允许编辑（修改内容后再发）或人工直接回复
        "send_email": InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject", "respond"],
        ),

        # get_server_status: 自动通过（不需要审批）
        # 不配置 = 自动通过，不需要显式写 False
    },

    # ★ HITL 必须配置 Checkpointer ★
    # interrupt() 依赖 Checkpointer 保存暂停时的 State
    checkpointer=InMemorySaver(),  # 生产用 PostgresSaver
)

# ================================================================
# 第 2 步：模拟一次需要审批的调用
# ================================================================

config = {
    "configurable": {
        "thread_id": "hitl_demo_001",
        "user_id": "operator_zhang",
    }
}

# 用户请求：包含危险操作
result = agent.invoke(
    {"messages": [HumanMessage("帮我删除用户 123 的所有记录，他在我们的黑名单上")]},
    config=config,
)

# ================================================================
# 第 3 步：检查是否触发了中断
# ================================================================

state = agent.get_state(config)
# state.interrupts 包含中断信息
# 如果有中断 → Agent 已暂停，等待人类决策
# 如果没有中断 → Agent 已完成（可能是工具不需要审批，或被拒绝了）

if state.interrupts:
    # === Agent 暂停了，显示审批信息 ===
    interrupt_info = state.interrupts[0]  # 可能有多个中断
    
    print("=" * 60)
    print("⏸️  Agent 暂停 — 需要人工审批")
    print("=" * 60)
    print(f"工具: {interrupt_info.value['action']}")
    print(f"参数: {interrupt_info.value['args']}")
    print(f"消息: {interrupt_info.value.get('message', '请审批')}")
    print()
    print("可选决策: approve / reject")
    print("=" * 60)

    # ================================================================
    # 第 4 步：人类做出决策并恢复执行
    # ================================================================

    # 决策 A：同意删除
    await agent.ainvoke(
        Command(resume={
            "decision": "approve",
            "reason": "该用户在黑名单上，已确认需要删除",
        }),
        config=config,
    )
    # → Agent 恢复 → 执行 delete_record → 返回结果

    # 决策 B：拒绝（如果选择拒绝）
    # await agent.ainvoke(
    #     Command(resume={
    #         "decision": "reject",
    #         "reason": "需要先确认用户是否确实在黑名单上",
    #     }),
    #     config=config,
    # )
    # → Agent 恢复 → 跳过 delete_record → 返回 "操作被拒绝"

else:
    # === Agent 没有中断 → 不需要审批 ===
    print("✅ Agent 已完成，无需审批")
    if result and "messages" in result:
        print(result["messages"][-1].content[:300])

# ================================================================
# 第 5 步：验证审计结果
# ================================================================

# 读取最终 State
final_state = agent.get_state(config)
print(f"\n📋 最终消息数: {len(final_state.values['messages'])}")
# 最后一条消息应该包含操作结果（删除成功 / 被拒绝）
```

### 14.6 HITL 配合 LangSmith 的效果

在 LangSmith 中，HITL 的每一步都是可见的：

```
LangSmith Trace 视图：

  ▼ Agent (RunnableSequence)                         8.5s
    ├─ ▼ ChatOpenAI                          1.2s
    │    Input: [SystemMsg, HumanMsg("删除用户 123")]
    │    Output: AIMsg(tool_calls=[delete_record])
    │
    ├─ ▼ ToolNode (delete_record)            0.0s
    │    ⏸️ INTERRUPTED — 等待人工审批
    │
    │  (时间流逝...人类审批中...)
    │
    ├─ ▶ Resume — decision: "approve"        0.0s
    │
    ├─ ▼ ToolNode (delete_record)            0.3s
    │    ✅ 执行成功
    │
    └─ ▼ ChatOpenAI                          1.1s
         Output: "用户 123 的记录已成功删除"
```

**LangSmith + HITL 带来的能力**：

| 能力 | 说明 |
|---|---|
| 审批历史可视化 | 每次审批决策都记录在 Trace 中（谁/何时/什么决策） |
| 审计合规 | 满足 SOC2/GDPR 的审计追溯要求 |
| 审批耗时统计 | 人类响应时间 = 瓶颈指标，用于 SLA 监控 |
| 异常审批告警 | 统计 reject 率 → 过高说明 Agent 行为需要调整 |

### 14.7 要点总结

```
Human-in-the-Loop 的核心机制：

  底层 LangGraph API:
    interrupt(value)  ──→ 暂停 Graph + 持久化 State + 返回 value
    Command(resume=x) ──→ 恢复 Graph + interrupt() 返回 x

  DeepAgents 配置层:
    interrupt_on = {
        "tool_name": True,                          # 简单开关
        "tool_name": InterruptOnConfig(             # 精细控制
            allowed_decisions=["approve","reject"],
            description="...",
        ),
    }

  四种决策类型:
    approve  → 同意执行
    edit     → 改参数后执行
    reject   → 拒绝执行
    respond  → 人工直接回复（不执行工具）

  必须配合 Checkpointer:
    无 Checkpointer → interrupt() 无法保存状态 → HITL 不可用

  典型工作流:
    Agent 推理 → 决定调危险工具 → interrupt 暂停
    → 人类审批 → Command(resume) 恢复 → 执行/拒绝
```

---

## 第十五章：框架选择决策指南 — DeepAgents vs LangChain

### 15.1 决策总览

```
你的任务特征是什么？
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  简单/确定                     复杂/动态                        │
│  ────────                     ────────                         │
│                                                               │
│  LangChain 原生                 DeepAgents                     │
│  ┌─────────────────┐           ┌─────────────────────────┐    │
│  │ prompt | llm     │           │ create_deep_agent(...)   │    │
│  │ | parser         │           │ 内置 文件系统+子Agent     │    │
│  │                  │           │ +记忆+规划+HITL          │    │
│  │ 轻量、快速、精准  │           │ 重量、强大、自主          │    │
│  │ 延迟: 1~3s       │           │ 延迟: 10s~几分钟          │    │
│  └─────────────────┘           └─────────────────────────┘    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 15.2 选 DeepAgents 的四种场景

#### 场景 1：逻辑必须动态生成

**什么意思**：任务的执行步骤在执行前无法预先确定，必须根据中间结果动态调整。

**为什么 DeepAgents 好**：

```
LangChain Chain:
  流程固定：A → B → C → D
  如果 B 的结果告诉你 "不需要 C，直接跳到 E" → Chain 做不到
  因为 Chain 是声明式管道，运行时不能改变拓扑

DeepAgents:
  Agent 自己决定下一步：
  "搜到 3 篇论文" → "需要再搜一篇" → "够了，开始写报告"
  每一步都是 LLM 根据当前状态动态决策的
  没有固定的执行路径
```

**典型例子**：

| 任务 | LangChain | DeepAgents |
|---|---|---|
| "帮我研究 XX 主题" | 固定搜索一次 → 固定返回格式 | 搜索 → 不够 → 再搜 → 读结果 → 不够 → 过滤 → 整理 → 写报告 |
| "分析这份代码仓库" | 读文件 → 分析 | ls 目录 → 读关键文件 → grep 搜索 → 发现依赖 → 检查依赖 → 写报告 |
| "排查这个线上故障" | 固定检查清单 | 查日志 → 发现异常 → 查数据库 → 发现慢查询 → 查配置 → 定位根因 → 写修复方案 |

#### 场景 2：需要高安全性

**什么意思**：操作不可逆、涉及敏感数据、需要合规审计。

**为什么 DeepAgents 好**：

```
LangChain:
  工具权限控制 = 你自己写 if/else
  HITL = 你自己写审批流
  审计 = 你自己写日志

DeepAgents:
  interrupt_on = {"delete_record": InterruptOnConfig(...)}
  → HITL 自动插入
  → 审批决策自动记录
  → LangSmith 中完整追溯
  → 一行配置搞定
```

**典型例子**：

| 任务 | LangChain | DeepAgents |
|---|---|---|
| 数据库写操作 | 手动在每个 `@tool` 里加审批逻辑 | `interrupt_on={"write_db": True}` |
| 删除用户数据（GDPR） | 手动实现审批流 + 审计 | `interrupt_on + InterruptOnConfig + LangSmith` |
| 重启生产服务 | 手动加确认逻辑 | `interrupt_on={"restart_service": True}` |

#### 场景 3：长链路、多步骤

**什么意思**：任务需要 5+ 个步骤，中间有依赖关系，需要跟踪进度。

**为什么 DeepAgents 好**：

```
LangChain:
  10 步任务 → 你自己管理状态 → 第 5 步失败 → 不知道前 4 步干了什么
  没有 TODO → 中断后无法恢复

DeepAgents:
  TODO.md 自动跟踪进度
  文件系统保存中间结果
  Checkpointer 持久化每一步状态
  中断后可以恢复
```

**典型例子**：

| 任务 | 步骤数 | LangChain | DeepAgents |
|---|---|---|---|
| 微服务迁移 | 15+ | 需人工协调 | Agent 自主规划 + 执行 |
| 年度技术报告 | 8+ | 需人工分步写 | Agent 自驱动：搜索→整理→写→审查 |
| CI/CD 修复 | 5~10 | 固定脚本 | Agent 动态排查 + 修复 |

#### 场景 4：依赖复杂环境与文件系统

**什么意思**：任务需要和文件系统大量交互（读/写/搜索/执行），或需要代码执行环境。

**为什么 DeepAgents 好**：

```
LangChain:
  你需要手动为每个文件操作写 @tool
  ls → @tool, read → @tool, write → @tool, grep → @tool, ...
  10 个文件操作 = 10 个 @tool 定义

DeepAgents:
  7 个文件工具自动注入
  大结果自动转存文件（不污染上下文）
  CompositeBackend 支持混合存储
  Sandbox 支持代码安全执行
```

**典型例子**：

| 任务 | LangChain | DeepAgents |
|---|---|---|
| 代码仓库分析 | 需手写 ls/grep/read 等工具 | 自动拥有完整文件系统 |
| 生成+测试代码 | 需手写 execute + 沙箱 | `LocalShellBackend` / `LangSmithSandbox` |
| 写一份报告并保存 | `write_file` 需手写 | `write()` 自动可用 |

---

### 15.3 选 LangChain 原生的四种场景

#### 场景 1：杀鸡不用牛刀 — RAG 检索步骤确定

**什么意思**：检索流程是固定的——查向量库 → 拼 Prompt → LLM 回答。不需要动态决策。

**为什么 LangChain 更好**：

```
LangChain RAG:
  retriever | format_docs | prompt | llm | parser
  延迟: 1~3 秒
  成本: 1 次 LLM 调用 + 1 次 Embedding

DeepAgents 做同样的事:
  Agent 推理 "用户想查资料" → 决定调 retriever → 调完 → 决定回答
  延迟: 5~10 秒（多 1 次推理循环）
  成本: 2 次 LLM 调用（多 1 次决策调用）
```

**什么时候用 LangChain**：

```
用户有明确问题 → 检索 → 回答
流程确定、不需要中间决策
例: "LangChain 的 Runnable 怎么用？" → 检索文档 → 回答
```

#### 场景 2：确定性 API 编排

**什么意思**：调用顺序是预先知道的——查天气 → 查空气质量 → 给穿衣建议。不需要 Agent 自己决定。

**为什么 LangChain 更好**：

```
LangChain Chain:
  {
    "weather": get_weather | format_weather,
    "aqi": get_aqi | format_aqi,
  }
  | build_advice_prompt | llm | parser
  延迟: 2 秒（3 个 API 并行调用）
  确定性: 100%（每次执行流程完全一样）

DeepAgents:
  Agent 推理 → 决定先查天气 → 等待 → 
  推理 → 决定再查空气质量 → 等待 → 
  推理 → 决定给建议 → 回答
  延迟: 6~8 秒（串行决策）
  确定性: 80~95%（有时会多查一次、有时会少查一次）
```

**什么时候用 LangChain**：

```
流程固定的 API 编排
例: "用户问天气 + 穿衣建议" → 并行调天气 API + 穿搭 API → 合并 → 回答
步骤确定、不依赖中间结果的判断
```

#### 场景 3：实时交互 — 等不起几分钟

**什么意思**：用户期望秒级响应。聊天机器人、客服、搜索。

**为什么 LangChain 更好**：

```
LangChain Chat:
  用户输入 → prompt | llm | parser → 1 秒返回
  支持流式输出，0.3 秒出第一个 token

DeepAgents:
  用户输入 → Agent 推理 → 可能调工具 → 再推理 → 返回
  延迟不确定（可能 5 秒，可能 30 秒）
  用户等待焦虑
```

**典型例子**：

| 场景 | 用 LangChain | 用 DeepAgents |
|---|---|---|
| 客服机器人 | 秒回，体验好 | 等 10 秒，流失用户 |
| 文档问答 | prompt | llm | parser → 1s | Agent 推理 5s → 过度设计 |
| 翻译 | prompt | llm | parser → 1s | 不需要文件/子Agent/规划 |

#### 场景 4：纯文本处理

**什么意思**：输入是文本，输出也是文本。不需要工具、不需要文件系统、不需要搜索。

**为什么 LangChain 更好**：

```
LangChain:
  翻译 → prompt | llm | parser, 1 秒
  摘要 → prompt | llm | parser, 2 秒
  分类 → prompt | llm | parser, 0.5 秒

DeepAgents:
  同样的任务：
  Agent 推理 "用户想翻译" → 决定直接回答 → 返回
  多消耗 1 次推理调用（浪费 Token 和时间）
```

### 15.4 完整对比矩阵

| 维度 | LangChain 原生 | DeepAgents |
|---|---|---|
| **启动时间** | <0.1 秒 | <0.5 秒 |
| **单次调用延迟** | 1~3 秒 | 5~30 秒（取决于任务复杂度） |
| **LLM 调用次数** | 尽量少（1~2 次） | 多（每次决策 1 次） |
| **Token 消耗** | 精确可控 | 较高（推理链 + TODO + 文件交互） |
| **代码量** | 少（管道声明式） | 极少（create_deep_agent 一行） |
| **灵活性** | 低（流程固定） | 极高（完全动态） |
| **可靠性** | 高（固定流程 = 确定行为） | 中高（Agent 可能走弯路） |
| **文件系统** | 需手写 | 内置 |
| **子 Agent** | 需 LangGraph 手写图 | 一行 SubAgent |
| **HITL** | 需手写 | interrupt_on 一行 |
| **审计合规** | 需手写 | LangSmith 内置 |
| **生产就绪度** | 高（流程可控） | 中（需监控 + 降级策略） |

### 15.5 实战决策树

```
你的应用：

1. 流程固定吗？
   ├─ 是 → LangChain（prompt | llm | parser 搞定）
   │      例: 翻译、摘要、分类、固定问答
   │
   └─ 否 → 继续判断 2

2. 需要文件系统吗？（读/写/搜索文件、执行代码）
   ├─ 否 → 继续判断 3
   └─ 是 → DeepAgents
           例: 代码生成+测试、报告撰写、代码仓库分析

3. 需要多步骤动态决策吗？
   ├─ 否 → LangChain（甚至不用 Agent，Chain 就够）
   └─ 是 → 继续判断 4

4. 步骤 > 5 且中间结果需要持久化？
   ├─ 否 → LangChain + LangGraph（自定义 Agent 流程）
   │      例: 3 步 Agent 查询 - 验证 - 回答
   └─ 是 → DeepAgents
           例: 研究任务、故障排查、多文件代码重构

5. 需要子 Agent 并行 / 安全审批？
   ├─ 否 → LangChain + LangGraph
   └─ 是 → DeepAgents
           例: 企业 IT 运维、多团队协作、合规操作
```

### 15.6 混合架构 — 最佳实践

**不是非黑即白的选择。最佳实践是把它们组合使用。**

```python
# 主流程用 DeepAgents：管理复杂任务、文件系统、子 Agent
main_agent = create_deep_agent(
    model="deepseek-v4-pro",
    subagents=[researcher, writer],
    ...
)

# 子 Agent 内部用 LangChain：执行确定的子任务
def my_deterministic_chain():
    return (
        prompt
        | ChatOpenAI(model="gpt-4o-mini")  # 便宜模型
        | StrOutputParser()
    )

# 确定性任务用 Chain，省 Token
# 复杂编排交给 DeepAgents
```

### 15.7 要点总结

```
DeepAgents 不是 LangChain 的替代品 — 它是 LangChain 之上的高级抽象

选 DeepAgents 的场景:
  ✓ 执行步骤无法预先确定（需要动态推理）
  ✓ 需要文件系统 + 代码执行环境
  ✓ 步骤多（5+），中间结果需要管理
  ✓ 需要子 Agent 并行 + 安全审批
  ✓ 需要持久记忆跨对话保持

选 LangChain 原生的场景:
  ✓ 流程固定（RAG、翻译、摘要、分类）
  ✓ 延迟敏感（用户等在屏幕前）
  ✓ 步骤少（1~3 步）
  ✓ 不需要文件系统/子 Agent/规划
  ✓ 成本敏感（Token 消耗需严格控制）

一句话：
  LangChain 是螺丝刀 — 精准、快速、简单任务一把搞定
  DeepAgents 是瑞士军刀 — 什么都能干，复杂任务时价值才体现
  不要用瑞士军刀拧螺丝，也不要用螺丝刀砍树
```
```
```