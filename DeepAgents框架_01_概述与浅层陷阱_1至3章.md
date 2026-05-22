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

