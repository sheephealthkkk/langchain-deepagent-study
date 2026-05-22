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

