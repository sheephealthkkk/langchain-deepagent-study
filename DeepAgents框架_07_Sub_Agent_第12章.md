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

