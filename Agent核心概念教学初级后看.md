# Agent 核心概念教学

## 第一章：Agent 五大核心能力

### 1.1 什么是 Agent？

**Agent = 能自主决策、使用工具、多步推理来完成目标的 AI 系统。**

与普通 LLM 调用的本质区别：

```
普通 LLM：用户问 → LLM 一次性回答（无工具、无循环、无自主决策）
Agent：   用户给目标 → Agent 自己规划步骤 → 调用工具 → 观察结果 → 调整计划 → ... → 交付结果
```

### 1.2 五大核心能力总览

```
                         ┌─────────────┐
                         │   自主性     │
                         │  Autonomy   │
                         └──────┬──────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
        ┌─────┴─────┐   ┌──────┴──────┐   ┌─────┴─────┐
        │  感知能力  │   │ 推理与规划   │   │  行动能力  │
        │Perception │   │Reasoning    │   │  Action   │
        └─────┬─────┘   └──────┬──────┘   └─────┬─────┘
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                         ┌──────┴──────┐
                         │  学习能力    │
                         │  Learning   │
                         └─────────────┘
```

---

### 1.3 能力一：自主性（Autonomy）

**定义**：不需要人工逐步指导，Agent 自己决定"下一步做什么"。

**和普通 LLM 调用的对比**：

| | 普通 LLM | Agent |
|---|---|---|
| 用户输入 | "北京天气怎么样？" | "帮我规划北京三日游" |
| 系统行为 | 一次回答 | 先查天气 → 查景点 → 查交通 → 排日程 → 给出完整方案 |
| 谁决定步骤 | 用户 | Agent 自主决定 |
| 工具调用 | 0 次 | N 次（自主选择、自主组合） |

**目前实现策略**：

| 策略 | 原理 | 自主程度 |
|---|---|---|
| **ReAct 循环**（基础） | 思考 → 行动 → 观察 → 再思考...循环直到完成 | 中 |
| **Plan-and-Execute**（升级） | 先做完整计划 → 按计划逐步执行 | 高 |
| **多 Agent 协作**（高级） | 多个 Agent 各自规划、互相委派任务 | 极高 |

---

### 1.4 能力二：感知能力（Perception）

**定义**：Agent 能理解当前"环境状态"——对话历史、工具返回结果、上下文变化。

**感知的内容**：

```
Agent 的"感官" = 消息列表

[
    SystemMessage("你是旅行规划助手。"),
    HumanMessage("帮我规划北京三日游。"),
    AIMessage("我先查天气。", tool_calls=[...]),
    ToolMessage("北京明天晴，25°C"),           ← 感知：天气信息
    AIMessage("根据天气，建议第一天户外。", tool_calls=[...]),
    ToolMessage("故宫门票已预约。"),            ← 感知：预订结果
    AIMessage("再查酒店。", tool_calls=[...]),
    ToolMessage("搜索到 127 家酒店。"),         ← 感知：大量结果
    AIMessage("结果太多，添加预算过滤。", tool_calls=[...]),
    ...
]
```

**目前实现策略**：

| 策略 | 原理 | 适用场景 |
|---|---|---|
| **消息列表作为上下文窗口** | 所有感知信息都在 `List[BaseMessage]` 中 | 短对话（< 上下文窗口） |
| **Summarization 中间件** | 对话历史太长 → 自动摘要压缩 | 长对话/多步任务 |
| **RAG 检索外部知识** | 工具返回不够 → 用 RAG 检索补充 | 需要专业知识 |
| **文件系统感知**（Deep Agents） | 维护虚拟文件系统，存储中间结果 | 复杂多步任务，结果复用 |
| **Checkpoint 持久化**（LangGraph） | 每一步状态持久化到磁盘 | 长时间运行，支持暂停/恢复 |

---

### 1.5 能力三：推理与规划（Reasoning & Planning）

**定义**：Agent 能把复杂目标拆解为可执行的步骤序列，并在执行中动态调整。

**推理的三个层次**：

```
层次1：反应式推理（Reactive）
  "用户问了天气 → 调 get_weather"
  策略：ReAct 循环
  特点：一步一看，灵活但缺乏全局规划

层次2：规划式推理（Planning）
  "规划旅行 → 第一步查天气 → 第二步定景点 → 第三步订酒店 → 第四步排日程"
  策略：Plan-and-Execute
  特点：全局最优，但计划可能跟不上变化

层次3：反思式推理（Reflective）
  "查了天气→结果不好→重新规划室内景点替代方案"
  策略：Reflection / Self-Critique
  特点：能自我纠错，适合开放性问题
```

**目前实现策略**：

| 策略 | 实现方式 | 适用 |
|---|---|---|
| **ReAct** | `思考→行动→观察` 循环 | 通用，LangChain `create_agent` 默认 |
| **Chain-of-Thought (CoT)** | Prompt 里要求 "Let's think step by step" | 数学、逻辑推理 |
| **Tree-of-Thought (ToT)** | 多路径探索，选择最佳推理链 | 复杂规划、创造性任务 |
| **Plan-and-Execute** | 先调用 Planner 生成计划 → Executor 逐步执行 | 多步任务，有明确目标 |
| **Reflexion** | 执行 → 失败 → 反思失败原因 → 修正 → 重试 | 需要自我纠错的场景 |
| **ReWOO**（Reason Without Observation） | 一次性生成全部工具调用计划 → 批量执行 | 减少 LLM 调用次数，降低成本 |

---

### 1.6 能力四：行动能力（Action）

**定义**：Agent 不仅"想"，还能"做"——调用工具改变外部世界。

**行动的分类**：

```
行动能力
├─ 只读操作（感知型）
│   ├─ 查数据库（SQL 查询）
│   ├─ 调搜索 API（Google / Bing / Tavily）
│   ├─ 读文件（本地 / 远程）
│   └─ 调外部 API（查天气 / 查股票）
│
└─ 写操作（改变型）
    ├─ 写数据库（INSERT / UPDATE）
    ├─ 发送消息（邮件 / Slack / 微信）
    ├─ 创建资源（GitHub Issue / Jira Ticket）
    ├─ 执行代码（Python / Shell）
    └─ 调用支付 / 下单 API  ← 高风险！
```

**目前实现策略**：

| 策略 | 原理 | 示例 |
|---|---|---|
| **Function Calling** | LLM 输出函数名 + 参数，框架执行 | `get_weather(city="北京")` |
| **代码执行沙箱** | Agent 写 Python/Shell → 沙箱中执行 → 拿结果 | 数据分析、文件处理 |
| **Human-in-the-Loop** | 写操作前暂停，等人类审批 | 支付、删除、发送邮件 |
| **工具权限分级** | 只读工具直接执行，写操作需确认 | `read_db` 自动，`delete_user` 确认 |

---

### 1.7 能力五：学习能力（Learning）

**定义**：Agent 能在交互中改进自己的行为——从错误中恢复、记住用户偏好、优化策略。

**这是当前最弱但最活跃的研究方向。**

**目前实现策略**：

| 策略 | 原理 | 成熟度 |
|---|---|---|
| **In-Context Learning**（上下文学习） | 示例放在 Prompt 里，模型模仿 | 成熟，常用 |
| **Few-Shot 示例** | 给 2~5 个 Q&A 示例，模型学习模式 | 成熟，常用 |
| **Reflexion**（反思） | 失败后让 LLM 分析原因 → 总结教训 → 下次避免 | 较成熟 |
| **记忆系统**（短期） | 对话历史就是短期记忆 | 成熟，`RunnableWithMessageHistory` |
| **记忆系统**（长期） | 向量库存储历史经验，检索相关记忆注入 Prompt | 发展中 |
| **Few-Shot 动态选择** | 根据当前问题，从示例库中检索最相关的 Few-Shot | 发展中 |
| **Fine-Tuning**（微调） | 用成功/失败数据训练模型 | 需大量数据，成本高 |
| **RLHF / DPO** | 强化学习对齐人类偏好 | 模型厂商层面 |

**LangChain 1.0 中实现记忆+学习的关键组件**：

```python
# 短期记忆：对话历史
from langchain_core.runnables.history import RunnableWithMessageHistory

# 长期记忆：向量库存储历史经验
from langchain_core.chat_history import BaseChatMessageHistory

# 反思：中间件自动分析错误
from langchain.agents.middleware import ToolRetry, Summarization
```

---

### 1.8 五大能力总结表

| 能力 | 一句话 | 当前最佳实现策略 | 成熟度 |
|---|---|---|---|
| **自主性** | 自己决定下一步 | ReAct + Plan-and-Execute + Multi-Agent | 高 |
| **感知能力** | 理解当前环境状态 | 消息列表 + Summarization + LangGraph Checkpoint | 高 |
| **推理与规划** | 拆解目标为步骤 | ReAct（通用）/ Plan-and-Execute（多步）/ Reflexion（纠错） | 中高 |
| **行动能力** | 调用工具改变世界 | Function Calling + 代码沙箱 + Human-in-the-Loop | 高 |
| **学习能力** | 从交互中改进 | Few-Shot + Reflexion + 向量记忆 | 低中 |

---

## 第二章：Agent 核心架构 = LLM + Tools + 思考循环

### 2.1 一句话架构

**Agent = LLM（大脑）+ Tools（手脚）+ Think-Act-Observe Loop（神经回路）**

```
                     ┌──────────┐
                     │   LLM    │  ← 大脑：推理、决策
                     │ (大脑)    │
                     └────┬─────┘
                          │ 思考 → 决定调用哪个工具
                          ▼
              ┌──────────────────────┐
              │     Think-Act-       │
              │   Observe Loop       │  ← 神经回路：循环
              │    (思考→行动→观察)   │
              └──────────┬───────────┘
                         │ 执行 → 拿到结果
                         ▼
              ┌──────────────────────┐
              │       Tools          │  ← 手脚：查天气、搜索、计算
              │  (查天气/搜索/计算)    │
              └──────────────────────┘
```

### 2.2 Think → Act → Observe 循环

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent 执行循环                            │
│                                                             │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐              │
│   │  THINK  │ ──→ │   ACT   │ ──→ │ OBSERVE │              │
│   │  思考   │     │  行动   │     │  观察   │              │
│   └─────────┘     └─────────┘     └─────────┘              │
│        ↑                                  │                │
│        └──────────── 循环 ────────────────┘                │
│                                                             │
│   每一步：                                                   │
│   THINK:  LLM 分析当前状态 → 判断"下一步该干什么"             │
│           → 要么调用工具（输出 tool_calls）                  │
│           → 要么给出最终回答（输出 content）                  │
│                                                             │
│   ACT:    框架执行 LLM 指定的工具调用                        │
│           → 把 tool_calls 转为实际函数执行                   │
│                                                             │
│   OBSERVE: 把工具返回结果追加到消息列表                      │
│           → Agent 看到新信息，回到 THINK                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 具体执行流程

用一个实际例子——用户问 "北京今天天气怎么样？适合户外运动吗？"：

```
═══════════════════════════════════════════════════════════════
循环开始
═══════════════════════════════════════════════════════════════

[THINK 第1轮]
  消息列表: [System("你是助手"), Human("北京今天天气怎么样？适合户外运动吗？")]
  LLM 推理: "我需要先获取北京的天气数据"
  LLM 决策: → tool_calls=[{name: "get_weather", args: {city: "北京"}}]
            → content=None（这次不直接回答）

[ACT]
  框架执行: get_weather(city="北京")
  返回: "北京：晴，25°C，湿度 45%，风速 2级"

[OBSERVE]
  追加: ToolMessage("北京：晴，25°C，湿度 45%，风速 2级", tool_call_id="c1")
  消息列表现在有 4 条消息了

═══════════════════════════════════════════════════════════════

[THINK 第2轮]
  消息列表: [System, Human, AIMessage(tool_calls=[...]), ToolMessage("北京：晴...")]
  LLM 推理: "拿到天气了。晴，25°C，微风，湿度适中"
  LLM 决策: "这些条件很适合户外运动！可以给用户具体建议"
  LLM 输出: → content="北京今天晴天，25°C，微风，非常适合户外运动！
                    建议去公园跑步、爬山或者骑行。注意防晒！"
            → tool_calls=None（不再需要工具）

  检测到 LLM 没有再要求调工具 → 循环结束！

═══════════════════════════════════════════════════════════════
循环结束 → 返回最终回答
═══════════════════════════════════════════════════════════════
```

### 2.4 循环终止条件

Agent 什么时候停止循环？四种情况：

```python
# 1. LLM 不再输出 tool_calls，只输出 content → 自然终止
AIMessage(content="北京今天晴天...", tool_calls=None)

# 2. 达到最大循环次数（max_tool_calls 中间件）
ToolCallLimit(max_tool_calls=10)  # 最多调用 10 次工具

# 3. 达到递归限制（RunnableConfig）
config={"recursion_limit": 25}

# 4. LLM 自己判断"任务完成"，显式结束
AIMessage(content="已完成所有任务。", tool_calls=[])
```

### 2.5 从代码看循环

```python
from langchain.agents import create_agent

# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}：晴，25°C"

@tool
def check_pollution(city: str) -> str:
    """获取空气质量"""
    return f"{city}：AQI 45，优"

# 创建 Agent
agent = create_agent(
    llm=ChatOpenAI(model="deepseek-v4-pro"),
    tools=[get_weather, check_pollution],
    system_prompt="你是生活助手。需要查询信息时主动调用工具。",
)

# 一次 invoke，内部自动完成 Think→Act→Observe 循环
result = agent.invoke({
    "messages": [
        HumanMessage("北京今天适合户外运动吗？")
    ]
})

# 用户只看到最终回答，循环过程对用户透明
print(result["messages"][-1].content)
# → "北京今天晴天 25°C，空气质量优，非常适合户外运动！建议..."
```

---

## 第三章：Agent 主流范式

### 3.1 范式一：ReAct（Reasoning + Acting）

**最经典、使用最广的范式。LangChain `create_agent` 的默认行为。**

```
ReAct = 思考（Reasoning）+ 行动（Acting）交替进行

Thought: 我需要查天气 → Action: get_weather("北京")
Thought: 天气不错，还需要查空气质量 → Action: check_pollution("北京")
Thought: 所有数据都有了 → 给出最终回答
```

**特点**：

| 优点 | 缺点 |
|---|---|
| 灵活，动态调整 | 多步任务时 LLM 调用次数多 |
| 模型自己能处理异常 | 没有全局计划，可能走弯路 |
| 实现简单，LangChain 一行搞定 | 对复杂多步任务效率不高 |

**代码**：

```python
agent = create_agent(llm=llm, tools=[...])
# 这就是 ReAct 模式，LangChain 默认
```

---

### 3.2 范式二：Function Calling（原生工具调用）

**利用 LLM API 原生的 Function Calling 能力，而非 Prompt 文本模拟。**

```
用户的 Prompt
     │
     ▼
LLM (API 层收到 tools 定义)
     │
     ├─→ 不需要工具 → 返回 content
     │
     └─→ 需要工具 → 返回 tool_calls[{name, args, id}]
              │
              ▼
         框架执行工具
              │
              ▼
         结果返回 LLM
```

**与 ReAct 的关系**：Function Calling 是**底层机制**，ReAct 是**上层范式**。ReAct 通常**基于** Function Calling 实现。

```python
# 底层：bind_tools → Function Calling
llm_with_tools = llm.bind_tools([get_weather, check_pollution])

# 高层：create_agent → ReAct（内部用的就是 bind_tools）
agent = create_agent(llm=llm, tools=[get_weather, check_pollution])
```

---

### 3.3 范式三：Plan-and-Execute（先规划后执行）

**先让 LLM 制定完整计划，再按计划逐步执行。**

```
用户目标: "帮我研究 LangChain 和 LlamaIndex 的差异，并写一篇对比文章"

[PLAN 阶段]
  规划 LLM 输出:
  Step 1: 搜索 LangChain 最新文档
  Step 2: 搜索 LlamaIndex 最新文档
  Step 3: 搜索两者的对比文章
  Step 4: 整理关键差异点
  Step 5: 写对比文章
  Step 6: 检查文章准确性

[EXECUTE 阶段]
  执行 Step 1 → 搜索 LangChain 最新文档
  执行 Step 2 → 搜索 LlamaIndex 最新文档
  执行 Step 3 → 搜索两者的对比文章
  执行 Step 4 → 整理关键差异点
  执行 Step 5 → 写对比文章
  执行 Step 6 → 检查文章准确性  ← 如果发现问题，修正

[REPLAN 阶段]（可选）
  如果执行中发现计划有问题 → 重新规划剩余步骤
```

**特点**：

| 优点 | 缺点 |
|---|---|
| 全局最优规划 | 计划可能跟不上实际情况变化 |
| 减少 LLM 调用（计划一次，执行多次） | 需要明确的、可分解的目标 |
| 适合复杂多步任务 | 简单任务杀鸡用牛刀 |

**实现**：LangGraph 中常见，用 `PlanNode` + `ExecuteNode` + 条件边构建。

---

### 3.4 范式四：Reflexion（反思 + 自我纠正）

**执行 → 失败 → 分析失败原因 → 生成改进策略 → 重试。**

```
任务: "写一个 Python 脚本读取 CSV 并计算平均值"

[尝试1]
  LLM 写代码 → 执行 → 报错: FileNotFoundError
  [Reflexion] "这个错误是因为我没有提供正确的文件路径，
               我需要先让用户提供文件路径，或者先列出目录中的文件。"

[尝试2]
  LLM 写代码 → 列出目录 → 找到 data.csv → 读取 → 执行 → 报错: KeyError
  [Reflexion] "CSV 的列名是 'score' 而不是 'value'，
               我应该先检查列名或使用 iloc。"

[尝试3]
  LLM 写代码 → 读取 data.csv → 用 'score' 列 → 计算平均值 → 成功！
  [Reflexion] "这次成功了。"
```

**特点**：

| 优点 | 缺点 |
|---|---|
| 能自我纠错 | 增加 LLM 调用次数（反思也消耗 Token） |
| 适合代码生成、数据分析 | 不适合简单任务 |
| 提高任务成功率 | 可能过度反思导致循环 |

**实现**：通过 LangGraph 构建 `ExecuteNode → CheckResultNode → ReflexionNode → (retry)` 循环。

---

### 3.5 范式五：Multi-Agent 协作

**多个 Agent 各自专精不同领域，互相协作完成复杂任务。**

```
                     ┌──────────────────┐
                     │   Supervisor     │  ← 总指挥：分配任务、汇总结果
                     │   (调度 Agent)    │
                     └──┬──────────┬────┘
                        │          │
            ┌───────────┘          └───────────┐
            ▼                                  ▼
   ┌────────────────┐                ┌────────────────┐
   │  Researcher    │                │  Writer        │
   │  (研究 Agent)  │                │  (写作 Agent)  │
   │                │                │                │
   │ 工具: 搜索API  │                │ 工具: 文档编辑 │
   │ 擅长: 搜集信息 │                │ 擅长: 生成文章 │
   └────────────────┘                └────────────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           ▼
                    ┌──────────────┐
                    │  Reviewer    │
                    │  (审核 Agent)│
                    │              │
                    │ 工具: 对比   │
                    │ 擅长: 检查   │
                    └──────────────┘
```

**特点**：

| 优点 | 缺点 |
|---|---|
| 每个 Agent 专精一个领域，质量高 | LLM 调用量大，成本高 |
| 天然支持复杂任务分解 | 通信开销：Agent 间需要传递上下文 |
| 易于扩展（加新 Agent 即可） | 调度逻辑复杂，可能出现死锁 |

---

### 3.6 五种范式对比

| 范式 | 核心循环 | LLM 调用 | 工具调用 | 适用场景 |
|---|---|---|---|---|
| **ReAct** | 思考→行动→观察 | 每次行动前调一次 | 每次 1 个 | 通用，日常任务 |
| **Function Calling** | 底层机制 | 判断 + 执行 | 每次 1~N 个 | 所有范式的基础层 |
| **Plan-and-Execute** | 先规划→逐步执行 | 规划 1 次 + 每步 1 次 | 每步 N 个 | 明确多步目标 |
| **Reflexion** | 尝试→失败→反思→重试 | 每次尝试 + 反思 | 每次 1~N 个 | 代码生成、数据分析 |
| **Multi-Agent** | 多 Agent 并行/串行 | 每个 Agent 各自调用 | 各 Agent 独有 | 复杂跨领域任务 |

### 3.7 选择指南

```
任务特征                              → 推荐范式
───────────────────────────────────────────────────
单步、有明确工具                      → Function Calling
多步、流程灵活、需要动态调整           → ReAct
多步、目标明确、步骤可预规划           → Plan-and-Execute
需要保证结果正确、失败可重试           → Reflexion
跨领域、任务可分解、不同专长           → Multi-Agent

日常使用：
  简单任务 → create_agent（ReAct）默认搞定
  复杂多步 → Plan-and-Execute
  代码生成 → Reflexion
```