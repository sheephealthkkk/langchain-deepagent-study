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

