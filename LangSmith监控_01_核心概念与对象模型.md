# LangSmith 监控与可观测性教学

## 第一章：LangSmith 核心概念

### 1.1 什么是 LangSmith

**LangSmith = LangChain 生态的官方可观测性平台**。它回答四个问题：

| 问题 | 没有 LangSmith | 有 LangSmith |
|---|---|---|
| 这条链内部发生了什么？ | 黑盒——只知道输入和输出 | 每一步的输入/输出/耗时完整可视 |
| 为什么这次回答质量差？ | 猜——可能 Prompt 不对、检索不准、LLM 幻觉 | 精准定位——看检索结果、看 Prompt 实际内容、看 LLM 参数 |
| 这次调用花了多少 Token/钱？ | 不知道——直到月底账单 | 实时——每次调用的 Token 明细和成本 |
| 用户对这个回答满意吗？ | 不知道——除非用户主动反馈 | 每条 Trace 绑定 thumbs up/down |

**一句话**：LangSmith 让 LLM 应用从"黑盒"变成"透明玻璃盒"——每一步都可看、可查、可优化。

### 1.2 两大核心功能

#### 功能 1：覆盖链路的追踪日志与实时分析

```
你的代码（零改动）                LangSmith 平台
─────────────────                ─────────────
                                  ┌─────────────────────────┐
chain.invoke("What is RAG?")      │  Trace #1                │
  │                               │  ├─ Run: retriever       │
  ├─ retriever 检索                │  │   input: "What is RAG?"│
  │    ↓                          │  │   output: [Doc, Doc]  │
  │  返回 [Doc, Doc]              │  │   latency: 0.23s      │
  │                               │  │                       │
  ├─ LLM 生成回答                  │  ├─ Run: ChatOpenAI     │
  │    ↓                          │  │   input: Prompt + Docs│
  │  返回 "RAG is..."              │  │   output: "RAG is..."│
  │                               │  │   tokens: 320 in/80 out│
  └─ 返回给用户                    │  │   cost: $0.002        │
                                  │  │   latency: 1.2s       │
                                  │  └───────────────────────│
                                  │  Total: 1.5s, $0.002     │
                                  └─────────────────────────┘
```

**实时分析能力**：
- 延迟分布：P50/P95/P99 延迟曲线
- Token 趋势：按小时/天查看 Token 消耗趋势
- 错误率监控：异常 Trace 自动标记
- 成本可视化：按项目/按模型/按用户拆分成本

#### 功能 2：构建集成的监控与调试环境

LangSmith 不只是"看"，还能"改"和"评估"：

```
┌─────────────────────────────────────────────────────────────┐
│                 LangSmith 监控调试工作流                       │
│                                                             │
│  发现问题                                                     │
│    │  用户反馈："这个回答不对"                                  │
│    ▼                                                        │
│  定位 Trace                                                   │
│    │  在 LangSmith 中找到该用户的调用记录                       │
│    ▼                                                        │
│  分析每一步                                                   │
│    │  展开 Run → 看检索结果 → 发现返回了不相关的文档             │
│    ▼                                                        │
│  调试修改                                                     │
│    │  修改 Retriever 参数（k=3→k=5）                          │
│    ▼                                                        │
│  对比回归                                                     │
│    │  用同一输入跑新旧两个版本 → 对比结果                       │
│    ▼                                                        │
│  添加评估                                                     │
│    │  为这个场景添加测试用例 → 持续监控                        │
│    ▼                                                        │
│  上线监控                                                     │
│       自动化评估 + 异常告警                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 第二章：核心对象模型

### 2.1 层级关系

```
┌─────────────────────────────────────────────────────────────┐
│                        Project                              │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                    Trace #1                            │ │
│  │  ┌──────────────────────────────────────────────┐    │ │
│  │  │              Run: retriever                   │    │ │
│  │  │  input → output, latency, tokens, metadata   │    │ │
│  │  └──────────────────────────────────────────────┘    │ │
│  │  ┌──────────────────────────────────────────────┐    │ │
│  │  │              Run: ChatOpenAI                   │    │ │
│  │  │  input → output, latency, tokens, metadata   │    │ │
│  │  │      ┌─────────────────────────────────┐     │ │
│  │  │      │  Run: tool_call (get_weather)    │     │ │
│  │  │      │  input → output                  │     │ │
│  │  │      └─────────────────────────────────┘     │ │
│  │  └──────────────────────────────────────────────┘    │ │
│  │  Feedback: 👍 (thumbs up)                            │ │
│  │  Tags: ["production", "v2"]                           │ │
│  │  Metadata: {"user_id": "alice", "session": "abc"}    │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                    Trace #2                            │ │
│  │  ...                                                   │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 五个核心对象详解

#### Project（项目）

**LangSmith 的顶层组织单元。** 一个 Project = 一个应用/服务/实验。

| 属性 | 说明 | 示例 |
|---|---|---|
| 名称 | 唯一标识 | `"my-rag-agent"`, `"customer-service-bot"` |
| 包含 | 所有 Trace | 该应用产生的所有调用记录 |
| 环境变量 | `LANGCHAIN_PROJECT` | 代码中设置，自动归属 |

**常见组织方式**：

```
方式 A（按应用）：
  my-rag-agent        ← 一个 Project
  customer-bot        ← 另一个 Project

方式 B（按环境）：
  my-app-dev          ← 开发环境
  my-app-staging      ← 预发布环境
  my-app-prod         ← 生产环境

方式 C（按实验）：
  rag-experiment-v1   ← 实验组 1
  rag-experiment-v2   ← 实验组 2（对比 A/B 测试）
```

#### Trace（追踪）

**一次完整的端到端调用。** 对应一次 `chain.invoke()` 或 `agent.invoke()`。

```
Trace = 用户请求 → 系统处理 → 返回响应 的完整记录

一次 Trace 包含：
  ├─ 一个根 Run
  ├─ N 个子 Run（嵌套层级 = 调用链深度）
  ├─ 0~N 个 Feedback（用户评价）
  ├─ Tags（分类标签）
  └─ Metadata（自定义元数据）
```

**Trace 的生命周期**：

```
创建 → 运行中 → 完成（成功/失败）
                   │
                   ├─ Feedback 追加（异步，可能在 Trace 完成后很久）
                   └─ 永久保留（除非主动删除）
```

#### Run（运行单元）

**Trace 中的一个执行步骤。** 每个 Runnable 的每次调用产生一个 Run。

```
chain.invoke(input)
  ├─ Run #1: retriever.invoke(input)      ← 子 Run
  ├─ Run #2: prompt.invoke(dict)          ← 子 Run
  ├─ Run #3: llm.invoke(messages)         ← 子 Run
  │    ├─ Run #3a: tool_call(get_weather) ← 子 Run 的子 Run
  │    └─ Run #3b: tool_call(search_web)
  └─ Run #4: parser.invoke(ai_message)    ← 子 Run
```

**Run 记录的核心数据**：

| 字段 | 说明 | 示例 |
|---|---|---|
| `name` | Runnable 名称 | `"ChatOpenAI"`, `"retriever"`, `"RunnableSequence"` |
| `run_type` | 运行类型 | `"llm"`, `"chain"`, `"tool"`, `"retriever"`, `"prompt"` |
| `inputs` | 输入数据 | `{"messages": [...]}` |
| `outputs` | 输出数据 | `AIMessage(content="...")` |
| `start_time` / `end_time` | 开始/结束时间 | 用于计算 latency |
| `error` | 错误信息 | `None` 或异常对象 |
| `total_tokens` / `prompt_tokens` / `completion_tokens` | Token 用量 | `350` / `300` / `50` |
| `parent_run_id` | 父 Run ID | 构建嵌套层级关系 |

#### Feedback（反馈）

**对 Trace 或 Run 的评价。** 可以来自用户（thumbs up/down）或自动评估。

```python
from langsmith import Client

client = Client()

# 人工反馈：用户点了 👍
client.create_feedback(
    run_id="abc-123",
    key="user_feedback",       # 反馈类型
    score=1,                   # 1=正面, 0=负面
    comment="回答很准确！",
)

# 自动反馈：评估系统打分
client.create_feedback(
    run_id="abc-123",
    key="correctness",         # 正确性评分
    score=0.95,                # 0~1 分数
    comment="自动评估：回答与参考答案匹配度 95%",
)
```

**Feedback 的用途**：

| 用途 | 反馈 Key | 来源 |
|---|---|---|
| 用户满意度 | `user_feedback` | 用户 👍/👎 |
| 回答正确性 | `correctness` | 自动评估（对比参考答案） |
| 回答相关性 | `relevance` | 自动评估 |
| 毒性检测 | `toxicity` | 自动评估 |
| 幻觉检测 | `hallucination` | 自动评估 |
| 人工审核 | `human_review` | 审核员标记 |

#### Tags（标签）与 Metadata（元数据）

**Tags = 分类维度，Metadata = 上下文信息。**

```python
# 在 invoke 时传入
chain.invoke(
    input,
    config={
        "tags": ["production", "v2.1"],         # ← Tags: 简短标签
        "metadata": {                           # ← Metadata: 结构化信息
            "user_id": "alice",
            "session_id": "chat_abc123",
            "deployment": "us-east-1",
            "experiment": "new_prompt_v3",
        },
    },
)
```

| | Tags | Metadata |
|---|---|---|
| **类型** | `list[str]` | `dict[str, Any]` |
| **用途** | 分类、过滤、分组 | 携带业务上下文 |
| **示例** | `["production", "v2"]` | `{"user_id":"alice", "plan":"enterprise"}` |
| **查询** | `按 tag="production" 过滤` | `按 metadata.user_id="alice" 过滤` |
| **典型场景** | A/B 测试分组、环境标识 | 用户 ID、会话 ID、部署区域 |

---

