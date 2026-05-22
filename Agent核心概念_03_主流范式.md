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