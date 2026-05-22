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