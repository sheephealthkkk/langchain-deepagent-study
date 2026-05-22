## 第六章：其他常用中间件

### 6.1 `TodoMiddleware` — 任务规划

```python
from langchain.agents.middleware import TodoMiddleware

# Agent 收到复杂任务时，自动生成 TODO 列表，按计划执行
agent = create_agent(
    model=llm,
    tools=[...],
    middleware=[TodoMiddleware()],
)
# 效果：用户问 "帮我写一篇 LangChain 对比文章" →
# Agent 先规划：
#   [TODO] 1. 搜索 LangChain 最新特性
#   [TODO] 2. 搜索竞品对比
#   [TODO] 3. 整理关键差异
#   [TODO] 4. 写文章
#   [TODO] 5. 检查准确性
# → 然后逐个完成
```

### 6.2 `ModelRetryMiddleware` — 模型重试

```python
from langchain.agents.middleware import ModelRetryMiddleware

# 与 ToolRetryMiddleware 完全对称，但针对 LLM 调用失败
agent = create_agent(
    model=llm,
    tools=[...],
    middleware=[
        ModelRetryMiddleware(
            max_retries=3,
            retry_on=(RateLimitError, APITimeoutError),
            backoff_factor=2.0,
        ),
    ],
)
```

### 6.3 `ShellToolMiddleware` — Shell 沙箱安全

```python
from langchain.agents.middleware import ShellToolMiddleware

# 对 Shell 工具添加执行策略（主机执行 / Docker / 沙箱）
agent = create_agent(
    model=llm,
    tools=[shell_tool],
    middleware=[
        ShellToolMiddleware(
            policy=DockerExecutionPolicy(image="python:3.12"),
            # 所有 Shell 命令在 Docker 容器中执行（隔离）
        ),
    ],
)
```

### 6.4 `FileSearchMiddleware` — 文件检索

```python
from langchain.agents.middleware import FileSearchMiddleware

# 为 Agent 添加文件搜索能力（类似 Claude 的文件检索功能）
agent = create_agent(
    model=llm,
    middleware=[
        FileSearchMiddleware(
            file_paths=["./docs/", "./codebase/"],
            max_results=5,
        ),
    ],
)
```

---

## 第七章：中间件组合策略

### 7.1 推荐的中间件栈

```python
# 生产级 Agent 中间件栈
agent = create_agent(
    model=primary_llm,
    tools=[...],
    middleware=[
        # === 第 1 层：安全（最外层，最先拦截）===
        HumanInTheLoopMiddleware(      # 高风险操作审批
            review_configs=[...]
        ),
        PIIMiddleware("email", strategy="redact"),  # PII 脱敏
        PIIMiddleware("credit_card", strategy="mask"),
        
        # === 第 2 层：可靠性 ===
        ModelFallbackMiddleware(       # LLM 故障切换
            backup_model_1,
            backup_model_2,
        ),
        ModelRetryMiddleware(          # LLM 重试
            max_retries=2,
        ),
        ToolRetryMiddleware(           # 工具重试
            max_retries=3,
        ),
        
        # === 第 3 层：资源控制 ===
        ModelCallLimitMiddleware(      # LLM 调用上限
            run_limit=15,
            thread_limit=100,
        ),
        ToolCallLimitMiddleware(       # 工具调用上限
            individual_limit={"send_email": 5},
        ),
        
        # === 第 4 层：上下文管理（最内层）===
        ToolSelectionMiddleware(       # 工具预选
            max_tools=5,
        ),
        SummarizationMiddleware(       # 历史压缩
            trigger_token_limit=4000,
        ),
    ],
)
```

### 7.2 按场景选择

| 场景 | 核心问题 | 推荐中间件 |
|---|---|---|
| **客服 Agent** | 长对话爆炸、PII 泄露 | Summarization + PII(redact) + HumanInTheLoop(退款) |
| **代码生成 Agent** | 工具执行失败、死循环 | ToolRetry + ToolCallLimit + ShellTool(Docker) |
| **数据分析 Agent** | 多步任务无规划、Token 超限 | Todo + Summarization + ContextEditing |
| **面向外部用户** | 安全、成本控制 | HumanInTheLoop(全写操作) + ModelCallLimit(thread) + PII(block) |

---

