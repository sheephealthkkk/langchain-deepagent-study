## 第四章：第三类 — `wrap_tool_call` 中间件

这一类中间件包裹在**工具调用外面**，可以拦截、重试、修改、跳过工具执行。

### 4.1 高风险操作的自动重试 — `ToolRetryMiddleware`

#### 工作原理

```
START
  │
  ▼
┌──────────────────────────────────────────────────────┐
│            ToolRetryMiddleware                        │
│                                                      │
│  wrap_tool_call(request, handler)                    │
│    │                                                 │
│    ├─ 1. 初始化 attempt=0，计算初始延迟                 │
│    │                                                 │
│    ├─ 2. 执行 handler(request)                        │
│    │     ├─ 成功 → 返回 ToolMessage                   │
│    │     └─ 失败 → 捕获异常                            │
│    │                                                 │
│    ├─ 3. 检查是否应该重试                              │
│    │     ├─ 异常类型在 retry_on 列表中？                │
│    │     │     → 是 → 继续                            │
│    │     │     → 否 → 立即抛出                         │
│    │     ├─ attempt < max_retries？                   │
│    │     │     → 是 → 继续                            │
│    │     │     → 否 → 根据 on_failure 处理             │
│    │     └─ 等待 backoff_factor ^ attempt 秒          │
│    │                                                 │
│    └─ 4. 重试循环直到成功或达到上限                     │
│                                                      │
└──────────────────────────────────────────────────────┘
  │
  ▼
 END
```

**退避策略**：

```
第 1 次失败 → 等待 1.0 秒 → 重试
第 2 次失败 → 等待 2.0 秒 → 重试  (backoff_factor=2.0)
第 3 次失败 → 等待 4.0 秒 → 重试
第 4 次失败 → 达到 max_retries=3 → 最终失败
```

#### 使用 vs 不使用的效果对比

```
工具：search_web("最新 LangChain 文档")

不使用 ToolRetryMiddleware:
┌─────────────────────────────────────────────────────┐
│ search_web → 网络超时 → 异常抛出                      │
│ Agent 崩溃 → 用户看到错误 "Request timed out"         │
│ 需要用户手动重新提问                                  │
└─────────────────────────────────────────────────────┘

使用 ToolRetryMiddleware(max_retries=3):
┌─────────────────────────────────────────────────────┐
│ search_web → 网络超时 → [等待 1s] → 重试             │
│ search_web → 网络超时 → [等待 2s] → 重试             │
│ search_web → 成功！→ 返回搜索结果                     │
│ Agent 继续 → 用户无感知 → 拿到正确答案               │
└─────────────────────────────────────────────────────┘
```

#### 完整示例

```python
# ================================================================
# 07_tool_retry_demo.py — 工具重试中间件
# ================================================================
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_openai import ChatOpenAI
from requests.exceptions import Timeout, ConnectionError

agent = create_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0.7),
    tools=[search_web, query_database, send_email],  # 这些都可能失败
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,            # 最多重试 3 次
            initial_delay=1.0,        # 第一次等 1 秒
            backoff_factor=2.0,       # 每次延迟 ×2（1s → 2s → 4s）
            max_delay=10.0,           # 最多等 10 秒
            retry_on=(                # 只在以下异常时重试
                Timeout,              # 网络超时 → 可能是临时的
                ConnectionError,      # 连接失败 → 可能是临时的
                OSError,              # 系统级错误 → 可能是临时的
            ),
            # RuntimeError、ValueError 等不重试（永久性错误重试无意义）
            on_failure="continue",    # 最终失败 → 返回错误 ToolMessage（不崩溃）
        ),
    ],
    system_prompt="你是可靠助手。外部工具偶发故障时会自动重试。",
)

# 调用 — 工具失败时自动重试
result = agent.invoke({
    "messages": [HumanMessage("搜索最新的 LangChain 1.0 文档")]
})
```

---

## 第五章：第四类 — `after_model` 中间件

这一类中间件在 **LLM 调用返回后**执行。可以校验输出、审批敏感操作、限流控制。

### 5.1 人工干预中间件 — `HumanInTheLoopMiddleware`

#### 工作原理

```
START
  │
  ▼
┌──────────────────────────────────────────────────────┐
│         HumanInTheLoopMiddleware                      │
│                                                      │
│  after_model(state, runtime)                         │
│    │                                                 │
│    ├─ 1. 检查 LLM 是否输出了 tool_calls               │
│    │     ├─ 否 → 正常结束（无工具需要审批）             │
│    │     └─ 是 → 继续                                │
│    │                                                 │
│    ├─ 2. 对每个 tool_call 检查是否需要审批             │
│    │     action_name 在 review_configs 中？            │
│    │     ├─ 否 → 放行                                 │
│    │     └─ 是 → 触发 interrupt                       │
│    │                                                 │
│    ├─ 3. 触发 interrupt → Agent 暂停                  │
│    │     返回 HITLRequest 给调用方：                   │
│    │       {                                         │
│    │         "action_name": "delete_user",            │
│    │         "args": {"user_id": "123"},              │
│    │         "allowed_decisions": ["approve","reject"]│
│    │       }                                         │
│    │                                                 │
│    └─ 4. 人工审批                                     │
│        ├─ approve → 继续执行工具                       │
│        ├─ edit    → 修改参数后执行                     │
│        ├─ reject  → 跳过（返回拒绝消息）               │
│        └─ respond → 人工直接回复（不调工具）            │
│                                                      │
└──────────────────────────────────────────────────────┘
  │
  ▼
 END
```

#### 使用 vs 不使用的效果对比

```
LLM 决定: tool_calls=[{name: "delete_user", args: {user_id: "123"}}]

不使用 HumanInTheLoopMiddleware:
┌─────────────────────────────────────────────────────┐
│ delete_user("123") → 立即执行 → 用户 123 被删除       │
│                                                     │
│ 风险: LLM 可能误判、用户 Prompt 可能被注入、不可逆操作  │
│ 后果: 删错了 → 无备份 → 数据丢失                      │
└─────────────────────────────────────────────────────┘

使用 HumanInTheLoopMiddleware:
┌─────────────────────────────────────────────────────┐
│ LLM 输出 delete_user → after_model 触发              │
│   → interrupt → Agent 暂停                          │
│   → UI 弹窗："确认删除用户 123？[确认] [拒绝]"         │
│   → 管理员点 [拒绝] → Agent 继续 → 输出"操作已拒绝"    │
│                                                     │
│ 效果: 高风险操作不会自动执行，人工把关                  │
└─────────────────────────────────────────────────────┘
```

#### 完整示例

```python
# ================================================================
# 08_human_in_the_loop_demo.py — 人工干预中间件
# ================================================================
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI

agent = create_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0),
    tools=[
        search_web,         # 只读 — 自动执行
        send_email,         # 发送 — 需要审批
        delete_user,        # 删除 — 需要审批
        create_order,       # 创建 — 需要审批
    ],
    middleware=[
        HumanInTheLoopMiddleware(
            # 审批策略：哪些工具需要人工审批，允许什么决策类型
            review_configs=[
                {
                    "action_name": "delete_user",
                    "allowed_decisions": ["approve", "reject"],
                    # 只能同意或拒绝（不可修改参数）
                },
                {
                    "action_name": "send_email",
                    "allowed_decisions": ["approve", "edit", "reject"],
                    # 可以编辑邮件内容后发送
                },
                {
                    "action_name": "create_order",
                    "allowed_decisions": ["approve", "edit", "reject"],
                    # 可以修改订单参数
                },
            ],
        ),
    ],
    system_prompt="你是管理员助手。写操作需要审批，读操作自动执行。",
)

# 调用 — Agent 输出 delete_user 时会暂停
import asyncio

async def demo():
    config = {"configurable": {"thread_id": "hitl_demo"}}
    
    # 第一次调用：Agent 尝试删除用户 → 触发 interrupt
    result = agent.invoke({
        "messages": [HumanMessage("删除用户 123，他违反了社区规定")]
    }, config=config)
    
    # 检查是否有待审批的操作
    state = agent.get_state(config)
    if state.interrupts:
        print(f"⏸️  Agent 暂停，等待审批：{state.interrupts}")
        
        # 人工决策：拒绝
        from langgraph.types import Command
        agent.invoke(
            Command(resume={"decision": "reject", "reason": "需要进一步核实"}),
            config=config,
        )
        print("❌ 操作被拒绝")
    else:
        print(f"✅ 自动完成：{result['messages'][-1].content[:100]}")
```

---

### 5.2 工具调用限制中间件 — `ToolCallLimitMiddleware`

#### 工作原理

```
START
  │
  ▼
┌──────────────────────────────────────────────────────┐
│          ToolCallLimitMiddleware                      │
│                                                      │
│  after_model(state, runtime)                         │
│    │                                                 │
│    ├─ 1. 从 state 读取工具调用计数                    │
│    │     counts = state["thread_tool_call_count"]    │
│    │     counts 是一个 dict: {"get_weather": 3, ...}  │
│    │                                                 │
│    ├─ 2. 检查当前需要执行的 tool_calls                │
│    │    对每个 tool_call:                             │
│    │      individual_count >= individual_limit ?     │
│    │      total_count >= all_tools_limit ?            │
│    │                                                 │
│    ├─ 3. 超限处理                                     │
│    │     ├─ "continue" → 超限工具返回错误 ToolMessage  │
│    │     ├─ "end"     → jump_to="end"（立即结束）     │
│    │     └─ "error"   → 抛出异常                      │
│    │                                                 │
│    └─ 4. 未超限 → 计数+1 → 放行工具执行               │
│                                                      │
└──────────────────────────────────────────────────────┘
  │
  ▼
 END
```

**与 ModelCallLimitMiddleware 的区别**：一个限制 LLM 调用次数，一个限制工具调用次数。两者互补。

#### 完整示例

```python
# ================================================================
# 09_tool_call_limit_demo.py — 工具调用限制中间件
# ================================================================
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI

agent = create_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0),
    tools=[search_web, calculate, send_email, delete_user],
    middleware=[
        ToolCallLimitMiddleware(
            # 每个工具单独限制
            individual_limit={
                "send_email": 5,      # 每会话最多发 5 封邮件
                "delete_user": 0,     # 完全禁止调用（设为 0）
                "__all__": 20,        # 所有工具总共最多 20 次
            },
            # 超限行为
            exit_behavior="continue", # 超限工具返回错误提示（不崩溃）
        ),
    ],
    system_prompt="你是助手。部分操作有次数限制。",
)

# 多次调用后，某些工具会被限制
config = {"configurable": {"thread_id": "limit_demo"}}
result = agent.invoke({
    "messages": [HumanMessage("搜索 LangChain 相关信息")]
}, config=config)
```

---

