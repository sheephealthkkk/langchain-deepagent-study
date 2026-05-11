# LangChain 中间件教学

## 第一章：中间件概览

### 1.1 什么是中间件

**中间件 = Agent 执行流程中的可插拔拦截器。** 类似 Java 中 Servlet Filter / Spring Interceptor / AOP 切面——在请求处理链的特定节点插入自定义逻辑，不修改 Agent 核心代码。

```
用户请求
  │
  ▼
┌──────────────────────────────────────────────────────┐
│                  Agent 执行流程                       │
│                                                      │
│  [before_agent]  ← 中间件钩子 1                      │
│        │                                             │
│        ▼                                             │
│  ┌─────────────────────┐                             │
│  │   before_model       │ ← 中间件钩子 2              │
│  ├─────────────────────┤                             │
│  │   wrap_model_call    │ ← 中间件钩子 3（拦截+替换） │
│  ├─────────────────────┤                             │
│  │   after_model        │ ← 中间件钩子 4              │
│  └─────────┬───────────┘                             │
│            │                                         │
│    ┌───────┴───────┐                                 │
│    │ 需要调工具吗？  │                                │
│    └───────┬───────┘                                 │
│     是     │     否                                   │
│      ▼            ▼                                   │
│  ┌─────────────────┐  ┌──────────┐                  │
│  │ wrap_tool_call   │  │  结束    │                  │
│  │ ← 中间件钩子 5   │  └──────────┘                  │
│  └────────┬────────┘                                 │
│           │                                          │
│           ▼ 回到 before_model（循环）                  │
│                                                      │
│  [after_agent]   ← 中间件钩子 6                      │
│                                                      │
└──────────────────────────────────────────────────────┘
  │
  ▼
输出响应
```

### 1.2 中间件分类

#### 按生命周期钩子分类

| 钩子 | 位置 | 能做什么 | 数据方向 |
|---|---|---|---|
| `before_agent` | Agent 启动前 | 初始化 State、注入上下文、权限校验 | State 更新 → Agent |
| `before_model` | 每次 LLM 调用前 | 修改 messages、动态选择 tools、拦截跳转 | State 更新 + 可能跳转 |
| `wrap_model_call` | 包裹 LLM 调用 | 重试、降级、缓存、短-路由 | 完整 request → 替换 response |
| `after_model` | 每次 LLM 调用后 | 校验输出、日志记录、提取结构化数据 | State 更新 |
| `wrap_tool_call` | 包裹工具调用 | 参数校验/修改、重试、缓存、权限拦截 | 修改 tool_call → 替换 ToolMessage |
| `after_agent` | Agent 结束后 | 清理资源、保存 State、审计日志 | State 更新 |
| `dynamic_prompt` | 动态生成 Prompt | 根据 State/Runtime 动态构建 system prompt | 新 Prompt 注入 |

#### 按关注点分类

| 分类 | 内置中间件 | 解决问题 |
|---|---|---|
| **模型可靠性** | `ModelRetryMiddleware`, `ModelFallbackMiddleware` | LLM 调用失败、超时、降级 |
| **工具可靠性** | `ToolRetryMiddleware`, `ToolSelectionMiddleware`, `ToolCallLimitMiddleware` | 工具调用失败、乱调、死循环 |
| **上下文管理** | `SummarizationMiddleware`, `ContextEditingMiddleware` | 对话过长、Token 超限 |
| **安全审计** | `HumanInTheLoopMiddleware`, `PIIMiddleware`, `ShellToolMiddleware` | 敏感操作审批、隐私脱敏 |
| **开发调试** | `TodoMiddleware`, `ToolEmulatorMiddleware`, `FileSearchMiddleware` | 任务规划、工具模拟、文件检索 |

### 1.3 中间件解决了什么问题

**问题 1：横切关注点散落各处**

没有中间件时，重试逻辑、日志、权限校验散落在每个工具和每个 LLM 调用中。

```python
# ❌ 没有中间件 — 每个工具都重复写重试逻辑
@tool
def tool_a(query):
    for attempt in range(3):        # 重复代码
        try:
            return do_a(query)
        except Exception:
            if attempt == 2: raise

@tool
def tool_b(query):
    for attempt in range(3):        # 重复代码
        try:
            return do_b(query)
        except Exception:
            if attempt == 2: raise

# ✅ 有中间件 — 重试逻辑集中在一处
ToolRetryMiddleware(max_retries=3)  # 所有工具自动获得重试
```

**问题 2：核心流程不可见**

没有中间件 → `agent.invoke()` 是黑盒 → 不知道每一步发生了什么。

有中间件 → 在每个钩子打印日志 → 完整执行轨迹。

**问题 3：行为不可动态调整**

没有中间件 → Agent 的 system prompt、工具列表启动后固定。

有中间件 → `before_model` 可以动态切换 tools、`dynamic_prompt` 可以根据 State 动态生成 prompt。

### 1.4 中间件的数据传输机制

中间件通过三种方式与 Agent 交互：

```
方式 1：State 更新（最常用）
  中间件返回 dict → 合并到 Agent State
  before_agent 返回 {"turn_count": 0} → State.turn_count = 0

方式 2：ModelRequest/ModelResponse（wrap_model_call 专用）
  中间件收到完整 request → 可以修改 → 调用 handler 或跳过
  返回替代 response → 下游拿到的是中间件替换后的结果

方式 3：跳转（JumpTo）
  中间件返回 {"jump_to": "end"} → Agent 直接结束
  跳过后续所有步骤
  有效目标：tools / model / end
```

**数据流详解**：

```
before_agent(state, runtime) → dict | None
  │ 输入：当前 State + Runtime（含 store、config 等）
  │ 输出：dict → 合并到 State；None → 无变化
  ▼

before_model(state, runtime) → dict | Command | None
  │ 输入：当前 State（含 messages） + Runtime
  │ 输出：State 更新 或 跳转指令
  │ 特殊：@hook_config(can_jump_to=["end","tools"]) 允许跳转
  ▼

wrap_model_call(request, handler) → ModelResponse | AIMessage
  │ 输入：ModelRequest（model, messages, tools, state, runtime）
  │ handler：调用它 = 执行模型；不调用 = 短路
  │ 输出：替换模型返回结果
  ▼

wrap_tool_call(request, handler) → ToolMessage | Command
  │ 输入：ToolCallRequest（tool_call dict, BaseTool, state, runtime）
  │ handler：调用它 = 执行工具；可以多次调用 = 重试
  │ 输出：工具执行结果
```

### 1.5 代理执行完整生命周期

```
═══════════════════════════════════════════════════════════════
                      Agent 生命周期
═══════════════════════════════════════════════════════════════

[1] 用户调用 agent.invoke({"messages": [...]}, config)
        │
[2]     ▼  before_agent(state, runtime)         ← 所有中间件按序执行
        │    ├─ SummarizationMiddleware: 裁剪过长历史
        │    └─ TodoMiddleware: 初始化 TODO 列表
        │
[3]     ▼  before_model(state, runtime)          ← 每次 LLM 调用前
        │    ├─ ContextEditingMiddleware: 修剪消息
        │    └─ ToolSelectionMiddleware: 动态选择工具
        │
[4]     ▼  dynamic_prompt(request)               ← 动态生成 Prompt
        │
[5]     ▼  wrap_model_call(request, handler)     ← 模型调用（可拦截）
        │    ├─ ModelRetryMiddleware: 失败重试
        │    ├─ ModelFallbackMiddleware: 切换模型
        │    └─ handler(request) → LLM API 调用
        │
[6]     ▼  after_model(state, runtime)           ← LLM 调用后
        │    ├─ 检查是否需要调工具
        │    ├─ 不需要 → 跳到 [8]
        │    └─ 需要 → 继续 [7]
        │
[7]     ▼  wrap_tool_call(request, handler)      ← 工具调用（可拦截）
        │    ├─ ToolRetryMiddleware: 失败重试
        │    ├─ HumanInTheLoopMiddleware: 需审批→暂停
        │    ├─ PIIMiddleware: 脱敏工具参数
        │    └─ handler(request) → 执行工具
        │    返回 ToolMessage → 追加到 messages → 回到 [3]
        │
[8]     ▼  after_agent(state, runtime)           ← Agent 结束
             ├─ SummarizationMiddleware: 生成摘要并保存
             └─ 清理、审计日志、返回最终结果
```

### 1.6 中间件与 AOP 的类比（Java 程序员视角）

| LangChain 中间件 | Java 类比 | 说明 |
|---|---|---|
| `AgentMiddleware` | `HandlerInterceptor` / `@Aspect` | 基类，定义拦截点 |
| `before_agent` | `preHandle()` / `@Before` | 入口拦截 |
| `before_model` | `@Before` 切点 | LLM 调用前 |
| `wrap_model_call` | `@Around` 切点 | 包裹 LLM 调用（可替换结果） |
| `after_model` | `@AfterReturning` 切点 | LLM 调用后 |
| `wrap_tool_call` | `@Around` 切点 | 包裹工具调用（可重试/跳过） |
| `after_agent` | `afterCompletion()` / `@After` | 出口拦截 |
| `@hook_config(can_jump_to=...)` | `response.sendRedirect()` | 控制流程跳转 |
| `middleware 列表` | `InterceptorRegistry` | 链式拦截，先注册=最外层 |

### 1.7 两种定义方式

**方式 1：类继承（完整控制）**

```python
from langchain.agents.middleware import AgentMiddleware

class MyRetryMiddleware(AgentMiddleware):
    """自定义重试中间件。"""

    def wrap_model_call(self, request, handler):
        """包裹 LLM 调用，失败时重试。"""
        for attempt in range(3):
            try:
                return handler(request)     # 执行真正的 LLM 调用
            except Exception:
                if attempt == 2:
                    raise                   # 最后一次还失败 → 抛出
```

**方式 2：函数装饰器（轻量快捷）**

```python
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def my_retry(request, handler):
    """与上面等价，但不需要继承类。"""
    for attempt in range(3):
        try:
            return handler(request)
        except Exception:
            if attempt == 2:
                raise
```

**选择指南**：

| | 继承 `AgentMiddleware` | 函数装饰器 |
|---|---|---|
| 需要多个钩子 | 是（一个类定义多个方法） | 否（每个装饰器一个钩子） |
| 需要内部状态 | 是（实例变量） | 否 |
| 需要 `state_schema` | 是 | 否（自动推断） |
| 代码量 | 多 | 少（一个函数） |
| 适用 | 复杂中间件 | 简单拦截逻辑 |

### 1.8 中间件执行顺序与洋葱模型

```
        请求进入
           │
    ┌──────┴──────┐
    │ 中间件 A     │  ← 最外层（先注册）
    │  ┌────────┐ │
    │  │中间件 B │ │  ← 第二层
    │  │ ┌────┐ │ │
    │  │ │核心 │ │ │  ← Agent 核心逻辑
    │  │ │流程 │ │ │
    │  │ └────┘ │ │
    │  └────────┘ │
    └─────────────┘
           │
        响应返回

执行顺序：
  before_agent:   A → B → ... (注册顺序)
  wrap_model_call: A 包裹 B 包裹 handler  (外层先拦截)
  after_agent:    ... → B → A (反向)
```

```python
# 注册顺序决定执行顺序
agent = create_agent(
    llm=llm,
    tools=[...],
    middleware=[
        SummarizationMiddleware(),  # ← 第 1 层（最外层）
        ModelRetryMiddleware(),     # ← 第 2 层
        ToolRetryMiddleware(),      # ← 第 3 层（最内层）
    ],
)

# wrap_model_call 执行流：
# SummarizationMiddleware.wrap_model_call 被调用
#   → handler = ModelRetryMiddleware.wrap_model_call
#       → handler = ToolRetryMiddleware.wrap_model_call
#           → handler = 真正的 LLM 调用
#       ← ToolRetryMiddleware 返回结果
#   ← ModelRetryMiddleware 返回结果
# ← SummarizationMiddleware 返回结果
```

---

## 第二章：第一类 — `before_model` 中间件

这一类中间件在**每次 LLM 调用之前**执行。可以修改 messages、动态选择 tools、拦截并跳转。

### 2.1 上下文压缩中间件 — `SummarizationMiddleware`

#### 工作原理

```
START
  │
  ▼
┌──────────────────────────────────────────────┐
│         SummarizationMiddleware              │
│                                              │
│  before_model(state, runtime)                │
│    │                                         │
│    ├─ 1. 计算当前 messages 的 token 数        │
│    │     token_count = count_tokens(messages) │
│    │                                         │
│    ├─ 2. token_count > threshold ?           │
│    │     ├─ 否 → 跳过（return None）          │
│    │     └─ 是 → 继续                        │
│    │                                         │
│    ├─ 3. 保留最近 N 条消息                    │
│    │     keep = messages[-keep_recent:]       │
│    │                                         │
│    ├─ 4. 旧消息 → 调用摘要 LLM 生成摘要        │
│    │     summary = summary_model.invoke(      │
│    │         old_messages                     │
│    │     )                                    │
│    │                                         │
│    ├─ 5. 替换消息：摘要 + 最近消息              │
│    │     messages = [summary_msg] + keep      │
│    │                                         │
│    └─ 6. 返回新 messages → Agent 继续         │
│                                              │
└──────────────────────────────────────────────┘
  │
  ▼
 END（进入 LLM 调用，但消息已被压缩）
```

**底层机制**：`before_model` 不是拦截调用，而是**在调用前修改 State**。它返回一个 dict 来替换 `messages` 字段，然后 Agent 用新 messages 去调 LLM。

#### 何时使用

- **长对话 Agent**（客服、教育辅导）：对话超过 20 轮后，Token 接近上下文窗口上限
- **多步任务 Agent**（代码生成、数据分析）：中间产生大量工具调用结果，快速膨胀
- **成本敏感场景**：每次 LLM 调用都带完整历史 → Token 浪费 → 需要压缩

#### 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model` | 与主 Agent 相同 | 用于生成摘要的 LLM（可指定更便宜的模型） |
| `trigger_token_limit` | 4000 | 触发压缩的 Token 阈值 |
| `keep_recent` | 20 | 保留最近多少条消息不压缩 |
| `summary_prompt` | 内置模板 | 自定义摘要 Prompt |
| `fallback_message_count` | 15 | 摘要模型也失败时，最多保留的消息数 |

#### 使用 vs 不使用的效果对比

```
场景：用户与客服 Agent 连续对话 50 轮，每轮都有工具调用

不使用 SummarizationMiddleware（50 轮后）：
┌─────────────────────────────────────────────────────┐
│ Messages: [System, Human1, AI1, Tool1, ..., Human50]│
│ Token 数: ~18,000                                    │
│ 效果: LLM 上下文接近极限，响应变慢，可能截断           │
│ 成本: 每次调用消耗 ~18K input tokens                  │
└─────────────────────────────────────────────────────┘

使用 SummarizationMiddleware（50 轮后）：
┌─────────────────────────────────────────────────────┐
│ Messages: [System, Summary(前 30 轮摘要), Human31,   │
│           AI31, ..., Human50]                        │
│ Token 数: ~6,000                                     │
│ 效果: LLM 拥有完整上下文（摘要保留了关键信息）         │
│ 成本: 每次调用消耗 ~6K input tokens（减少 67%）       │
└─────────────────────────────────────────────────────┘
```

#### 完整示例

```python
# ================================================================
# 01_summarization_demo.py — 上下文压缩中间件演示
# ================================================================
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_openai import ChatOpenAI

# ---- 1. 创建 Agent ----
# 主模型（用于回答用户问题）
main_llm = ChatOpenAI(
    model="deepseek-v4-pro",
    temperature=0.7,
)

# 摘要模型（用于压缩历史，可指定更便宜的模型降低成本）
# 如果不指定，默认复用 main_llm
summary_llm = ChatOpenAI(
    model="deepseek-v4-pro",    # 可以换成更便宜的模型
    temperature=0.0,             # 摘要不需要创意，温度=0
    max_tokens=1024,             # 摘要输出有限
)

agent = create_agent(
    model=main_llm,
    tools=[search_web, get_weather, calculate],
    middleware=[
        SummarizationMiddleware(
            model=summary_llm,            # 用什么模型做摘要
            trigger_token_limit=4000,     # Token 超过 4000 就触发压缩
            keep_recent=20,               # 保留最近 20 条消息不压缩
            # 内置摘要 Prompt 会让模型提取：
            #   SESSION INTENT（会话目标）
            #   SUMMARY（关键决策和结论）
            #   ARTIFACTS（创建/修改的文件）
            #   NEXT STEPS（待完成任务）
        ),
    ],
    system_prompt="你是智能客服助手。用户可能进行长时间对话，你的记忆会被自动压缩。",
)

# ---- 2. 对比演示 ----
import tiktoken

# 模拟长对话：用户连续提问 30 轮
# 每轮返回大量工具数据（模拟膨胀）
config = {"configurable": {"thread_id": "long_chat_demo"}}

for round_num in range(1, 31):
    user_msg = f"第 {round_num} 轮：帮我查一下关于 Python 3.{min(round_num, 12)} 的新特性"
    
    result = agent.invoke(
        {"messages": [HumanMessage(user_msg)]},
        config=config,
    )
    
    # 监控消息数量和 token 估算
    state = agent.get_state(config)
    if state and state.values:
        msg_count = len(state.values["messages"])
        # 粗略估算 token（1 token ≈ 4 字符）
        total_chars = sum(
            len(m.content or "") 
            for m in state.values["messages"] 
            if hasattr(m, "content") and m.content
        )
        est_tokens = total_chars // 4
        
        print(f"  第 {round_num} 轮后：{msg_count} 条消息，≈{est_tokens} tokens")
        
        # 当消息数突然减少时 → 压缩发生了！
        if round_num > 1 and msg_count < prev_msg_count:
            print(f"  ★ 第 {round_num} 轮触发了压缩！")
            print(f"    消息数从 {prev_msg_count} → {msg_count}")
        
        prev_msg_count = msg_count

# 验证：即使经过 30 轮对话，Agent 仍然能正常工作
final_result = agent.invoke(
    {"messages": [HumanMessage("总结一下我们这 30 轮讨论的主要内容")]},
    config=config,
)
print(f"\n最终回答: {final_result['messages'][-1].content[:300]}...")
```

---

### 2.2 PII 信息脱敏中间件 — `PIIMiddleware`

#### 工作原理

```
START
  │
  ▼
┌──────────────────────────────────────────────────────┐
│              PIIMiddleware                           │
│                                                      │
│  before_model(state, runtime)                        │
│    │                                                 │
│    ├─ 1. 遍历最新的 HumanMessage 和 AIMessage         │
│    │     (SystemMessage 通常不含 PII，跳过)            │
│    │                                                 │
│    ├─ 2. 对每条消息执行 PII 检测                      │
│    │     ├─ detect_email(text)      → 匹配 email     │
│    │     ├─ detect_credit_card(text) → Luhn 算法校验  │
│    │     ├─ detect_ip(text)         → IP 格式校验     │
│    │     ├─ detect_mac_address(text)→ MAC 格式       │
│    │     └─ detect_url(text)        → URL 格式       │
│    │                                                 │
│    ├─ 3. 根据配置的策略处理每个匹配                     │
│    │     ├─ strategy="redact" → 替换为 [REDACTED]     │
│    │     ├─ strategy="mask"   → 部分掩码              │
│    │     ├─ strategy="hash"   → 确定性哈希替换         │
│    │     └─ strategy="block"  → 抛出异常，拒绝处理      │
│    │                                                 │
│    └─ 4. 返回修改后的 messages → LLM 看到的已脱敏      │
│                                                      │
└──────────────────────────────────────────────────────┘
  │
  ▼
 END（LLM 收到的消息中 PII 已被处理）
```

#### 支持的模式类型

| PII 类型 | 检测方式 | 示例 | 策略 |
|---|---|---|---|
| `email` | 正则匹配 | `alice@company.com` | `redact/mask/hash/block` |
| `credit_card` | 正则 + Luhn 算法校验 | `4111-1111-1111-1111` | `redact/mask/hash/block` |
| `ip` | stdlib `ipaddress` 模块校验 | `192.168.1.1` | `redact/mask/hash/block` |
| `mac_address` | 正则匹配 | `00:1A:2B:3C:4D:5E` | `redact/mask/hash/block` |
| `url` | 正则匹配 | `https://internal.company.com` | `redact/mask/hash/block` |

#### 四种处理策略

| 策略 | 输入 | 输出 | 能否识别同一人 | 适用 |
|---|---|---|---|---|
| `block` | `alice@acme.com` | `PIIDetectionError` 异常 | — | 完全禁止 PII 进入 LLM |
| `redact` | `alice@acme.com` | `[REDACTED_EMAIL]` | 否 | 合规场景、日志脱敏 |
| `mask` | `4111-1111-1111-1234` | `****-****-****-1234` | 否 | 客服 UI、人工可读 |
| `hash` | `alice@acme.com` | `<email_hash:a1b2c3d4>` | 是 | 分析、调试、去重 |

#### 使用 vs 不使用的效果对比

```
用户输入: "我的邮箱是 alice@company.com，信用卡 4111-1111-1111-1234"

不使用 PIIMiddleware:
┌─────────────────────────────────────────────────┐
│ LLM 收到的消息:                                  │
│   "我的邮箱是 alice@company.com，                │
│    信用卡 4111-1111-1111-1234"                   │
│                                                 │
│ 风险:                                           │
│  - PII 进入了第三方 LLM API（数据出境风险）       │
│  - LLM 可能在后续回复中复述这些 PII               │
│  - 日志中记录了明文 PII（合规风险）               │
└─────────────────────────────────────────────────┘

使用 PIIMiddleware(strategy="mask"):
┌─────────────────────────────────────────────────┐
│ LLM 收到的消息:                                  │
│   "我的邮箱是 a****@company.com，                │
│    信用卡 ****-****-****-1234"                   │
│                                                 │
│ 效果:                                           │
│  - 敏感信息从未离开你的服务器                     │
│  - LLM 看不到完整 PII，无法泄露                   │
│  - 日志中已是脱敏数据                             │
└─────────────────────────────────────────────────┘
```

#### 完整示例

```python
# ================================================================
# 02_pii_demo.py — PII 信息脱敏中间件演示
# ================================================================
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_openai import ChatOpenAI

# ---- 1. 创建带 PII 保护的 Agent ----
# 对不同类型的 PII 使用不同策略
agent = create_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0),
    tools=[customer_lookup, order_query],
    middleware=[
        # email：完全脱敏（替换为 [REDACTED_EMAIL]）
        PIIMiddleware("email", strategy="redact"),
        
        # credit_card：部分掩码（保留后 4 位，客服可人工核对）
        PIIMiddleware("credit_card", strategy="mask"),
        
        # ip：哈希（需要调试时能识别同一 IP 的多次请求）
        PIIMiddleware("ip", strategy="hash"),
        
        # url：脱敏（隐藏内部系统地址）
        PIIMiddleware("url", strategy="redact"),
    ],
    system_prompt="你是客服助手，所有用户信息已经过脱敏处理。",
)

# ---- 2. 演示 ----
config = {"configurable": {"thread_id": "pii_demo"}}

# 用户消息包含多种 PII
safe_result = agent.invoke({
    "messages": [HumanMessage(
        "你好，我的邮箱是 alice.johnson@acmecorp.com，"
        "请帮我查一下订单。我的信用卡后四位是 1234，"
        "完整卡号是 4111-1111-1111-1234。"
        "你可以在 https://internal.acmecorp.com/orders 查到。"
    )],
}, config=config)

# 查看脱敏后的实际消息（在 State 中）
state = agent.get_state(config)
for msg in state.values["messages"]:
    if hasattr(msg, "content") and msg.content:
        content = msg.content
        # 检查是否还有原始 PII
        if "alice.johnson@acmecorp.com" in content:
            print(f"⚠️  未脱敏！消息中仍有原始邮箱")
        elif "4111-1111-1111-1234" in content:
            print(f"⚠️  未脱敏！消息中仍有原始卡号")
        else:
            print(f"✅ 已脱敏。片段：{content[:100]}...")
```

---

### 2.3 模型调用限制中间件 — `ModelCallLimitMiddleware`

#### 工作原理

```
START
  │
  ▼
┌──────────────────────────────────────────────────────┐
│          ModelCallLimitMiddleware                    │
│                                                      │
│  before_model(state, runtime)                        │
│    │                                                 │
│    ├─ 1. 读取计数                                    │
│    │     thread_count = state["thread_model_call_count"]│
│    │     run_count = state["run_model_call_count"]   │
│    │                                                 │
│    ├─ 2. 检查限制                                    │
│    │     ├─ thread_count >= thread_limit ?           │
│    │     │     → 是：触发限制                         │
│    │     ├─ run_count >= run_limit ?                 │
│    │     │     → 是：触发限制                         │
│    │     └─ 否 → 继续（return None）                  │
│    │                                                 │
│    ├─ 3. 触发限制时的处理                             │
│    │     ├─ exit_behavior="continue" → 发警告消息     │
│    │     ├─ exit_behavior="end"     → jump_to="end"  │
│    │     └─ exit_behavior="error"   → 抛出异常        │
│    │                                                 │
│    └─ 4. 未触发限制 → 计数+1 → 继续                   │
│                                                      │
└──────────────────────────────────────────────────────┘
  │
  ▼
 END（正常调用 LLM 或直接终止）
```

**两层计数**：

| 层级 | 存储 | 生命周期 | 用途 |
|---|---|---|---|
| `run_model_call_count` | UntrackedValue（不持久化） | 单次 `invoke()` 调用内 | 防止单次请求中的无限循环 |
| `thread_model_call_count` | Checkpoint（持久化） | 整个 thread 生命周期 | 防止单用户消耗过多资源 |

#### 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `run_limit` | `None` | 单次 run 的 LLM 调用上限 |
| `thread_limit` | `None` | 单个 thread 的 LLM 调用总上限 |
| `exit_behavior` | `"continue"` | 超限后行为：`continue`/`end`/`error` |

#### 完整示例

```python
# ================================================================
# 03_model_call_limit_demo.py — 模型调用限制中间件
# ================================================================
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_openai import ChatOpenAI

agent = create_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0),
    tools=[search_web, calculate, get_weather],
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=10,           # 单次 invoke 内最多调 10 次 LLM
            thread_limit=50,        # 整个会话最多调 50 次 LLM
            exit_behavior="end",    # 超限时优雅结束（不发 error）
        ),
    ],
    system_prompt="你是任务执行助手。",
)

config = {"configurable": {"thread_id": "limit_demo"}}

# 正常对话（不会触发限制）
result = agent.invoke(
    {"messages": [HumanMessage("北京天气怎么样？")]},
    config=config,
)
print(f"✅ 正常完成: {result['messages'][-1].content[:100]}")

# 如果用户在同一个 run 中触发了很多次工具调用循环
# 达到 run_limit=10 时，Agent 会自动结束
# 而不是无限循环消耗 API 费用
```

---

## 第三章：第二类 — `wrap_model_call` 中间件

这一类中间件**包裹在 LLM 调用外面**，可以拦截、替换、重试、缓存模型调用结果。

### 3.1 管理上下文大小中间件 — `ContextEditingMiddleware`

#### 工作原理

```
START
  │
  ▼
┌──────────────────────────────────────────────────────┐
│         ContextEditingMiddleware                     │
│                                                      │
│  wrap_model_call(request, handler)                   │
│    │                                                 │
│    ├─ 1. 计算 request.messages 的 token 数            │
│    │     token_count = count_tokens(messages)         │
│    │                                                 │
│    ├─ 2. token_count > trigger_limit ?               │
│    │     ├─ 否 → 直接 handler(request)（不做任何事）   │
│    │     └─ 是 → 继续                                │
│    │                                                 │
│    ├─ 3. 对消息历史执行编辑策略                        │
│    │     ClearToolUsesEdit:                          │
│    │       - 找到旧的 ToolMessage（工具返回值）         │
│    │       - 将内容替换为 "[cleared]"                  │
│    │       - 保留最近 keep_recent 条不清理              │
│    │                                                 │
│    └─ 4. 用清理后消息创建新的 ModelRequest             │
│         → handler(new_request) → 返回                 │
│                                                      │
└──────────────────────────────────────────────────────┘
  │
  ▼
 END
```

**与 SummarizationMiddleware 的区别**：

| | SummarizationMiddleware | ContextEditingMiddleware |
|---|---|---|
| 钩子 | `before_model` | `wrap_model_call` |
| 处理方式 | 生成摘要替换旧消息 | 直接把旧工具结果标记为 `[cleared]` |
| 是否调 LLM | 是（调摘要 LLM） | 否 |
| 信息保留 | 摘要中有语义信息 | 旧工具结果内容丢失 |
| 适用 | 需要保留旧上下文语义 | 旧工具结果不重要，只需释放空间 |

#### 完整示例

```python
# ================================================================
# 04_context_editing_demo.py — 上下文编辑中间件
# ================================================================
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware
from langchain.agents.middleware.context_editing import ClearToolUsesEdit
from langchain_openai import ChatOpenAI

agent = create_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0.7),
    tools=[search_web, file_reader, code_executor],
    middleware=[
        ContextEditingMiddleware(
            # 当消息超过 4000 token 时触发清理
            trigger_token_limit=4000,
            # 清理策略：清除旧的工具返回值（保留函数名，内容变 [cleared]）
            edits=[
                ClearToolUsesEdit(
                    keep_recent=10,    # 最近 10 轮的工具结果不清除
                    placeholder="[cleared]",  # 替换文本
                ),
            ],
        ),
    ],
    system_prompt="你是研究助手。对话可能很长，旧数据会自动清理。",
)
```

---

### 3.2 模型故障自动切换中间件 — `ModelFallbackMiddleware`

#### 工作原理

```
START
  │
  ▼
┌──────────────────────────────────────────────────────┐
│          ModelFallbackMiddleware                      │
│                                                      │
│  wrap_model_call(request, handler)                   │
│    │                                                 │
│    ├─ 1. 尝试用 primary_model 调用 handler            │
│    │     try: return handler(request)                │
│    │                                                 │
│    ├─ 2. primary_model 失败（超时/限流/服务不可用）     │
│    │     → 捕获异常                                   │
│    │                                                 │
│    ├─ 3. 切换到 first_model                          │
│    │     new_request = request.override(             │
│    │         model=first_model                       │
│    │     )                                           │
│    │     try: return handler(new_request)            │
│    │                                                 │
│    ├─ 4. first_model 也失败 → 切换到 additional_models│
│    │     依次尝试直到成功 或 全部失败                    │
│    │                                                 │
│    └─ 5. 全部模型失败 → 抛出最后一个异常               │
│                                                      │
└──────────────────────────────────────────────────────┘
  │
  ▼
 END
```

#### 使用 vs 不使用的效果对比

```
场景：GPT-4 主模型突然限流（429 Too Many Requests）

不使用 ModelFallbackMiddleware:
┌─────────────────────────────────────────────────────┐
│ llm.invoke() → 429 Rate Limit Error → 异常抛出       │
│ Agent 崩溃 → 用户看到错误 → 需要重试整个对话          │
│ 所有对话上下文丢失（未被 Checkpointer 保存到这一步）   │
└─────────────────────────────────────────────────────┘

使用 ModelFallbackMiddleware:
┌─────────────────────────────────────────────────────┐
│ llm.invoke() → 429 Error → 捕获                      │
│ → 自动切换 deepseek-v4-pro → 成功 → 返回结果         │
│ Agent 继续运行 → 用户无感知                           │
│ 对话上下文完整保留                                   │
└─────────────────────────────────────────────────────┘
```

#### 完整示例

```python
# ================================================================
# 05_model_fallback_demo.py — 模型故障自动切换中间件
# ================================================================
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_openai import ChatOpenAI

# 主模型：GPT-4（质量和速度最优，但可能限流或宕机）
primary_model = ChatOpenAI(model="gpt-4", temperature=0.7, max_retries=1)

# 备用模型：按优先级排列
agent = create_agent(
    model=primary_model,  # 主模型
    tools=[search_web, calculate],
    middleware=[
        ModelFallbackMiddleware(
            # 第一降级：DeepSeek（便宜，兼容 OpenAI 协议）
            ChatOpenAI(
                model="deepseek-v4-pro",
                base_url="https://api.deepseek.com",
                temperature=0.7,
                max_retries=1,
            ),
            # 第二降级：GPT-4o-mini（最便宜，最后兜底）
            "openai:gpt-4o-mini",
            # 字符串格式会被 init_chat_model 自动解析
        ),
    ],
    system_prompt="你是智能助手。",
)

# 正常调用 — 用户完全无感知切换
result = agent.invoke({
    "messages": [HumanMessage("解释量子计算的基本原理")]
})
print(result["messages"][-1].content[:200])
```

---

### 3.3 智能工具选择中间件 — `ToolSelectionMiddleware`

#### 工作原理

```
START
  │
  ▼
┌──────────────────────────────────────────────────────┐
│          ToolSelectionMiddleware                      │
│                                                      │
│  wrap_model_call(request, handler)                   │
│    │                                                 │
│    ├─ 1. 提取最后一条 HumanMessage                    │
│    │     (用户当前的问题)                              │
│    │                                                 │
│    ├─ 2. 用一个小 LLM（选择器）判断哪些工具相关         │
│    │     selection_model.invoke(                      │
│    │       system="选择最相关的工具",                  │
│    │       user=last_human_message,                   │
│    │       schema={                                   │
│    │         tool_a: Literal[True],  ← 是否选中       │
│    │         tool_b: Literal[True],                   │
│    │         ...                                      │
│    │       }                                          │
│    │     )                                            │
│    │                                                 │
│    ├─ 3. 过滤：只保留被选中的工具                      │
│    │     selected_tools = [t for t in all_tools       │
│    │                       if selection[t.name]]      │
│    │                                                 │
│    └─ 4. 用选中的工具子集创建 new_request              │
│        → handler(new_request) → 返回                  │
│                                                      │
└──────────────────────────────────────────────────────┘
  │
  ▼
 END
```

**为什么需要**：当 Agent 有 50+ 个工具时，全部传给 LLM → 上下文膨胀 + 选择困难 → 准确率下降。预选 3~5 个最相关的 → LLM 选择范围小 → 准确率高。

#### 使用 vs 不使用的效果对比

```
50 个工具的 Agent，用户问 "今天北京天气怎么样？"

不使用 ToolSelectionMiddleware:
  LLM 收到 50 个工具定义（~2000 tokens）
  包括：天气、搜索、代码执行、邮件、Slack、Jira、数据库、文件操作...
  LLM 需要从 50 个中选 → 可能选错（概率 2% × 50 = 有 63% 的概率至少一个误判）

使用 ToolSelectionMiddleware:
  选择器选出 3 个工具：get_weather、get_aqi、search_web（~150 tokens）
  LLM 只需从 3 个中选 → 准确率 > 99%
  上下文从 2000 tokens → 150 tokens（减少 92%）
```

#### 完整示例

```python
# ================================================================
# 06_tool_selection_demo.py — 智能工具选择中间件
# ================================================================
from langchain.agents import create_agent
from langchain.agents.middleware import ToolSelectionMiddleware
from langchain_openai import ChatOpenAI

# 主模型（功能完整）
main_llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0.7)

# 选择器模型（只做工具选择，可以用小/便宜模型）
selector_llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0)

agent = create_agent(
    model=main_llm,
    tools=[
        # 假设有 20+ 个工具
        get_weather, get_aqi, get_time, get_stock,
        search_web, search_arxiv, search_wikipedia,
        read_file, write_file, delete_file,
        send_email, send_slack, create_jira,
        query_db, execute_python, calculate,
        translate, summarize, generate_image,
    ],
    middleware=[
        ToolSelectionMiddleware(
            model=selector_llm,    # 用什么模型做选择（可便宜）
            max_tools=5,           # 最多选 5 个工具传给主 LLM
            # 选择器内部做的事：
            # 1. 分析用户问题
            # 2. 从 20+ 工具中选出最相关的 ≤5 个
            # 3. 只有这 5 个工具传给主 LLM
        ),
    ],
    system_prompt="你是全能助手。",
)

# 调用时，主 LLM 看到的工具从 20+ → 5
result = agent.invoke({
    "messages": [HumanMessage("今天北京天气怎么样？适合户外跑步吗？")]
})
```

---

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

## 第八章：自定义中间件 — 参数传递机制详解

理解自定义中间件，必须先掌握四类数据对象：**ModelRequest**（调用请求）、**ModelResponse**（调用结果）、**AgentState**（全局账本）、**Command**（流程控制指令）+ 一个核心回调 **handler**。

### 8.1 ModelRequest — 单次 LLM 调用的"请求快照"

#### 总览

| 维度 | 说明 |
|---|---|
| **作用域** | 单次 LLM 调用内（一次性、不可变） |
| **数据内容** | 本次 LLM 调用所需的全部输入：model、messages、tools、state、runtime... |
| **类比** | Java Servlet 的 `HttpServletRequest` —— 封装了本次请求的所有信息 |
| **典型来源** | Agent 在每次调用 LLM 前自动构建，传入 `wrap_model_call` 的 `request` 参数 |
| **修改方式** | `request.override(model=..., messages=...)` → 返回新实例（不可变模式） |

#### 结构示例

```python
# ModelRequest 实例（简化展示）
request = ModelRequest(
    model=<ChatOpenAI model="deepseek-v4-pro" temperature=0.7>,
    messages=[                              # ← 不包含 SystemMessage！
        HumanMessage("帮我规划北京三日游"),
        AIMessage("我来查天气。", tool_calls=[...]),
        ToolMessage("北京：晴，25°C", tool_call_id="c1"),
        AIMessage("再查景点。", tool_calls=[...]),
        ToolMessage("故宫推荐...", tool_call_id="c2"),
    ],
    system_message=SystemMessage("你是旅行规划助手。"),  # ← SystemMessage 单独存放
    tool_choice=None,                       # None="auto", "any", "none", 或指定
    tools=[get_weather, search_web, get_attractions, book_hotel],
    response_format=None,                   # 结构化输出 schema
    state={"messages": [...], "user_id": "alice"},
    runtime=<Runtime context={...}>,
    model_settings={},                      # 传给 API 的额外参数
)
```

#### 重点：`runtime` — 传递上下文信息

`runtime` 是 ModelRequest 中最重要的字段之一。它携带了**框架级上下文信息**，中间件可以通过它访问：

```python
def wrap_model_call(self, request, handler):
    # runtime 是什么？
    #   runtime = 框架持有的"运行时环境"对象
    #   包含 store（BaseStore）、config（RunnableConfig）、
    #   stream_writer、context（用户自定义上下文）等
    
    # 1. 访问全局 Store（BaseStore）
    store = request.runtime.store
    # 可以在 LLM 调用前后读写全局记忆

    # 2. 访问用户自定义上下文
    if request.runtime.context:
        user_id = request.runtime.context.get("user_id")

    # 3. 流式写入器
    writer = request.runtime.stream_writer
    # 可以中途向客户端推送自定义事件
    
    # 4. 当前配置
    config = request.runtime.config
    thread_id = config["configurable"]["thread_id"]
```

### 8.2 ModelResponse — 单次 LLM 调用的"响应结果"

#### 总览

| 维度 | 说明 |
|---|---|
| **作用域** | 单次 LLM 调用内（一次性，封装 LLM 返回） |
| **数据内容** | `result`（消息列表，通常一个 AIMessage）+ `structured_response`（可选结构化输出） |
| **类比** | Java Servlet 的 `HttpServletResponse` —— 封装了本次请求的返回结果 |
| **典型来源** | `handler(request)` 的返回值 |
| **修改方式** | 构造新的 `ModelResponse(result=[...], structured_response=...)` 替换 |

#### 结构示例

```python
# handler(request) 返回的 ModelResponse
response = ModelResponse(
    result=[
        AIMessage(
            content="北京三日游建议：第一天故宫，第二天长城，第三天颐和园。",
            tool_calls=None,  # 不需要再调工具了
            response_metadata={"token_usage": {"prompt_tokens": 300, "completion_tokens": 80}},
        ),
    ],
    structured_response=None,  # 没要求结构化输出
)
```

#### 数据内容详解

```python
# result 字段（核心）
# 类型: list[BaseMessage]
# 内容: LLM 返回的消息（通常一条 AIMessage，可能包含 tool_calls）
response.result       # [AIMessage(content=..., tool_calls=[...])]
response.result[0]    # AIMessage
response.result[0].content           # 文本回复
response.result[0].tool_calls        # 工具调用请求
response.result[0].usage_metadata    # Token 用量

# structured_response 字段（可选）
# 类型: Any（Pydantic BaseModel | dict | None）
# 内容: 当 request.response_format 不为 None 时，LLM 输出的结构化数据
response.structured_response  # WeatherReport(city="北京", temp=25)
```

### 8.3 handler — 调用"下一步"的回调函数

#### 总览

| 维度 | 说明 |
|---|---|
| **类型签名** | `Callable[[ModelRequest], ModelResponse]`（同步）/ `Callable[[ModelRequest], Awaitable[ModelResponse]]`（异步） |
| **核心作用** | 调用它 = 执行"下一个中间件 + 真正的 LLM 调用" |
| **什么时候调用** | 中间件想继续正常流程时调用；不想继续时**不调用**（短路） |
| **调用次数** | 0 次（短路）、1 次（正常）、N 次（重试） |
| **类比** | Java FilterChain 的 `chain.doFilter(request, response)`；AOP 的 `ProceedingJoinPoint.proceed()` |

#### handler 的本质：洋葱的下一层

```python
def wrap_model_call(self, request, handler):
    # handler 不是一个固定的函数指针
    # 它是"下一个中间件的 wrap_model_call + 最终 LLM 调用"的组合
    
    # 调用 handler(request) →
    #   1. 下一个中间件的 wrap_model_call 收到 request
    #   2. 如果还有更多中间件 → 继续传递
    #   3. 最后一个中间件 → handler = 真正的 LLM API 调用
    #   4. LLM 返回 → 结果沿洋葱层反向传回
    #   5. 你拿到最终的 ModelResponse
    
    # 所以：
    #   - 调用 handler 之前 = 在 LLM 调用之前修改 request
    #   - 调用 handler 之后 = 在 LLM 返回之后修改 response
    #   - 不调用 handler = 完全跳过 LLM（短路）
    
    return handler(request)
```

#### 三种 handler 使用模式

```python
# 模式 1：调用前修改 request（类似 before_model，但更灵活）
def wrap_model_call(self, request, handler):
    # 修改 request 后调用 handler
    new_request = request.override(
        messages=request.messages[-10:],              # 只传最近 10 条
        model_settings={"temperature": 0.0},          # 强制低温
    )
    return handler(new_request)                       # 用修改后的请求调 LLM

# 模式 2：调用后修改 response
def wrap_model_call(self, request, handler):
    response = handler(request)                       # 先正常调 LLM
    # 修改 LLM 的回复
    modified_msg = AIMessage(content=f"[已审核] {response.result[0].content}")
    return ModelResponse(result=[modified_msg])       # 返回修改后的

# 模式 3：不调用 handler（短路 / 缓存命中）
def wrap_model_call(self, request, handler):
    last_human_msg = request.messages[-1].content
    if cached := cache.get(last_human_msg):
        # 缓存命中 → 直接返回，不调用 LLM（省 1 次 API 调用！）
        return ModelResponse(result=[AIMessage(content=cached)])
    return handler(request)                           # 缓存未命中 → 正常调用
```

### 8.4 AgentState — 全局状态容器

#### 总览

| 维度 | 说明 |
|---|---|
| **作用域** | **整个 thread 生命周期**（跨多轮对话、跨多次 LLM 调用） |
| **数据内容** | `messages`（消息列表）+ 任意自定义字段（`user_id`, `turn_count`, `summary`...） |
| **类比** | Java Web 的 `HttpSession` —— 会话级别持久化；或数据库的 `accounts` 表 —— 每个 thread 有一本独立"账本" |
| **典型来源** | `request.state`（在 `wrap_model_call` 中）、`state` 参数（在 `before_agent`/`after_model` 等钩子中） |
| **修改方式** | 返回 `dict` → 框架按 reducer 规则合并（`add_messages` 追加、`add` 累加、无注解则覆盖） |

#### 结构示例与字段详解

```python
# AgentState 实例（典型结构）
state = {
    # ---- 核心字段 ----
    "messages": [                     # ← add_messages reducer: 新消息自动追加
        SystemMessage("你是助手。"),
        HumanMessage("北京天气？"),
        AIMessage("查一下...", tool_calls=[...]),
        ToolMessage("晴 25°C"),
        AIMessage("北京晴，25°C"),
    ],
    
    # ---- 持久化自定义字段（由 Checkpointer 保存）----
    "user_id": "alice",               # 会话绑定用户
    "turn_count": 5,                  # add reducer: 每次 new_turn_count = old + new
    "summary": "用户在北京，已查询天气和景点",  # 覆盖式：新值覆盖旧值
    "extracted_entities": [           # 业务数据累积
        {"entity": "故宫", "type": "attraction"},
        {"entity": "Python", "type": "language"},
    ],
    
    # ---- 非持久化字段（用 UntrackedValue/EphemeralValue 标记）----
    "jump_to": "end",                 # EphemeralValue: 用完即弃
}
```

#### State 的"全局账本"本质

```
Thread A (chat_1) ───── 全局账本 A ─────
  │                        │
  ├─ Round 1                │ messages += [Human1, AI1]
  ├─ Round 2                │ messages += [Human2, AI2], turn_count += 1
  └─ Round 3                │ messages += [Human3, AI3], turn_count += 1
                             │
                    Checkpointer 每轮保存
                    
Thread B (chat_2) ───── 全局账本 B ─────
  │                        │
  └─ Round 1                │ messages = [Human1, AI1]
                             │
                    两个账本完全独立、互不可见
                    
BaseStore ───── 跨 Thread 共享 ─────
  namespace=("users", "alice", "profile")
  key="latest"
  → Thread A 和 Thread B 都能读
```

### 8.5 Command — 中间件的"指令控制语言"

#### 总览

| 维度 | 说明 |
|---|---|
| **作用域** | 当前中间件钩子内（返回后由 Agent 框架解释执行） |
| **数据内容** | `update`（State 修改）、`goto`（跳转目标）、`resume`（恢复中断） |
| **类比** | HTTP 重定向 `302 Location: /end`；或 Spring MVC 的 `return "redirect:/home"` |
| **典型来源** | 中间件方法的返回值 |
| **何时用** | 需要改变 Agent 默认流程时——跳转、注入 State、恢复中断 |

#### 四种 Command 使用模式

```python
# 模式 1: goto — 跳转到指定节点
from langgraph.types import Command

def after_model(self, state, runtime):
    if detect_toxicity(state["messages"][-1]):
        return Command(
            goto="end",                    # ← 直接结束 Agent
            update={"messages": [AIMessage("无法处理该请求")]},
        )
    return None  # 正常流程

# 模式 2: goto + Send — 跳转并带入新数据
def before_model(self, state, runtime):
    if needs_retry(state):
        return Command(
            goto=[Send("model", {"messages": state["messages"][:5]})],
        )

# 模式 3: resume — 恢复 Human-in-the-Loop 中断
@app.post("/approve")
async def approve(thread_id: str, decision: str):
    await agent.ainvoke(
        Command(resume={"decision": decision}),
        config={"configurable": {"thread_id": thread_id}},
    )

# 模式 4: update — 只修改 State，不跳转
def before_agent(self, state, runtime):
    return Command(
        update={"turn_count": 1, "user_id": runtime.config["configurable"]["user_id"]},
    )
```

### 8.6 四者对比总结

| | ModelRequest | ModelResponse | AgentState | Command |
|---|---|---|---|---|
| **作用域** | 单次 LLM 调用 | 单次 LLM 调用 | 整个 thread 生命周期 | 当前钩子返回后 |
| **谁创建** | Agent 框架 | handler 返回 | Agent + Checkpointer | 中间件返回 |
| **谁消费** | 中间件 + handler | 中间件 + Agent | 所有中间件 + 所有钩子 | Agent 框架 |
| **生命周期** | 一次性 | 一次性 | 持久化（跨轮次） | 一次性（执行后销毁） |
| **可变性** | 不可变（override 创建新实例） | 只读 | 可读可写 | 只写（创建后交由框架） |
| **核心目的** | 封装"本次调什么" | 封装"本次调出什么" | 维护"整个对话进展到哪" | 控制"下一步去哪" |
| **类比** | HttpServletRequest | HttpServletResponse | HttpSession | response.sendRedirect() |
| **可包含** | model/messages/tools/state/runtime | result/structured_response | messages/自定义字段 | update/resume/goto |

### 8.7 各参数在各钩子中的可用性

```
钩子                    │ Request │ Response │ State │ Command │ handler
───────────────────────┼─────────┼──────────┼───────┼─────────┼────────
before_agent           │    ✗    │    ✗     │  ✓ rw │    ✓    │   ✗
before_model           │    ✗    │    ✗     │  ✓ rw │    ✓    │   ✗
wrap_model_call        │    ✓ r  │    ✓ w   │  ✓ r  │    ✗*   │   ✓
after_model            │    ✗    │    ✗     │  ✓ rw │    ✓    │   ✗
wrap_tool_call         │    ✓ r  │    ✓ w   │  ✓ r  │    ✓    │   ✓
after_agent            │    ✗    │    ✗     │  ✓ rw │    ✓    │   ✗
dynamic_prompt         │    ✓ r  │    ✗     │  ✗    │    ✗    │   ✗

✓ = 可用, ✗ = 不可用, r = 只读, rw = 读写, w = 写
✗* = wrap_model_call 不能直接返回 Command，但可以通过 ExtendedModelResponse 携带
```

### 8.8 哪些中间件用哪种参数

| 内置中间件 | 主要钩子 | 操作的核心参数 |
|---|---|---|
| `SummarizationMiddleware` | `before_model` | **State**（读 messages → 压缩 → 写回） |
| `PIIMiddleware` | `before_model` | **State**（读 messages → 脱敏 → 写回） |
| `ModelCallLimitMiddleware` | `before_model` | **State**（读计数）+ **Command**（超限跳转） |
| `ModelRetryMiddleware` | `wrap_model_call` | **handler**（多次调用）+ **Request**（修改） |
| `ModelFallbackMiddleware` | `wrap_model_call` | **handler**（多次调用）+ **Request**（换 model） |
| `ToolSelectionMiddleware` | `wrap_model_call` | **Request**（修改 tools 列表） |
| `ContextEditingMiddleware` | `wrap_model_call` | **Request**（修改 messages） |
| `ToolRetryMiddleware` | `wrap_tool_call` | **handler**（多次调用） |
| `HumanInTheLoopMiddleware` | `after_model` | **Command**（resume 恢复中断） |
| `ToolCallLimitMiddleware` | `after_model` | **State**（读计数）+ **Command**（超限跳转） |

### 8.9 参数关系图解

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AgentState（全局账本）                             │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ messages: [SystemMsg, HumanMsg, AIMsg, ToolMsg, AIMsg, ...]  │ │
│  │ user_id: "alice"                                              │ │
│  │ turn_count: 5          ← add reducer（新+旧）                  │ │
│  │ summary: "用户在北京..."  ← 覆盖式 reducer                      │ │
│  └───────────────────────────────────────────────────────────────┘ │
│      ▲                        ▲                        │           │
│      │ 写入                   │ 读取                    │ 写入       │
│      │                        │                        ▼           │
│  before_model             wrap_model_call          after_model     │
│  (压缩/脱敏)              (读 state 做缓存判断)     (更新计数器)     │
└─────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ 单次模型调用
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│              单次模型调用内部（wrap_model_call）                       │
│                                                                     │
│  ┌──────────┐     ┌──────────┐     ┌──────────────┐                │
│  │ Model    │────→│ handler  │────→│ Model        │                │
│  │ Request  │     │ 调用1次   │     │ Response     │                │
│  │          │     │          │     │              │                │
│  │ .model   │     │ 调用N次   │     │ .result      │                │
│  │ .msgs    │     │ = 重试    │     │ .structured_ │                │
│  │ .tools   │     │          │     │   response   │                │
│  │ .state───┼───→ │ 调用0次   │     │              │                │
│  │ .runtime─┼───→ │ = 短路    │     │              │                │
│  │          │     │          │     │              │                │
│  └──────────┘     └──────────┘     └──────────────┘                │
│       │                │                   │                       │
│       │   修改后传     │                   │                       │
│       │   override()   │                   │                       │
│       └────────────────┘                   │                       │
│                                            │                       │
│                        ┌───────────────────┘                       │
│                        ▼                                           │
│                 ┌──────────┐                                       │
│                 │ Command  │  ← 中间件返回（控制下一步）              │
│                 │          │                                       │
│                 │ .goto    │  → "end" / "model" / "tools"         │
│                 │ .update  │  → 修改 AgentState                    │
│                 │ .resume  │  → 恢复 HumanInTheLoop 中断           │
│                 └──────────┘                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

图解说明：

1. AgentState = 全局账本
   - 贯穿整个 thread 生命周期，由 Checkpointer 持久化
   - 每个钩子都可以读写它：before_model 写（压缩消息）、wrap_model_call 读（缓存判断）、after_model 写（更新计数器）
   
2. 单次模型调用内部（wrap_model_call）是一条"请求→处理→响应"的流水线：
   - ModelRequest：封装本次调用的全部输入（model、messages、tools、state、runtime）
     中间件可以在调用 handler 之前用 request.override() 修改（换模型、裁剪消息、过滤工具）
   - handler：调用它 = 执行下一层中间件 + 最终 LLM 调用
     调用 1 次 = 正常；调用 N 次 = 重试；调用 0 次 = 短路（缓存/拦截）
   - ModelResponse：handler 返回的 LLM 结果
     中间件可以修改 response 再返回（审核内容、追加元信息）
   
3. Command = 中间件的"返回指令"：
   - 在 wrap_model_call 中不能直接返回 Command，但可以通过 ExtendedModelResponse 携带
   - 在 before_model/after_model/before_agent 中可以直接返回 Command
   - Command.goto 改变流程走向（跳到 end/model/tools）
   - Command.update 直接修改 AgentState
   - Command.resume 恢复 HumanInTheLoop 暂停
   
4. 关键关系：
   ModelRequest.state 和 AgentState 指向同一个对象，
   所以 wrap_model_call 中 request.state["messages"] 看到的
   和 before_model 中 state["messages"] 看到的是同一份数据。
   区别在于：before_model 可以直接"写"（返回 dict → 合并到 State），
   wrap_model_call 中是通过 request.state 只读访问。
```

### 8.10 自定义中间件完整示例

结合以上四种参数，写一个"LLM 调用缓存中间件"：

```python
# ================================================================
# custom_cache_middleware.py — 自定义缓存中间件
# 演示：ModelRequest(读) + handler(调用/跳过) + AgentState(缓存存储) + Command(不适用)
# ================================================================
import hashlib
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

class CacheMiddleware(AgentMiddleware):
    """
    LLM 调用缓存中间件。

    解决问题：
      相同/相似的问题反复问 → 每次都调 LLM → 浪费 Token 和延迟。
      这个中间件缓存 LLM 的响应，命中时跳过 LLM 调用。

    参数传递：
      - ModelRequest: 从 request.messages 提取最后一条 HumanMessage 做缓存 key
      - handler: 缓存未命中时调用（正常调用 LLM）；命中时跳过（0 次调用）
      - AgentState: 不直接使用。缓存存储在 request.runtime.store 中（全局 BaseStore）
      - Command: 不适用（wrap_model_call 中不走 Command，只替换 ModelResponse）
    """

    def __init__(self, cache_ttl: int = 3600):
        """
        Args:
            cache_ttl: 缓存有效期（秒），默认 1 小时
        """
        self.cache_ttl = cache_ttl  # 缓存有效期

    def _make_cache_key(self, messages: list) -> str:
        """
        从消息列表生成缓存 Key。

        策略：取最后一条 HumanMessage 的内容做 MD5 哈希。
        MD5 保证相同问题 → 相同 Key → 缓存命中。
        """
        last_msg = messages[-1].content if messages else ""
        key = hashlib.md5(last_msg.encode()).hexdigest()
        return f"cache:{key}"

    def wrap_model_call(self, request: ModelRequest, handler):
        """
        包裹模型调用 — 缓存命中时短路。

        request: 包含 model、messages、tools、state、runtime（都可以读）
        handler: 调用它 = 真正执行 LLM。不调用 = 短路。
        
        流程:
          1. 从 request.messages 生成缓存 Key
          2. 从 request.runtime.store（BaseStore）查缓存
          3. 命中 → 直接返回缓存的 response（不调用 handler！省 1 次 LLM 调用）
          4. 未命中 → 调用 handler → 存入缓存 → 返回
        """
        # === 读取 request 的信息 ===
        # 从 ModelRequest 中提取缓存 Key（最后一条 HumanMessage 的内容）
        cache_key = self._make_cache_key(request.messages)

        # === 从 runtime.store 查缓存 ===
        # runtime.store 是 BaseStore（跨线程的全局存储）
        # 用 store 而不是 AgentState 存缓存，因为 AgentState 只跟 thread
        # 而缓存应该跨 thread 共享（不同线程的相同问题都命中）
        if request.runtime and hasattr(request.runtime, 'store') and request.runtime.store:
            cached_item = request.runtime.store.get(
                namespace=("cache", "llm_responses"),
                key=cache_key,
            )

            # 检查缓存是否过期
            if cached_item:
                import time
                cached_time = cached_item.value.get("timestamp", 0)
                if time.time() - cached_time < self.cache_ttl:
                    # ★ 缓存命中 → 返回缓存的 response，不调用 handler ★
                    print(f"  ⚡ 缓存命中：{request.messages[-1].content[:50]}...")
                    return ModelResponse(
                        result=[
                            AIMessage(content=cached_item.value["response"])
                        ],
                    )

        # === 缓存未命中 → 调用 handler（执行真正的 LLM）===
        print(f"  💰 LLM 调用：{request.messages[-1].content[:50]}...")
        response = handler(request)  # ★ 这里实际调用 LLM API ★

        # === 存入缓存 ===
        if request.runtime and hasattr(request.runtime, 'store') and request.runtime.store:
            request.runtime.store.put(
                namespace=("cache", "llm_responses"),
                key=cache_key,
                value={
                    "response": response.result[0].content,
                    "timestamp": time.time(),
                    "question": request.messages[-1].content[:100],
                },
                # 不设 ttl，用 self.cache_ttl 在读取时检查
            )

        return response
```

---

## 第九章：自定义中间件实战

### 9.1 装饰器方式：动态提示词切换中间件

#### 场景

同一个 Agent 面对不同角色的用户，需要切换 System Prompt 风格：

- **气象专家** → 专业术语、数据分析、不解释基础概念
- **旅游爱好者** → 通俗语言、景点推荐、穿衣/出行建议
- **小白用户** → 极度简单、避免术语、每步都解释

#### 设计方案

利用 `before_model` 钩子 + 函数装饰器 `@before_model`：
- 从 State 中读取 `user_role`（由配置注入）
- 根据角色动态生成对应风格的 SystemMessage
- 将新 SystemMessage 覆盖到 State 中

#### 完整代码

```python
# ================================================================
# 01_dynamic_prompt_middleware.py — 动态提示词切换中间件
# ================================================================
from langchain.agents.middleware import before_model
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import SystemMessage

# ---- 1. 角色 → Prompt 映射表 ----
# 不同角色对应的 System Prompt 模板
ROLE_PROMPTS = {
    "weather_expert": (
        "你是一位**资深气象学家**，拥有 20 年天气预报经验。\n\n"
        "## 你的风格\n"
        "- 使用专业气象术语（等压线、锋面、气旋、对流层等）\n"
        "- 引用数值天气预报模型（ECMWF、GFS）的分析结果\n"
        "- 给出具体的气象数据（温度范围、降水量、风速风向、相对湿度）\n"
        "- 不要解释基础气象概念，假设用户有气象学背景\n"
        "- 如果不确定，明确标注「基于当前模型的概率预报」\n\n"
        "## 你的输出格式\n"
        "1. 天气概况（一句话）\n"
        "2. 详细气象分析\n"
        "3. 数据总结（温度/降水/风力表格）\n"
        "4. 预报可信度评估"
    ),
    "traveler": (
        "你是一位**热情的旅行规划师**，游历过 50+ 个国家。\n\n"
        "## 你的风格\n"
        "- 用通俗生动的语言，像朋友聊天一样\n"
        "- 着重推荐景点、美食、交通建议、拍照打卡点\n"
        "- 根据天气给出穿衣和出行建议\n"
        "- 提醒注意事项（防晒、雨具、旺季拥挤等）\n"
        "- 适当加入本地人的小贴士\n\n"
        "## 你的输出格式\n"
        "1. 一句话亮点推荐\n"
        "2. 行程建议（上午/下午/晚上）\n"
        "3. 穿衣/装备提醒\n"
        "4. 美食推荐"
    ),
    "beginner": (
        "你是一位**耐心的科普老师**，像教小学生一样解释问题。\n\n"
        "## 你的风格\n"
        "- 用最简单的词，就像在跟 10 岁小朋友说话\n"
        "- 每个专业词汇都要用括号解释（比如「锋面（冷空气和热空气相遇的地方）」）\n"
        "- 多用比喻和生活中的例子\n"
        "- 不要一次说太多，每次最多 3 个要点\n"
        "- 主动问「这样解释清楚了吗？需要我再解释哪个部分吗？」\n\n"
        "## 你的输出格式\n"
        "1. 用生活化比喻一句话回答\n"
        "2. 具体解释（带括号注解）\n"
        "3. 「记住这个就够了」的核心要点"
    ),
}

# 默认角色 Prompt
DEFAULT_SYSTEM_PROMPT = "你是智能助手，根据用户需求提供帮助。"


# ---- 2. @before_model 装饰器：把普通函数变成中间件 ----
@before_model
def dynamic_prompt_middleware(
    state: AgentState,
    runtime,  # ← 框架自动注入，包含 config
) -> dict | None:
    """
    动态提示词中间件。

    这是一个 @before_model 装饰器创建的轻量中间件。
    等价于继承 AgentMiddleware 并重写 before_model()，
    但代码量少得多（一个函数 vs 一个类）。

    执行时机：每次 LLM 调用之前
    执行逻辑：
      1. 从 runtime.config 读取当前用户的角色
      2. 根据角色选择对应的 System Prompt
      3. 生成 SystemMessage → 注入 State（替换旧的 SystemMessage）
      4. 返回 dict → Agent 框架自动合并到 State

    参数说明：
      state: 当前 AgentState（含 messages、user_id 等）
      runtime: 运行时上下文（含 config、store 等）
    """
    # === 步骤 1：从运行时配置中获取用户角色 ===
    # runtime.config 是 RunnableConfig 对象
    # configurable 字典在 agent.invoke(config={"configurable": {...}}) 时传入
    user_role = runtime.config.get("configurable", {}).get("user_role", "beginner")

    # === 步骤 2：根据角色选择 System Prompt ===
    prompt_text = ROLE_PROMPTS.get(user_role, DEFAULT_SYSTEM_PROMPT)

    # === 步骤 3：检查是否真的需要切换 ===
    # 如果当前 messages 的第一条已经是同样的 SystemMessage → 跳过
    current_messages = state.get("messages", [])
    if current_messages and isinstance(current_messages[0], SystemMessage):
        if current_messages[0].content == prompt_text:
            return None  # 没有变化，不更新 State（避免无效写入）

    # === 步骤 4：返回 State 更新 ===
    # 返回 dict → Agent 框架用 add_messages reducer 合并
    # SystemMessage 是新增的 → 会被放到消息列表最前面
    # 旧的 SystemMessage 需要手动删除（这里用覆盖方式：告诉框架替换）
    print(f"  🎭 切换角色：{user_role} → Prompt 长度：{len(prompt_text)} 字符")
    return {
        "messages": [SystemMessage(content=prompt_text)],
    }


# ---- 3. 使用 ----
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

agent = create_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0.7),
    tools=[get_weather, search_web],
    middleware=[dynamic_prompt_middleware],  # ← 函数直接作为中间件！
    system_prompt=DEFAULT_SYSTEM_PROMPT,     # 初始值，会被中间件覆盖
)

# === 测试三种角色 ===
test_questions = [
    ("weather_expert", "北京明天天气怎么样？"),
    ("traveler", "北京明天天气怎么样？"),
    ("beginner", "北京明天天气怎么样？"),
]

for role, question in test_questions:
    config = {
        "configurable": {
            "thread_id": f"role_test_{role}",
            "user_role": role,           # ★ 角色从配置注入 ★
        }
    }
    result = agent.invoke(
        {"messages": [HumanMessage(question)]},
        config=config,
    )
    
    print(f"\n{'='*60}")
    print(f"👤 角色：{role}")
    print(f"❓ 问题：{question}")
    print(f"🤖 回答：{result['messages'][-1].content[:300]}...")
    print(f"{'='*60}")
```

#### 执行效果对比

```
同一个问题 "北京明天天气怎么样？"，不同角色的回答：

👤 weather_expert（气象专家）：
  "根据 ECMWF 00Z 初始场预报，北京地区明日受西风槽东移影响，
   850hPa 温度场显示有弱冷空气渗透。地面气压场分析，气压梯度较大，
   预计风力 3-4 级，阵风 5-6 级。降水概率 15%，主要集中在对流层中层...
   | 要素 | 预报值 |
   | 温度 | 22°C ~ 28°C |
   | 降水 | <0.5mm |
   | 相对湿度 | 45%~65% |"

👤 traveler（旅游爱好者）：
  "哇，明天北京天气超棒！☀️
   上午：趁凉快去爬长城，带件薄外套就够，记得涂防晒！
   中午：后海附近找个小馆子吃炸酱面，本地人都去那～
   下午：颐和园划船，阳光下的昆明湖太美了！
   小贴士：早晚有温差，带件薄外套准没错👌"

👤 beginner（小白用户）：
  "明天北京是晴天哦！太阳公公会出来（这就是我们说的「晴天」），
   气温 22 到 28 度。什么意思呢？就是早上有点凉（像冰箱冷藏室），
   中午会很暖和（像春天的阳光照在身上）。建议穿一件短袖，再带一件
   薄外套，这样就不会冷也不会热啦～这样解释清楚了吗？😊"
```

---

### 9.2 装饰器方式：动态模型切换中间件

#### 场景

用户的问题复杂度不同：

- "你好" → 不需要大模型回答，用便宜模型即可
- "帮我分析这份财报的风险因素" → 需要强推理能力，用高端模型

**目标**：简单问题用便宜模型（省钱），复杂问题用高端模型（保证质量）。

#### 设计方案

利用 `wrap_model_call` 钩子 + 关键词分析器：
- 在调用 LLM 之前拦截 `ModelRequest`
- 分析最后一条 HumanMessage 的内容
- 简单 → 用 `request.override(model=cheap_model)`
- 复杂 → 保持原 model（不做 override）

#### 完整代码

```python
# ================================================================
# 02_dynamic_model_switch_middleware.py — 动态模型切换中间件
# ================================================================
from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

# ---- 1. 模型池 ----
# 高性能模型（贵）：用于复杂推理任务
premium_model = ChatOpenAI(
    model="deepseek-v4-pro",
    temperature=0.3,       # 推理用低温
    max_tokens=4096,
)

# 经济型模型（便宜）：用于简单对话
budget_model = ChatOpenAI(
    model="deepseek-v4-pro",  # 实际可换 gpt-3.5-turbo / 本地模型
    temperature=0.7,
    max_tokens=512,
)


# ---- 2. 复杂度判断器 ----
# 关键词策略：包含以下关键词 → 判定为需要高级模型
COMPLEX_KEYWORDS = [
    # 任务复杂度信号
    "分析", "评估", "对比", "总结", "解释为什么", "详细说明",
    "写"，"生成", "翻译", "优化", "重构", "设计",
    # 领域专业信号
    "代码", "算法", "架构", "财报", "法律", "数学", "统计",
    "风险", "安全", "合规", "性能", "投资", "决策",
    # 长度信号（用户愿意打长问题 = 复杂任务）
]

def is_complex_question(text: str) -> bool:
    """
    判断问题是否复杂。

    判断策略（多维度加权）：
      1. 关键词匹配：命中 ≥ 2 个复杂关键词 → 可能是复杂任务
      2. 问题长度：> 50 字符 → 更可能是复杂问题
      3. 问句类型：以"为什么/如何/请分析/请解释"开头 → 推理型问题

    返回值：
      True  → 用 premium_model（贵但强）
      False → 用 budget_model（便宜）
    """
    text_lower = text.lower()
    
    # 维度 1：关键词匹配
    keyword_hits = sum(1 for kw in COMPLEX_KEYWORDS if kw in text_lower)
    
    # 维度 2：问题长度
    is_long = len(text) > 50
    
    # 维度 3：推理型问句
    reasoning_starters = ["为什么", "如何", "请分析", "请解释", "请评估", 
                          "how", "why", "explain", "analyze"]
    is_reasoning = any(text_lower.startswith(s) for s in reasoning_starters)
    
    # 综合判断：满足任意 2 个条件 → 复杂
    score = sum([keyword_hits >= 2, is_long, is_reasoning])
    return score >= 2


# ---- 3. @wrap_model_call 装饰器创建中间件 ----
@wrap_model_call
def dynamic_model_switch(
    request: ModelRequest,       # ← 本次 LLM 调用的完整请求
    handler,                     # ← 调用它 = 执行 LLM（可多次调用）
) -> ModelResponse | AIMessage:
    """
    动态模型切换中间件。

    执行时机：每次 LLM 调用之前（包裹调用）
    执行逻辑：
      1. 从 request.messages 中提取最后一条 HumanMessage
      2. 用 is_complex_question() 判断复杂度
      3. 简单 → request.override(model=budget_model) 后调 handler
      4. 复杂 → 不修改 request，直接调 handler（用原始 premium_model）

    参数说明：
      request: ModelRequest，包含 model/messages/tools/state/runtime
              可以用 request.override(model=...) 创建修改后的副本
      handler: 调用它 = 执行真正的 LLM API 调用
    """
    # === 步骤 1：提取用户最后一条消息 ===
    # request.messages 不包含 SystemMessage（SystemMessage 单独存放）
    user_messages = [
        msg for msg in request.messages
        if hasattr(msg, "type") and msg.type == "human"
    ]
    if not user_messages:
        # 没有 HumanMessage → 无法判断 → 用 premium model（安全）
        return handler(request)
    
    last_user_msg = user_messages[-1].content or ""

    # === 步骤 2：判断复杂度 ===
    if is_complex_question(last_user_msg):
        # 复杂问题 → 保持原 model（premium_model）
        print(f"  🧠 复杂问题 → 使用 premium 模型")
        print(f"     问题: {last_user_msg[:80]}...")
        return handler(request)  # request.model 不变 = premium
    
    # === 步骤 3：简单问题 → 切换到预算模型 ===
    print(f"  💰 简单问题 → 切换到 budget 模型")
    print(f"     问题: {last_user_msg[:80]}...")
    
    # request.override() 创建新请求（不可变模式）
    # override 只改 model 字段，其他字段（messages/tools/state）保持不变
    budget_request = request.override(model=budget_model)
    return handler(budget_request)  # 用预算模型执行


# ---- 4. 使用 ----
agent = create_agent(
    model=premium_model,       # 默认用 premium（兜底）
    tools=[get_weather, search_web],
    middleware=[dynamic_model_switch],
    system_prompt="你是智能助手。",
)

# 测试：同一个 Agent，不同复杂度的问题用不同模型
test_cases = [
    "你好",                                               # → budget
    "帮我详细分析一下 Python 和 Go 在微服务架构中的优劣势对比，包括性能、生态、学习曲线",  # → premium
    "今天天气怎么样",                                        # → budget
    "解释量子计算的核心原理并从信息论角度分析其安全性"            # → premium
]

for question in test_cases:
    config = {"configurable": {"thread_id": f"model_test_{hash(question)}"}}
    result = agent.invoke(
        {"messages": [HumanMessage(question)]},
        config=config,
    )
    print(f"  回答: {result['messages'][-1].content[:100]}...\n")
```

---

### 9.3 继承方式：会话持久化中间件

#### 场景

Agent 默认通过 Checkpointer 持久化 State。但如果你想在每次对话前后做**额外的持久化操作**（如写审计日志、同步用户档案到外部系统），就需要自定义持久化中间件。

#### 设计方案

继承 `AgentMiddleware`，利用 `before_agent` 和 `after_agent` 两个钩子：
- `before_agent`：从 BaseStore 恢复用户上下文 → 注入 State
- `after_agent`：从 State 提取关键信息 → 写入 BaseStore + 审计日志

#### 完整代码

```python
# ================================================================
# 03_session_persistence_middleware.py — 会话持久化中间件（继承方式）
# ================================================================
import json
from datetime import datetime, timezone
from typing import Annotated
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from langgraph.runtime import Runtime

class SessionPersistenceMiddleware(AgentMiddleware):
    """
    会话持久化中间件（继承 AgentMiddleware 方式）。

    与装饰器方式的区别：
      - 装饰器：一个函数 = 一个钩子，适合简单逻辑
      - 继承：一个类可以实现多个钩子（before_agent + after_agent），
              可以有实例变量、辅助方法、内部状态

    这个中间件做的事：
      before_agent:  从 BaseStore 恢复用户上下文 → 注入到 State
      after_agent:   从 State 提取关键信息 → 写入 BaseStore + 审计日志

    类比 Java：
      继承 AgentMiddleware ≈ 继承 OncePerRequestFilter 并实现 preHandle + postHandle
      装饰器 @before_model ≈ 用 @Aspect @Before 注解单个方法
    """

    def __init__(self, audit_file: str = "./audit_log.jsonl"):
        """
        Args:
            audit_file: 审计日志文件路径（生产环境换成数据库）
        """
        self.audit_file = audit_file
        self._start_time = None  # 实例变量：记录会话开始时间（用于计算耗时）

    # ================================================================
    # 钩子 1：Agent 启动前 → 恢复用户上下文
    # ================================================================
    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict | None:
        """
        在 Agent 开始处理之前执行。

        做的事：
          1. 记录会话开始时间（用于 after_agent 计算耗时）
          2. 从 runtime.store（BaseStore）读取用户的持久化档案
          3. 将用户档案注入到 State（后续所有钩子都能读到）

        参数传递：
          state:    全局账本，可以读写。返回 dict 即修改。
          runtime:  运行时上下文。runtime.store 是 BaseStore（跨线程存储），
                    runtime.config 包含 thread_id、user_id 等配置。
        """
        # 记录开始时间（存到实例变量，after_agent 读取）
        self._start_time = datetime.now(timezone.utc)

        # 从 config 中获取身份标识
        config = runtime.config.get("configurable", {})
        user_id = config.get("user_id", "anonymous")
        thread_id = config.get("thread_id", "unknown")

        # === 从 BaseStore 恢复用户上下文 ===
        # 如果有之前存储的档案 → 注入到 State
        user_context = {}
        if hasattr(runtime, 'store') and runtime.store:
            item = runtime.store.get(
                namespace=("users", user_id, "context"),
                key="latest",
            )
            if item:
                user_context = item.value

        # === 写入审计日志：会话开始 ===
        self._write_audit({
            "event": "session_start",
            "user_id": user_id,
            "thread_id": thread_id,
            "timestamp": self._start_time.isoformat(),
            "restored_context": user_context,
        })

        print(f"  📂 会话开始：user={user_id}, thread={thread_id}")
        if user_context:
            print(f"     ↳ 恢复了用户上下文：{json.dumps(user_context, ensure_ascii=False)[:100]}")

        # 返回 State 更新 → 将用户上下文注入到 State
        return {
            "user_id": user_id,
            "user_context": user_context,
            "session_started_at": self._start_time.isoformat(),
        }

    # ================================================================
    # 钩子 2：Agent 结束后 → 持久化 + 审计
    # ================================================================
    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict | None:
        """
        在 Agent 完成处理之后执行。

        做的事：
          1. 从 State 中提取本轮对话的关键信息
          2. 更新 BaseStore 中的用户持久化档案
          3. 写入审计日志：会话结束 + 统计信息
        """
        end_time = datetime.now(timezone.utc)
        elapsed = (end_time - self._start_time).total_seconds() if self._start_time else 0

        config = runtime.config.get("configurable", {})
        user_id = config.get("user_id", "anonymous")
        thread_id = config.get("thread_id", "unknown")

        # === 从 State 提取关键信息 ===
        messages = state.get("messages", [])
        msg_count = len(messages)
        turn_count = state.get("turn_count", 0)
        
        # 提取本轮对话中的实体（生产环境用 NER 模型）
        entities = state.get("extracted_entities", [])

        # === 写入 BaseStore：更新用户持久化档案 ===
        if hasattr(runtime, 'store') and runtime.store:
            # 读取旧档案
            old_context = state.get("user_context", {})
            # 合并新信息（用 LLM 提取的实体更新）
            old_entities = old_context.get("entities", [])
            merged_entities = list(set(old_entities + entities))

            updated_context = {
                **old_context,
                "entities": merged_entities,
                "last_active_at": end_time.isoformat(),
                "total_messages": old_context.get("total_messages", 0) + msg_count,
                "total_sessions": old_context.get("total_sessions", 0) + 1,
            }

            # 写回 BaseStore
            runtime.store.put(
                namespace=("users", user_id, "context"),
                key="latest",
                value=updated_context,
            )

        # === 写入审计日志：会话结束 ===
        self._write_audit({
            "event": "session_end",
            "user_id": user_id,
            "thread_id": thread_id,
            "timestamp": end_time.isoformat(),
            "elapsed_seconds": elapsed,
            "message_count": msg_count,
            "turn_count": turn_count,
            "entities_extracted": entities,
        })

        print(f"  📁 会话结束：user={user_id}, 耗时={elapsed:.1f}s, "
              f"消息数={msg_count}, 实体={entities}")

        return None  # after_agent 不需要更新 State

    # ================================================================
    # 辅助方法：写入审计日志
    # ================================================================
    def _write_audit(self, record: dict):
        """将审计记录追加写入 JSONL 文件。"""
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ 审计日志写入失败：{e}")


# ---- 使用 ----
from langgraph.store.memory import InMemoryStore

# 注意：SessionPersistenceMiddleware 需要 BaseStore（runtime.store）
# 所以创建 Agent 时必须传 store 参数
store = InMemoryStore()

agent = create_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0.7),
    tools=[get_weather, search_web],
    middleware=[SessionPersistenceMiddleware(audit_file="./audit.jsonl")],
    store=store,                          # ← 必须传 store
    checkpointer=InMemorySaver(),         # ← 短期记忆
)

# 多轮对话测试
config = {
    "configurable": {
        "thread_id": "persist_demo",
        "user_id": "alice",              # ★ 用户身份标识 ★
    }
}

# 第 1 轮：Agent 首次为 Alice 服务（before_agent 恢复上下文为空）
result_1 = agent.invoke(
    {"messages": [HumanMessage("我是 Alice，我喜欢 Python 和 RAG 技术")]},
    config=config,
)

# 第 2 轮：Agent 再次为 Alice 服务（before_agent 恢复了第 1 轮的上下文）
result_2 = agent.invoke(
    {"messages": [HumanMessage("你还记得我是谁吗？")]},
    config=config,
)

# 审计日志自动写入 audit.jsonl：
# {"event": "session_start", "user_id": "alice", ...}
# {"event": "session_end", "user_id": "alice", ...}
```

---

### 9.4 装饰器 vs 继承：选择指南

| | `@before_model` 装饰器 | 继承 `AgentMiddleware` |
|---|---|---|
| **代码量** | 1 个函数 | 1 个类 + 多个方法 |
| **支持多个钩子** | 否（一个装饰器 = 一个钩子） | 是（一个类实现多个钩子） |
| **实例变量/内部状态** | 否（函数无状态） | 是（类可以有属性） |
| **可复用性** | 函数可单独测试 | 类可被继承和扩展 |
| **类型安全** | 靠类型注解 | 完整的泛型支持 |
| **适用场景** | 轻量拦截、快速原型 | 复杂业务逻辑、需要内部状态 |
| **本章示例** | 动态提示词切换、动态模型切换 | 会话持久化 |

**选择策略**：

```
你的中间件需要：
├─ 只拦截一个钩子 + 无内部状态
│   └─ → 装饰器（@before_model / @wrap_model_call / ...）
│
├─ 需要多个钩子协同（before + after）
│   └─ → 继承 AgentMiddleware
│
├─ 需要内部状态（计数器、缓存、连接池）
│   └─ → 继承 AgentMiddleware
│
└─ 需要访问 state_schema 或其他基类特性
    └─ → 继承 AgentMiddleware
```

---

## 第十章：多中间件组合实战 — IT 运维 Agent

### 10.1 场景概述

构建一个企业级 IT 运维 Agent，集成 7 个中间件 + RBAC 权限体系。完整代码见 `it_ops_agent.py`。

**角色定义**：

| 角色 | 权限范围 | 典型用户 |
|---|---|---|
| `admin` | 所有权限（读写删 + 用户管理） | IT 主管 |
| `operator` | 运维操作（查状态/重启/查日志/写数据库/看指标） | 运维工程师 |
| `viewer` | 只读（查状态/看日志/看指标） | 数据分析师 |
| `auditor` | 只读 + 审计日志 | 合规审计员 |

**可用运维工具**：`get_server_status`, `restart_service`, `view_logs`, `get_system_metrics`, `query_database`, `write_database`, `delete_record`

### 10.2 中间件执行流程

```
                    用户请求
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ [1] before_agent（Agent 启动前，只执行一次）                    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SecurityCheckMiddleware                             │    │
│  │   ├─ 解析 JWT/Token → UserContext                    │    │
│  │   ├─ 验证用户身份有效性                              │    │
│  │   ├─ RBAC: 角色 → 权限列表查询                       │    │
│  │   └─ 将 user_id/role/permissions 注入 AgentState    │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ AuditLogMiddleware.before_agent                      │    │
│  │   └─ 记录操作开始：用户 + 角色 + 时间戳               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ [2] wrap_model_call（每次 LLM 调用前，洋葱模型）               │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ DynamicPromptMiddleware（最外层）                     │    │
│  │   └─ 根据 user_role 选择 Prompt 注入 SystemMessage   │    │
│  │       admin → 决策果断、强调安全审计                   │    │
│  │       operator → 按手册执行、先读后写                   │    │
│  │       viewer → 数据分析、引导联系运维                   │    │
│  │       auditor → 合规审计、标记不合规                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SmartModelSwitchMiddleware                           │    │
│  │   └─ 分析问题复杂度 → 切换 premium/budget 模型         │    │
│  │       危险操作 / 复杂分析 / 长问题 → premium           │    │
│  │       简单查询 → budget（省钱）                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ContextManagementMiddleware（最内层）                 │    │
│  │   └─ Token 超阈值 → 裁剪旧消息                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ [3] 模型执行（LLM API 调用）                                   │
│                                                              │
│   LLM 收到: 角色 Prompt + 裁剪后的消息 + 工具列表               │
│   → 决定调用哪些工具 / 直接回答                                │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ [4] after_model（每次 LLM 调用后）                             │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ResponseValidationMiddleware                         │    │
│  │   └─ 检查 LLM 输出是否泄露敏感信息（密码/Token/密钥）  │    │
│  │       泄露 → 拦截 → 返回安全提示                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ AuditLogMiddleware.after_model                       │    │
│  │   └─ 记录操作完成：工具调用次数 + 耗时 + 结果摘要     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ [5] wrap_tool_call（工具调用前）                               │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SafetyGuardrailMiddleware                            │    │
│  │   ├─ 权限校验：用户是否有执行此工具的权限？             │    │
│  │   │   无权限 → 拦截 → 返回权限不足提示                 │    │
│  │   ├─ 危险操作确认：restart/delete/write → 审计标记    │    │
│  │   └─ 通过检查 → 正常执行工具                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 10.3 RBAC 设计详解

#### 为什么需要 RBAC

IT 运维系统的核心安全问题是**"谁能做什么"**。没有 RBAC → 任何一个用户都可以重启服务器、删除数据库 → 灾难。

#### 三层映射结构

```
用户 (UserContext)
  │
  └─→ 角色 (Role: admin/operator/viewer/auditor)
        │
        └─→ 权限集合 (Permission: server:read, server:restart, ...)
              │
              └─→ 可调用的工具 (Tool: get_server_status, restart_service, ...)
```

#### 代码中的实现

```python
# 第一层：角色 → 权限
ROLE_PERMISSIONS = {
    "admin":     {SERVER_READ, SERVER_RESTART, DB_READ, DB_WRITE, DB_DELETE, ...},
    "operator":  {SERVER_READ, SERVER_RESTART, DB_READ, DB_WRITE, ...},
    "viewer":    {SERVER_READ, LOG_VIEW, METRICS_VIEW},
    "auditor":   {SERVER_READ, LOG_VIEW, AUDIT_VIEW, ...},
}

# 第二层：工具 → 所需权限
TOOL_PERMISSION_MAP = {
    "restart_service":  Permission.SERVER_RESTART,   # 需要 server:restart 权限
    "delete_record":    Permission.DB_DELETE,         # 需要 database:delete 权限
    "get_server_status": Permission.SERVER_READ,      # 需要 server:read 权限
}

# 权限校验在 SafetyGuardrailMiddleware.wrap_tool_call 中执行：
# 1. 取出 tool_name
# 2. 查 TOOL_PERMISSION_MAP 找到 required_permission
# 3. 检查 required_permission 是否在 user_permissions 中
# 4. 不在 → 拦截，返回权限不足提示
```

### 10.4 安全护栏设计详解

#### 双层安全检查

```
第 1 层：权限校验（permission check）
  "你有权执行这个操作吗？"
  → 查看 role → 查看 permissions → 匹配 required_permission

第 2 层：危险操作确认（danger confirmation）
  "你知道这个操作的危险性吗？"
  → 查 DANGEROUS_OPERATIONS → 标记级别 → 审计日志记录
```

#### 危险操作分级

| 操作 | 危险等级 | 处理方式 |
|---|---|---|
| `delete_record` | critical | 拦截 + 审计 + 不可逆警告 |
| `restart_service` | high | 需确认 + 审计 + 提示影响范围 |
| `configure_server` | high | 需确认 + 审计 + 提示稳定性风险 |
| `write_database` | medium | 审计标记 + 提示数据准确性 |
| `get_server_status` | none | 直接放行（只读操作） |

### 10.5 审计日志设计

#### 双钩子保证可靠性

```
before_agent:  记录操作开始（即使 Agent 崩溃也有记录）
after_model:   记录操作完成（包含工具调用次数、耗时、结果摘要）
```

**为什么这样设计**：

- 只用 after_model → Agent 崩溃时日志丢失，无法追溯
- 只用 before_agent → 只知道开始，不知道结果和耗时
- before + after 双钩子 → 完整追踪每次操作的生命周期

#### 日志格式

```
[10:23:01] 📝 [审计] 操作开始 | 用户: 张管理(admin_zhang) | 角色: admin | 部门: IT运维部
[10:23:05] ⚠️ 危险操作: restart_service | 等级: high | 参数: {"service_name":"nginx"}
[10:23:08] 📝 [审计] 操作完成 | 用户: admin_zhang | 工具调用: 2次 | 耗时: 7.2s
```

### 10.6 角色 Prompt 差异化

同一个问题 "查看服务器状态"，不同角色得到不同风格的回复：

```
admin（管理员）:
  "所有服务器状态总览已完成。⚠️ db-server-01 CPU 使用率 72.3%，
   建议今晚 22:00 执行维护窗口，添加 CPU 资源。已为你生成扩容工单模板。

   影响范围: 仅 db-server-01
   回滚方案: 保留原配置快照，出问题一键回滚"

operator（运维工程师）:
  "📊 4 台服务器状态检查完毕：
   ✅ web-01/web-02: 正常
   ⚠️ db-server-01: CPU 偏高 (72.3%)，内存 81.5%
   
   建议操作:
   1. 先查看 db-server-01 的错误日志确认原因
   2. 如果是流量突增，联系 DBA 评估扩容
   3. 不是紧急问题，可在工作时间内处理"

viewer（观察员）:
  "📊 系统状态报告：
   3 台服务器健康，1 台需要注意（db-server-01）。
   
   详细分析:
   - db-server-01 CPU 使用率呈上升趋势（过去 4 小时从 45% → 72.3%）
   - 可能与 10:00 的数据导入任务相关
   
   ⚠️ 建议联系运维团队检查 db-server-01，已整理好问题描述。"

auditor（审计员）:
  "✅ 已记录本次状态查询操作。审计摘要：
   - 操作: 服务器状态查询
   - 时间: 2026-05-10 10:23 UTC
   - 操作人: auditor_li
   - 结果: 4 台服务器状态已记录
   
   合规性: 本次查询操作不涉及变更，符合流程。"
```

### 10.7 关键设计决策

| 决策 | 做法 | 原因 |
|---|---|---|
| **中间件全部用继承** | 7 个类，非装饰器 | 每个中间件有复杂的内部逻辑、需要多个钩子、需要实例变量 |
| **权限存在 ROLE_PERMISSIONS 字典** | 非数据库 | 教学示例简化。生产环境存数据库 + 缓存 |
| **get_current_user 从 runtime 取** | 非 State 取 | runtime.config 来自 JWT，不可被 LLM 篡改。安全字段走框架注入通道 |
| **危险操作在 wrap_tool_call 拦截** | 非 before_model | 工具调用是最终执行点，在此之前拦截可能被绕过 |
| **双钩子审计日志** | before_agent + after_model | 确保 Agent 崩溃时也有记录 |
| **角色 Prompt 差异** | 4 套完整 System Prompt | 同一 Agent，不同角色的行为边界完全不同 |
| **权限不足返回自然语言** | 非抛异常 | LLM 收到权限不足提示后可以解释给用户，而非直接崩溃 |

---

## 第十一章：中间件编排 — 顺序、分层与优先级

### 11.1 洋葱模型：注册顺序 ≠ 执行顺序

中间件的注册顺序决定了它们在洋葱中的位置。**先注册 = 最外层 = 最先拦截请求、最后处理响应**。

```
# 注册顺序（从左到右）：
middleware = [A, B, C]

# 实际执行（洋葱模型）：
#
# 请求进入 ──────────────────────────────→
#   ┌─ A ──────────────────────────────┐
#   │  ┌─ B ────────────────────────┐  │
#   │  │  ┌─ C ──────────────────┐  │  │
#   │  │  │    核心 Agent 流程     │  │  │
#   │  │  └──────────────────────┘  │  │
#   │  └────────────────────────────┘  │
#   └──────────────────────────────────┘
# ← 响应返回 ──────────────────────────────

# 5 个钩子的具体执行路径：
#
# before_agent:         A → B → C         （注册顺序 = 正序）
# before_model:         A → B → C         （注册顺序 = 正序）
# wrap_model_call:     A 包 B 包 C 包 LLM  （外层先拦截）
# after_model:          C → B → A         （注册顺序 = 反序！）
# wrap_tool_call:      A 包 B 包 C 包 Tool （外层先拦截）
# after_agent:          C → B → A         （注册顺序 = 反序）
```

**为什么 after/end 钩子是反序？**

就像 Java Servlet Filter 的 `doFilter` 之后执行 `finally` 块——最外层的 Filter 最后才拿到响应，因为它把控制权交给了内层，内层全部完成后才返回到外层。

### 11.2 五个钩子的执行特性对比

| 钩子 | 执行频率 | 所在层级 | 能否阻止后续 | 数据流向 | 控制颗粒度 |
|---|---|---|---|---|---|
| `before_agent` | 1 次/会话 | Agent 入口 | 是（Command.goto end） | State 写入 → 全局 | 粗——控制整个会话 |
| `before_model` | 每次 LLM 调用 | LLM 入口 | 是（Command.goto） | State 写入 → 全局 | 中——控制单次模型调用 |
| `wrap_model_call` | 每次 LLM 调用 | LLM 包裹 | 是（不调用 handler = 短路） | 修改 Request → 影响本次调用 | 细——完全控制本次调用 |
| `after_model` | 每次 LLM 调用 | LLM 出口 | 是（Command.goto） | State 写入 + 日志 | 中——控制调用后行为 |
| `wrap_tool_call` | 每次工具调用 | Tool 包裹 | 是（不调用 handler = 短路） | 修改 Tool 输入/输出 | 最细——控制每个工具调用 |

**控制颗粒度层次**：

```
粗（会话级）：   before_agent, after_agent
  ↓
中（LLM 调用级）： before_model, after_model
  ↓
细（单次调用级）： wrap_model_call
  ↓
最细（工具级）：  wrap_tool_call
```

### 11.3 分层架构：四层防线

```
┌─────────────────────────────────────────────────────────────┐
│                    中间件分层架构                             │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │          第 1 层：安全边界（Guard Layer）               │ │
│  │                                                       │ │
│  │  中间件: SecurityCheck, PII, Auth, RateLimit          │ │
│  │  优先级: P0 — 不通过则拒绝，不进业务层                  │ │
│  │  原则:  安全 > 一切。安全拦截后直接返回，不浪费资源。    │ │
│  │  钩子:   mainly before_agent + before_model           │ │
│  └───────────────────────────────────────────────────────┘ │
│                         │ 通过                               │
│                         ▼                                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │          第 2 层：业务逻辑（Business Layer）            │ │
│  │                                                       │ │
│  │  中间件: DynamicPrompt, ToolSelection, ModelSwitch    │ │
│  │  优先级: P1 — 核心业务功能，影响用户体验                │ │
│  │  原则:  业务 > 成本。先保证质量，再考虑省钱。            │ │
│  │  钩子:   mainly wrap_model_call + before_model         │ │
│  └───────────────────────────────────────────────────────┘ │
│                         │ 通过                               │
│                         ▼                                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │          第 3 层：成本控制（Cost Layer）                │ │
│  │                                                       │ │
│  │  中间件: ModelSwitch(budget), Summarization, Cache    │ │
│  │  优先级: P2 — 在保证业务质量的前提下降低成本            │ │
│  │  原则:  成本 < 业务。省钱的前提是不影响回答质量。       │ │
│  │  钩子:   mainly wrap_model_call                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                         │ 通过                               │
│                         ▼                                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │        第 4 层：监控与日志（Observability Layer）       │ │
│  │                                                       │ │
│  │  中间件: AuditLog, Metrics, ResponseValidation        │ │
│  │  优先级: P3 — 不影响业务，但必须执行                    │ │
│  │  原则:  监控不干扰业务。日志失败不能导致请求失败。       │ │
│  │  钩子:   mainly after_model + after_agent             │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**四层对应的生产级注册顺序**：

```python
agent = create_agent(
    model=llm,
    tools=[...],
    middleware=[
        # ===== 第 1 层：安全边界（最外层，最先拦截）=====
        SecurityCheckMiddleware(),
        PIIMiddleware("email", strategy="redact"),
        # ===== 第 2 层：业务逻辑 =====
        DynamicPromptMiddleware(),
        ToolSelectionMiddleware(...),
        # ===== 第 3 层：成本控制 =====
        SmartModelSwitchMiddleware(...),
        SummarizationMiddleware(...),
        CacheMiddleware(),
        # ===== 第 4 层：监控与日志（最内层）=====
        ResponseValidationMiddleware(),
        AuditLogMiddleware(),
    ],
)
```

### 11.4 六大优先级原则

**原则 1：安全优先** — 安全 > 一切。拦截后直接返回，不浪费 LLM 调用。

**原则 2：提前失败（Fail Fast）** — 在最可能失败的钩子中最先检查：before_agent 查认证 → wrap_model_call 查缓存 → wrap_tool_call 查权限。

**原则 3：缓存优先（Cache First）** — 缓存命中 → 0 Token 消耗 → 响应 < 1ms。放在 wrap_model_call 最外层。

**原则 4：写操作依赖正确** — 先读后写、先备份后改、先验证后执行。中间件顺序体现依赖关系。

**原则 5：中断操作置后** — HumanInTheLoop 放在 after_model：先让 LLM 充分推理 → 再暂停审批。

**原则 6：性能开销降序** — 高开销中间件放内层（只在必要时执行）；低开销放外层（每次都执行）。

### 11.5 中间件之间的数据依赖

中间件通过 **AgentState** 共享数据，不是孤立运行：

```
SecurityCheckMiddleware (before_agent)
  写入: user_id="alice", user_role="admin", user_permissions=[...]
        │
        ▼ AgentState（全局账本）
        │
DynamicPromptMiddleware (wrap_model_call)
  读取: state["user_role"] → "admin" → 选择对应 Prompt
        │
        ▼
SafetyGuardrailMiddleware (wrap_tool_call)
  读取: state["user_permissions"] → 检查工具权限
        │
        ▼
AuditLogMiddleware (after_model)
  读取: state["user_id"] → 写入审计日志
```

依赖管理四策略：上游写下游读、防御性读取（默认值）、显式文档化（docstring 声明依赖）、双钩子保底。

### 11.6 编排决策树

```
设计新中间件 → 放哪一层？

├─ 安全/认证/权限？→ 第 1 层。钩子: before_agent 或 before_model
├─ 业务逻辑/用户体验？→ 第 2 层。可短路的放外层（Cache），需LLM调用后的放内层
├─ 成本/资源控制？→ 第 3 层。钩子: wrap_model_call
├─ 只记录/观察？→ 第 4 层。钩子: after_model / after_agent
└─ 跨层协调？→ 多个钩子（如 AuditLog 同时 before_agent + after_model）
```

---

## 第十二章：扩展 — 企业中常用的中间件思路

以下五个方向帮助你在实际项目中快速判断"这里该不该抽象成中间件"：

| 中间件 | 所在层 | 钩子 | 做什么 | 编排原因 |
|---|---|---|---|---|
| **意图理解** | 第 2 层 | before_model | 分析意图 → 路由分支 | 安全通过后、LLM 调用前决策 |
| **文档解析** | 第 2 层 | wrap_model_call | 上传文件 → 自动解析 → 注入 State | 预处理在 LLM 推理之前 |
| **知识抽取** | 第 4 层 | after_model | 从回复提取实体 → 存长期记忆库 | 不阻塞用户响应 |
| **语义检索** | 第 2 层 | wrap_model_call | 自动检索知识库 → 注入 Prompt | 需用户身份做权限过滤 |
| **代码理解** | 第 2 层 | wrap_tool_call | 静态分析代码安全 → 通过后才执行 | 权限检查之后、代码执行之前 |

---

## 第十三章：大总结 — 中间件的工程化价值

### 13.1 工程化对比

```
没有中间件                    有中间件
─────────                    ────────
重试逻辑散落各工具             ToolRetryMiddleware 一行注册
安全规则硬编码在 Prompt         SecurityCheckMiddleware 统一拦截
改日志格式 → 改 20 个文件      改 1 个 AuditLogMiddleware
加缓存 → 改每个 LLM 调用点     加 1 个 CacheMiddleware
```

### 13.2 五大工程化原则

单一职责 / 开闭原则 / 依赖倒置 / 分层隔离 / 声明式组合 — 每个都在中间件体系中有具体体现。

### 13.3 架构跃迁

```
阶段 1：一把梭     → 一个函数搞定全部
阶段 2：手工切分   → 散落在各处，难以维护
阶段 3：中间件化   → 独立开发/测试/部署/组合 → 这就是架构
```

### 13.4 中间件模块知识体系

```
第一章：概览       → 什么是中间件、分类、生命周期
第二~五章：内置   → Summarization/PII/Model/Tool/HITL
第六章：其他常用   → Todo/ModelRetry/Shell
第七章：组合策略   → 推荐栈 + 按场景选择
第八章：参数传递   → Request/Response/State/Command 详解
第九章：自定义实战 → 装饰器 vs 继承 + 3 个完整示例
第十章：企业实战   → IT Ops Agent + RBAC + 7 个中间件
第十一章：编排     → 洋葱模型 + 分层 + 优先级 + 依赖
第十二章：扩展思路 → 意图/文档/知识/检索/代码 5 个方向
第十三章：大总结   → 工程化价值 + 五大原则 + 跃迁路径

核心收获：
• 横切关注点 → 中间件 → 声明式组合 → 可插拔架构
• 安全 > 业务 > 成本 > 监控 → 四层防线
• 洋葱模型 → 外层先拦截、内层后执行
• AgentState 是全局账本 → 中间件通过它解耦通信
• 装饰器搞轻量拦截、继承搞复杂业务、编排搞企业级

── 中间件模块结束 ──
```