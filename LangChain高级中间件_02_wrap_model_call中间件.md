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

