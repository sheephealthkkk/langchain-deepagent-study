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

