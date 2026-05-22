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

