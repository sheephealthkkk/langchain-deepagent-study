# LangChain 中级教程：Tools 工具体系

## 第一章：Tools 工具的集成

### 1.1 三种定义方式

LangChain 1.0 提供三种定义 Tool 的方式。本质一样，灵活度和适用场景不同。

```
@tool 装饰器          StructuredTool          BaseTool 继承
  (最简)                (中等灵活)              (完全控制)
     │                      │                      │
     └──────────────────────┼──────────────────────┘
                            │
                    底层都是 BaseTool
                  (实现 Runnable 协议)
```

### 1.2 方式一：@tool 装饰器（推荐首选）

**把一个普通 Python 函数变成 Tool。** 99% 的场景用这个就够了。

```python
from langchain.tools import tool

# 同步工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气。返回温度、天气状况、湿度。"""
    return f"{city}：晴，25°C，湿度 45%"

# 异步工具
@tool
async def search_database(query: str, limit: int = 10) -> str:
    """搜索内部数据库。limit 控制返回数量。"""
    await asyncio.sleep(0.3)
    return f"关于 '{query}' 的 {limit} 条结果"

# 使用
get_weather.invoke({"city": "北京"})           # → "北京：晴，25°C"
await search_database.ainvoke({"query": "AI"}) # → "关于 'AI' 的 10 条结果"
```

**`@tool` 自动做了什么**：

| 自动推导 | 来源 |
|---|---|
| `name` | 函数名（`get_weather`） |
| `description` | docstring（`获取指定城市的实时天气...`） |
| `args_schema` | 函数签名（`city: str` → `{"type":"string"}`） |

### 1.3 方式二：StructuredTool — 从 Runnable 构建

**当你已经有一个 Runnable（如 Chain），想把它包装成 Tool 时用。** 不需要重写函数，直接包一层。

```python
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableLambda

# 已有的 Runnable
translate_pipeline = (
    ChatPromptTemplate.from_template("翻译为{target_lang}：{text}")
    | llm
    | StrOutputParser()
)

# 直接包成 Tool
translate_tool = StructuredTool.from_function(
    func=lambda text, target_lang: translate_pipeline.invoke({
        "text": text, "target_lang": target_lang
    }),
    name="translate",
    description="将文本翻译为目标语言。text: 待翻译文本, target_lang: 目标语言",
)

# 还支持协程函数
translate_tool = StructuredTool.from_function(
    coroutine=async_translate,
    name="translate_async",
    description="异步翻译工具",
)
```

**`StructuredTool` 还支持 `args_schema` 显式声明**：

```python
from pydantic import BaseModel, Field

class TranslateInput(BaseModel):
    text: str = Field(description="待翻译文本")
    target_lang: str = Field(description="目标语言，如 '中文'、'English'")

translate_tool = StructuredTool.from_function(
    func=my_translate,
    name="translate",
    description="多语言翻译工具",
    args_schema=TranslateInput,       # ← 显式声明参数 Schema
)
```

### 1.4 方式三：继承 BaseTool（完全控制 — 面向对象的方式）

#### 什么时候需要继承 BaseTool

前两种方式的本质都是"把一个函数包成 Tool"。但有些场景函数做不到：

```
场景 A：工具需要**内部状态**（计数器、连接池、缓存）
  @tool 是函数，函数无状态。每次调用都是全新的，不能"记住上一次"。

场景 B：工具需要**控制流式输出**（逐块返回结果，而非一次性返回）
  @tool 函数只返回一个 str，框架帮你处理流式。但如果你想要自定义流式行为，必须继承。

场景 C：工具需要在**调用前后加钩子**（如鉴权、日志、参数校验）
  函数只有一行逻辑，没有 pre/post 钩子可以挂载。

场景 D：工具是**一个复杂的类**（从 Java 迁移过来的 Service、Repository 等）
  直接继承 BaseTool 比先函数化再 @tool 更自然。
```

#### 类图：BaseTool 的继承链

```
Runnable                ← 所有组件的基类（invoke/batch/stream）
   └─ BaseTool           ← 工具的抽象基类，定义了工具契约
        │
        │ 你必须重写的：
        │   _run(input) → str              同步执行逻辑
        │
        │ 你可选重写的：
        │   _arun(input) → str             异步执行逻辑
        │   _stream(input) → Iterator[str]  自定义流式输出
        │
        │ 你必须声明的（类属性）：
        │   name: str                      工具名（LLM 用这个名称调用）
        │   description: str               工具描述（LLM 据此判断何时调用）
        │   args_schema: BaseModel          参数 Schema（LLM 知道要传什么参数）
        │
        │ 你可以添加的：
        │   自定义实例属性（计数器、连接池、缓存、配置...）
        │   自定义方法（辅助函数、校验逻辑、钩子...）
        │
        ▼
   你的 CalculatorTool / DatabaseTool / WeatherTool ...
```

#### 完整实战：一个"可观测的"数据库查询工具

这个例子展示继承 BaseTool 的真正威力——不是简单计算，而是一个有连接池、计数器、日志、慢查询告警的生产级工具。

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Literal
import time, logging

logger = logging.getLogger(__name__)

# ================================================================
# 第 1 步：定义参数 Schema（与 @tool 的 args_schema 完全一样）
# ================================================================
class DatabaseQueryInput(BaseModel):
    """
    数据库查询工具的输入参数。

    这个 Schema 会被 LangChain 自动转为 LLM 看得懂的 JSON Schema。
    每个 Field 的 description 帮助 LLM 理解：这个参数是干什么的、取值范围是什么。
    """
    sql: str = Field(
        description="要执行的 SQL 查询语句。仅支持 SELECT，禁止 INSERT/UPDATE/DELETE",
        min_length=1,
        max_length=2000,
    )
    limit: int = Field(
        default=100,
        description="返回结果的最大行数",
        ge=1,      # ≥ 1
        le=1000,   # ≤ 1000
    )


# ================================================================
# 第 2 步：继承 BaseTool，实现工具类
# ================================================================
class DatabaseQueryTool(BaseTool):
    """
    安全的数据库查询工具。

    这个类展示了继承 BaseTool 的完整能力：
    ┌─────────────────────────────────────────────┐
    │ 类属性（必须声明）                              │
    │   name / description / args_schema           │
    │                                              │
    │ 实例属性（自定义状态）                          │
    │   connection_pool : 数据库连接池（复用连接）     │
    │   query_count     : 本实例累计执行的查询次数     │
    │   slow_query_threshold : 慢查询阈值（秒）       │
    │                                              │
    │ 必须重写的方法                                  │
    │   _run(input) → str                          │
    │                                              │
    │ 可选重写的方法                                  │
    │   _arun(input) → str   (异步版)               │
    │                                              │
    │ 自定义方法                                     │
    │   _check_sql_safety : SQL 安全检查             │
    │   _log_query        : 查询日志记录             │
    │   _maybe_warn_slow  : 慢查询告警               │
    └─────────────────────────────────────────────┘
    """

    # ===== 必须声明的 3 个类属性 =====
    # 这些属性替代了 @tool 的自动推导
    name: str = "query_database"
    description: str = (
        "在数据库中执行只读 SQL 查询。"
        "仅支持 SELECT 语句。"
        "返回 CSV 格式的前 N 行结果。"
        "如果查询超时或语法错误，返回错误信息。"
    )
    args_schema: type[BaseModel] = DatabaseQueryInput

    # ===== 自定义实例属性（@tool 做不到的）=====
    # 这些属性在 __init__ 中初始化，整个实例生命周期内保持

    connection_pool: object = None      # 数据库连接池（真实项目用 SQLAlchemy）
    query_count: int = 0                # 累计查询次数（内部计数器）
    slow_query_threshold: float = 3.0   # 慢查询阈值（秒），超过就打 WARN 日志

    # ===== 必须重写：同步执行逻辑 =====
    def _run(self, sql: str, limit: int = 100) -> str:
        """
        执行 SQL 查询。

        这是 BaseTool 要求子类实现的**唯一必须方法**。
        框架调用 invoke() → 框架做参数校验 → _run() → 框架包装返回值。

        参数：
          sql   : LLM 传入的 SQL 语句（由 args_schema 自动校验）
          limit : LLM 传入的行数限制（默认 100）

        返回值：
          str : 给 LLM 看的查询结果（LLM 会把这个结果放入上下文继续推理）
        """
        # ---- 步骤 1：调用前置钩子（安全校验）----
        # 这些逻辑在 @tool 函数里也可以写，但放在类方法中更清晰
        error_msg = self._check_sql_safety(sql)
        if error_msg:
            return f"❌ SQL 安全检查失败: {error_msg}"

        # ---- 步骤 2：执行查询 + 计时 ----
        self.query_count += 1       # 累加计数器（实例状态！）
        start = time.monotonic()

        try:
            # 模拟数据库查询（生产环境替换为真实的 connection_pool.execute）
            results = self._execute_sql(sql, limit)

            # ---- 步骤 3：调用后置钩子（日志 + 告警）----
            elapsed = time.monotonic() - start
            self._log_query(sql, elapsed, len(results))
            self._maybe_warn_slow(sql, elapsed)

            # ---- 步骤 4：格式化返回结果 ----
            if not results:
                return "查询成功，但没有匹配的记录。"
            return f"查询成功（{elapsed:.2f}s），返回 {len(results)} 行:\n" + results

        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error(f"[query_database] 查询失败 (耗时 {elapsed:.2f}s): {e}")
            # 返回自然语言错误（不是 traceback），以便 LLM 理解并调整
            return f"❌ 查询执行失败: {e}。请检查 SQL 语法后重试。"

    # ===== 可选重写：异步执行 =====
    async def _arun(self, sql: str, limit: int = 100) -> str:
        """
        异步版执行逻辑。

        为什么需要单独实现？
          _run 是同步的 → 在 asyncio 事件循环中会阻塞。
          _arun 是异步的 → 在等待数据库响应时让出控制权，不阻塞其他协程。

        当前简单转发给 _run（生产环境应改为真正的 async DB driver）。
        """
        return self._run(sql, limit)

    # ===== 自定义私有方法：安全检查 =====
    def _check_sql_safety(self, sql: str) -> str | None:
        """
        检查 SQL 是否安全。

        返回 None = 安全，返回 str = 错误信息（会阻止执行）。

        为什么放在工具内部而不是外部中间件？
          这个检查是此工具特有的——只有 database 类工具需要 SQL 安全检查。
          外部中间件是"横切关注点"，内部校验是"业务规则"。
        """
        sql_upper = sql.upper().strip()

        # 禁止写操作（这个工具只读）
        dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE"]
        for keyword in dangerous_keywords:
            if sql_upper.startswith(keyword):
                return f"禁止执行 {keyword} 操作。此工具仅支持 SELECT 查询。"

        # 禁止多语句查询（防注入）
        if ";" in sql.rstrip(";"):   # 允许末尾有一个分号
            return "禁止执行多条 SQL 语句。"

        return None   # 通过校验

    # ===== 自定义私有方法：日志 =====
    def _log_query(self, sql: str, elapsed: float, row_count: int):
        """记录查询日志。"""
        logger.info(
            f"[query_database] SQL: {sql[:80]} | "
            f"耗时: {elapsed:.3f}s | 结果: {row_count}行 | "
            f"累计: 第{self.query_count}次查询"
        )

    # ===== 自定义私有方法：告警 =====
    def _maybe_warn_slow(self, sql: str, elapsed: float):
        """慢查询告警。"""
        if elapsed > self.slow_query_threshold:
            logger.warning(
                f"⚠️ 慢查询告警: {elapsed:.2f}s (阈值: {self.slow_query_threshold}s) "
                f"SQL: {sql[:100]}"
            )

    # ===== 私有辅助方法 =====
    def _execute_sql(self, sql: str, limit: int) -> str:
        """模拟数据库查询。生产环境替换为真实数据库调用。"""
        # 这里用真实数据库连接池：
        # with self.connection_pool.acquire() as conn:
        #     cursor = conn.execute(sql)
        #     rows = cursor.fetchmany(limit)
        return "模拟结果: id=1, name=Alice\n模拟结果: id=2, name=Bob"


# ================================================================
# 第 3 步：使用 —— 和 @tool 完全一样
# ================================================================

# 实例化（可以传构造函数参数，如真实的连接池）
db_tool = DatabaseQueryTool(
    slow_query_threshold=2.0,   # 覆盖默认的 3.0 秒
)

# LLM 看到的是 name/description/args_schema，不知道里面有连接池和计数器
agent = create_agent(llm=llm, tools=[db_tool])

# 第一次调用 — 状态在实例内部自动累加
r1 = db_tool.invoke({"sql": "SELECT * FROM users WHERE active=1", "limit": 50})
print(f"查询次数: {db_tool.query_count}")  # → 1

# 第二次调用 — 同一个实例，计数器累加
r2 = db_tool.invoke({"sql": "SELECT * FROM orders WHERE amount > 100", "limit": 10})
print(f"查询次数: {db_tool.query_count}")  # → 2

# 危险操作 — SQL 安全检查自动拦截
r3 = db_tool.invoke({"sql": "DELETE FROM users WHERE id=1"})
# → "❌ SQL 安全检查失败: 禁止执行 DELETE 操作。此工具仅支持 SELECT 查询。"
```

#### 三种实现方法的关系

不管哪种方式，最终都实现了 `Runnable` 协议。LLM 看到的都是 `name + description + args_schema`，不知道内部是用 `@tool` 还是 `BaseTool`：

```
@tool            →  框架自动包装为  →  BaseTool 子类
StructuredTool   →  框架自动包装为  →  BaseTool 子类
继承 BaseTool    →  你手动定义     →  BaseTool 子类
                        │
                        ▼
                  都是 Runnable
              LLM 无差别调用
```

### 1.5 三种方式对比

| | `@tool` | `StructuredTool` | 继承 `BaseTool` |
|---|---|---|---|
| 代码量 | 最少（3 行） | 中等 | 最多 |
| 自动推导 name/description/schema | 是 | 需手动指定 | 需手动指定 |
| 支持异步 | 是（async def） | 是（coroutine 参数） | 是（`_arun` 方法） |
| 支持内部状态 | 否 | 否 | 是（实例属性） |
| 从现有 Runnable 构建 | 否 | 是 | 需要额外代码 |
| 自定义流式逻辑 | 否 | 否 | 是（`_stream` 方法） |
| 生命周期钩子（pre/post） | 否 | 否 | 是（重写 `_run` 前后加逻辑） |
| 面向对象设计 | 否（函数式） | 半（函数 + 参数） | 是（完整的类） |
| 适用 | **99% 的场景** | 包装已有组件 | 需要状态/流式/复杂逻辑 |

**选择指南**（结合实际场景）：

```
函数就是工具                          → @tool
  "我有一个 get_weather 函数，想让它被 LLM 调"
  
一个已有的 Chain 想当成工具用           → StructuredTool.from_function
  "我已经写了一个翻译 Pipeline，想直接给 Agent 用"
  
工具需要计数器/连接池/缓存/流式/钩子     → 继承 BaseTool
  "我的数据库工具需要连接池复用、慢查询告警、调用计数"
  "我的文件工具需要断点续传进度"
  "从 Java Service 迁移过来的类，继承比包装更自然"
```

---

## 第二章：多工具管理

### 2.1 定义多个工具

```python
@tool
def get_weather(city: str) -> str:
    """获取指定城市天气"""
    return f"{city}：晴，25°C"

@tool
def get_time(city: str) -> str:
    """获取指定城市当前时间"""
    return f"{city}：北京时间 14:30"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，如 '2+3*4'"""
    return f"结果: {eval(expression)}"

@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息"""
    return f"关于 '{query}' 的搜索结果: ..."

# 工具列表
tools = [get_weather, get_time, calculate, search_web]
```

### 2.2 绑定到 LLM

```python
llm_with_tools = llm.bind_tools(tools)

# LLM 根据用户问题，自动选择调用哪个工具
response = llm_with_tools.invoke("北京现在几点了？")
# → AIMessage(content=None, tool_calls=[{
#       "name": "get_time",
#       "args": {"city": "北京"},
#       "id": "call_001"
#   }])
```

### 2.3 Agent 自动调度多工具

```python
from langchain.agents import create_agent

agent = create_agent(
    llm=llm,
    tools=tools,
    system_prompt="你是智能助手，需要查询信息时主动调用工具。",
)

# 一个问题可能触发多个工具
result = agent.invoke({
    "messages": [HumanMessage("北京天气怎么样？帮我算 15*8")]
})

# Agent 内部自动完成：
# 1. 调用 get_weather(city="北京")
# 2. 拿到天气结果
# 3. 调用 calculate(expression="15*8")
# 4. 拿到计算结果
# 5. 整合两个结果，生成最终回答
```

---

## 第三章：Tool 的定义策略

### 3.1 name — 唯一标识符

```python
# name 是工具的"身份证"，LLM 用它指定要调用的工具
@tool
def get_weather(city: str) -> str:      # name = "get_weather"
    ...

# 显式指定 name
@tool("weather_tool")                    # name = "weather_tool"
def get_weather(city: str) -> str:
    ...

# StructuredTool
tool = StructuredTool.from_function(
    func=my_func,
    name="weather_tool",                # 显式指定
    ...
)
```

**策略**：

| 规则 | 好 | 坏 |
|---|---|---|
| 用动词+名词 | `get_weather`, `search_database` | `tool1`, `func` |
| 避免与内置函数重名 | `calculate_expression` | `eval`, `exec` |
| 下划线分隔 | `send_email` | `sendEmail` |
| 长度适中 | `search_web` | `search_the_internet_for_information` |

### 3.2 description — 模型判断调用的唯一依据

**这是最重要的字段！** LLM 根据 description 判断"什么时候该用这个工具、这个工具能干什么"。

```python
# ❌ 糟糕的 description — 模型不知道什么时候用
@tool
def f(x: str) -> str:
    """处理数据。"""
    ...

# ✅ 好的 description — 模型能准确判断
@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气信息。返回温度(°C)、天气状况(晴/雨/阴)、
    湿度(%)、风速(级)。需要传入城市中文名称，如 '北京'、'上海'。"""
    ...

# ✅ 含使用限制的 description
@tool
def delete_user(user_id: str) -> str:
    """永久删除指定用户及其所有数据。WARNING: 不可逆操作！
    仅管理员可调用。user_id: 用户唯一标识符。"""
    ...
```

**策略**：

```
好的 description = 做什么 + 输入什么 + 返回什么 + 何时用 + 注意事项
```

### 3.3 args_schema — 显式声明参数 Schema

默认情况下，`@tool` 从函数签名自动推导 Schema。但显式声明可以提供更多控制：

```python
from pydantic import BaseModel, Field
from typing import Literal

# === 自动推导（默认）===
@tool
def search(query: str, limit: int = 10) -> str:
    """搜索数据库。"""
    ...

# LLM 看到的：query: string, limit: integer (default: 10)

# === 显式声明（更精确的控制）===
class SearchInput(BaseModel):
    """搜索参数 — 这个 docstring 也会被模型看到！"""
    query: str = Field(
        description="搜索关键词，支持中文和英文，多个词用空格分隔",
        min_length=1,
        max_length=200,
        examples=["LangChain 教程", "Python async"],
    )
    limit: int = Field(
        default=10,
        description="返回结果数量上限",
        ge=1,           # ≥ 1
        le=100,         # ≤ 100
    )
    source: Literal["web", "database", "arxiv"] = Field(
        default="web",
        description="搜索来源",
    )

@tool(args_schema=SearchInput)
def search(query: str, limit: int = 10, source: str = "web") -> str:
    """搜索信息。"""
    return f"在 {source} 中搜索 '{query}'，返回 {limit} 条结果"
```

**什么时候显式声明 args_schema**：

| 场景 | 默认推导 | 显式声明 |
|---|---|---|
| 简单参数 | 够用 | 不需要 |
| 需要参数校验（ge/le/min_length） | 不支持 | **必须** |
| 需要枚举（Literal） | 不支持 | **必须** |
| 需要默认值 | 支持 | 支持 |
| 需要 examples | 不支持 | **推荐** |

### 3.4 参数校验

```python
class TransferInput(BaseModel):
    from_account: str = Field(description="转出账户", min_length=10, max_length=20)
    to_account: str = Field(description="转入账户", min_length=10, max_length=20)
    amount: float = Field(description="转账金额", gt=0)        # > 0
    currency: Literal["CNY", "USD", "EUR"] = Field(description="币种")

@tool(args_schema=TransferInput)
def transfer(from_account: str, to_account: str, amount: float, currency: str) -> str:
    """转账"""
    return f"已从 {from_account} 转 {amount} {currency} 到 {to_account}"

# 参数不合法 → Pydantic 自动校验 → 返回清晰错误
transfer.invoke({
    "from_account": "123",          # ← 太短，min_length=10
    "to_account": "45678901234",
    "amount": -100,                 # ← 负数，gt=0
})
# → ValidationError: from_account 长度不足, amount 必须 > 0
```

### 3.5 异步优先

```python
# ❌ 同步阻塞（在 Agent 循环中会阻塞整个流程）
@tool
def fetch_data(query: str) -> str:
    time.sleep(2)                    # 阻塞 2 秒！
    return http_client.get(query)

# ✅ 异步（Agent 循环在等待时可以处理其他任务）
@tool
async def fetch_data(query: str) -> str:
    await asyncio.sleep(0)           # 让出控制权
    return await http_client.get(query)
```

**同步工具会被 LangChain 自动在线程池中执行，但对于大量 I/O 操作，原生的 `async def` 更高效。**

### 3.6 文档清晰

```python
# ❌ 文档不清晰 — 模型容易误判
@tool
def f(x: str) -> str:
    """do something"""
    ...

# ✅ 文档清晰
@tool
def send_email(
    to: str = Field(description="收件人邮箱地址"),
    subject: str = Field(description="邮件主题"),
    body: str = Field(description="邮件正文，支持 Markdown"),
) -> str:
    """发送电子邮件。发送成功返回 'sent'，失败返回错误信息。

    使用场景：需要通知用户、发送报告、确认操作时调用。

    限制：每天最多发送 100 封，超出会返回错误。
    """
    ...
```

### 3.7 返回值友好

```python
# ❌ 返回技术细节，模型难以理解
@tool
def query_db(sql: str) -> str:
    return str(cursor.fetchall())  # → "[(1, 'Alice'), (2, 'Bob')]"

# ✅ 返回自然语言描述
@tool
def query_db(sql: str) -> str:
    rows = cursor.fetchall()
    if not rows:
        return "查询结果为空。"
    return f"找到 {len(rows)} 条记录: " + ", ".join(
        f"ID={r[0]}, 姓名={r[1]}" for r in rows
    )
    # → "找到 2 条记录: ID=1, 姓名=Alice, ID=2, 姓名=Bob"
```

**策略**：Tool 的返回值是**给 LLM 看的**（不是给用户看的）。格式越清晰，LLM 越能正确理解和引用。

### 3.8 调试清晰

```python
import logging

logger = logging.getLogger(__name__)

@tool
def complex_search(query: str, filters: dict = None) -> str:
    """复杂搜索。"""
    logger.info(f"[complex_search] query='{query}', filters={filters}")

    try:
        results = do_search(query, filters)
        logger.info(f"[complex_search] 成功，{len(results)} 条结果")
        return format_results(results)
    except Exception as e:
        logger.error(f"[complex_search] 失败: {e}", exc_info=True)
        # 返回错误信息给 LLM，让 Agent 能据此调整
        return f"搜索失败: {e}。建议尝试简化查询条件或更换关键词。"
```

**调试策略**：

| 策略 | 实现 |
|---|---|
| 日志记录每次调用 | `logger.info(f"[{self.name}] 参数={...}")` |
| 记录耗时 | `time.monotonic()` 前后打点 |
| 错误返回自然语言 | 不要 `raise Exception`，返回错误描述给 LLM |
| 建议替代方案 | 失败后告诉 LLM "可以尝试 XXX" |

---

## 第四章：官方预置工具速查

LangChain 社区提供了 150+ 个开箱即用的工具。以下是常用分类：

### 4.1 搜索工具

```python
from langchain_community.tools import (
    WikipediaQueryRun,        # Wikipedia 百科查询
    ArxivQueryRun,            # Arxiv 论文查询
    DuckDuckGoSearchRun,      # DDG 搜索
    GoogleSearchRun,          # Google 搜索（需 API Key）
    GoogleSerperRun,          # Google Serper（搜索 API）
    TavilySearchResults,      # Tavily AI 搜索（推荐，专为 Agent 设计）
    BraveSearch,              # Brave 搜索（隐私优先）
    BingSearchRun,            # Bing 搜索
)

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# Wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia.invoke("LangChain")  # → 'LangChain is a framework for...'

# Arxiv: 搜最新论文
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
arxiv.invoke("RAG LLM agent 2024")
```

### 4.2 代码/Shell 工具

```python
from langchain_community.tools import (
    ShellTool,                # 执行 Shell 命令（危险，需沙箱）
)

# Shell: 只在沙箱中使用！
shell = ShellTool()
shell.invoke("ls -la /tmp")
```

### 4.3 数据库工具

```python
from langchain_community.tools import (
    QuerySQLDatabaseTool,     # 查询 SQL 数据库
    InfoSQLDatabaseTool,      # 获取数据库表结构
    ListSQLDatabaseTool,      # 列出所有表
    QuerySQLCheckerTool,      # 检查 SQL 语句是否正确
)
```

### 4.4 文件管理工具

```python
from langchain_community.tools.file_management import (
    ReadFileTool,             # 读文件
    WriteFileTool,            # 写文件
    CopyFileTool,             # 复制文件
    MoveFileTool,             # 移动文件
    DeleteFileTool,           # 删除文件
    FileSearchTool,           # 搜索文件
    ListDirectoryTool,        # 列出目录内容
)
```

### 4.5 网络请求工具

```python
from langchain_community.tools import (
    RequestsGetTool,          # HTTP GET
    RequestsPostTool,         # HTTP POST
    RequestsPutTool,          # HTTP PUT
    RequestsPatchTool,        # HTTP PATCH
    RequestsDeleteTool,       # HTTP DELETE
)

# 以 GET 为例
http_get = RequestsGetTool()
http_get.invoke("https://api.github.com/repos/langchain-ai/langchain")
```

### 4.6 通讯工具

```python
from langchain_community.tools import (
    GmailSendMessage,         # 发送 Gmail
    GmailSearch,              # 搜索 Gmail
    SlackSendMessage,         # 发送 Slack 消息
    JiraAction,               # Jira 操作（创建/查询 Issue）
)
```

### 4.7 多媒体工具

```python
from langchain_community.tools import (
    YouTubeSearchTool,                              # YouTube 搜索
    GoogleCloudTextToSpeechTool,                    # 文本转语音
    AzureAiServicesSpeechToTextTool,                # 语音转文本
)
```

### 4.8 常见使用模式

```python
# 预置工具 + 自定义工具混用
from langchain_community.tools import WikipediaQueryRun, TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}：晴，25°C"

# 混编
tools = [
    get_weather,                                              # 自定义
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),     # 预置
    TavilySearchResults(api_key=os.getenv("TAVILY_KEY")),    # 预置（需 API Key）
]

agent = create_agent(llm=llm, tools=tools)
```

---

## 第五章：Tool 开发 Checklist

按这个清单检查每个 Tool：

```
□ name: 动词+名词，下划线分隔，全局唯一
□ description: 做什么 + 输入什么 + 返回什么 + 何时用 + 注意事项
□ args_schema: 简单参数用默认推导，需要校验/枚举/示例时显式声明
□ 参数校验: 用 Pydantic Field 的 ge/le/min_length/pattern 等
□ 异步: I/O 操作用 async def
□ 返回值: 自然语言描述，给 LLM 看的不是给用户看的
□ 错误处理: 不抛异常，返回错误描述 + 建议
□ 日志: 记录每次调用的输入输出和耗时
□ 文档: docstring 有使用场景和限制说明
```

---

