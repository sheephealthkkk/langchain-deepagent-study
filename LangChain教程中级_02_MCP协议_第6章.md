## 第六章：MCP（Model Context Protocol）接入 LangChain

### 6.1 什么是 MCP

**MCP = 工具的标准通信协议**。类比：USB-C 协议统一了充电线，MCP 统一了"LLM 如何发现和调用工具"。

在没有 MCP 之前：

```
你的 Agent ──→ 调用 get_weather（自己写的 LangChain @tool）
你的 Agent ──→ 调用 search_web（自己写的 LangChain @tool）
你的 Agent ──→ 调用 GitHub API（需要自己写一个 @tool 包装）
你的 Agent ──→ 调用 Slack API（需要自己写一个 @tool 包装）
            ↑
    每个外部服务都需要自己包装，格式各自不同
```

有了 MCP 之后：

```
你的 Agent ──→ MCP Client ──┬──→ MCP Server A（提供 GitHub 工具）
                            ├──→ MCP Server B（提供 Slack 工具）
                            ├──→ MCP Server C（提供数据库工具）
                            └──→ MCP Server D（提供文件管理工具）
            ↑
    所有工具通过统一的 MCP 协议发现和调用，格式完全一致
```

**核心价值**：一次开发 MCP Client，就可以用所有 MCP Server 提供的工具。服务端（MCP Server）和客户端（Agent）解耦。

### 6.2 MCP 核心字段

**`StdioServerParameters`** — 定义一个 MCP 服务的位置和启动方式：

```python
from mcp import StdioServerParameters

# 本地 MCP 服务配置
StdioServerParameters(
    command="python",                    # ← 启动 MCP Server 的命令
    args=[                               # ← 传给命令的参数
        "-m", "my_mcp_server",           # 以模块方式运行
        "--port", "8000",
    ],
    env={"API_KEY": "sk-xxx"},           # ← 环境变量（可选）
)
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `command` | `str` | 启动 MCP Server 的可执行命令（`python`、`node`、`uv` 等） |
| `args` | `list[str]` | 传给命令的参数列表 |
| `env` | `dict[str,str]` | 环境变量，用于传 API Key 等敏感信息 |

**MCP 协议的标准字段**（每个 Tool 被 MCP 包装后的统一结构）：

```python
Tool(
    name="get_weather",          # ← 同 LangChain 的 name
    description="获取天气...",    # ← 同 LangChain 的 description
    inputSchema={                # ← 同 LangChain 的 args_schema
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名"}
        },
        "required": ["city"],
    },
)
```

### 6.3 本地部署 MCP 服务 — 完整开发流程

这一节从头搭建一个可运行的 MCP 本地环境：**你写一个 Server 提供工具，再写一个 Client 消费这些工具**。

#### 6.3.1 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                      你的开发机器                                │
│                                                                 │
│  ┌──────────────────────────┐    ┌──────────────────────────┐  │
│  │  weather_server.py       │    │  agent_client.py         │  │
│  │  (MCP Server — 提供方)    │    │  (MCP Client — 消费方)    │  │
│  │                          │    │                          │  │
│  │  FastMCP("Weather")      │    │  ClientSessionGroup      │  │
│  │    ├─ get_weather()      │    │    ├─ connect(weather)    │  │
│  │    └─ get_air_quality()  │    │    ├─ connect(file)       │  │
│  │                          │    │    └─ 聚合所有工具          │  │
│  │  mcp.run(stdio)          │    │                          │  │
│  └──────────┬───────────────┘    │  MCPToolkit →            │  │
│             │                    │  create_agent(tools)     │  │
│             │   stdio 通信        └──────────────────────────┘  │
│             └─────────→ stdin/stdout ← ──────────               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**关键理解**：Server 和 Client 是**两个独立的 Python 进程**。Server 通过 stdio（标准输入输出）与 Client 通信。Client 启动 Server 进程，往它的 stdin 写 JSON-RPC 请求，从它的 stdout 读响应。

#### 6.3.2 第一步：创建 MCP Server

```python
# ================================================================
# weather_server.py — MCP Server（工具提供方）
# 运行方式：python weather_server.py
# 这个进程启动后，等待 Client 通过 stdio 发来 JSON-RPC 请求
# ================================================================
from mcp.server.fastmcp import FastMCP

# === 1. 创建 FastMCP 实例 ===
# FastMCP 是官方提供的高层封装，底层是 asyncio + JSON-RPC over stdio
# 参数 name: 会在 MCP 协议的 initialize 响应中返回给 Client
mcp = FastMCP("Weather Service")

# === 2. 用 @mcp.tool() 装饰器注册工具 ===
# 和 LangChain 的 @tool 几乎一样：函数名 → name，docstring → description
# 区别：这些工具在 Client 端通过 MCP 协议发现和调用，不在当前进程执行

@mcp.tool()
def get_weather(city: str) -> str:
    """
    获取指定城市的实时天气信息。

    返回该城市的温度、天气状况。
    支持全国主要城市，如 北京、上海、广州、深圳。
    """
    # 模拟天气数据库（生产环境换成真实 API 调用）
    weather_db = {
        "北京": "晴，25°C，湿度 40%，风力 2级",
        "上海": "多云，28°C，湿度 65%，风力 3级",
        "广州": "阵雨，30°C，湿度 80%，风力 1级",
        "深圳": "晴转多云，29°C，湿度 70%，风力 2级",
    }
    return weather_db.get(city, f"未找到 {city} 的天气数据。支持的城市：{', '.join(weather_db.keys())}")

@mcp.tool()
def get_air_quality(city: str) -> str:
    """
    获取指定城市的空气质量指数（AQI）和级别。

    返回 AQI 数值和级别描述（优/良/轻度污染/中度污染/重度污染）。
    """
    aqi_db = {
        "北京": "AQI 52，级别：良",
        "上海": "AQI 45，级别：优",
        "广州": "AQI 68，级别：良",
        "深圳": "AQI 35，级别：优",
    }
    return aqi_db.get(city, f"未找到 {city} 的空气质量数据。")

# === 3. 启动 Server ===
# transport="stdio" 表示通过标准输入输出与 Client 通信
# 这个调用会阻塞当前进程，等待 Client 连接
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

#### 6.3.3 第二步：创建 MCP Client

```python
# ================================================================
# agent_client.py — MCP Client（工具消费方）
# 运行方式：python agent_client.py
# 这个进程启动 weather_server.py，通过 stdio 连接它，加载工具
# ================================================================
import asyncio
from mcp import StdioServerParameters
from mcp.client.session_group import ClientSessionGroup
from langchain_mcp import MCPToolkit
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

async def main():
    # === 1. 配置要连接的 MCP Server 列表 ===
    # 每个 StdioServerParameters 描述一个 MCP 服务的位置和启动方式
    # command + args 一起构成启动该服务的命令行
    # ClientSessionGroup 会自动执行这些命令，启动子进程，建立 stdio 连接
    servers = [
        StdioServerParameters(
            command="python",                       # 用什么命令启动
            args=["weather_server.py"],             # 传给命令的参数
            # 等效于在终端执行: python weather_server.py
        ),
        # 可以添加更多 MCP Server——都通过 stdio 连接
        # StdioServerParameters(
        #     command="python",
        #     args=["file_server.py"],
        # ),
    ]

    # === 2. 创建 ClientSessionGroup ===
    # ClientSessionGroup 是管理多个 MCP 连接的容器。
    # 它负责：
    #   - 启动子进程（执行 command + args）
    #   - 建立 stdio 双向通信
    #   - 自动初始化 MCP 协议握手（initialize → initialized → list_tools）
    #   - 聚合所有 Server 的工具列表
    # async with 退出时自动关闭所有子进程

    async with ClientSessionGroup() as group:
        # --- 2a. 连接所有 Server ---
        # connect_to_server 内部做的事：
        #   ① subprocess.Popen(server.command, server.args)  → 启动子进程
        #   ② stdio_client(server)                            → 建立通信通道
        #   ③ ClientSession(read_stream, write_stream)        → 创建会话
        #   ④ session.initialize()                           → MCP 协议握手
        #   ⑤ session.list_tools()                           → 获取工具列表
        for server in servers:
            await group.connect_to_server(server)

        # --- 2b. 从所有 Server 加载工具 ---
        # group._tools 是一个 dict: {ClientSession: [MCP Tool 对象列表]}
        # MCPToolkit 把原始的 MCP Tool 转换为 LangChain BaseTool
        # 这样就能直接传给 create_agent()
        all_tools = []
        for session, mcp_tools in group._tools.items():
            # MCPToolkit 包装了一个 ClientSession
            # initialize() → 调用 session.list_tools() → 拿到工具列表
            # get_tools() → 把 MCP Tool 转为 LangChain BaseTool
            toolkit = MCPToolkit(session=session)
            await toolkit.initialize()       # 获取工具列表
            all_tools.extend(toolkit.get_tools())  # 转为 LangChain 工具

        print(f"✅ 已加载 {len(all_tools)} 个 MCP 工具：")
        for t in all_tools:
            print(f"   • {t.name}: {t.description[:60]}...")

        # --- 2c. 创建 Agent ---
        llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0.7)
        agent = create_agent(
            llm=llm,
            tools=all_tools,
            system_prompt="你是智能助手，可以使用天气查询工具。",
        )

        # --- 2d. 测试调用 ---
        result = agent.invoke({
            "messages": [HumanMessage("北京今天天气怎么样？空气质量好吗？")]
        })
        print(f"\n🤖 Agent 回答: {result['messages'][-1].content}")

# 入口
asyncio.run(main())
```

#### 6.3.4 关键通信机制

```
┌──────────────┐                    ┌──────────────────┐
│ agent_client │                    │ weather_server   │
│  (父进程)     │                    │  (子进程)         │
└──────┬───────┘                    └────────┬─────────┘
       │                                     │
       │  ① subprocess.Popen("python",       │
       │     ["weather_server.py"])           │
       │  ─────────────────────────────────→  │ 启动子进程
       │                                     │
       │  ② initialize 请求（JSON-RPC）        │
       │  {"method":"initialize", ...}        │
       │  ───────── stdin ──────────→         │
       │                                     │
       │  ③ initialize 响应                   │
       │  {"result":{"serverInfo":...}}       │
       │  ←──────── stdout ──────────         │
       │                                     │
       │  ④ list_tools 请求（JSON-RPC）        │
       │  {"method":"tools/list"}             │
       │  ───────── stdin ──────────→         │
       │                                     │
       │  ⑤ 工具列表                          │
       │  {"tools":[{"name":"get_weather"},   │
       │    {"name":"get_air_quality"}]}      │
       │  ←──────── stdout ──────────         │
       │                                     │
       │  ⑥ Agent 调用 get_weather            │
       │  {"method":"tools/call",             │
       │   "params":{"name":"get_weather",    │
       │   "arguments":{"city":"北京"}}}      │
       │  ───────── stdin ──────────→         │
       │                                     │
       │  ⑦ 工具结果                          │
       │  {"content":[{"text":"北京：晴..."}]} │
       │  ←──────── stdout ──────────         │
       │                                     │
```

**stdio 通信的本质**：Client 把 JSON-RPC 请求写入 Server 的 `stdin`，Server 把结果写入自己的 `stdout`，Client 从 Server 的 `stdout` 读取。整个过程对开发者透明——你只操作 `MCPToolkit` 返回的 LangChain 工具，底层通信由 MCP 协议栈处理。

---

### 6.4 MCP 失败时的自动降级策略

#### 为什么需要降级

MCP Server 是独立进程，可能因为各种原因不可用：

```
网络断开 → 远程 MCP Server 不可达
子进程崩溃 → 本地 MCP Server 异常退出
超时 → Server 响应太慢
```

如果 Agent 唯一依赖 MCP 工具，这些故障等于 Agent 瘫痪。降级策略让 Agent 在 MCP 故障时切换到本地备用实现。

#### 降级包装器实现

```python
from langchain_core.tools import StructuredTool
from typing import Callable
import logging

logger = logging.getLogger(__name__)

class MCPWithFallback:
    """
    MCP 工具降级包装器。

    设计思路：
      try: MCP 工具（主路径）→ 功能完整、数据最新
      except: 本地 fallback（备用路径）→ 功能降级但可用

    类比：
      微服务架构中的 Circuit Breaker（断路器）。
      主服务挂了 → 自动切到本地缓存 / 兜底逻辑。
    """

    def __init__(
        self,
        mcp_tool,                       # MCP 工具（主路径）
        fallback_func: Callable,        # 本地 fallback 函数（备用路径）
        max_retries: int = 1,           # MCP 失败后重试次数
    ):
        self.mcp_tool = mcp_tool
        self.fallback = fallback_func
        self.max_retries = max_retries
        self._failure_count = 0         # 累计失败次数（用于监控告警）

    def to_tool(self) -> StructuredTool:
        """
        生成一个包装后的 LangChain 工具。

        返回的工具有三个特征：
          1. 优先走 MCP 路径
          2. MCP 连续失败 max_retries+1 次后自动切 fallback
          3. 对调用方完全透明（Agent 不知道内部用了 MCP 还是 fallback）

        生成的工具签名（name/description/args_schema）沿用 MCP 工具的定义，
        所以 Agent 看到的是同一个工具，不知道底层有主备切换。
        """
        async def _call_with_fallback(**kwargs):
            # === 尝试走 MCP 主路径 ===
            for attempt in range(self.max_retries + 1):
                try:
                    result = await self.mcp_tool.ainvoke(kwargs)
                    self._failure_count = 0  # 成功后重置计数器
                    return result
                except Exception as e:
                    self._failure_count += 1
                    logger.warning(
                        f"[MCP降级] {self.mcp_tool.name} 第{attempt+1}次尝试失败: {e}"
                    )
                    if attempt < self.max_retries:
                        continue  # 还有重试次数 → 重试
                    # 重试耗尽 → 走 fallback

            # ===  MCP 失败 → 自动切换 fallback ===
            logger.warning(
                f"[MCP降级] {self.mcp_tool.name} 已切换至本地 fallback "
                f"（累计失败 {self._failure_count} 次）"
            )
            return self.fallback(**kwargs)

        # 用 StructuredTool.from_function 生成 LangChain 工具
        # args_schema 沿用 MCP 工具的定义 → Agent 看到同样的参数
        return StructuredTool.from_function(
            coroutine=_call_with_fallback,
            name=self.mcp_tool.name,
            description=f"{self.mcp_tool.description}（注：当前可能运行在降级模式）",
            args_schema=self.mcp_tool.args_schema,
        )

    @property
    def is_degraded(self) -> bool:
        """当前是否处于降级模式（用于外部监控）。"""
        return self._failure_count > 0


# === 使用示例 ===
# 从 ClientSessionGroup 拿到了原始的 MCP 工具
mcp_weather_tool = all_tools[0]  # get_weather（MCP 版本）

# 包装为带降级的工具
safe_weather = MCPWithFallback(
    mcp_tool=mcp_weather_tool,
    # fallback：本地静态数据（不需要网络，保证可用）
    fallback_func=lambda city: (
        f"{city}（本地降级模式）：数据暂不可用，"
        f"建议稍后重试或联系管理员。"
    ),
    max_retries=1,
).to_tool()

# 把降级工具传给 Agent
agent = create_agent(llm=llm, tools=[safe_weather])
# Agent 无感知——正常时走 MCP，故障时自动切 fallback
```

---

### 6.5 把自己的 LangChain 工具暴露为 MCP 服务

#### 反向流程：LangChain → MCP

前面讲的是"消费 MCP 工具"（Client 端）。这里讲反向的——**把你已有的 LangChain 工具暴露出去，让其他 Agent 通过 MCP 协议调用**。

```
你的 LangChain @tool ──→ 注册到 FastMCP Server ──→ 其他 Agent 通过 MCP 调用
```

**使用场景**：
- 团队 A 维护了一套 LangChain 工具 → 暴露为 MCP → 团队 B 的 Agent 直接调用
- 微服务架构中，每个服务暴露自己的 MCP Server → 一个统一的 Agent 聚合所有服务

#### 完整实现

```python
# ================================================================
# expose_as_mcp.py — 把 LangChain 工具反向暴露为 MCP 服务
# ================================================================
from mcp.server.fastmcp import FastMCP
from langchain.tools import tool

# === 1. 定义你已有的 LangChain 工具 ===
# 这些工具可能已经在你项目中使用了很久，不需要修改

@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式。

    支持四则运算、幂运算（**）、取余（%）。
    示例: '2 + 3 * 4', '10 ** 3', '100 % 7'。
    返回计算结果。
    """
    try:
        # 安全限制：只允许数字、运算符、括号、空格
        allowed = set("0123456789+-*/().% **")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符。"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    搜索内部知识库。

    参数 query: 搜索查询（支持自然语言）
    参数 top_k: 返回结果数量（1~20）
    返回匹配的文档标题和摘要。
    """
    # 模拟知识库搜索
    results = [
        ("Python 编码规范", "本文档定义了公司 Python 代码风格..."),
        ("部署手册 v3.2", "生产环境部署步骤：1. 构建镜像..."),
        ("API 文档", "REST API 接口说明：base_url=/api/v1/..."),
    ]
    matched = [f"• {title}: {snippet[:60]}..." for title, snippet in results[:top_k]]
    return f"搜索 '{query}' 的结果（共 {len(matched)} 条）：\n" + "\n".join(matched)

# === 2. 创建 MCP Server 并注册工具 ===
# 关键点：LangChain @tool → MCP tool 的转换
# add_tool() 需要的参数：
#   fn:         工具的实际执行函数（calculate.func → 去掉 @tool 装饰器后的原始函数）
#   name:       工具名称（LLM 用这个名称调用）
#   description: 工具描述（LLM 据此判断何时调用，使用 LangChain 工具的 docstring）

mcp = FastMCP("Internal Tool Service")

# 注册 calculate 工具
mcp.add_tool(
    fn=calculate.func,            # ← .func 是原始函数（去掉 @tool 包装）
    name="calculate",
    description=calculate.description,  # ← 沿用 LangChain @tool 的 docstring
)

# 注册 search_knowledge_base 工具
mcp.add_tool(
    fn=search_knowledge_base.func,
    name="search_knowledge_base",
    description=search_knowledge_base.description,
)

# 也可以手动注册一个未用 @tool 装饰的普通函数
def get_system_time() -> str:
    """获取系统当前时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

mcp.add_tool(
    fn=get_system_time,
    name="get_system_time",
    description="获取服务器当前系统时间。不需要参数。",
)

# === 3. 启动 MCP Server ===
if __name__ == "__main__":
    print("🚀 启动 Internal Tool MCP Server...")
    mcp.run(transport="stdio")
    # 其他 Agent 现在可以通过 MCP 协议调用 calculate / search_kb / get_system_time
```

---

### 6.6 远程 MCP 服务 + 本地混合配置

#### 架构场景

生产环境中，不是所有 MCP Server 都跑在本地。典型的混合架构：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Agent 所在的机器                            │
│                                                                 │
│  ┌──────────────────────┐    网络     ┌──────────────────────┐  │
│  │  本地 MCP Server      │            │  远程 MCP Server      │  │
│  │  weather_server.py    │           │  search-api.example.com│  │
│  │  file_server.py       │           │  db-api.internal.com   │  │
│  │  (通过 stdio 连接)     │           │  (通过 HTTP 连接)       │  │
│  └──────────────────────┘            └──────────────────────┘  │
│           │                                     │               │
│           └────────────┬────────────────────────┘               │
│                        ▼                                        │
│               ClientSessionGroup                                │
│               （统一聚合所有工具）                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 完整实现

```python
# ================================================================
# hybrid_mcp_agent.py — 本地 + 远程 MCP 统一接入
# ================================================================
import asyncio, contextlib, os
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
from mcp.client.session_group import ClientSessionGroup
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp import MCPToolkit
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# === 1. 定义所有 MCP Server 配置 ===
# 本地 Server 用 StdioServerParameters（启动子进程）
# 远程 Server 用 HTTP URL（通过 HTTP/SSE 连接）
SERVER_CONFIGS = {
    # ─── 本地 MCP 服务（stdio）───
    # StdioServerParameters 会让 Client 启动一个子进程并建立 stdio 通信
    "weather": StdioServerParameters(
        command="python",
        args=["weather_server.py"],
        # 可选：传环境变量
        # env={"API_KEY": os.getenv("WEATHER_API_KEY", "")},
    ),
    "calculator": StdioServerParameters(
        command="python",
        args=["calculator_server.py"],
    ),

    # ─── 远程 MCP 服务（HTTP）───
    # 远程 Server 已经以 HTTP 方式运行在别的机器上
    # URL 指向它的 streamable_http_path（默认 /mcp）
    "enterprise_search": "https://search-api.internal.company.com/mcp",
    "user_database": "https://user-db.internal.company.com/mcp",
}


async def create_hybrid_mcp_agent():
    """
    创建混合 MCP Agent。

    统一处理本地和远程 MCP Server：
      - 本地（StdioServerParameters）→ 用 stdio_client 建立连接
      - 远程（HTTP URL）           → 用 streamablehttp_client 建立连接
      - 所有工具自动聚合，前缀防重名
    """
    all_tools = []
    exit_stack = contextlib.AsyncExitStack()

    for name, config in SERVER_CONFIGS.items():
        if isinstance(config, StdioServerParameters):
            # =========================================
            # 本地 MCP Server：启动子进程 + stdio 连接
            # =========================================

            # 第 1 步：建立 stdio 传输通道
            # stdio_client(config) 内部：
            #   ① 启动子进程
            #   ② 返回 (read_stream, write_stream) 双向通信对
            transport = await exit_stack.enter_async_context(
                stdio_client(config)
            )

            # 第 2 步：在传输通道上创建 MCP 会话
            session = await exit_stack.enter_async_context(
                ClientSession(transport[0], transport[1])
            )

            # 第 3 步：通过 MCPToolkit 获取工具
            toolkit = MCPToolkit(session=session)
            await toolkit.initialize()

        elif isinstance(config, str) and config.startswith("http"):
            # =========================================
            # 远程 MCP Server：HTTP/SSE 连接
            # =========================================

            # streamablehttp_client 返回异步生成器
            # 每次迭代返回 (read_stream, write_stream, get_id)
            gen = streamablehttp_client(
                url=config,
                headers={
                    "Authorization": f"Bearer {os.getenv('MCP_AUTH_TOKEN', '')}",
                    "X-Client-Version": "1.0.0",
                },
                timeout=30.0,              # 请求超时
                sse_read_timeout=300.0,     # 长连接读取超时（Agent 可能长时间等待）
                terminate_on_close=True,    # 连接关闭时通知服务端
            )
            transport = await exit_stack.enter_async_context(gen)

            session = await exit_stack.enter_async_context(
                ClientSession(transport[0], transport[1])
            )
            await session.initialize()
            toolkit = MCPToolkit(session=session)
            await toolkit.initialize()

        else:
            print(f"⚠️ 跳过未知配置类型: {name}")
            continue

        # === 加前缀防重名 ===
        # 不同 MCP Server 可能有同名工具（如两个 Server 都有 search）
        # 加前缀后：weather_search 和 enterprise_search 不会冲突
        for t in toolkit.get_tools():
            t.name = f"{name}_{t.name}"
            all_tools.append(t)

        print(f"  ✅ [{name}] 加载了 {len(toolkit.get_tools())} 个工具")

    # === 创建 Agent ===
    llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0.7)
    agent = create_agent(
        llm=llm,
        tools=all_tools,
        system_prompt=(
            "你是企业智能助手，拥有天气查询、计算、搜索和数据库访问能力。"
            "需要查询信息时主动使用可用工具。"
        ),
    )

    print(f"\n📦 共加载 {len(all_tools)} 个 MCP 工具（来自 {len(SERVER_CONFIGS)} 个 Server）")
    return agent, exit_stack


async def main():
    agent, exit_stack = await create_hybrid_mcp_agent()

    # 测试：Agent 能同时使用本地和远程工具
    config = {"configurable": {"thread_id": "hybrid_demo"}}
    result = agent.invoke(
        {"messages": [HumanMessage("北京今天天气怎么样？帮我算 15*8")]},
        config=config,
    )
    print(f"\n🤖 Agent: {result['messages'][-1].content}")

    # 清理资源
    await exit_stack.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

---

### 6.7 HTTP MCP 传输详解

#### 6.7.1 两种 HTTP 传输方式

MCP 协议支持两种基于 HTTP 的传输方式，不同场景选不同方式：

| | Streamable HTTP | SSE (Server-Sent Events) |
|---|---|---|
| **通信方向** | 双向（全双工） | 单向（服务端 → 客户端推送） |
| **连接模式** | 一个 POST + 一个 SSE 长连接 | 纯 SSE 长连接 |
| **适用场景** | Agent 需要收发流式数据 | 客户端只读推送 |
| **协议版本** | MCP 2024-11-05+ | MCP 早期版本 |
| **推荐度** | ★★★★ 推荐 | 遗留兼容 |

**Streamable HTTP 的工作原理**：

```
Client                                           Server
  │                                                 │
  │ ① POST /mcp (JSON-RPC initialize 请求)          │
  │ ──────────────────────────────────────────────→ │
  │                                                 │
  │ ② 200 OK (JSON-RPC initialize 响应)             │
  │    Header: Mcp-Session-Id: abc123               │
  │ ←────────────────────────────────────────────── │
  │                                                 │
  │ ③ POST /mcp (JSON-RPC tools/list 请求)          │
  │    Header: Mcp-Session-Id: abc123               │
  │ ──────────────────────────────────────────────→ │
  │                                                 │
  │ ④ 200 OK (工具列表 JSON-RPC 响应)                │
  │ ←────────────────────────────────────────────── │
  │                                                 │
  │ ⑤ POST /mcp (JSON-RPC tools/call 请求)          │
  │    Header: Mcp-Session-Id: abc123               │
  │    Accept: text/event-stream                    │
  │ ──────────────────────────────────────────────→ │
  │                                                 │
  │ ⑥ 200 OK (SSE 流式响应)                          │
  │    包含工具执行的流式结果                           │
  │ ← ─ ─ ─ SSE Stream ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    │
```

#### 6.7.2 配置参数详解

```python
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client(
    # ===== 连接参数 =====
    url="https://mcp-server.example.com/mcp",
    # ↑ MCP Server 的 HTTP 端点地址
    # 对应 FastMCP 的 streamable_http_path 参数（默认 "/mcp"）

    # ===== 认证参数 =====
    headers={
        "Authorization": f"Bearer {os.getenv('MCP_TOKEN')}",
        # ↑ 身份认证：Bearer Token（生产环境必须）
        "X-Client-ID": "agent-prod-01",
        # ↑ 客户端标识：便于服务端追踪和日志
        "X-Client-Version": "1.0.0",
        # ↑ 客户端版本：便于服务端做兼容性处理
    },

    # ===== 超时参数 =====
    timeout=30.0,
    # ↑ 单次 HTTP 请求的超时时间（秒）
    # 包括：建立连接 + 发送请求 + 接收响应头
    # 太短：正常慢请求被截断 → Agent 误判工具故障
    # 太长：故障时 Agent 等待过久 → 用户体验差
    # 推荐值：30s（REST API 通常的响应时间上限）

    sse_read_timeout=300.0,
    # ↑ SSE 流读取超时（秒）
    # SSE 是长连接——Server 持续推送事件
    # 工具执行可能需要几分钟（如大型数据库查询）
    # 这个超时控制的是"两次事件之间的最大间隔"
    # 推荐值：300s（5 分钟），让长时间工具执行有足够窗口

    # ===== 生命周期参数 =====
    terminate_on_close=True,
    # ↑ Client 关闭连接时是否通知 Server
    # True：发送关闭信号 → Server 清理会话资源 → 避免资源泄漏
    # False：直接断开 → Server 需要等超时才能清理（不推荐）

    # ===== 高级参数 =====
    # httpx_client_factory: 自定义 HTTP 客户端工厂
    #   → 可传入自定义 httpx.AsyncClient（带代理、自定义 TLS 等）
    # auth: httpx.Auth 对象
    #   → 除了 headers 中的 Bearer Token，也可用 httpx 原生的 Auth 机制
) as (read_stream, write_stream, get_session_id):
    # read_stream:  从 Server 接收消息（MemoryObjectReceiveStream）
    # write_stream: 向 Server 发送消息（MemoryObjectSendStream）
    # get_session_id: 返回当前会话 ID 或 None
    session = ClientSession(read_stream, write_stream)
    await session.initialize()
    tools = (await session.list_tools()).tools
```

#### 6.7.3 本地 HTTP 调试配置

在本地开发时，Server 和 Client 通常在同一台机器上。FastMCP 可以同时启动 HTTP 模式而非 stdio：

```python
# 开发/调试时用 HTTP 模式（可以 curl 测试）：
if __name__ == "__main__":
    mcp.run(transport="streamable-http")  # 默认监听 127.0.0.1:8000/mcp
    # 启动后可以通过 curl http://localhost:8000/mcp 测试

# 生产环境通常用 stdio（更安全，不需要暴露端口）：
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 6.8 MCP 标准 vs MultiServerMCP（ClientSessionGroup）

| 维度 | MCP 标准（单 Server） | MultiServerMCP（ClientSessionGroup） |
|---|---|---|
| **顶层结构** | 一个 `ClientSession` 连一个 Server | 一个 `ClientSessionGroup` 管理多个 `ClientSession` |
| **服务器配置** | 单个 `StdioServerParameters` | `List[StdioServerParameters | HTTP URL]` |
| **连接方式** | `stdio_client(params)` → 一对 transport | `group.connect_to_server(params)` → 内部管理所有连接 |
| **工具聚合** | 需要手动收集每个 Server 的工具 | 自动聚合：`group._tools` 包含所有 Server 的工具 |
| **命名冲突** | 不存在（只有一个 Server） | 需手动加前缀防止重名：`f"{server_name}_{tool_name}"` |
| **适用场景** | 单一外部服务（如只接 GitHub API） | Agent 需要多种能力（天气 + 搜索 + 数据库 + 文件） |
| **失败隔离** | 一个服务挂 → 全部不可用 | 一个服务挂 → 其他服务正常，可加 fallback |

---

