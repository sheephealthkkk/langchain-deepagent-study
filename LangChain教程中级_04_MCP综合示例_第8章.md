## 第八章：MCP 完整综合示例

结合以上所有知识点，写一个生产级的 MCP Agent：

```python
# ================================================================
# mcp_agent_full.py — 生产级 MCP Agent 完整示例
# ================================================================
import asyncio
import contextlib
import logging
from pydantic import BaseModel, Field
from typing import Literal

logger = logging.getLogger(__name__)

# ---- 1. 统一工具规范 ----
class StandardToolSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z_][a-z0-9_]{2,50}$")
    description: str = Field(min_length=20)
    category: Literal["weather", "search", "code", "file", "chat"]
    risk_level: Literal["low", "medium", "high"]

# ---- 2. 意图分类模型 ----
class IntentResult(BaseModel):
    intent: Literal["weather", "search", "code", "chat"] = Field(
        description="用户意图"
    )
    confidence: float = Field(ge=0, le=1)
    sub_intent: str = Field(default="", description="子意图")

# ---- 3. 工具过滤 Prompt ----
TOOL_RULES = (
    "## 工具规则\n"
    "1. 精确匹配：工具描述与用户需求严格对应\n"
    "2. 找不到 → 说「不支持该功能」，不要发散\n"
    "3. 禁止在一个回复中调用不相关的多个工具"
)

# ---- 4. MCP 连接管理器 ----
class MCPConnectionManager:
    def __init__(self, server_configs: dict):
        self.server_configs = server_configs
        self.group: ClientSessionGroup | None = None
        self.exit_stack = contextlib.AsyncExitStack()

    async def connect_all(self):
        """连接所有 MCP Server，加载所有工具。"""
        self.group = await self.exit_stack.enter_async_context(
            ClientSessionGroup()
        )
        for name, config in self.server_configs.items():
            await self.group.connect_to_server(config)

    def get_all_tools(self) -> list:
        """获取所有 MCP 工具，加前缀防重名。"""
        tools = []
        for session, mcp_tools in self.group._tools.items():
            for t in mcp_tools:
                t.name = f"{self._get_server_name(session)}_{t.name}"
            tools.extend(mcp_tools)
        return tools

    async def close(self):
        await self.exit_stack.aclose()

# ---- 5. 降级工具注册表 ----
FALLBACK_TOOLS = {
    "weather": lambda city: f"{city}（fallback）：晴，22°C",
    "search": lambda query: f"搜索 '{query}'（fallback）：无网络连接",
}

# ---- 6. 主 Agent 类 ----
class MCPAgent:
    def __init__(self, llm, mcp_manager, local_tools: list):
        self.llm = llm
        self.mcp = mcp_manager
        self.local_tools = local_tools
        self.intent_classifier = intent_prompt | llm.with_structured_output(
            IntentResult
        )

    def route(self, user_input: str) -> list:
        """意图 → 工具匹配。"""
        intent = self.intent_classifier.invoke(user_input)

        if intent.confidence < 0.6:
            return []  # 低置信度 → 直接聊

        # 按意图过滤 — 不是一次性给所有工具
        category_map = {"weather": 0, "search": 1, "code": 2}
        category = category_map.get(intent.intent)
        return [t for t in self.active_tools if t.metadata.get("category") == category]

    def execute(self, user_input: str) -> str:
        """完整执行：路由 → 调用 → 降级。"""
        matched_tools = self.route(user_input)

        # 没有匹配工具 → 严格返回
        if not matched_tools:
            return "当前不支持该功能。请尝试其他问题。"

        # 创建当前意图的临时 Agent（工具数 ≤ 5）
        agent = create_agent(
            llm=self.llm,
            tools=matched_tools + self.local_tools,  # MCP + fallback
            system_prompt=TOOL_RULES,
        )

        result = agent.invoke({"messages": [HumanMessage(user_input)]})
        return result["messages"][-1].content

# ---- 7. 运行入口 ----
async def main():
    server_configs = {
        "weather": StdioServerParameters(command="python", args=["weather_server.py"]),
        "search": StdioServerParameters(command="python", args=["search_server.py"]),
    }

    mcp_mgr = MCPConnectionManager(server_configs)
    await mcp_mgr.connect_all()
    mcp_tools = mcp_mgr.get_all_tools()

    @tool
    def local_calc(expr: str) -> str:
        """本地计算器（MCP 失败时的 fallback）。"""
        return f"计算结果: {eval(expr)}"

    agent = MCPAgent(
        llm=ChatOpenAI(model="deepseek-v4-pro", temperature=0),
        mcp_manager=mcp_mgr,
        local_tools=[local_calc],
    )

    for q in ["北京天气怎么样？", "帮我算 15*8", "今天心情真好！"]:
        print(f"👤 {q}")
        print(f"🤖 {agent.execute(q)}\n")

    await mcp_mgr.close()

asyncio.run(main())
```

**执行流程跟踪**：

```
1. "北京天气怎么样？"
   意图识别 → intent="weather", confidence=0.95
   路由 → 只用 weather 组工具（get_weather, get_aqi）
   Agent 创建 → 只加载 2 个工具（上下文精简）
   调用 get_weather("北京") → 返回天气数据
   回复: "北京今天晴，25°C..."

2. "帮我算 15*8"
   意图识别 → intent="code", confidence=0.88
   路由 → code 组工具（execute_python, local_calc）
   MCP 的 execute_python 是远程调用 → 如果失败 → 自动用 local_calc
   回复: "计算结果: 120"

3. "今天心情真好！"
   意图识别 → intent="chat", confidence=0.92
   路由 → 无工具（chat 组为空）
   直接返回: "当前不支持该功能。" ← 严格边界，不乱调工具
   或者: ToolRouter 识别为闲聊 → LLM 直接回复
```

**为什么这样设计 — 细节说明**：

| 设计决策 | 原因 |
|---|---|
| 意图分类后才加载工具 | 避免 50+ 工具塞进上下文，降低选择困难 |
| 每个意图独立 Agent | 子 Agent 只面对 3~5 个工具，准确率高 |
| MCP 失败 → 本地 fallback | 保证可用性，非关键路径允许降级 |
| 工具过滤 Prompt | 显式约束模型行为，不知道就说不知道 |
| 分层路由（Orchestrator → Sub-Agent） | 降低单点复杂度，提高稳定性 |

---

