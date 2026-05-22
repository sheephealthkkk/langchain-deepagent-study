## 第七章：进阶 — ToolRouter 与工具调用的可靠性

### 7.1 问题：为什么 Agent 会"乱调工具"

```
用户: "今天心情真好！"
Agent: → 调用 get_weather     ← 用户没问天气，Agent 乱调
Agent: → 调用 search_web      ← 用户没让搜索
Agent: → 调用 delete_user     ← 危险！用户没授权删除

根本原因：LLM 看到一堆工具，根据概率选了一个——但选错了。
```

### 7.2 ToolRouter：意图驱动的工具调度

```
                    用户输入
                       │
                       ▼
              ┌────────────────┐
              │   意图识别      │  ← "用户想干什么？"
              │  (分类模型/LLM) │
              └───────┬────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
     ┌────────┐ ┌────────┐ ┌────────┐
     │ 天气   │ │ 搜索   │ │ 闲聊   │
     │ tools  │ │ tools  │ │ (直接  │
     │        │ │        │ │  回答) │
     └────────┘ └────────┘ └────────┘
```

### 7.3 完整 ToolRouter 实现

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent

# === 第1步：意图识别 ===
class Intent(BaseModel):
    """意图分类结果"""
    intent: Literal["weather", "search", "calculation", "chat"] = Field(
        description="用户意图类型"
    )
    confidence: float = Field(description="置信度", ge=0, le=1)

intent_chain = intent_prompt | llm.with_structured_output(Intent)

# === 第2步：工具分组（每个意图对应一组工具）===
tool_groups = {
    "weather": [get_weather, get_air_quality],
    "search": [search_web, search_database, search_arxiv],
    "calculation": [calculate, convert_units],
    "chat": [],   # 闲聊不需要工具
}

# === 第3步：ToolRouter 核心逻辑 ===
class ToolRouter:
    """意图驱动的工具路由器。"""

    def __init__(self, llm, tool_groups: dict[str, list], intent_chain):
        self.llm = llm
        self.tool_groups = tool_groups
        self.intent_chain = intent_chain

    def route(self, user_input: str) -> dict:
        """根据意图选择合适的工具组。"""
        # 1. 意图识别
        intent_result = self.intent_chain.invoke(user_input)

        # 2. 置信度阈值过滤 — 低置信度默认当闲聊
        if intent_result.confidence < 0.6:
            return {
                "intent": "chat",
                "tools": [],
                "reason": f"置信度 {intent_result.confidence} < 0.6，降级为闲聊"
            }

        # 3. 找到对应工具组
        tools = self.tool_groups.get(intent_result.intent, [])

        # 4. 如果意图匹配但没有工具 → 告知用户
        if not tools:
            return {
                "intent": intent_result.intent,
                "tools": [],
                "reason": f"意图 '{intent_result.intent}' 没有对应工具，直接回复"
            }

        return {"intent": intent_result.intent, "tools": tools}

    def execute(self, user_input: str) -> str:
        """完整执行流程。"""
        route_result = self.route(user_input)

        if not route_result["tools"]:
            # 无工具可用 → LLM 直接回答
            return self.llm.invoke(user_input).content

        # 创建临时 Agent，只用当前组的工具
        agent = create_agent(
            llm=self.llm,
            tools=route_result["tools"],
            system_prompt=f"当前意图: {route_result['intent']}。只用可用工具回答。",
        )
        result = agent.invoke({"messages": [HumanMessage(user_input)]})
        return result["messages"][-1].content

# 使用
router = ToolRouter(llm, tool_groups, intent_chain)
response = router.execute("北京今天天气怎么样？")
# → 路由到 weather 组 → 调用 get_weather → 返回天气信息

response = router.execute("今天心情真好！")
# → 路由到 chat（无工具）→ LLM 直接回复，不会乱调工具
```

### 7.4 ToolRouter 的策略细节

**策略 1：意图识别 — 高精度分类**

```python
class PreciseIntent(BaseModel):
    intent: Literal["data_query", "code_gen", "file_op", "comm", "chat"] = Field(
        description="细分意图类型"
    )
    sub_intent: str = Field(description="子意图，如 'weather', 'stock'")
    requires_tools: bool = Field(description="是否需要工具")
    reason: str = Field(description="分类理由")

# 多维度分析
intent = llm.with_structured_output(PreciseIntent).invoke(user_input)
```

**策略 2：工具匹配 — 精确到子组**

```python
# 每个意图不再只有一个工具组，而是动态匹配
def match_tools(intent_result):
    # 1. 先按 intent 粗筛
    candidates = tool_groups.get(intent_result.intent, [])

    # 2. 再按 sub_intent 精筛
    if intent_result.sub_intent:
        candidates = [t for t in candidates
                      if intent_result.sub_intent.lower() in t.description.lower()
                      or intent_result.sub_intent.lower() in t.name.lower()]

    # 3. 如果精筛后为空 → 回退到粗筛结果
    return candidates or tool_groups.get(intent_result.intent, [])
```

**策略 3：参数验证 + 错误处理**

```python
def safe_tool_call(tool, args: dict) -> str:
    """安全的工具调用包装。"""
    try:
        # 参数验证
        schema = tool.args_schema
        if schema:
            validated = schema(**args)  # Pydantic 自动校验
            args = validated.model_dump()

        # 执行
        result = tool.invoke(args)
        return result

    except ValidationError as e:
        # 参数不对 → 告诉 LLM 纠正
        return f"参数错误: {e}。请检查参数格式后重试。"

    except Exception as e:
        # 执行失败 → 建议
        return f"工具执行失败: {e}。建议尝试替代方案或简化参数。"
```

**策略 4：严格返回 — 没找到就是没找到**

```python
NO_TOOL_RESPONSE = (
    "我没有找到处理这个请求的工具。请尝试以下操作:\n"
    "1. 用更具体的关键词重新描述你的需求\n"
    "2. 检查请求是否在我支持的功能范围内\n"
    "3. 如果是闲聊，我会直接回答你"
)

# 在路由结果中
if not matched_tools:
    return {"response": NO_TOOL_RESPONSE, "tools": []}
```

### 7.5 动态工具加载 — 避免上下文过长

**问题**：工具太多（100+）→ 全部塞进 Prompt → 上下文爆炸 + 模型选择困难。

```python
class DynamicToolLoader:
    """按需加载工具，避免上下文过长。"""

    def __init__(self, llm, tool_registry: dict):
        self.llm = llm
        self.tool_registry = tool_registry  # {category: [tools]}
        self.active_tools: list = []

    def load_for_intent(self, intent: str, max_tools: int = 5) -> list:
        """根据意图动态加载工具。"""
        # 1. 粗筛：按类别过滤
        candidates = self.tool_registry.get(intent, [])

        # 2. 排序：按使用频率 + 最近使用时间
        candidates = sorted(candidates, key=lambda t: t.use_score, reverse=True)

        # 3. 截断：只取 top-N
        self.active_tools = candidates[:max_tools]
        return self.active_tools

    def unload(self):
        """释放工具，清空上下文。"""
        self.active_tools.clear()
```

**三种加载策略**：

| 策略 | 做法 | 适用 |
|---|---|---|
| **意图驱动** | 先识别意图 → 只加载该意图的工具组 | 通用 |
| **分页加载** | 先加载 top-5，不够再加载下 5 个 | 工具数量 50+ |
| **关键词匹配** | 用户输入的关键词与工具 description 做向量匹配 | 工具数量 100+ |

### 7.6 统一工具规范 — 好的 Schema 长什么样

```python
class StandardToolSpec(BaseModel):
    """所有工具必须遵循的规范。"""

    name: str = Field(
        pattern=r"^[a-z_][a-z0-9_]{2,50}$",  # 命名规范
        description="工具名：小写字母+下划线，2~50字符",
    )
    description: str = Field(
        min_length=20,   # 至少 20 字符，必须有足够说明
        description="必须包含：功能、适用场景、输入参数说明、返回值说明、限制条件",
    )
    category: Literal["data", "code", "file", "comm", "system"] = Field(
        description="工具分类，用于动态加载",
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="风险等级：high/critical 级别需 Human-in-the-Loop",
    )
    version: str = Field(default="1.0.0", description="版本号")
```

**一个好 Schema 的强制要求**：

```
1. name: 全局唯一，语义化（动词_名词）
2. description: ≥20 字符，包含 5 要素（功能/场景/输入/输出/限制）
3. args_schema: 每个参数都有 description，复杂参数有 examples
4. category: 用于动态加载分组
5. risk_level: 用于权限控制（low 自动执行，high 需确认）
```

### 7.7 "工具过滤 Prompt" — 修饰模型行为

```python
TOOL_FILTER_PROMPT = (
    "## 工具使用规则（严格遵守）\n\n"
    "1. **按需调用**：只在需要外部信息或执行操作时才调用工具。\n"
    "2. **匹配意图**：工具的功能必须与用户需求精确匹配。\n"
    "   - 用户问天气 → 只用 weather 相关工具\n"
    "   - 用户让搜索 → 只用 search 相关工具\n"
    "   - 用户闲聊 → 不调用任何工具，直接回复\n"
    "3. **严禁发散**：\n"
    "   - 如果找不到匹配的工具，直接回复「当前不支持该功能」\n"
    "   - 不要猜测、不要勉强调用、不要张冠李戴\n"
    "4. **边界清晰**：\n"
    "   - 工具能做什么就做什么，不能做的不要编造\n"
    "   - 工具返回什么就用什么，不要添加额外信息\n"
)

agent = create_agent(
    llm=llm,
    tools=tools,
    system_prompt=TOOL_FILTER_PROMPT,  # ← 注入工具使用规则
)
```

### 7.8 层次化多级 Agent — 降低单 Agent 复杂度

```
        ┌───────────────────┐
        │   Orchestrator    │  ← 第1级：总调度，分配任务
        │   (调度 Agent)    │      用 5 个工具组（非具体工具）
        └──────┬────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌──────┐ ┌──────┐ ┌──────┐
│天气  │ │搜索  │ │代码  │      ← 第2级：子 Agent
│Agent │ │Agent │ │Agent │      每个只有 3~5 个具体工具
└──────┘ └──────┘ └──────┘
   3个工具   4个工具   5个工具
```

```python
class HierarchicalAgent:
    """层次化多级 Agent。"""

    def __init__(self, llm):
        # 第1级：Orchestrator — 只做分发，不直接调工具
        self.orchestrator = create_agent(
            llm=llm,
            tools=[self._dispatch_to_sub_agent],  # 唯一的"工具"：分发
            system_prompt="你是总调度。分析用户需求，分发给对应的子 Agent。",
        )

        # 第2级：子 Agent — 每个只有少量工具
        self.sub_agents = {
            "weather": create_agent(llm=llm, tools=[get_weather, get_aqi]),
            "search": create_agent(llm=llm, tools=[search_web, search_arxiv]),
            "code": create_agent(llm=llm, tools=[execute_python, format_code]),
        }

    def execute(self, user_input: str) -> str:
        """执行：Orchestrator 分发 → 子 Agent 执行 → 汇总返回。"""
        dispatch = self.orchestrator.invoke(
            {"messages": [HumanMessage(user_input)]}
        )

        # 解析 Orchestrator 的分发决定
        target = self._parse_dispatch(dispatch["messages"][-1].content)
        if target in self.sub_agents:
            result = self.sub_agents[target].invoke(
                {"messages": [HumanMessage(user_input)]}
            )
            return result["messages"][-1].content

        return dispatch["messages"][-1].content  # 直接回答
```

**层次化收益**：单 Agent 面对 20 个工具 → 易混淆。拆分后每个子 Agent 只面对 3~5 个 → 准确率大幅提升。

---

