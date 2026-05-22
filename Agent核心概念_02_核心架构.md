## 第二章：Agent 核心架构 = LLM + Tools + 思考循环

### 2.1 一句话架构

**Agent = LLM（大脑）+ Tools（手脚）+ Think-Act-Observe Loop（神经回路）**

```
                     ┌──────────┐
                     │   LLM    │  ← 大脑：推理、决策
                     │ (大脑)    │
                     └────┬─────┘
                          │ 思考 → 决定调用哪个工具
                          ▼
              ┌──────────────────────┐
              │     Think-Act-       │
              │   Observe Loop       │  ← 神经回路：循环
              │    (思考→行动→观察)   │
              └──────────┬───────────┘
                         │ 执行 → 拿到结果
                         ▼
              ┌──────────────────────┐
              │       Tools          │  ← 手脚：查天气、搜索、计算
              │  (查天气/搜索/计算)    │
              └──────────────────────┘
```

### 2.2 Think → Act → Observe 循环

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent 执行循环                            │
│                                                             │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐              │
│   │  THINK  │ ──→ │   ACT   │ ──→ │ OBSERVE │              │
│   │  思考   │     │  行动   │     │  观察   │              │
│   └─────────┘     └─────────┘     └─────────┘              │
│        ↑                                  │                │
│        └──────────── 循环 ────────────────┘                │
│                                                             │
│   每一步：                                                   │
│   THINK:  LLM 分析当前状态 → 判断"下一步该干什么"             │
│           → 要么调用工具（输出 tool_calls）                  │
│           → 要么给出最终回答（输出 content）                  │
│                                                             │
│   ACT:    框架执行 LLM 指定的工具调用                        │
│           → 把 tool_calls 转为实际函数执行                   │
│                                                             │
│   OBSERVE: 把工具返回结果追加到消息列表                      │
│           → Agent 看到新信息，回到 THINK                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 具体执行流程

用一个实际例子——用户问 "北京今天天气怎么样？适合户外运动吗？"：

```
═══════════════════════════════════════════════════════════════
循环开始
═══════════════════════════════════════════════════════════════

[THINK 第1轮]
  消息列表: [System("你是助手"), Human("北京今天天气怎么样？适合户外运动吗？")]
  LLM 推理: "我需要先获取北京的天气数据"
  LLM 决策: → tool_calls=[{name: "get_weather", args: {city: "北京"}}]
            → content=None（这次不直接回答）

[ACT]
  框架执行: get_weather(city="北京")
  返回: "北京：晴，25°C，湿度 45%，风速 2级"

[OBSERVE]
  追加: ToolMessage("北京：晴，25°C，湿度 45%，风速 2级", tool_call_id="c1")
  消息列表现在有 4 条消息了

═══════════════════════════════════════════════════════════════

[THINK 第2轮]
  消息列表: [System, Human, AIMessage(tool_calls=[...]), ToolMessage("北京：晴...")]
  LLM 推理: "拿到天气了。晴，25°C，微风，湿度适中"
  LLM 决策: "这些条件很适合户外运动！可以给用户具体建议"
  LLM 输出: → content="北京今天晴天，25°C，微风，非常适合户外运动！
                    建议去公园跑步、爬山或者骑行。注意防晒！"
            → tool_calls=None（不再需要工具）

  检测到 LLM 没有再要求调工具 → 循环结束！

═══════════════════════════════════════════════════════════════
循环结束 → 返回最终回答
═══════════════════════════════════════════════════════════════
```

### 2.4 循环终止条件

Agent 什么时候停止循环？四种情况：

```python
# 1. LLM 不再输出 tool_calls，只输出 content → 自然终止
AIMessage(content="北京今天晴天...", tool_calls=None)

# 2. 达到最大循环次数（max_tool_calls 中间件）
ToolCallLimit(max_tool_calls=10)  # 最多调用 10 次工具

# 3. 达到递归限制（RunnableConfig）
config={"recursion_limit": 25}

# 4. LLM 自己判断"任务完成"，显式结束
AIMessage(content="已完成所有任务。", tool_calls=[])
```

### 2.5 从代码看循环

```python
from langchain.agents import create_agent

# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}：晴，25°C"

@tool
def check_pollution(city: str) -> str:
    """获取空气质量"""
    return f"{city}：AQI 45，优"

# 创建 Agent
agent = create_agent(
    llm=ChatOpenAI(model="deepseek-v4-pro"),
    tools=[get_weather, check_pollution],
    system_prompt="你是生活助手。需要查询信息时主动调用工具。",
)

# 一次 invoke，内部自动完成 Think→Act→Observe 循环
result = agent.invoke({
    "messages": [
        HumanMessage("北京今天适合户外运动吗？")
    ]
})

# 用户只看到最终回答，循环过程对用户透明
print(result["messages"][-1].content)
# → "北京今天晴天 25°C，空气质量优，非常适合户外运动！建议..."
```

---

