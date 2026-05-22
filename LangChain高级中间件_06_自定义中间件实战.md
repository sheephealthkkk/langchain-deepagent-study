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

