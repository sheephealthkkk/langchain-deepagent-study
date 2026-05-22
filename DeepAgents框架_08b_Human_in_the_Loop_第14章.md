## 第十四章：Human-in-the-Loop — 人工干预

### 14.1 什么场景需要人工干预

Agent 不是万能的。某些操作必须经过人类确认才能执行——这不是技术限制，而是业务规则和安全要求。

```
需要人工干预的典型场景：

┌─────────────────────────────────────────────────────────────┐
│ 场景                      示例                               │
├─────────────────────────────────────────────────────────────┤
│ 涉及金钱/资源              删除生产数据库记录                  │
│                           退款/转账操作                       │
│                           修改计费配置                        │
│                                                             │
│ 影响线上服务               重启生产服务器                      │
│                           修改 K8s Deployment                 │
│                           切换 DNS / 负载均衡                 │
│                                                             │
│ 不可逆操作                 删除用户数据（GDPR）                │
│                           强制合并/覆盖 Git 分支               │
│                           发送全员邮件通知                     │
│                                                             │
│ 合规要求                   修改审计日志保留策略                │
│                           变更安全组/防火墙规则               │
│                           导出敏感数据                        │
│                                                             │
│ 高风险代码执行             eval/exec 用户提供的代码             │
│                           数据库 Migration                    │
│                           SQL 写操作（UPDATE/DELETE）          │
└─────────────────────────────────────────────────────────────┘
```

### 14.2 底层机制 — LangGraph 的 `interrupt()` + `Command.resume`

DeepAgents 的 HITL 直接使用 LangGraph 的中断/恢复机制：

```python
# LangGraph 的中断机制（DeepAgents 内部调用的底层 API）

# 第 1 步：Agent 执行到需要审批的节点
# HumanInTheLoopMiddleware 在 before_model 钩子中调用：
from langgraph.types import interrupt

# interrupt() 会：
#   1. 暂停当前 Graph 执行
#   2. 将当前 State 通过 Checkpointer 持久化
#   3. 把 prompt 信息返回给调用方（人类审批者）
#   4. 等待外部 resume 信号
interrupt_value = interrupt({
    "action": "delete_record",
    "args": {"table": "users", "id": "123"},
    "message": "确认删除用户 123？此操作不可逆！",
})

# 第 2 步：人类审批后，外部程序调用：
from langgraph.types import Command
agent.invoke(
    Command(resume={"decision": "approve", "reason": "已核实"}),
    config=config,  # 同一个 thread_id
)

# 第 3 步：Graph 从暂停点恢复执行
# interrupt() 返回人类输入的值
# print(interrupt_value)  → {"decision": "approve", "reason": "已核实"}
# Agent 继续执行后续逻辑
```

**核心流转图**：

```
Agent 执行                    人类审批者
──────────                    ──────────

  正常推理
    │
    ▼
  决定执行工具
    │
    ▼
  HumanInTheLoopMiddleware
  检查：这个工具需要审批吗？
    ├─ 不需要 → 直接执行
    └─ 需要   → interrupt()
                  │
                  │  State 持久化到 Checkpointer
                  │  "等待人工决策"
                  │
                  ▼ (暂停) ──────────────→ 👤 收到审批通知
                                               │
                                          approve / edit / reject
                                               │
                  ◄────────────────────────────┘
                  │  Command(resume={...})
                  │  State 从 Checkpointer 恢复
                  ▼
  根据审批结果继续：
    approve → 执行工具 → 返回结果
    edit    → 修改参数 → 执行工具
    reject  → 跳过工具 → 返回拒绝说明
    respond → 人工直接回复 → 返回人工回复
```

### 14.3 `interrupt_on` — HITL 的配置开关

`interrupt_on` 是 DeepAgents 中把 HITL 插入执行流程的唯一入口。它是一个字典，key 是工具名，value 是配置。

```python
from deepagents import create_deep_agent
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig

agent = create_deep_agent(
    model="deepseek-v4-pro",
    tools=[...],
    # ★ interrupt_on: HITL 的开关
    interrupt_on={
        # === 用法 1：bool 值（简单开关）===
        "delete_record": True,
        # ↑ True = 启用审批，允许所有决策类型（approve/edit/reject/respond）

        "get_server_status": False,
        # ↑ False = 自动通过（和没配置一样，显式声明而已）

        # === 用法 2：InterruptOnConfig（精细控制）===
        "restart_service": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            # ↑ 只允许同意或拒绝，不允许编辑参数（防止改错服务名）
            description="⚠️ 重启服务 {tool_name}？参数: {tool_args}",
            # ↑ 审批时显示的描述信息，支持模板变量
        ),

        "send_email": InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"],
            # ↑ 允许编辑（修改邮件内容/收件人后再发送）
        ),

        "write_database": InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject", "respond"],
            # ↑ 全部允许，包括人工直接回复（respond = 不执行工具，人工写回答）
            description=lambda tool_call, state, runtime: (
                f"数据写入操作:\n"
                f"  表: {tool_call['args'].get('table', 'unknown')}\n"
                f"  操作: {tool_call['args'].get('operation', 'unknown')}\n"
                f"  操作人: {state.get('user_id', 'unknown')}\n"
                f"  时间: {datetime.now().isoformat()}"
            ),
        ),
    },
)
```

### 14.4 `interrupt_on` 的四种决策类型

| 决策类型 | 含义 | Agent 收到后的行为 |
|---|---|---|
| `approve` | 同意执行 | 正常执行工具 → 返回结果给 LLM |
| `edit` | 修改参数后执行 | 使用人类修改的参数执行工具 → 返回结果 |
| `reject` | 拒绝执行 | 跳过工具 → 返回 "操作被拒绝" 给 LLM |
| `respond` | 人工直接回复 | 不执行工具 → 把人类的回复直接作为工具结果 |

### 14.5 完整示例

```python
# ================================================================
# hitl_demo.py — Human-in-the-Loop 完整演示
# ================================================================
import asyncio
from datetime import datetime
from deepagents import create_deep_agent
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage

# ================================================================
# 第 1 步：创建带 HITL 的 Agent
# ================================================================

agent = create_deep_agent(
    model=ChatOpenAI(model="deepseek-v4-pro", temperature=0),

    # ★ 关键配置：哪些工具需要人工审批 ★
    interrupt_on={
        # delete_record: 只需要同意/拒绝（不可编辑——防止改错 ID）
        "delete_record": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            description=(
                "⚠️⚠️ 不可逆操作警告 ⚠️⚠️\n"
                "工具: delete_record\n"
                "此操作将永久删除数据，无法恢复。\n"
                "请仔细确认后决定。"
            ),
        ),

        # restart_service: 需要同意或拒绝
        "restart_service": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
        ),

        # send_email: 允许编辑（修改内容后再发）或人工直接回复
        "send_email": InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject", "respond"],
        ),

        # get_server_status: 自动通过（不需要审批）
        # 不配置 = 自动通过，不需要显式写 False
    },

    # ★ HITL 必须配置 Checkpointer ★
    # interrupt() 依赖 Checkpointer 保存暂停时的 State
    checkpointer=InMemorySaver(),  # 生产用 PostgresSaver
)

# ================================================================
# 第 2 步：模拟一次需要审批的调用
# ================================================================

config = {
    "configurable": {
        "thread_id": "hitl_demo_001",
        "user_id": "operator_zhang",
    }
}

# 用户请求：包含危险操作
result = agent.invoke(
    {"messages": [HumanMessage("帮我删除用户 123 的所有记录，他在我们的黑名单上")]},
    config=config,
)

# ================================================================
# 第 3 步：检查是否触发了中断
# ================================================================

state = agent.get_state(config)
# state.interrupts 包含中断信息
# 如果有中断 → Agent 已暂停，等待人类决策
# 如果没有中断 → Agent 已完成（可能是工具不需要审批，或被拒绝了）

if state.interrupts:
    # === Agent 暂停了，显示审批信息 ===
    interrupt_info = state.interrupts[0]  # 可能有多个中断
    
    print("=" * 60)
    print("⏸️  Agent 暂停 — 需要人工审批")
    print("=" * 60)
    print(f"工具: {interrupt_info.value['action']}")
    print(f"参数: {interrupt_info.value['args']}")
    print(f"消息: {interrupt_info.value.get('message', '请审批')}")
    print()
    print("可选决策: approve / reject")
    print("=" * 60)

    # ================================================================
    # 第 4 步：人类做出决策并恢复执行
    # ================================================================

    # 决策 A：同意删除
    await agent.ainvoke(
        Command(resume={
            "decision": "approve",
            "reason": "该用户在黑名单上，已确认需要删除",
        }),
        config=config,
    )
    # → Agent 恢复 → 执行 delete_record → 返回结果

    # 决策 B：拒绝（如果选择拒绝）
    # await agent.ainvoke(
    #     Command(resume={
    #         "decision": "reject",
    #         "reason": "需要先确认用户是否确实在黑名单上",
    #     }),
    #     config=config,
    # )
    # → Agent 恢复 → 跳过 delete_record → 返回 "操作被拒绝"

else:
    # === Agent 没有中断 → 不需要审批 ===
    print("✅ Agent 已完成，无需审批")
    if result and "messages" in result:
        print(result["messages"][-1].content[:300])

# ================================================================
# 第 5 步：验证审计结果
# ================================================================

# 读取最终 State
final_state = agent.get_state(config)
print(f"\n📋 最终消息数: {len(final_state.values['messages'])}")
# 最后一条消息应该包含操作结果（删除成功 / 被拒绝）
```

### 14.6 HITL 配合 LangSmith 的效果

在 LangSmith 中，HITL 的每一步都是可见的：

```
LangSmith Trace 视图：

  ▼ Agent (RunnableSequence)                         8.5s
    ├─ ▼ ChatOpenAI                          1.2s
    │    Input: [SystemMsg, HumanMsg("删除用户 123")]
    │    Output: AIMsg(tool_calls=[delete_record])
    │
    ├─ ▼ ToolNode (delete_record)            0.0s
    │    ⏸️ INTERRUPTED — 等待人工审批
    │
    │  (时间流逝...人类审批中...)
    │
    ├─ ▶ Resume — decision: "approve"        0.0s
    │
    ├─ ▼ ToolNode (delete_record)            0.3s
    │    ✅ 执行成功
    │
    └─ ▼ ChatOpenAI                          1.1s
         Output: "用户 123 的记录已成功删除"
```

**LangSmith + HITL 带来的能力**：

| 能力 | 说明 |
|---|---|
| 审批历史可视化 | 每次审批决策都记录在 Trace 中（谁/何时/什么决策） |
| 审计合规 | 满足 SOC2/GDPR 的审计追溯要求 |
| 审批耗时统计 | 人类响应时间 = 瓶颈指标，用于 SLA 监控 |
| 异常审批告警 | 统计 reject 率 → 过高说明 Agent 行为需要调整 |

### 14.7 要点总结

```
Human-in-the-Loop 的核心机制：

  底层 LangGraph API:
    interrupt(value)  ──→ 暂停 Graph + 持久化 State + 返回 value
    Command(resume=x) ──→ 恢复 Graph + interrupt() 返回 x

  DeepAgents 配置层:
    interrupt_on = {
        "tool_name": True,                          # 简单开关
        "tool_name": InterruptOnConfig(             # 精细控制
            allowed_decisions=["approve","reject"],
            description="...",
        ),
    }

  四种决策类型:
    approve  → 同意执行
    edit     → 改参数后执行
    reject   → 拒绝执行
    respond  → 人工直接回复（不执行工具）

  必须配合 Checkpointer:
    无 Checkpointer → interrupt() 无法保存状态 → HITL 不可用

  典型工作流:
    Agent 推理 → 决定调危险工具 → interrupt 暂停
    → 人类审批 → Command(resume) 恢复 → 执行/拒绝
```

---

