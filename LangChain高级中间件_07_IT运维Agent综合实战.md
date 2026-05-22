## 第十章：多中间件组合实战 — IT 运维 Agent

### 10.1 场景概述

构建一个企业级 IT 运维 Agent，集成 7 个中间件 + RBAC 权限体系。完整代码见 `it_ops_agent.py`。

**角色定义**：

| 角色 | 权限范围 | 典型用户 |
|---|---|---|
| `admin` | 所有权限（读写删 + 用户管理） | IT 主管 |
| `operator` | 运维操作（查状态/重启/查日志/写数据库/看指标） | 运维工程师 |
| `viewer` | 只读（查状态/看日志/看指标） | 数据分析师 |
| `auditor` | 只读 + 审计日志 | 合规审计员 |

**可用运维工具**：`get_server_status`, `restart_service`, `view_logs`, `get_system_metrics`, `query_database`, `write_database`, `delete_record`

### 10.2 中间件执行流程

```
                    用户请求
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ [1] before_agent（Agent 启动前，只执行一次）                    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SecurityCheckMiddleware                             │    │
│  │   ├─ 解析 JWT/Token → UserContext                    │    │
│  │   ├─ 验证用户身份有效性                              │    │
│  │   ├─ RBAC: 角色 → 权限列表查询                       │    │
│  │   └─ 将 user_id/role/permissions 注入 AgentState    │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ AuditLogMiddleware.before_agent                      │    │
│  │   └─ 记录操作开始：用户 + 角色 + 时间戳               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ [2] wrap_model_call（每次 LLM 调用前，洋葱模型）               │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ DynamicPromptMiddleware（最外层）                     │    │
│  │   └─ 根据 user_role 选择 Prompt 注入 SystemMessage   │    │
│  │       admin → 决策果断、强调安全审计                   │    │
│  │       operator → 按手册执行、先读后写                   │    │
│  │       viewer → 数据分析、引导联系运维                   │    │
│  │       auditor → 合规审计、标记不合规                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SmartModelSwitchMiddleware                           │    │
│  │   └─ 分析问题复杂度 → 切换 premium/budget 模型         │    │
│  │       危险操作 / 复杂分析 / 长问题 → premium           │    │
│  │       简单查询 → budget（省钱）                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ContextManagementMiddleware（最内层）                 │    │
│  │   └─ Token 超阈值 → 裁剪旧消息                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ [3] 模型执行（LLM API 调用）                                   │
│                                                              │
│   LLM 收到: 角色 Prompt + 裁剪后的消息 + 工具列表               │
│   → 决定调用哪些工具 / 直接回答                                │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ [4] after_model（每次 LLM 调用后）                             │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ResponseValidationMiddleware                         │    │
│  │   └─ 检查 LLM 输出是否泄露敏感信息（密码/Token/密钥）  │    │
│  │       泄露 → 拦截 → 返回安全提示                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ AuditLogMiddleware.after_model                       │    │
│  │   └─ 记录操作完成：工具调用次数 + 耗时 + 结果摘要     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ [5] wrap_tool_call（工具调用前）                               │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SafetyGuardrailMiddleware                            │    │
│  │   ├─ 权限校验：用户是否有执行此工具的权限？             │    │
│  │   │   无权限 → 拦截 → 返回权限不足提示                 │    │
│  │   ├─ 危险操作确认：restart/delete/write → 审计标记    │    │
│  │   └─ 通过检查 → 正常执行工具                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 10.3 RBAC 设计详解

#### 为什么需要 RBAC

IT 运维系统的核心安全问题是**"谁能做什么"**。没有 RBAC → 任何一个用户都可以重启服务器、删除数据库 → 灾难。

#### 三层映射结构

```
用户 (UserContext)
  │
  └─→ 角色 (Role: admin/operator/viewer/auditor)
        │
        └─→ 权限集合 (Permission: server:read, server:restart, ...)
              │
              └─→ 可调用的工具 (Tool: get_server_status, restart_service, ...)
```

#### 代码中的实现

```python
# 第一层：角色 → 权限
ROLE_PERMISSIONS = {
    "admin":     {SERVER_READ, SERVER_RESTART, DB_READ, DB_WRITE, DB_DELETE, ...},
    "operator":  {SERVER_READ, SERVER_RESTART, DB_READ, DB_WRITE, ...},
    "viewer":    {SERVER_READ, LOG_VIEW, METRICS_VIEW},
    "auditor":   {SERVER_READ, LOG_VIEW, AUDIT_VIEW, ...},
}

# 第二层：工具 → 所需权限
TOOL_PERMISSION_MAP = {
    "restart_service":  Permission.SERVER_RESTART,   # 需要 server:restart 权限
    "delete_record":    Permission.DB_DELETE,         # 需要 database:delete 权限
    "get_server_status": Permission.SERVER_READ,      # 需要 server:read 权限
}

# 权限校验在 SafetyGuardrailMiddleware.wrap_tool_call 中执行：
# 1. 取出 tool_name
# 2. 查 TOOL_PERMISSION_MAP 找到 required_permission
# 3. 检查 required_permission 是否在 user_permissions 中
# 4. 不在 → 拦截，返回权限不足提示
```

### 10.4 安全护栏设计详解

#### 双层安全检查

```
第 1 层：权限校验（permission check）
  "你有权执行这个操作吗？"
  → 查看 role → 查看 permissions → 匹配 required_permission

第 2 层：危险操作确认（danger confirmation）
  "你知道这个操作的危险性吗？"
  → 查 DANGEROUS_OPERATIONS → 标记级别 → 审计日志记录
```

#### 危险操作分级

| 操作 | 危险等级 | 处理方式 |
|---|---|---|
| `delete_record` | critical | 拦截 + 审计 + 不可逆警告 |
| `restart_service` | high | 需确认 + 审计 + 提示影响范围 |
| `configure_server` | high | 需确认 + 审计 + 提示稳定性风险 |
| `write_database` | medium | 审计标记 + 提示数据准确性 |
| `get_server_status` | none | 直接放行（只读操作） |

### 10.5 审计日志设计

#### 双钩子保证可靠性

```
before_agent:  记录操作开始（即使 Agent 崩溃也有记录）
after_model:   记录操作完成（包含工具调用次数、耗时、结果摘要）
```

**为什么这样设计**：

- 只用 after_model → Agent 崩溃时日志丢失，无法追溯
- 只用 before_agent → 只知道开始，不知道结果和耗时
- before + after 双钩子 → 完整追踪每次操作的生命周期

#### 日志格式

```
[10:23:01] 📝 [审计] 操作开始 | 用户: 张管理(admin_zhang) | 角色: admin | 部门: IT运维部
[10:23:05] ⚠️ 危险操作: restart_service | 等级: high | 参数: {"service_name":"nginx"}
[10:23:08] 📝 [审计] 操作完成 | 用户: admin_zhang | 工具调用: 2次 | 耗时: 7.2s
```

### 10.6 角色 Prompt 差异化

同一个问题 "查看服务器状态"，不同角色得到不同风格的回复：

```
admin（管理员）:
  "所有服务器状态总览已完成。⚠️ db-server-01 CPU 使用率 72.3%，
   建议今晚 22:00 执行维护窗口，添加 CPU 资源。已为你生成扩容工单模板。

   影响范围: 仅 db-server-01
   回滚方案: 保留原配置快照，出问题一键回滚"

operator（运维工程师）:
  "📊 4 台服务器状态检查完毕：
   ✅ web-01/web-02: 正常
   ⚠️ db-server-01: CPU 偏高 (72.3%)，内存 81.5%
   
   建议操作:
   1. 先查看 db-server-01 的错误日志确认原因
   2. 如果是流量突增，联系 DBA 评估扩容
   3. 不是紧急问题，可在工作时间内处理"

viewer（观察员）:
  "📊 系统状态报告：
   3 台服务器健康，1 台需要注意（db-server-01）。
   
   详细分析:
   - db-server-01 CPU 使用率呈上升趋势（过去 4 小时从 45% → 72.3%）
   - 可能与 10:00 的数据导入任务相关
   
   ⚠️ 建议联系运维团队检查 db-server-01，已整理好问题描述。"

auditor（审计员）:
  "✅ 已记录本次状态查询操作。审计摘要：
   - 操作: 服务器状态查询
   - 时间: 2026-05-10 10:23 UTC
   - 操作人: auditor_li
   - 结果: 4 台服务器状态已记录
   
   合规性: 本次查询操作不涉及变更，符合流程。"
```

### 10.7 关键设计决策

| 决策 | 做法 | 原因 |
|---|---|---|
| **中间件全部用继承** | 7 个类，非装饰器 | 每个中间件有复杂的内部逻辑、需要多个钩子、需要实例变量 |
| **权限存在 ROLE_PERMISSIONS 字典** | 非数据库 | 教学示例简化。生产环境存数据库 + 缓存 |
| **get_current_user 从 runtime 取** | 非 State 取 | runtime.config 来自 JWT，不可被 LLM 篡改。安全字段走框架注入通道 |
| **危险操作在 wrap_tool_call 拦截** | 非 before_model | 工具调用是最终执行点，在此之前拦截可能被绕过 |
| **双钩子审计日志** | before_agent + after_model | 确保 Agent 崩溃时也有记录 |
| **角色 Prompt 差异** | 4 套完整 System Prompt | 同一 Agent，不同角色的行为边界完全不同 |
| **权限不足返回自然语言** | 非抛异常 | LLM 收到权限不足提示后可以解释给用户，而非直接崩溃 |

---

## 第十一章：中间件编排 — 顺序、分层与优先级

### 11.1 洋葱模型：注册顺序 ≠ 执行顺序

中间件的注册顺序决定了它们在洋葱中的位置。**先注册 = 最外层 = 最先拦截请求、最后处理响应**。

```
# 注册顺序（从左到右）：
middleware = [A, B, C]

# 实际执行（洋葱模型）：
#
# 请求进入 ──────────────────────────────→
#   ┌─ A ──────────────────────────────┐
#   │  ┌─ B ────────────────────────┐  │
#   │  │  ┌─ C ──────────────────┐  │  │
#   │  │  │    核心 Agent 流程     │  │  │
#   │  │  └──────────────────────┘  │  │
#   │  └────────────────────────────┘  │
#   └──────────────────────────────────┘
# ← 响应返回 ──────────────────────────────

# 5 个钩子的具体执行路径：
#
# before_agent:         A → B → C         （注册顺序 = 正序）
# before_model:         A → B → C         （注册顺序 = 正序）
# wrap_model_call:     A 包 B 包 C 包 LLM  （外层先拦截）
# after_model:          C → B → A         （注册顺序 = 反序！）
# wrap_tool_call:      A 包 B 包 C 包 Tool （外层先拦截）
# after_agent:          C → B → A         （注册顺序 = 反序）
```

**为什么 after/end 钩子是反序？**

就像 Java Servlet Filter 的 `doFilter` 之后执行 `finally` 块——最外层的 Filter 最后才拿到响应，因为它把控制权交给了内层，内层全部完成后才返回到外层。

### 11.2 五个钩子的执行特性对比

| 钩子 | 执行频率 | 所在层级 | 能否阻止后续 | 数据流向 | 控制颗粒度 |
|---|---|---|---|---|---|
| `before_agent` | 1 次/会话 | Agent 入口 | 是（Command.goto end） | State 写入 → 全局 | 粗——控制整个会话 |
| `before_model` | 每次 LLM 调用 | LLM 入口 | 是（Command.goto） | State 写入 → 全局 | 中——控制单次模型调用 |
| `wrap_model_call` | 每次 LLM 调用 | LLM 包裹 | 是（不调用 handler = 短路） | 修改 Request → 影响本次调用 | 细——完全控制本次调用 |
| `after_model` | 每次 LLM 调用 | LLM 出口 | 是（Command.goto） | State 写入 + 日志 | 中——控制调用后行为 |
| `wrap_tool_call` | 每次工具调用 | Tool 包裹 | 是（不调用 handler = 短路） | 修改 Tool 输入/输出 | 最细——控制每个工具调用 |

**控制颗粒度层次**：

```
粗（会话级）：   before_agent, after_agent
  ↓
中（LLM 调用级）： before_model, after_model
  ↓
细（单次调用级）： wrap_model_call
  ↓
最细（工具级）：  wrap_tool_call
```

### 11.3 分层架构：四层防线

```
┌─────────────────────────────────────────────────────────────┐
│                    中间件分层架构                             │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │          第 1 层：安全边界（Guard Layer）               │ │
│  │                                                       │ │
│  │  中间件: SecurityCheck, PII, Auth, RateLimit          │ │
│  │  优先级: P0 — 不通过则拒绝，不进业务层                  │ │
│  │  原则:  安全 > 一切。安全拦截后直接返回，不浪费资源。    │ │
│  │  钩子:   mainly before_agent + before_model           │ │
│  └───────────────────────────────────────────────────────┘ │
│                         │ 通过                               │
│                         ▼                                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │          第 2 层：业务逻辑（Business Layer）            │ │
│  │                                                       │ │
│  │  中间件: DynamicPrompt, ToolSelection, ModelSwitch    │ │
│  │  优先级: P1 — 核心业务功能，影响用户体验                │ │
│  │  原则:  业务 > 成本。先保证质量，再考虑省钱。            │ │
│  │  钩子:   mainly wrap_model_call + before_model         │ │
│  └───────────────────────────────────────────────────────┘ │
│                         │ 通过                               │
│                         ▼                                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │          第 3 层：成本控制（Cost Layer）                │ │
│  │                                                       │ │
│  │  中间件: ModelSwitch(budget), Summarization, Cache    │ │
│  │  优先级: P2 — 在保证业务质量的前提下降低成本            │ │
│  │  原则:  成本 < 业务。省钱的前提是不影响回答质量。       │ │
│  │  钩子:   mainly wrap_model_call                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                         │ 通过                               │
│                         ▼                                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │        第 4 层：监控与日志（Observability Layer）       │ │
│  │                                                       │ │
│  │  中间件: AuditLog, Metrics, ResponseValidation        │ │
│  │  优先级: P3 — 不影响业务，但必须执行                    │ │
│  │  原则:  监控不干扰业务。日志失败不能导致请求失败。       │ │
│  │  钩子:   mainly after_model + after_agent             │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**四层对应的生产级注册顺序**：

```python
agent = create_agent(
    model=llm,
    tools=[...],
    middleware=[
        # ===== 第 1 层：安全边界（最外层，最先拦截）=====
        SecurityCheckMiddleware(),
        PIIMiddleware("email", strategy="redact"),
        # ===== 第 2 层：业务逻辑 =====
        DynamicPromptMiddleware(),
        ToolSelectionMiddleware(...),
        # ===== 第 3 层：成本控制 =====
        SmartModelSwitchMiddleware(...),
        SummarizationMiddleware(...),
        CacheMiddleware(),
        # ===== 第 4 层：监控与日志（最内层）=====
        ResponseValidationMiddleware(),
        AuditLogMiddleware(),
    ],
)
```

### 11.4 六大优先级原则

**原则 1：安全优先** — 安全 > 一切。拦截后直接返回，不浪费 LLM 调用。

**原则 2：提前失败（Fail Fast）** — 在最可能失败的钩子中最先检查：before_agent 查认证 → wrap_model_call 查缓存 → wrap_tool_call 查权限。

**原则 3：缓存优先（Cache First）** — 缓存命中 → 0 Token 消耗 → 响应 < 1ms。放在 wrap_model_call 最外层。

**原则 4：写操作依赖正确** — 先读后写、先备份后改、先验证后执行。中间件顺序体现依赖关系。

**原则 5：中断操作置后** — HumanInTheLoop 放在 after_model：先让 LLM 充分推理 → 再暂停审批。

**原则 6：性能开销降序** — 高开销中间件放内层（只在必要时执行）；低开销放外层（每次都执行）。

### 11.5 中间件之间的数据依赖

中间件通过 **AgentState** 共享数据，不是孤立运行：

```
SecurityCheckMiddleware (before_agent)
  写入: user_id="alice", user_role="admin", user_permissions=[...]
        │
        ▼ AgentState（全局账本）
        │
DynamicPromptMiddleware (wrap_model_call)
  读取: state["user_role"] → "admin" → 选择对应 Prompt
        │
        ▼
SafetyGuardrailMiddleware (wrap_tool_call)
  读取: state["user_permissions"] → 检查工具权限
        │
        ▼
AuditLogMiddleware (after_model)
  读取: state["user_id"] → 写入审计日志
```

依赖管理四策略：上游写下游读、防御性读取（默认值）、显式文档化（docstring 声明依赖）、双钩子保底。

### 11.6 编排决策树

```
设计新中间件 → 放哪一层？

├─ 安全/认证/权限？→ 第 1 层。钩子: before_agent 或 before_model
├─ 业务逻辑/用户体验？→ 第 2 层。可短路的放外层（Cache），需LLM调用后的放内层
├─ 成本/资源控制？→ 第 3 层。钩子: wrap_model_call
├─ 只记录/观察？→ 第 4 层。钩子: after_model / after_agent
└─ 跨层协调？→ 多个钩子（如 AuditLog 同时 before_agent + after_model）
```

---

## 第十二章：扩展 — 企业中常用的中间件思路

以下五个方向帮助你在实际项目中快速判断"这里该不该抽象成中间件"：

| 中间件 | 所在层 | 钩子 | 做什么 | 编排原因 |
|---|---|---|---|---|
| **意图理解** | 第 2 层 | before_model | 分析意图 → 路由分支 | 安全通过后、LLM 调用前决策 |
| **文档解析** | 第 2 层 | wrap_model_call | 上传文件 → 自动解析 → 注入 State | 预处理在 LLM 推理之前 |
| **知识抽取** | 第 4 层 | after_model | 从回复提取实体 → 存长期记忆库 | 不阻塞用户响应 |
| **语义检索** | 第 2 层 | wrap_model_call | 自动检索知识库 → 注入 Prompt | 需用户身份做权限过滤 |
| **代码理解** | 第 2 层 | wrap_tool_call | 静态分析代码安全 → 通过后才执行 | 权限检查之后、代码执行之前 |

---

## 第十三章：大总结 — 中间件的工程化价值

### 13.1 工程化对比

```
没有中间件                    有中间件
─────────                    ────────
重试逻辑散落各工具             ToolRetryMiddleware 一行注册
安全规则硬编码在 Prompt         SecurityCheckMiddleware 统一拦截
改日志格式 → 改 20 个文件      改 1 个 AuditLogMiddleware
加缓存 → 改每个 LLM 调用点     加 1 个 CacheMiddleware
```

### 13.2 五大工程化原则

单一职责 / 开闭原则 / 依赖倒置 / 分层隔离 / 声明式组合 — 每个都在中间件体系中有具体体现。

### 13.3 架构跃迁

```
阶段 1：一把梭     → 一个函数搞定全部
阶段 2：手工切分   → 散落在各处，难以维护
阶段 3：中间件化   → 独立开发/测试/部署/组合 → 这就是架构
```

### 13.4 中间件模块知识体系

```
第一章：概览       → 什么是中间件、分类、生命周期
第二~五章：内置   → Summarization/PII/Model/Tool/HITL
第六章：其他常用   → Todo/ModelRetry/Shell
第七章：组合策略   → 推荐栈 + 按场景选择
第八章：参数传递   → Request/Response/State/Command 详解
第九章：自定义实战 → 装饰器 vs 继承 + 3 个完整示例
第十章：企业实战   → IT Ops Agent + RBAC + 7 个中间件
第十一章：编排     → 洋葱模型 + 分层 + 优先级 + 依赖
第十二章：扩展思路 → 意图/文档/知识/检索/代码 5 个方向
第十三章：大总结   → 工程化价值 + 五大原则 + 跃迁路径

核心收获：
• 横切关注点 → 中间件 → 声明式组合 → 可插拔架构
• 安全 > 业务 > 成本 > 监控 → 四层防线
• 洋葱模型 → 外层先拦截、内层后执行
• AgentState 是全局账本 → 中间件通过它解耦通信
• 装饰器搞轻量拦截、继承搞复杂业务、编排搞企业级

── 中间件模块结束 ──
```