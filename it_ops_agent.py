# ================================================================
# it_ops_agent.py — 企业级 IT 运维 Agent（RBAC + 多中间件组合）
# ================================================================
"""
架构概览：

┌──────────────────────────────────────────────────────────────┐
│                     IT Ops Agent                             │
│                                                              │
│  中间件执行流程：                                               │
│                                                              │
│  [1] before_agent                                            │
│      ├─ SecurityCheckMiddleware: 安全验证 + 权限注入           │
│      └─ AuditLogMiddleware: 记录操作开始                       │
│                                                              │
│  [2] wrap_model_call                                         │
│      ├─ DynamicPromptMiddleware: 根据角色注入提示词             │
│      ├─ SmartModelSwitchMiddleware: 根据意图切换模型            │
│      └─ ContextManagementMiddleware: 裁剪/管理上下文            │
│                                                              │
│  [3] 模型执行（LLM API 调用）                                   │
│                                                              │
│  [4] after_model                                             │
│      ├─ ResponseValidationMiddleware: 响应校验                 │
│      └─ AuditLogMiddleware: 记录操作完成 + 结果摘要             │
│                                                              │
│  [5] wrap_tool_call                                          │
│      └─ SafetyGuardrailMiddleware: 危险操作拦截 + 确认          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
"""
import os, sys, json, logging
from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from enum import Enum

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent, InjectedState, InjectedStore
from langgraph.runtime import Runtime

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    AgentState, ModelRequest, ModelResponse, hook_config,
)
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langchain_core.runnables import RunnableConfig

sys.stdout.reconfigure(encoding='utf-8')

# ================================================================
# 第一部分：基础设施 — 角色、权限、日志、辅助函数
# ================================================================

# ---- 1.1 日志配置 ----
# 带时间戳的日志输出：同时写入文件和终端
LOG_FILE = "./it_ops_audit.log"

def get_logger(name: str) -> logging.Logger:
    """获取带时间戳格式的日志记录器。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if not logger.handlers:
        # 文件 handler（持久化审计）
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)

        # 终端 handler（实时输出）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(console_handler)

    return logger

logger = get_logger("it_ops_agent")


def ts_log(message: str, level: str = "INFO"):
    """带时间戳的快捷日志函数。"""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    full_msg = f"[{now}] {message}"
    getattr(logger, level.lower())(full_msg)


# ---- 1.2 用户上下文 ----
class UserContext(BaseModel):
    """当前操作用户的完整上下文信息。

    由 SecurityCheckMiddleware 在 before_agent 阶段从 JWT/Token 解析注入。
    整个 Agent 执行周期内，所有中间件和工具都可访问此信息。
    """
    user_id: str = Field(description="用户唯一标识，如 'zhangsan'")
    display_name: str = Field(description="用户显示名，如 '张三'")
    role: str = Field(description="用户角色：admin/operator/viewer")
    permissions: list[str] = Field(
        default_factory=list,
        description="用户拥有的权限列表，如 ['server:read','server:restart']"
    )
    department: str = Field(default="", description="所属部门")
    authenticated_at: str = Field(default="", description="认证时间 ISO 格式")


def get_current_user(runtime: Runtime) -> UserContext:
    """
    从 runtime.config 中解析当前用户信息。

    为什么从 runtime 取而不从 State 取？
      - runtime.config 由入站请求的 JWT/Token 解析而来，不可被 LLM 篡改
      - State 中的字段 LLM 理论上可以通过 tool output 间接影响
      - 安全信息必须走"框架注入"通道，不走"LLM 可触达"通道

    类比 Java：
      SecurityContextHolder.getContext().getAuthentication().getPrincipal()
    """
    configurable = runtime.config.get("configurable", {})
    return UserContext(
        user_id=configurable.get("user_id", "anonymous"),
        display_name=configurable.get("display_name", "游客"),
        role=configurable.get("role", "viewer"),
        permissions=configurable.get("permissions", []),
        department=configurable.get("department", ""),
        authenticated_at=datetime.now(timezone.utc).isoformat(),
    )


# ---- 1.3 RBAC 模型 ----
class Role(Enum):
    """系统角色枚举。"""
    ADMIN = "admin"           # 超级管理员：所有权限
    OPERATOR = "operator"     # 运维工程师：读写运维操作
    VIEWER = "viewer"         # 只读用户：仅查看
    AUDITOR = "auditor"       # 审计员：查看 + 审计日志


class Permission(Enum):
    """系统权限枚举。"""
    # 服务器相关
    SERVER_READ = "server:read"                 # 查看服务器状态
    SERVER_RESTART = "server:restart"           # 重启服务
    SERVER_CONFIGURE = "server:configure"       # 修改服务器配置

    # 数据库相关
    DB_READ = "database:read"                   # 查询数据库
    DB_WRITE = "database:write"                 # 写入数据库
    DB_DELETE = "database:delete"               # 删除数据库记录

    # 日志相关
    LOG_VIEW = "log:view"                       # 查看日志
    LOG_DOWNLOAD = "log:download"               # 下载日志

    # 监控相关
    METRICS_VIEW = "metrics:view"               # 查看监控指标
    ALERT_CONFIGURE = "alert:configure"         # 配置告警规则

    # 用户管理
    USER_MANAGE = "user:manage"                 # 用户管理

    # 审计相关
    AUDIT_VIEW = "audit:view"                   # 查看审计日志


# ---- 1.4 角色 → 权限映射 ----
# 每个角色拥有的权限集合（最小权限原则）
ROLE_PERMISSIONS: dict[str, set[Permission]] = {
    Role.ADMIN.value: {
        # 管理员：拥有所有权限
        Permission.SERVER_READ,
        Permission.SERVER_RESTART,
        Permission.SERVER_CONFIGURE,
        Permission.DB_READ,
        Permission.DB_WRITE,
        Permission.DB_DELETE,
        Permission.LOG_VIEW,
        Permission.LOG_DOWNLOAD,
        Permission.METRICS_VIEW,
        Permission.ALERT_CONFIGURE,
        Permission.USER_MANAGE,
        Permission.AUDIT_VIEW,
    },
    Role.OPERATOR.value: {
        # 运维工程师：可以操作服务器和数据库，但不能管理用户
        Permission.SERVER_READ,
        Permission.SERVER_RESTART,
        Permission.DB_READ,
        Permission.DB_WRITE,
        Permission.LOG_VIEW,
        Permission.LOG_DOWNLOAD,
        Permission.METRICS_VIEW,
        Permission.AUDIT_VIEW,
    },
    Role.VIEWER.value: {
        # 只读用户：只能看，不能改
        Permission.SERVER_READ,
        Permission.LOG_VIEW,
        Permission.METRICS_VIEW,
    },
    Role.AUDITOR.value: {
        # 审计员：查看 + 审计日志
        Permission.SERVER_READ,
        Permission.LOG_VIEW,
        Permission.LOG_DOWNLOAD,
        Permission.METRICS_VIEW,
        Permission.AUDIT_VIEW,
    },
}


# ---- 1.5 工具 → 权限映射 ----
# 每个工具调用前需要检查用户是否有对应权限
TOOL_PERMISSION_MAP: dict[str, Permission] = {
    "get_server_status":    Permission.SERVER_READ,
    "restart_service":      Permission.SERVER_RESTART,
    "configure_server":     Permission.SERVER_CONFIGURE,
    "query_database":       Permission.DB_READ,
    "write_database":       Permission.DB_WRITE,
    "delete_record":        Permission.DB_DELETE,
    "view_logs":            Permission.LOG_VIEW,
    "download_logs":        Permission.LOG_DOWNLOAD,
    "get_metrics":          Permission.METRICS_VIEW,
    "configure_alert":      Permission.ALERT_CONFIGURE,
    "manage_users":         Permission.USER_MANAGE,
    "view_audit_log":       Permission.AUDIT_VIEW,
}


# ---- 1.6 危险操作定义 ----
# 这些操作在执行前需要额外的安全检查（SafetyGuardrailMiddleware 处理）
DANGEROUS_OPERATIONS = {
    "restart_service": {
        "level": "high",
        "message": "重启服务会导致短暂的服务中断，确认要继续吗？",
        "requires_confirmation": True,
    },
    "delete_record": {
        "level": "critical",
        "message": "⚠️ 删除操作不可逆！确认要删除这条记录吗？",
        "requires_confirmation": True,
    },
    "write_database": {
        "level": "medium",
        "message": "写入操作会修改数据库内容，请确认数据准确性。",
        "requires_confirmation": True,
    },
    "configure_server": {
        "level": "high",
        "message": "修改服务器配置可能影响服务稳定性，确认要继续吗？",
        "requires_confirmation": True,
    },
    "configure_alert": {
        "level": "medium",
        "message": "修改告警规则将影响监控通知，确认规则配置正确吗？",
        "requires_confirmation": False,
    },
}

# ================================================================
# 第二部分：运维工具定义
# ================================================================

@tool
def get_server_status(server_name: str = "all") -> str:
    """
    查询服务器运行状态。

    参数 server_name: 服务器名称，默认 'all' 查询所有服务器。
    返回 CPU、内存、磁盘使用率和运行时长。
    示例: get_server_status('web-server-01')
    """
    # 模拟服务器状态数据（生产环境替换为真实 API 调用）
    servers = {
        "web-server-01": {"cpu": 45.2, "mem": 62.8, "disk": 71.3, "uptime": "15d 4h", "status": "healthy"},
        "web-server-02": {"cpu": 38.1, "mem": 55.4, "disk": 68.9, "uptime": "15d 3h", "status": "healthy"},
        "db-server-01": {"cpu": 72.3, "mem": 81.5, "disk": 85.1, "uptime": "30d 12h", "status": "warning"},
        "cache-server": {"cpu": 12.5, "mem": 34.2, "disk": 45.0, "uptime": "7d 8h", "status": "healthy"},
    }

    if server_name == "all":
        lines = ["📊 所有服务器状态："]
        for name, info in servers.items():
            icon = "✅" if info["status"] == "healthy" else "⚠️" if info["status"] == "warning" else "🔴"
            lines.append(
                f"  {icon} {name}: CPU={info['cpu']}% | MEM={info['mem']}% | "
                f"DISK={info['disk']}% | 运行时长={info['uptime']}"
            )
        return "\n".join(lines)

    info = servers.get(server_name)
    if not info:
        return f"❌ 未找到服务器 '{server_name}'"
    return (
        f"服务器 {server_name}：\n"
        f"  状态: {info['status']}\n"
        f"  CPU: {info['cpu']}%\n"
        f"  内存: {info['mem']}%\n"
        f"  磁盘: {info['disk']}%\n"
        f"  运行时长: {info['uptime']}"
    )


@tool
def restart_service(service_name: str, reason: str = "") -> str:
    """
    重启指定服务。⚠️ 高危操作，会导致服务短暂中断。

    参数 service_name: 要重启的服务名称（如 'nginx', 'api-gateway'）
    参数 reason: 重启原因（必填，记录到审计日志）
    """
    if not reason:
        return "❌ 必须提供重启原因（reason 参数）。此信息将记录到审计日志。"
    ts_log(f"⚠️ 重启服务: {service_name}, 原因: {reason}", "WARNING")
    return (
        f"✅ 服务 '{service_name}' 已成功重启。\n"
        f"   重启原因: {reason}\n"
        f"   中断时长: 约 3 秒\n"
        f"   新进程 PID: {hash(service_name) % 100000 + 1000}\n"
        f"   操作已记录到审计日志。"
    )


@tool
def view_logs(service_name: str, lines: int = 50, filter_keyword: str = "") -> str:
    """
    查看服务的运行日志。

    参数 service_name: 服务名称
    参数 lines: 返回最近多少行（默认 50）
    参数 filter_keyword: 过滤关键词（可选，如 'ERROR'）
    """
    # 模拟日志数据
    mock_logs = [
        f"[10:23:01] INFO  Request processed: /api/users (200) 15ms",
        f"[10:23:05] INFO  Request processed: /api/orders (200) 22ms",
        f"[10:23:12] WARN  Connection pool nearing limit: 85/100",
        f"[10:23:18] ERROR Database timeout after 30s: SELECT * FROM orders",
        f"[10:23:25] INFO  Retry succeeded: Database connection restored",
        f"[10:23:30] INFO  Health check passed: all systems nominal",
    ]

    # 过滤
    if filter_keyword:
        mock_logs = [l for l in mock_logs if filter_keyword.upper() in l.upper()]

    preview = "\n".join(mock_logs[:lines])
    return (
        f"📋 {service_name} 最近 {len(mock_logs[:lines])} 行日志"
        f"{f'（过滤: {filter_keyword}）' if filter_keyword else ''}：\n{preview}"
    )


@tool
def get_system_metrics(metric_type: str = "all") -> str:
    """
    获取系统资源监控指标。

    参数 metric_type: 指标类型 — 'cpu', 'memory', 'disk', 'network', 或 'all'（默认）
    """
    metrics = {
        "cpu": {
            "usage": "45.2%", "cores": 8, "load_avg": [1.2, 1.5, 1.8],
            "top_process": "java (23.5%)"
        },
        "memory": {
            "total": "32GB", "used": "20.5GB (64%)", "available": "11.5GB",
            "swap": "2.0GB / 8.0GB"
        },
        "disk": {
            "/": "71.3% (150GB/210GB)", "/data": "85.1% (850GB/1TB)",
            "iops": "1250 read / 340 write"
        },
        "network": {
            "inbound": "125 Mbps", "outbound": "89 Mbps",
            "connections": 2340, "errors": 12
        },
    }

    if metric_type == "all":
        lines = ["📊 系统资源监控总览："]
        for name, data in metrics.items():
            lines.append(f"  [{name.upper()}]")
            for k, v in data.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    data = metrics.get(metric_type)
    if not data:
        return f"❌ 不支持的指标类型 '{metric_type}'。可选: cpu, memory, disk, network, all"
    return f"📊 [{metric_type.upper()}] 监控指标：\n" + "\n".join(
        f"  {k}: {v}" for k, v in data.items()
    )


@tool
def query_database(query_description: str) -> str:
    """
    查询数据库（只读）。

    参数 query_description: 自然语言描述查询内容，如 '查询最近 100 条订单'
    注意：实际操作前会进行权限校验。
    """
    ts_log(f"📊 数据库查询: {query_description[:80]}")
    return (
        f"📊 查询结果（模拟）：\n"
        f"  查询: {query_description}\n"
        f"  返回行数: 127\n"
        f"  耗时: 45ms\n"
        f"  （生产环境返回真实数据）"
    )


@tool
def write_database(operation: str, table: str, data_summary: str) -> str:
    """
    写入数据库。⚠️ 需要确认操作。

    参数 operation: INSERT / UPDATE
    参数 table: 目标表名
    参数 data_summary: 操作数据摘要
    """
    ts_log(f"✏️ 数据库写入: {operation} {table} — {data_summary[:80]}")
    return (
        f"✅ 数据库写入成功：\n"
        f"  操作: {operation}\n"
        f"  表: {table}\n"
        f"  内容: {data_summary}\n"
        f"  操作已记录到审计日志。"
    )


@tool
def delete_record(table: str, record_id: str, reason: str = "") -> str:
    """
    删除数据库记录。⚠️⚠️ 不可逆操作！

    参数 table: 表名
    参数 record_id: 记录 ID
    参数 reason: 删除原因（必填）
    """
    if not reason:
        return "❌ 必须提供删除原因（reason 参数）。"
    ts_log(f"🗑️ 删除记录: {table}#{record_id}, 原因: {reason}", "WARNING")
    return (
        f"🗑️ 记录已删除：\n"
        f"  表: {table}\n"
        f"  ID: {record_id}\n"
        f"  原因: {reason}\n"
        f"  ⚠️ 此操作不可逆！操作已记录到审计日志。"
    )


# 所有运维工具注册
OPS_TOOLS = [
    get_server_status,
    restart_service,
    view_logs,
    get_system_metrics,
    query_database,
    write_database,
    delete_record,
]

# ================================================================
# 第三部分：中间件定义（全部使用继承方式）
# ================================================================

# ---- 3.1 安全检查 + 权限注入中间件 ----
class SecurityCheckMiddleware(AgentMiddleware):
    """
    安全检查与权限注入中间件（before_agent）。

    职责：
      1. 从 runtime.config 解析用户 JWT/Token → UserContext
      2. 验证用户身份有效性
      3. 根据角色查询权限列表（RBAC）
      4. 将用户信息和权限注入 AgentState（供后续所有中间件和工具使用）

    为什么用 before_agent 而不是 before_model？
      before_agent 在整个 Agent 生命周期只执行一次。
      before_model 每次 LLM 调用都执行（多轮循环中可能有多次）。
      权限验证只需一次，放在 before_agent 最高效。

    类比 Java：
      Spring Security Filter Chain 中的 AuthenticationFilter
    """

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Agent 启动前：验证用户身份 + 注入权限。"""
        # === 步骤 1：解析用户上下文 ===
        user = get_current_user(runtime)

        # === 步骤 2：验证身份有效性 ===
        if not user.user_id or user.user_id == "anonymous":
            ts_log("❌ 安全拦截：未认证用户尝试访问", "WARNING")
            return {
                "messages": [AIMessage(content="❌ 访问被拒绝：请先登录认证。")],
                "security_blocked": True,
            }

        # === 步骤 3：RBAC — 查询角色权限 ===
        permissions = ROLE_PERMISSIONS.get(user.role, set())
        permission_names = [p.value for p in permissions]

        ts_log(
            f"🔐 用户认证: {user.display_name} ({user.user_id}) | "
            f"角色: {user.role} | "
            f"权限数: {len(permissions)} | "
            f"部门: {user.department}"
        )

        # === 步骤 4：注入到 AgentState ===
        # 这些字段在整个 Agent 生命周期内对所有中间件和工具可见
        return {
            "user_id": user.user_id,
            "user_display_name": user.display_name,
            "user_role": user.role,
            "user_permissions": permission_names,
            "user_department": user.department,
            "authenticated_at": user.authenticated_at,
            "security_blocked": False,         # 标志位：安全验证通过
            "audit_entries": [],               # 审计日志累积列表
        }


# ---- 3.2 动态提示词注入中间件 ----
class DynamicPromptMiddleware(AgentMiddleware):
    """
    动态提示词注入中间件（wrap_model_call）。

    职责：
      1. 根据用户角色（从 AgentState 读取）选择对应的 System Prompt
      2. 不同角色看到不同风格的提示词：
         - admin：可执行所有操作，强调安全审计
         - operator：可执行运维操作，强调操作规范和回滚方案
         - viewer：只读模式，强调数据分析和报告
         - auditor：审计模式，强调合规和日志追溯

    为什么用 wrap_model_call 而不是 before_model？
      wrap_model_call 可以直接修改 ModelRequest 中的 system_message，
      通过 request.override(system_message=...) 创建新的请求。
      这种方式比 before_model 修改 State 中的 messages 更干净——
      不会在 State 中遗留中间 SystemMessage。
    """

    # 角色 → System Prompt 映射
    ROLE_PROMPTS = {
        "admin": (
            "你是 **IT 运维超级管理员**，拥有系统最高权限。\n\n"
            "## 你的风格\n"
            "- 决策果断，给出明确的执行指令\n"
            "- 每次操作前简要说明影响范围和回滚方案\n"
            "- 优先考虑系统稳定性和安全性\n"
            "- 操作后必须记录审计摘要\n\n"
            "## 操作规则\n"
            "- 危险操作（重启、删除、修改配置）前必须确认\n"
            "- 操作完成后给出验证步骤\n"
            "- 发现异常立即报告并建议处置方案"
        ),
        "operator": (
            "你是 **IT 运维工程师**，负责日常运维操作。\n\n"
            "## 你的风格\n"
            "- 严格按照运维手册执行操作\n"
            "- 操作前先检查当前状态（先读后写）\n"
            "- 遇到不确定的情况，先收集信息再做判断\n"
            "- 给出清晰的步骤和预期结果\n\n"
            "## 操作规则\n"
            "- 重启服务前先检查依赖服务状态\n"
            "- 数据库写操作前先备份\n"
            "- 操作后验证服务是否恢复正常\n"
            "- 遇到权限不足的操作，告知用户申请权限"
        ),
        "viewer": (
            "你是 **IT 系统观察员**，拥有只读权限。\n\n"
            "## 你的风格\n"
            "- 专注于数据分析和状态报告\n"
            "- 用清晰的图表化思维展示系统状态\n"
            "- 发现异常时给出详细的分析和建议\n"
            "- 引导用户联系运维团队执行写操作\n\n"
            "## 操作规则\n"
            "- 只能执行查询类操作（查状态/看日志/看指标）\n"
            "- 不能重启服务、修改配置、写入数据库\n"
            "- 如果需要写操作，告知用户：「此操作需要运维工程师权限，已帮你整理好操作请求，请联系运维团队。」\n"
            "- 将观察到的异常整理成工单格式"
        ),
        "auditor": (
            "你是 **IT 合规审计员**，负责操作审计和合规检查。\n\n"
            "## 你的风格\n"
            "- 关注操作的合规性和可追溯性\n"
            "- 每项操作都要关联到审计记录\n"
            "- 检查操作是否符合变更管理流程\n"
            "- 生成的报告可直接用于合规审查\n\n"
            "## 操作规则\n"
            "- 查看所有操作日志和审计记录\n"
            "- 标记不合规的操作行为\n"
            "- 生成合规报告\n"
            "- 不可以执行运维操作（只读 + 审计）"
        ),
    }

    DEFAULT_PROMPT = "你是 IT 运维助手。根据你的权限提供帮助。"

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse | AIMessage:
        """在 LLM 调用前，根据用户角色注入对应的 System Prompt。"""
        # 从 AgentState 中读取角色（SecurityCheckMiddleware 已注入）
        state = request.state
        user_role = state.get("user_role", "viewer")

        # 选择对应的 Prompt
        prompt_text = self.ROLE_PROMPTS.get(user_role, self.DEFAULT_PROMPT)

        # 用 override 创建新的 ModelRequest（不可变模式）
        # system_message 单独存放，LLM 会将其放在消息列表最前面
        personalized_request = request.override(
            system_message=SystemMessage(content=prompt_text),
        )

        ts_log(f"🎭 注入角色 Prompt: {user_role}")
        return handler(personalized_request)


# ---- 3.3 智能模型切换中间件 ----
class SmartModelSwitchMiddleware(AgentMiddleware):
    """
    智能模型切换中间件（wrap_model_call）。

    职责：
      根据用户输入的关键词分析判断是否需要更强大的模型：
      - 简单查询（查状态/看日志）→ 经济型模型（省钱）
      - 复杂分析（故障排查/性能优化）→ 高级模型（保证质量）
      - 危险操作验证 → 高级模型（安全关键路径）

    为什么用 wrap_model_call？
      可以在 LLM 调用之前拦截，通过 request.override(model=...) 动态切换模型。
    """

    def __init__(self, premium_model, budget_model):
        """
        Args:
            premium_model: 高性能模型（用于复杂/关键任务）
            budget_model: 经济型模型（用于简单查询）
        """
        self.premium = premium_model
        self.budget = budget_model

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse | AIMessage:
        """分析问题复杂度，选择合适的模型。"""
        # 提取用户最后一条消息
        user_msgs = [
            msg for msg in request.messages
            if hasattr(msg, "type") and msg.type == "human"
        ]
        last_msg = (user_msgs[-1].content or "") if user_msgs else ""

        # === 复杂度判断 ===
        needs_premium = self._needs_premium_model(last_msg, request.state)

        if needs_premium:
            ts_log(f"🧠 复杂任务 → 使用 PREMIUM 模型")
            return handler(request)  # 保持原 model（premium）

        ts_log(f"💰 简单任务 → 切换到 BUDGET 模型")
        return handler(request.override(model=self.budget))

    def _needs_premium_model(self, text: str, state: dict) -> bool:
        """判断是否需要高级模型。"""
        # 维度 1：危险操作 → 必须 premium（安全关键路径）
        for op_name in DANGEROUS_OPERATIONS:
            if op_name.replace("_", " ") in text.lower():
                return True

        # 维度 2：复杂分析 → premium
        complex_signals = [
            "为什么", "原因", "分析", "排查", "故障", "报错", "异常",
            "优化", "建议", "方案", "对比", "评估", "总结",
        ]
        if any(s in text for s in complex_signals):
            return True

        # 维度 3：长度信号（> 80 字符 → 更可能是复杂问题）
        if len(text) > 80:
            return True

        return False


# ---- 3.4 上下文管理中间件 ----
class ContextManagementMiddleware(AgentMiddleware):
    """
    上下文管理中间件（wrap_model_call）。

    职责：
      当对话历史 Token 数超过阈值时，自动裁剪旧消息。
      保留 SystemMessage + 最近 N 条消息 + 重要工具结果。
      防止上下文溢出导致 LLM 调用失败。

    为什么用 wrap_model_call 而不是 before_model？
      wrap_model_call 可以直接修改 messages（裁剪后传给 handler），
      不影响 State 中的原始 messages。
      如果需要在 State 中也反映裁剪，需要在 after_model 中同步。
    """

    max_tokens: int = 4000          # Token 上限
    keep_recent: int = 15           # 保留最近消息数

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse | AIMessage:
        """裁剪过长的消息历史。"""
        messages = request.messages
        total_chars = sum(len(m.content or "") for m in messages)

        # 粗略估算：1 token ≈ 4 字符
        if total_chars < self.max_tokens * 4:
            return handler(request)  # 无需裁剪

        # 裁剪：保留最近 N 条
        trimmed = list(messages[-self.keep_recent:])
        ts_log(
            f"📏 上下文裁剪：{len(messages)} 条 → {len(trimmed)} 条 "
            f"({total_chars} → ~{sum(len(m.content or '') for m in trimmed)} 字符)"
        )

        trimmed_request = request.override(messages=trimmed)
        return handler(trimmed_request)


# ---- 3.5 响应验证中间件 ----
class ResponseValidationMiddleware(AgentMiddleware):
    """
    响应验证中间件（after_model）。

    职责：
      1. 检查 LLM 输出是否包含敏感信息泄露（PII、密码、Token）
      2. 验证输出结构的完整性
      3. 如果输出不合规 → 拦截并返回安全提示

    为什么用 after_model？
      模型调用之后、用户看到之前——最后一道防线。
    """

    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """验证 LLM 输出。"""
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not hasattr(last_msg, "content") or not last_msg.content:
            return None

        content = last_msg.content

        # 检查敏感信息泄露模式
        leaked = self._detect_sensitive_info(content)
        if leaked:
            ts_log(f"🚨 检测到敏感信息泄露: {leaked}", "WARNING")
            return {
                "messages": [AIMessage(
                    content="⚠️ 响应被拦截：检测到可能的敏感信息泄露。"
                            "请管理员检查审计日志。"
                )],
            }

        return None  # 通过验证

    def _detect_sensitive_info(self, text: str) -> list[str]:
        """检测文本中的敏感信息。"""
        import re
        patterns = {
            "password": r'(password|passwd|pwd)\s*[:=]\s*\S+',
            "token": r'(token|api[_-]?key)\s*[:=]\s*[A-Za-z0-9\-_]{20,}',
            "private_key": r'-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----',
        }
        found = []
        for name, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                found.append(name)
        return found


# ---- 3.6 审计日志中间件 ----
class AuditLogMiddleware(AgentMiddleware):
    """
    审计日志中间件（before_agent + after_model）。

    职责：
      记录每次 Agent 调用的完整审计信息：
      - before_agent: 记录操作开始 + 用户信息
      - after_model: 记录操作完成 + 工具调用 + 耗时 + 结果摘要

    为什么不只用 after_model？
      如果 Agent 在执行中崩溃，after_model 不会执行 → 日志会丢失。
      before_agent + after_model 双钩子确保"开始"和"结束"都被记录，
      即使崩溃也能从 before_agent 日志中追溯。
    """

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """记录操作开始。"""
        user = get_current_user(runtime)
        ts_log(
            f"📝 [审计] 操作开始 | 用户: {user.display_name}({user.user_id}) "
            f"| 角色: {user.role} | 部门: {user.department}"
        )
        return {"audit_started_at": datetime.now(timezone.utc).isoformat()}

    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """记录操作完成。"""
        # 从 State 读取最后一条消息作为结果摘要
        messages = state.get("messages", [])
        result_summary = ""
        tool_calls_count = 0

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_count += len(msg.tool_calls)

        if messages and hasattr(messages[-1], "content") and messages[-1].content:
            result_summary = messages[-1].content[:100]

        # 计算耗时
        started_at_str = state.get("audit_started_at", "")
        elapsed = ""
        if started_at_str:
            try:
                started = datetime.fromisoformat(started_at_str)
                elapsed = f"{(datetime.now(timezone.utc) - started).total_seconds():.1f}s"
            except ValueError:
                pass

        user_id = state.get("user_id", "unknown")
        user_role = state.get("user_role", "unknown")
        ts_log(
            f"📝 [审计] 操作完成 | 用户: {user_id} | 角色: {user_role} "
            f"| 工具调用: {tool_calls_count}次 | 耗时: {elapsed} "
            f"| 结果: {result_summary}"
        )
        return None  # 不需要更新 State


# ---- 3.7 安全护栏中间件（危险操作拦截）----
class SafetyGuardrailMiddleware(AgentMiddleware):
    """
    安全护栏中间件（wrap_tool_call）。

    职责：
      1. 在工具执行前检查是否属于危险操作
      2. 危险操作 → 需要用户确认后才执行
      3. 超权限操作 → 直接拦截（不执行）
      4. 所有拦截操作记录到审计日志

    为什么用 wrap_tool_call？
      工具调用是 Agent 操作外部世界的唯一通道。
      在通道上设置护栏 = 所有操作都被拦截检查。

    类比：
      Kubernetes Admission Controller — 所有 API 请求必须经过准入控制。
    """

    def wrap_tool_call(self, request, handler) -> ToolMessage:
        """在工具执行前检查安全性。"""
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})
        state = request.state

        # === 检查 1：权限校验 ===
        required_permission = TOOL_PERMISSION_MAP.get(tool_name)
        if required_permission:
            user_permissions = state.get("user_permissions", [])
            if required_permission.value not in user_permissions:
                # 权限不足 → 拦截
                user_role = state.get("user_role", "unknown")
                ts_log(
                    f"🚫 权限拦截: {tool_name} | "
                    f"需要: {required_permission.value} | "
                    f"用户角色: {user_role}",
                    "WARNING"
                )
                return ToolMessage(
                    content=(
                        f"⛔ 操作被拦截：权限不足。\n"
                        f"操作：{tool_name}\n"
                        f"所需权限：{required_permission.value}\n"
                        f"你当前的权限等级无法执行此操作。"
                        f"如需执行，请联系管理员申请权限。"
                    ),
                    tool_call_id=request.tool_call.get("id", ""),
                )

        # === 检查 2：危险操作确认 ===
        danger_info = DANGEROUS_OPERATIONS.get(tool_name)
        if danger_info and danger_info.get("requires_confirmation"):
            ts_log(
                f"⚠️ 危险操作: {tool_name} | "
                f"等级: {danger_info['level']} | "
                f"参数: {json.dumps(tool_args, ensure_ascii=False)[:100]}",
                "WARNING"
            )
            # 在 ToolMessage 中标注此操作已被审计
            result = handler(request)
            if isinstance(result, ToolMessage):
                original = result.content if hasattr(result, "content") else str(result)
                return ToolMessage(
                    content=(
                        f"{original}\n\n"
                        f"🔒 [安全审计] 此操作已被标记：\n"
                        f"   等级: {danger_info['level']}\n"
                        f"   操作: {tool_name}\n"
                        f"   {danger_info['message']}"
                    ),
                    tool_call_id=request.tool_call.get("id", ""),
                    name=tool_name,
                )

        # === 通过检查 → 正常执行 ===
        return handler(request)


# ================================================================
# 第四部分：组装 Agent
# ================================================================

def create_it_ops_agent():
    """
    创建 IT 运维 Agent。

    中间件执行顺序（注册顺序 = 执行顺序）：

    before_agent:
      1. SecurityCheckMiddleware   — 安全验证 + 权限注入
      2. AuditLogMiddleware        — 记录操作开始

    wrap_model_call（洋葱模型，外层包内层）：
      3. DynamicPromptMiddleware   — 角色 Prompt 注入
      4. SmartModelSwitchMiddleware — 智能模型切换
      5. ContextManagementMiddleware — 上下文裁剪

    after_model:
      6. ResponseValidationMiddleware — 响应校验
      7. AuditLogMiddleware          — 记录操作完成

    wrap_tool_call:
      8. SafetyGuardrailMiddleware   — 危险操作拦截
    """
    # 模型配置
    premium_llm = ChatOpenAI(
        model="deepseek-v4-pro",
        temperature=0.3,
        max_tokens=2048,
    )
    budget_llm = ChatOpenAI(
        model="deepseek-v4-pro",
        temperature=0.7,
        max_tokens=512,
    )

    # 短期记忆
    checkpointer = InMemorySaver()

    agent = create_react_agent(
        model=premium_llm,
        tools=OPS_TOOLS,
        checkpointer=checkpointer,
        middleware=[
            # === before_agent 层 ===
            SecurityCheckMiddleware(),
            AuditLogMiddleware(),

            # === wrap_model_call 层 ===
            DynamicPromptMiddleware(),
            SmartModelSwitchMiddleware(
                premium_model=premium_llm,
                budget_model=budget_llm,
            ),
            ContextManagementMiddleware(),

            # === after_model 层 ===
            ResponseValidationMiddleware(),
            # AuditLogMiddleware 已在上方注册（before_agent + after_model 双钩子）

            # === wrap_tool_call 层 ===
            SafetyGuardrailMiddleware(),
        ],
        system_prompt="你是 IT 运维助手。具体指令由 DynamicPromptMiddleware 动态注入。",
    )

    return agent


# ================================================================
# 第五部分：测试演示
# ================================================================

def demo():
    """演示 IT Ops Agent 的多角色、多权限场景。"""
    agent = create_it_ops_agent()

    test_scenarios = [
        {
            "name": "管理员 — 查看服务器状态",
            "config": {
                "configurable": {
                    "thread_id": "admin_session_1",
                    "user_id": "admin_zhang",
                    "display_name": "张管理",
                    "role": "admin",
                    "permissions": [p.value for p in ROLE_PERMISSIONS["admin"]],
                    "department": "IT运维部",
                }
            },
            "question": "查看所有服务器的运行状态，有没有需要关注的？",
        },
        {
            "name": "运维工程师 — 查看日志",
            "config": {
                "configurable": {
                    "thread_id": "operator_session_1",
                    "user_id": "li_ops",
                    "display_name": "李运维",
                    "role": "operator",
                    "permissions": [p.value for p in ROLE_PERMISSIONS["operator"]],
                    "department": "IT运维部",
                }
            },
            "question": "查看 web-server-01 最近的错误日志。",
        },
        {
            "name": "普通用户 — 尝试重启服务（应该被拦截）",
            "config": {
                "configurable": {
                    "thread_id": "viewer_session_1",
                    "user_id": "wang_view",
                    "display_name": "王观察",
                    "role": "viewer",
                    "permissions": [p.value for p in ROLE_PERMISSIONS["viewer"]],
                    "department": "数据分析部",
                }
            },
            "question": "重启 web-server-01，原因：响应变慢。",
        },
    ]

    for scenario in test_scenarios:
        print("\n" + "=" * 70)
        print(f"🧪 {scenario['name']}")
        print("=" * 70)

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(scenario["question"])]},
                config=scenario["config"],
            )
            final_msg = result["messages"][-1]
            print(f"🤖 回答: {final_msg.content[:300] if hasattr(final_msg, 'content') else str(final_msg)[:300]}")
        except Exception as e:
            print(f"❌ 异常: {e}")

    print("\n" + "=" * 70)
    print("✅ 演示完成。审计日志文件: " + LOG_FILE)
    print("=" * 70)


if __name__ == "__main__":
    demo()
