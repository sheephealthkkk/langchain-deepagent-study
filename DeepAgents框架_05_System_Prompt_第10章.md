## 第十章：System Prompt — DeepAgents 的"宪法"与三大中间件契约

### 10.1 为什么 System Prompt 是 DeepAgents 的核心

在 LangChain 中，System Prompt 是你自己写的几行文字。在 DeepAgents 中，System Prompt 不是一段静态文本——它是 **FilesystemMiddleware、MemoryMiddleware、SubAgentMiddleware 三大中间件的协作契约**。

```
┌─────────────────────────────────────────────────────────────┐
│              DeepAgents System Prompt 分层架构               │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  BASE_AGENT_PROMPT（基础行为层）                       │ │
│  │  "你是 deep agent，用工具帮助用户完成任务"               │ │
│  │  定义：角色本质、行为准则、任务执行模型                   │ │
│  └──────────────────────────┬────────────────────────────┘ │
│                             │ append_to_system_message()    │
│  ┌──────────────────────────┴────────────────────────────┐ │
│  │  _FILESYSTEM_SYSTEM_PROMPT_TEMPLATE（文件系统层）      │ │
│  │  "你有文件系统工具：ls/read/write/edit/glob/grep"      │ │
│  │  定义：文件操作规范、命名约定、错误处理                   │ │
│  └──────────────────────────┬────────────────────────────┘ │
│                             │ append_to_system_message()    │
│  ┌──────────────────────────┴────────────────────────────┐ │
│  │  EXECUTION_SYSTEM_PROMPT（代码执行层）                  │ │
│  │  "你有 execute 工具，在沙箱中运行 shell 命令"            │ │
│  │  定义：沙箱安全边界、超时限制、输出格式                   │ │
│  └──────────────────────────┬────────────────────────────┘ │
│                             │                               │
│  ┌──────────────────────────┴────────────────────────────┐ │
│  │  MemoryMiddleware 注入（记忆层）                        │ │
│  │  "你可以记住和回忆信息，跨对话保持上下文"                 │ │
│  └──────────────────────────┬────────────────────────────┘ │
│                             │                               │
│  ┌──────────────────────────┴────────────────────────────┐ │
│  │  SubAgentMiddleware 注入（委派层）                      │ │
│  │  "你可以调用子 Agent 处理复杂子任务"                     │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  最终 System Prompt = BASE + Filesystem + Execute +         │
│                        Memory + SubAgent + 用户自定义         │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 BASE_AGENT_PROMPT — 角色本质定义

这是 DeepAgents 的"出厂默认人格"，通过 `dynamic_prompt` 钩子在每次 LLM 调用前注入。与 LangChain 手写 System Prompt 的区别是——它精确约束了 Agent 的行为模式，不是"建议"而是"规则"。

**完整内容与逐段解读**：

```
You are a deep agent, an AI assistant that helps users
accomplish tasks using tools.
```

> **角色定位**："deep agent" = 能深入完成任务的 Agent，不是简单问答 ChatBot。"using tools" = 主动使用工具的 Agent，不是被动等待指令。

```
## Core Behavior

- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble ("Sure!", "Great question!",
  "I'll now...").
- Don't say "I'll now do X" — just do it.
- If the request is underspecified, ask only the minimum
  followup needed to take the next useful action.
- If asked how to approach something, explain first, then act.
```

> **行为准则**：禁止废话前缀（"好的，我来帮你..."）——这是 Agent 而不是客服。先理解再行动。不明确时只问最少的问题。

```
## Doing Tasks

When the user asks you to do something:

1. **Understand first** — read relevant files, check existing
   patterns. Quick but thorough.
2. **Act** — implement the solution. Work quickly but accurately.
3. **Verify** — check your work against what was asked, not
   against your own output.

Keep working until the task is fully complete. Don't stop
partway and explain what you would do — just do it.
```

> **任务模型**：三阶段循环（Understand → Act → Verify）。强调"做到完成为止，不要半路停下来解释"——这是 Agent 和 Chain 的本质区别。

```
**When things go wrong:**
- If something fails repeatedly, stop and analyze *why*
  — don't keep retrying the same approach.
- If you're blocked, tell the user what's wrong and ask.
```

> **失败处理**：不要无脑重试（传统 Agent 的最大问题之一）。分析原因 → 向用户求助。

```
## Professional Objectivity

- Prioritize accuracy over validating the user's beliefs
- Disagree respectfully when the user is incorrect
- Avoid unnecessary superlatives, praise, or emotional validation
```

> **职业客观性**：不迎合用户、不用过度赞美。准确 > 情绪价值。

**BASE_AGENT_PROMPT 的核心设计思想**：

| 传统 LLM System Prompt | BASE_AGENT_PROMPT |
|---|---|
| "你是一个友好的助手，用中文回复" | "你是 deep agent，用工具帮助用户完成任务" |
| 角色描述（模糊） | 行为规则（精确） |
| 被动等待指令 | 主动使用工具、自主执行到完成 |
| 不控制输出风格 | 禁止废话前缀、先做后说 |
| 无失败处理 | 分析原因 → 求助（不无脑重试） |

### 10.3 提示词 = 三大中间件的协作契约

DeepAgents 的三个核心中间件（Filesystem、Memory、SubAgent）不是通过代码耦合协同的——它们通过 **System Prompt 注入各自的"使用说明"** 来协同。每个中间件在 `before_model` 阶段把自己的指令追加到 SystemMessage 末尾。

```python
# FilesystemMiddleware 的 before_model 钩子（简化）
def before_model(self, state, runtime):
    # 构建文件系统提示词
    prompt = _FILESYSTEM_SYSTEM_PROMPT_TEMPLATE.format(
        available_tools=tool_list,    # 动态注入当前可用的工具名
    )
    # 如果有 execute 工具 → 追加执行提示词
    if has_execute_tool:
        prompt += "\n\n" + EXECUTION_SYSTEM_PROMPT
    
    # 追加到已有的 SystemMessage（不覆盖！）
    return {"messages": [SystemMessage(content=prompt)]}

# MemoryMiddleware 和 SubAgentMiddleware 同理——
# 各自追加自己的"使用说明"，最终构成完整契约。
```

**为什么用 Prompt 而不是代码耦合？**

| 代码耦合方式 | Prompt 契约方式 |
|---|---|
| 中间件 A 直接调用中间件 B 的方法 | 中间件 A 注入"如何使用 B 的说明" → LLM 自己决定 |
| B 的接口改了 → A 的代码崩溃 | B 的注入改了 → LLM 自动适应新的说明 |
| 加新中间件 C → 需要修改 A 和 B | 加新中间件 C → C 注入自己的说明，A 和 B 不受影响 |
| 中间件的组合方式是硬编码的 | LLM 根据聚合的 Prompt 动态决定调用策略 |

### 10.4 文件系统提示词详解

文件系统提示词分为两部分：**通用文件操作规范** + **每个工具的使用说明**。

**通用规范**（`_FILESYSTEM_SYSTEM_PROMPT_TEMPLATE`）：

```
## Following Conventions

- Read files before editing — understand existing content
  before making changes
- Mimic existing style, naming conventions, and patterns

## Filesystem Tools `ls`, `read_file`, `write_file`,
  `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with
using these tools.
All file paths must start with a /. Follow the tool docs
for the available tools, and use pagination (offset/limit)
when reading large files.
```

> **解决什么问题**：
> - "先读再改"规则 → 防止 Agent 盲目覆盖文件
> - "模仿已有风格" → 保证生成代码/文档的一致性
> - "分页读大文件" → 防止 Agent 一次读入 10000 行文本撑爆上下文

**七个文件系统工具描述**：

| 工具 | 描述要求 | 解决的问题 |
|---|---|---|
| `ls` | "探索文件系统的正确方式。几乎总是应该先 ls 再 read/edit" | 防止 Agent 直接猜文件路径 |
| `read_file` | "假设这个工具能读任何文件。读不存在的文件会返回错误" | 让 Agent 放心尝试，相信错误提示 |
| `write_file` | "创建新文件。优先用 edit_file 修改已有文件" | 防止无谓的新文件创建 |
| `edit_file` | "精确字符串替换。必须先用 read_file 读过才能 edit" | 防止盲目编辑、强制先读 |
| `glob` | "通配符模式匹配。`**` 匹配任意层级目录" | 防止 Agent 手动遍历目录树 |
| `grep` | "按文本模式搜索文件，非正则" | 防止 Agent 读每个文件后手动搜索 |
| `execute` | "沙箱中运行 shell 命令，返回输出和退出码" | 定义安全边界、超时和输出格式 |

### 10.5 执行工具提示词 — 沙箱安全契约

```
## Execute Tool `execute`

You have access to an `execute` tool for running shell commands
in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and
other shell operations.

- execute: run a shell command in the sandbox
  (returns output and exit code)

Before executing, please consider:
1. Is the command safe?
2. Will it complete within a reasonable time?
3. Is the working directory correct?
```

> **解决什么问题**：
> - "沙箱环境"提示 → Agent 知道命令不会影响真实系统（敢执行）
> - "返回输出和退出码" → Agent 知道怎么判断成功/失败
> - 三个检查 → 防止危险命令、防止超时等待、防止路径错误

### 10.6 子 Agent 调用协议

SubAgentMiddleware 注入的不是固定的 prompt，而是**动态生成的工具描述**——每个 SubAgent 在主 Agent 眼中就是一个"能处理某类任务的工具"：

```python
# SubAgentMiddleware 内部（简化）
# 用户定义的 SubAgent：
#   SubAgent(name="researcher", description="搜索和整理信息...")
# 注入到主 Agent 的 System Prompt：
#   "你可以调用 researcher 工具来搜索和整理信息..."
#   "researcher 是这个子 Agent 的描述..."
```

**协议的具体内容**（LLM 视角）：

```
## Sub-Agent Tools

You can delegate complex sub-tasks to specialized sub-agents.
Each sub-agent appears as a tool you can call.

- researcher: Search and compile information. Best for
  literature review and data collection.
- code-reviewer: Review code quality, security, and best
  practices.
- writer: Write reports based on research materials.

To use a sub-agent, call its tool with the task description.
The sub-agent will execute independently and return results.
```

> **这个协议达成什么效果**：
> - 主 Agent 不需要知道子 Agent 内部如何实现（黑盒调用）
> - 子 Agent 失败了 → 主 Agent 看到的是工具返回错误（和解法一样）
> - 加新子 Agent → 只是工具列表多了一项，主 Agent 逻辑不动

---

### 10.7 System Prompt 动态注入流程

```
每次 LLM 调用前（before_model 钩子）：

[1] BASE_AGENT_PROMPT（来自 create_deep_agent 或 HarnessProfile）
      │
      ▼ SystemMessage(content=BASE_AGENT_PROMPT)
      │
[2] FilesystemMiddleware.before_model()
      │ 检测可用工具 → 拼装 _FILESYSTEM_SYSTEM_PROMPT_TEMPLATE
      │ 如果有 execute → 追加 EXECUTION_SYSTEM_PROMPT
      │ append_to_system_message() 追加到已有 SystemMessage
      ▼
[3] MemoryMiddleware.before_model()
      │ 如果启用了 memory → 追加记忆相关提示词
      │ append_to_system_message() 追加
      ▼
[4] SubAgentMiddleware.before_model()
      │ 如果有子 Agent → 为每个 SubAgent 生成工具描述
      │ append_to_system_message() 追加
      ▼
[5] 用户自定义 middleware
      │ 如果有自定义中间件的 before_model → 继续追加
      ▼
[6] 最终 SystemMessage 发送给 LLM
```

**与传统 LangChain Prompt 的差异**：

| | LangChain | DeepAgents |
|---|---|---|
| System Prompt 来源 | 手动写一段文本 | 多层动态拼装 |
| 修改方式 | 改字符串 → 重启 | 加中间件 → 自动注入 |
| 不同环境差异 | 手写多套 Prompt | Backend + 中间件自动适配 |
| 工具描述 | 手动写 description | 中间件自动生成 + 注入 |
| Prompt 可观测性 | 启动时固定，运行时不可见 | 每次 LLM 调用前动态生成 |

### 10.8 LangChain vs DeepAgents System Prompt 对比

```python
# ===== LangChain 方式：手动写一切 =====
system_prompt = """
你是 AI 研究助手。

规则：
- 先读文件再修改
- 错误时分析原因，不要重试
- 保持简洁

工具：
- ls: 列出目录
- read_file: 读取文件
- write_file: 写入文件
- search_web: 搜索网络
"""

agent = create_agent(
    llm=llm,
    tools=[ls_tool, read_tool, write_tool, search_tool],
    system_prompt=system_prompt,
)
# 问题：
# 1. 工具描述是你手写的 — 和实际工具可能不同步
# 2. 规则是你手写的 — 忘了加"先读再写"就是忘了
# 3. 加新工具需要改 Prompt — 忘了改就不同步

# ===== DeepAgents 方式：中间件自动注入 =====
agent = create_deep_agent(
    model="deepseek-v4-pro",
    tools=[search_web],  # 不需要手动写文件系统工具
    system_prompt="你是 AI 研究助手。",  # 只写业务相关的
)
# 内部自动注入：
# ✓ BASE_AGENT_PROMPT（角色 + 行为规则）
# ✓ Filesystem 提示词（7 个工具 + 使用规范）
# ✓ Execute 提示词（沙箱安全）
# ✓ Memory 提示词（如果启用）
# ✓ SubAgent 提示词（如果配置）
# → 最终 System Prompt 可能是你写的那行的 10 倍长，
#   但每一段都是对应中间件自动生成的，保证一致性。
```

---

### 10.9 要点总结

```
System Prompt 在 DeepAgents 中的角色演变：

LangChain 时代：System Prompt = 一段固定的文字
DeepAgents 时代：System Prompt = 中间件的协作契约

每个中间件通过 before_model 注入自己的 "使用说明"：
  FilesystemMiddleware → 文件操作规范 + 7 个工具描述
  MemoryMiddleware    → 记忆存取指令
  SubAgentMiddleware  → 子 Agent 调用协议

好处：
  ✓ 中间件解耦 — 通过 Prompt 协同而非代码耦合
  ✓ 自动同步 — 工具列表变了，Prompt 自动更新
  ✓ 动态适配 — 不同 Backend 有不同工具，Prompt 自动适配
  ✓ 可观测 — 每次 LLM 调用的完整 Prompt 可在 LangSmith 中查看
```

---

