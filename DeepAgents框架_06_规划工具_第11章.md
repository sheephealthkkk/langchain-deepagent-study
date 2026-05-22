## 第十一章：规划工具 — 工作流思维链与 TODO.md

### 11.1 核心：工作流思维链是什么

DeepAgents 的"规划"不是写在某个 Python 模块里的硬编码逻辑——它是 **Agent 利用文件系统工具自己写给自己的 TODO 列表**。

```
BASE_AGENT_PROMPT 里的"理解→行动→验证"三阶段循环
        +
文件系统工具（write_file / edit_file / read_file）
        +
Agent 的自主推理能力
        =
Agent 自己写出 plan.md → 按计划执行 → 完成后划掉
```

**这就像你（Claude Code）的 TodoWrite 工具**——我每接到一个复杂任务，先列出步骤清单，然后逐个执行，完成一个划掉一个。DeepAgents 用完全相同的方式工作，只是它的"TodoWrite"是 `write_file("/workspace/plan.md", ...)`。

### 11.2 对比：传统 Agent vs DeepAgents 规划

```
传统 ReAct Agent:
  用户: "帮我写一个 Flask API，包含认证、数据库、测试"
  
  Thought: 先创建 app.py
  Action: write_file("app.py", "from flask import Flask...")
  Thought: 再加认证
  Action: edit_file("app.py", ...)
  Thought: 还要数据库
  Action: write_file("models.py", "...")
  Thought: 好像忘了什么...测试！
  Action: write_file("test_app.py", "...")
  
  问题：
    ✗ 没有整体规划 → 想到哪做到哪
    ✗ 没有进度跟踪 → 做到一半不知道还剩什么
    ✗ 容易漏步骤 → 最后才想起测试
    ✗ 无法恢复 → 中断后不知道从哪继续


DeepAgents (文件系统规划):
  用户: "帮我写一个 Flask API，包含认证、数据库、测试"
  
  [1] Understand: 读项目现有文件
       ls("/workspace/") → []
       "这是一个新项目，从零开始"
  
  [2] Plan: 写 TODO 文件
       write_file("/workspace/TODO.md",
         "# Flask API 项目计划\n"
         "- [ ] 1. 创建 app.py 主文件\n"
         "- [ ] 2. 实现 JWT 用户认证\n"
         "- [ ] 3. 创建数据库模型 models.py\n"
         "- [ ] 4. 实现 CRUD API 端点\n"
         "- [ ] 5. 编写单元测试 test_app.py\n"
         "- [ ] 6. 添加 requirements.txt\n"
         "- [ ] 7. 最终检查：启动 + 测试"
       )
  
  [3] Execute: 逐个完成
       edit_file("/workspace/TODO.md", 
         "- [ ] 1. 创建 app.py", 
         "- [x] 1. 创建 app.py")  ← 划掉已完成的
       ...逐个执行...
  
  [4] Verify: 对照 TODO 检查
       read_file("/workspace/TODO.md")
       "全部 [x] → 任务完成！"
  
  优势：
    ✓ 先规划后执行 → 不遗漏
    ✓ 进度可见 → TODO.md 就是进度条
    ✓ 可恢复 → 中断后读 TODO.md 继续
    ✓ 可审计 → TODO.md 记录了完整执行过程
```

### 11.3 TODO.md — Agent 的"外部大脑"

TODO.md 不只是记录——它是 Agent 的 **外部工作记忆**。Agent 的上下文窗口有限（~128K tokens），但文件系统无限。把计划写进文件 = 把大脑"外挂"到磁盘。

```
Agent 的两种记忆：

  内部记忆（上下文窗口）              外部记忆（文件系统）
  ─────────────────────               ──────────────────
  对话历史（messages 列表）            /workspace/TODO.md
  容量: 128K tokens                   容量: 无限（磁盘有多大就能多大）
  生命周期: 对话结束 = 丢失             生命周期: 持久化（除非主动删除）
  用途: 当前推理                      用途: 长期规划、跨对话保持
  
  ★ DeepAgents 的创新：用文件系统弥补上下文窗口的容量限制
```

**TODO.md 的典型结构**（Agent 自己写的，不是模板）：

```markdown
# 项目: 构建 RAG 对比报告

## 进度: 3/5 完成

- [x] 1. 搜索 LangChain RAG 最新文档
      → 结果保存在 /workspace/notes/langchain_rag.md
- [x] 2. 搜索 LlamaIndex RAG 最新文档  
      → 结果保存在 /workspace/notes/llamaindex_rag.md
- [x] 3. 搜索 RAG 学术论文
      → 结果保存在 /workspace/notes/papers.md
- [ ] 4. 整理对比分析 → 写报告
      → 待写: /workspace/report.md
- [ ] 5. 验证报告引用 → 修改 → 完成

## 发现
- LangChain 侧重快速原型，LlamaIndex 侧重数据管道
- 2024 RAG 趋势: Agentic RAG, Graph RAG, Multimodal RAG

## 阻塞
- (无)
```

**TODO.md 的几个关键特性**：

| 特性 | 实现方式 | 为什么重要 |
|---|---|---|
| 进度追踪 | `[ ]` → `[x]` 标记 | Agent 一眼看到进展 |
| 中间结果锚定 | 记录文件路径 | 不会"搜了但忘了结果在哪" |
| 发现记录 | `## 发现` 章节 | 关键结论不丢，不被后续消息冲掉 |
| 阻塞标记 | `## 阻塞` 章节 | 中断后恢复，知道为什么停 |
| 跨对话 | 文件持久化 | 下次对话读 TODO.md 继续 |

### 11.4 企业级规划模式

#### 模式 1：层级 TODO（大任务拆小任务）

```markdown
# 项目: 微服务迁移

- [x] 1. 评估现有架构 → /workspace/evaluation.md
- [ ] 2. 用户服务迁移
      - [x] 2.1 写 Dockerfile
      - [ ] 2.2 写 K8s 部署配置
      - [ ] 2.3 集成测试
- [ ] 3. 订单服务迁移
- [ ] 4. 数据迁移
- [ ] 5. 监控和告警配置
```

#### 模式 2：SOP 驱动（标准操作流程模板化）

企业可以将常见任务的 SOP 写成模板文件，Agent 读取模板后按步骤执行：

```markdown
# SOP: 新服务上线路由检查清单
# (保存在 /workspace/templates/deploy_sop.md)

- [ ] 1. 检查 Docker 镜像构建状态
- [ ] 2. 验证 K8s manifest 文件语法
- [ ] 3. 检查依赖服务健康状态
- [ ] 4. 执行滚动更新
- [ ] 5. 验证 /health 端点
- [ ] 6. 检查日志无异常
- [ ] 7. 更新变更管理工单
```

Agent 读取这个 SOP → 逐条执行 → 逐条标记完成 → 生成执行报告。

#### 模式 3：多 Agent 并行规划

```markdown
# 项目: 年度技术报告

## 分配给 research-agent
- [x] 搜索 AI 趋势
- [x] 搜索云原生趋势
- [ ] 搜索安全趋势

## 分配给 writer-agent  
- [ ] 写 AI 章节（等待 research-agent 完成）

## 分配给 reviewer-agent
- [ ] 初审报告
- [ ] 事实核查
```

### 11.5 规划工具在企业中的实际应用

**场景 1：代码审查自动化**

```
Agent 收到 PR → 
  写 TODO.md:
    - [ ] 读 PR 描述
    - [ ] 读变更的文件列表
    - [ ] 逐个文件审查
    - [ ] 运行测试
    - [ ] 生成审查报告
  → 执行 → 报告写入 /workspace/review_report.md
```

**场景 2：故障排查 Runbook**

```
Agent 收到 "服务响应 5xx" →
  写 TODO.md:
    - [ ] 查服务状态
    - [ ] 查最近日志（grep ERROR）
    - [ ] 查系统资源（CPU/内存/磁盘）
    - [ ] 查依赖服务
    - [ ] 定位根因
    - [ ] 写故障报告 + 建议处置
  → 执行 → 报告 + 告警
```

**场景 3：数据管道监控**

```
Agent 定时触发 →
  写 TODO.md:
    - [ ] 检查昨天的数据导入是否完成
    - [ ] 验证数据质量（行数、空值率）
    - [ ] 检查下游依赖任务状态
    - [ ] 如有异常 → 生成告警 + 写原因分析
  → 执行 → 日报 + 异常告警
```

### 11.6 要点总结

```
规划工具的核心思想：

  传统方式：Prompt 中写 "Let's think step by step"
            → LLM 在脑子里想，不记录 → 容易忘、不可恢复

  DeepAgents：文件系统 = 外部大脑
            → write_file("TODO.md") = 记住计划
            → edit_file("TODO.md") = 更新进度  
            → 任何时候 read_file("TODO.md") = 知道做到哪了

类比：
  你（Claude Code）的 TodoWrite 工具 ≈ DeepAgents 的 write_file("TODO.md")
  都是 —— 把计划写下来 → 按计划执行 → 完成标记 → 不遗漏、可恢复

企业级应用：
  SOP 模板 — 标准化操作流程
  层级 TODO — 大任务拆小任务
  多 Agent 并行 — 各子 Agent 独立 TODO
  故障 Runbook — 自动化故障排查
  数据管道 — 定时巡检日报
```

---

