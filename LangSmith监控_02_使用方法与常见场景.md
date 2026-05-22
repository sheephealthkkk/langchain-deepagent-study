## 第三章：使用方法与配置

### 3.1 最小化配置（3 行代码启用）

```bash
# 1. 安装
pip install langsmith

# 2. 设置环境变量
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="lsv2_pt_..."    # 从 https://smith.langchain.com 获取
export LANGCHAIN_PROJECT="my-rag-agent"   # Trace 归属的 Project
```

**代码零改动。** 设置这三个环境变量后，所有 `chain.invoke()` / `agent.invoke()` 自动上报到 LangSmith。

```
环境变量设置前：                                设置后：
  chain.invoke() → 结果                          chain.invoke() → 结果
                                                    │
                                                    ▼
                                              LangSmith 自动捕获
                                              (Trace → Run → Token → Latency)
```

### 3.2 代码中显式配置

```python
import os
from langsmith import Client

# === 方式 1：环境变量（推荐，全局生效）===
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_your_key"
os.environ["LANGCHAIN_PROJECT"] = "my-rag-agent"

# === 方式 2：代码中传 config（每次调用单独配置）===
from langchain_core.runnables import RunnableConfig

chain.invoke(
    input,
    config=RunnableConfig(
        tags=["production", "v2.1"],
        metadata={
            "user_id": "alice",
            "session_id": "chat_abc",
        },
        run_name="alice_question_about_rag",  # ← 给这次 Trace 取个名字
    ),
)

# === 方式 3：在 Runnable 上绑定配置 ===
tagged_chain = chain.with_config(
    tags=["production"],
    metadata={"environment": "prod"},
)
# 之后 tagged_chain 的所有调用都带这些 tags/metadata
```

### 3.3 自定义 Trace 上报

```python
from langsmith import traceable

# === 方式 1：装饰器（监控任意函数）===
@traceable(run_type="tool", name="my_custom_tool")
def my_function(query: str) -> str:
    """这个函数的每次调用都会作为 Run 上报到 LangSmith。"""
    return f"处理 {query}"

# === 方式 2：手动创建 Trace ===
from langsmith import Client

client = Client()

# 手动创建一个 Run
run = client.create_run(
    name="manual_check",
    run_type="chain",
    inputs={"question": "What is RAG?"},
    project_name="my-rag-agent",
)
# ... 执行业务逻辑 ...
client.update_run(
    run.id,
    outputs={"answer": "RAG is Retrieval-Augmented Generation."},
    end_time=...,  # 自动记录耗时
)
```

### 3.4 在 LangSmith 平台中查看

#### 查看 Trace 详情

```
Trace 详情页：
┌─────────────────────────────────────────────────────────────┐
│ Trace: alice_question_about_rag                              │
│ Project: my-rag-agent                                       │
│ Tags: [production, v2.1]                                    │
│ Latency: 2.3s │ Tokens: 520 │ Cost: $0.003                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ▼ RunnableSequence (chain)                          2.3s   │
│    ├─ Input: {"question": "What is RAG?"}                   │
│    ├─ Output: "RAG (Retrieval-Augmented Generation)..."     │
│    │                                                        │
│    ▼ retriever                                    0.23s     │
│      ├─ Input: "What is RAG?"                               │
│      ├─ Output: [Document("RAG is..."), Document("...")]    │
│      └─ Metadata: {"k": 4, "search_type": "similarity"}    │
│                                                             │
│    ▼ ChatOpenAI                                   1.07s     │
│      ├─ Input: [SystemMessage, HumanMessage]                │
│      ├─ Output: AIMessage("RAG (Retrieval-Augmented...)")   │
│      ├─ Tokens: 320 prompt + 80 completion = 400 total     │
│      ├─ Model: deepseek-v4-pro                              │
│      ├─ Temperature: 0.7                                    │
│      └─ Cost: $0.002                                        │
│                                                             │
│  Feedback: 👍 (thumbs up, "回答很准确")                      │
└─────────────────────────────────────────────────────────────┘
```

#### 过滤与搜索

```
LangSmith 查询语法：

# 按 Tags 过滤
tags:production

# 按 Metadata 过滤
metadata.user_id:alice

# 按 Run 类型过滤
run_type:llm

# 按错误过滤
error:*

# 组合查询
tags:production AND run_type:llm AND error:*

# 按时间过滤
start_time:[2026-05-01 TO 2026-05-10]
```

### 3.5 采样配置（控制上报量）

```python
# 环境变量控制采样率（生产环境推荐）
os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = "0.1"
# ↑ 只上报 10% 的 Trace。减少存储和网络开销。
# 生产环境高并发时建议 0.01~0.1
```

---

## 第四章：常见使用场景

### 场景 1：定位性能瓶颈

```
步骤：
1. 在 LangSmith 中打开 Project
2. 按 "Latency" 降序排列 Trace
3. 展开最慢的几个 Trace
4. 看哪个 Run 耗时最长 → 这就是瓶颈

示例：
  Trace latency: 8.5s
  ├─ retriever: 0.3s  ✓ 正常
  ├─ LLM call #1: 1.2s ✓ 正常
  ├─ LLM call #2: 6.8s ✗ 异常！分析：这次 LLM 输入 token 过多 → 需要上下文裁剪
  └─ parser: 0.1s   ✓ 正常

结论：LLM 第二次调用的 Prompt 太长。优化 → 加 SummarizationMiddleware。
```

### 场景 2：对比实验（A/B 测试）

```python
# 实验 A：用不同 Prompt
chain_a = (
    prompt_v1 | llm | parser
).with_config(tags=["experiment:prompt_v1"])

# 实验 B：用不同 Prompt
chain_b = (
    prompt_v2 | llm | parser
).with_config(tags=["experiment:prompt_v2"])

# 同时跑 100 条测试
for q in test_questions:
    chain_a.invoke(q)  # → LangSmith 中标签 experiment:prompt_v1
    chain_b.invoke(q)  # → LangSmith 中标签 experiment:prompt_v2

# 然后在 LangSmith 中：
# 1. 按 tags 分组 → 查看两组 Trace
# 2. 对比平均 latency、Token 用量、Feedback 评分
# 3. 决定哪个 Prompt 更好
```

### 场景 3：Debug 单条错误 Trace

```
用户反馈："这个回答完全不对！"

调试流程：
1. 用用户 ID + 时间范围找到该 Trace
2. 展开 Run → 看 retriever 的输出
   → 发现：检索到的 4 个 Document 和用户问题完全不相关
3. 看 embedding 模型 → 发现用了英文模型处理中文问题
4. 修复：换成 BAAI/bge-small-zh-v1.5（中文嵌入模型）
5. 重新测试 → 检索准确 → 回答正确

整个流程不需要在生产环境加日志、不需要重新部署、不需要复现。
```

### 场景 4：成本追踪

```
LangSmith 自动展示每次 LLM 调用的 Token 用量和成本：

1. 按 Project 查看总成本趋势图
2. 按 Model 查看各模型成本占比
3. 按 Trace 查看单次调用成本
4. 设置成本告警：日消费 > $50 → 发通知

优化方向：
  - 发现 80% 成本来自 20% 的长 Trace → 优化高频问题的缓存
  - 发现某个用户占用 40% Token → 检查是否有滥用
```

---

