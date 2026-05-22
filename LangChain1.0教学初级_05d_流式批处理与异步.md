## 一、三种调用模式对比

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0.8)

# ===== 模式1：invoke — 等全部完成，一次性拿结果 =====
result = llm.invoke("Hello")          # 阻塞 2~5 秒
print(result.content)                 # → 完整回答

# ===== 模式2：stream — 边生成边拿，逐 token 返回 =====
for chunk in llm.stream("讲个故事"):
    print(chunk.content, end="")      # 一个字一个字蹦出来

# ===== 模式3：batch — 并发处理多个 =====
results = llm.batch(["Hello", "Hi", "Hey"])  # 3 个请求并发
for r in results:
    print(r.content)
```

| | `invoke` | `stream` | `batch` |
|---|---|---|---|
| 返回时机 | 全完成后 | 逐 token | 全完成后 |
| 用户感知 | 等 | 实时输出 | 等（但总时间短） |
| 内存占用 | 一次完整结果 | 逐块，低 | N 个完整结果 |
| 适用 | 单次问答 | 聊天 UI、长文本 | 批量评估、离线处理 |

## 二、流式处理详解

### 2.1 基础流式：`stream`

```python
# 同步流式
for chunk in llm.stream("解释相对论"):
    print(chunk.content, end="", flush=True)
    # chunk 是 AIMessageChunk，content 是本次新增的 token 片段

# 异步流式
async for chunk in llm.astream("解释相对论"):
    print(chunk.content, end="", flush=True)
```

**`stream` 返回的不是最终 AIMessage，而是 `AIMessageChunk`** —— 每个 chunk 是最终消息的一小段。chunk 可以累加：

```python
full: AIMessageChunk = None
for chunk in llm.stream("Hello"):
    full = chunk if full is None else full + chunk
# full 现在等价于 llm.invoke("Hello") 的结果
```

### 2.2 带事件的流式：`astream_events` — 完整的生命周期

这是最强大的流式模式。**每个组件（LLM、Retriever、Tool、Prompt、Chain）的每次执行都会触发 start → stream → end 三阶段事件。**

```python
async for event in chain.astream_events("What is RAG?", version="v2"):
    print(f"[{event['event']}] {event['name']}")
```

**全部 21 种标准事件**：

```
7 种 Runnable 类型 × 3 个生命周期阶段 = 21 种事件

on_chain_start        on_chain_stream        on_chain_end
on_chat_model_start   on_chat_model_stream   on_chat_model_end
on_llm_start          on_llm_stream          on_llm_end
on_prompt_start       on_prompt_stream       on_prompt_end
on_tool_start         on_tool_stream         on_tool_end
on_retriever_start    on_retriever_stream    on_retriever_end
on_embedding_start    on_embedding_stream    on_embedding_end

+ on_custom_event（用户自定义）
```

**每种事件携带的数据（`event['data']`）**：

| 事件阶段 | data 中的字段 | 含义 |
|---|---|---|
| `start` | `input` | Runnable 收到的输入 |
| `stream` | `chunk` | 本次流式块 |
| `end` | `output` + `input` | 最终输出（以及可能已知的输入） |
| 错误时 | `error` | 异常对象 |

### 2.3 实战：用 `astream_events` 调试 RAG 链

```python
async def debug_rag_chain(chain, question: str):
    """逐事件打印 RAG 链的执行过程。"""
    async for event in chain.astream_events(question, version="v2"):
        etype = event["event"]
        ename = event["name"]
        data = event["data"]
        parents = event["parent_ids"]

        indent = "  " * len(parents)  # 根据嵌套深度缩进

        if etype == "on_chain_start":
            print(f"{indent}▶ {ename} 开始")
            if data.get("input"):
                inp = str(data["input"])[:100]
                print(f"{indent}  输入: {inp}")

        elif etype == "on_chat_model_start":
            print(f"{indent}🤖 {ename} 调用中...")

        elif etype == "on_retriever_start":
            print(f"{indent}🔍 {ename} 检索中...")

        elif etype == "on_chat_model_stream":
            chunk_data = data.get("chunk", {})
            if hasattr(chunk_data, "content") and chunk_data.content:
                print(f"{indent}  💬 {chunk_data.content}", end="", flush=True)

        elif etype == "on_chain_end":
            out = str(data.get("output", ""))[:150]
            print(f"\n{indent}◀ {ename} 结束 → {out}")

        elif etype == "on_tool_start":
            print(f"{indent}🔧 {ename} 调用工具...")

        elif etype == "on_tool_end":
            tool_out = str(data.get("output", ""))[:100]
            print(f"{indent}🔧 {ename} 工具返回: {tool_out}")
```

**实际输出效果**（以我们的 RAG 链为例）：

```
▶ RunnableSequence 开始
  输入: What is RAG?
  ▶ history_aware_retriever 开始
    🤖 ChatOpenAI 调用中...
    💬 独立问题：What is RAG?
    ◀ ChatOpenAI 结束
    🔍 retriever 检索中...
    ◀ retriever 结束
  ▶ stuff_documents_chain 开始
    🤖 ChatOpenAI 调用中...
    💬 RAG (Retrieval-Augmented Generation) 是...
    ◀ ChatOpenAI 结束
  ◀ RunnableSequence 结束
```

### 2.4 `astream_events` 的过滤参数

```python
chain.astream_events(
    input,
    version="v2",
    include_types=["chat_model"],          # 只要 LLM 相关事件
    include_names=["ChatOpenAI"],          # 只要指定名字的 Runnable
    include_tags=["production"],           # 只要打了特定标签的
    exclude_types=["chain"],               # 排除 Chain 级别事件
    exclude_names=["rewrite_chain"],       # 排除特定名
)
```

## 三、事件体系可以干什么

### 用途 1：调试 LLM 流程 — 看清每一步的输入输出

```python
# 快速定位哪个环节出了问题
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_end":
        print(f"Prompt: {event['data']['input']}")     # 模型实际收到的 Prompt
        print(f"Response: {event['data']['output']}")  # 模型返回的完整响应
```

### 用途 2：Token 用量实时监控

```python
total_tokens = 0
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_end":
        output = event["data"]["output"]
        if hasattr(output, "usage_metadata"):
            usage = output.usage_metadata
            total_tokens += usage.get("total_tokens", 0)
            print(f"[本次消耗 {usage['total_tokens']} tokens，累计 {total_tokens}]")
```

### 用途 3：性能剖析 — 每一步的耗时

```python
import time

timing = {}
async for event in chain.astream_events(input, version="v2"):
    run_id = event["run_id"]
    if event["event"].endswith("_start"):
        timing[run_id] = time.monotonic()
    elif event["event"].endswith("_end"):
        elapsed = time.monotonic() - timing.pop(run_id, 0)
        print(f"{event['name']}: {elapsed:.2f}s")
```

### 用途 4：自定义业务事件

```python
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableLambda

def my_step(x):
    dispatch_custom_event("user_trace", {"stage": "data_validated", "count": len(x)})
    return x

chain_with_trace = RunnableLambda(my_step) | llm
# astream_events 会包含 on_custom_event
```

### 用途 5：流式 UI — 前端实时展示

```python
# 后端用 astream_events 分发给前端
# → 前端：打字机效果 + "正在检索..."状态 + token 计数
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_retriever_start":
        yield {"status": "retrieving"}
    elif event["event"] == "on_chat_model_stream":
        yield {"status": "streaming", "token": chunk.content}
    elif event["event"] == "on_chain_end":
        yield {"status": "done", "answer": event["data"]["output"]}
```

---

## 四、批处理

### 4.1 基础用法

```python
# 同步批处理
results = chain.batch([
    "What is LangChain?",
    "What is LangGraph?",
    "What is RAG?",
])
# 内部自动并行，总时间 ≈ max(单个时间)，不是 sum

# 异步批处理
results = await chain.abatch(["Q1", "Q2", "Q3"])

# 控制并发数
results = chain.batch(inputs, config={"max_concurrency": 2})
```

### 4.2 批处理配置

```python
from langchain_core.runnables import RunnableConfig

results = chain.batch(
    inputs,
    config=RunnableConfig(
        max_concurrency=3,     # 最多同时跑 3 个
        timeout=30,            # 单个超时 30s
        max_retries=2,         # 失败重试 2 次
    ),
    return_exceptions=True,   # 某个失败不终止，返回异常对象
)
```

---

## 五、异步并发处理

### 5.1 同步 vs 异步的底层差异

```python
# === 同步 ===
# 所有操作在当前线程串行执行，线程阻塞等待
r1 = llm.invoke("Q1")  # 阻塞 3s，CPU 空闲
r2 = llm.invoke("Q2")  # 阻塞 3s，CPU 空闲
r3 = llm.invoke("Q3")  # 阻塞 3s，CPU 空闲
# 总耗时 ≈ 9s，CPU 全程空闲

# === 异步 ===
# 一个线程管理多个并发请求，等待时不阻塞
import asyncio
async def main():
    tasks = [llm.ainvoke(q) for q in ["Q1", "Q2", "Q3"]]
    r1, r2, r3 = await asyncio.gather(*tasks)
# 总耗时 ≈ 3s（最慢的那个），3 个请求同时在网络传输中
```

**核心区别**：LLM 调用的主要耗时是**网络 I/O**（send request → wait → receive response），CPU 基本闲置。同步时 CPU 傻等；异步时 CPU 在等 A 回复的间隙去发 B 的请求、收 C 的回复。

### 5.2 LLM 调用本质上都是异步的

`ChatOpenAI` 底层使用 `httpx`（支持同步/异步双模式）。同步 `invoke` 内部实际是 `asyncio.run(self._agenerate(...))`：

```python
# ChatOpenAI 内部（简化）
def _generate(self, messages, **kwargs):
    # 同步方法，内部其实跑了一个 async loop
    return asyncio.run(self._agenerate(messages, **kwargs))

async def _agenerate(self, messages, **kwargs):
    # 真正的异步 HTTP 请求
    response = await self.async_client.chat.completions.create(...)
```

### 5.3 什么时候异步才能提速

| 场景 | 同步耗时 | 异步耗时 | 提速？ |
|---|---|---|---|
| **1 个 LLM 调用** | 3s | 3s | 不，单请求无并发 |
| **3 个 LLM 调用（同一 API）** | 9s | 3s | **是，3 倍** |
| **1 个 RAG：retrieve → LLM → parse**（串行依赖） | 4s | 4s | 不，有依赖不能并行 |
| **CPU 密集型（如本地 embedding 1000 条）** | 60s | 60s | 不，CPU 瓶颈，需多进程 |
| **混合：retrieve + LLM summary + LLM translation** | 8s | 3s | **是，3 个独立任务** |
| **同一 API 有并发限制（5 QPS）** | — | — | **可能更慢**，被限流 |

**结论：异步只在 I/O 密集型 + 多独立任务 + 无并发限制时提速。CPU 密集（本地 Embedding）需要多进程；有依赖关系（串联 Chain）无法加速。**

### 5.4 异步编程模式

```python
# 模式1：asyncio.gather — 并发跑多个独立任务
tasks = [chain.ainvoke(q) for q in questions]
results = await asyncio.gather(*tasks)

# 模式2：asyncio.Semaphore — 控制并发数
sem = asyncio.Semaphore(3)
async def limited_invoke(q):
    async with sem:
        return await chain.ainvoke(q)
results = await asyncio.gather(*[limited_invoke(q) for q in questions])

# 模式3：asyncio.as_completed — 先完成的先用
for coro in asyncio.as_completed([chain.ainvoke(q) for q in questions]):
    result = await coro
    print(f"完成: {result}")

# 模式4：asyncio.wait_for — 加超时
try:
    result = await asyncio.wait_for(chain.ainvoke(q), timeout=10)
except asyncio.TimeoutError:
    result = "超时"
```

### 5.5 异步 + 流式

```python
async def stream_all(questions: list[str]):
    """同时对多个问题流式输出。"""
    async def stream_one(q):
        async for chunk in chain.astream(q):
            yield (q, chunk)

    tasks = [stream_one(q) for q in questions]
    # 多个流并行处理
```

---

## 六、RunnableConfig 完整配置项

```python
from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {
    # ===== 并发控制 =====
    "max_concurrency": 5,         # batch/并发时的最大并行数

    # ===== 超时与重试 =====
    "timeout": 30.0,              # 单个操作超时（秒）
    "max_retries": 3,             # 失败自动重试次数

    # ===== 追踪与调试 =====
    "run_name": "my_rag_chain",   # 本次运行的名称（出现在事件中）
    "tags": ["production", "v2"], # 标签，可在事件中过滤
    "metadata": {                 # 自定义元数据
        "user_id": "123",
        "session_id": "abc",
        "version": "1.0.0",
    },
    "run_id": "custom-uuid",     # 指定 run_id，否则自动生成

    # ===== 回调 =====
    "callbacks": [                # CallbackHandler 列表
        MyLoggingHandler(),
        MyMetricsHandler(),
    ],

    # ===== 可配置字段 =====
    "configurable": {             # 传递给 Runnable 的运行时参数
        "session_id": "user_123",
        "llm": "deepseek-v4-pro",
    },

    # ===== 递归限制 =====
    "recursion_limit": 25,        # Agent 循环的最大递归次数
}
```

### 各配置项的应用场景

| 配置项 | 典型场景 |
|---|---|
| `max_concurrency` | 同一 API 有并发限制时设为限制值（如 5 QPS 设 4） |
| `timeout` | 生产环境保护，防止单个 slow request 卡死整个服务 |
| `max_retries` | API 不稳定时的自动容错 |
| `tags` | 区分环境（`dev`/`staging`/`prod`），按标签过滤事件 |
| `metadata` | 记录用户 ID、请求来源，用于日志聚合和成本分摊 |
| `configurable` | 同一 Chain 在不同 session 间切换参数 |
| `recursion_limit` | Agent 用，防止无限工具调用循环 |

### config 的传递方式

```python
# 方式1：invoke 时传
chain.invoke(input, config={"tags": ["prod"], "max_concurrency": 3})

# 方式2：.with_config() 绑定到链
production_chain = chain.with_config(
    tags=["production"],
    metadata={"env": "prod"},
    timeout=60,
)

# 方式3：.with_fallbacks() 降级
robust_chain = chain.with_fallbacks([backup_chain])
robust_chain = chain.with_retry(stop_after_attempt=3)
```

---

## 七、流式 vs 批处理 vs 并发：决策树

```
你需要的是？
├─ 单次问答
│   └─ invoke() 或 stream()（用户要看打字机效果）
│
├─ 多个独立问题
│   ├─ 同步简单 → batch()（自动并行）
│   ├─ 需要控制并发 → chain.batch(inputs, config={"max_concurrency": 5})
│   ├─ 需要更细粒度控制 → asyncio.gather + ainvoke
│   └─ 需要在 async 框架中 → abatch() / ainvoke
│
├─ 需要监控每一步
│   └─ astream_events(version="v2")
│       + include_types/include_names 过滤
│
├─ 长文本生成 + 实时展示
│   └─ stream() / astream()
│
└─ 离线评估（100+ 条）
    └─ batch() + max_concurrency=API限制值
        + return_exceptions=True
        + metadata 记录每条耗时
```

---

## 八、网络与资源考虑

### API 限流（Rate Limit）

大多数 LLM API 有并发限制：

| 厂商 | 免费/开发版 | 付费版 |
|---|---|---|
| OpenAI | 3 RPM / 200 TPM | 500+ RPM |
| DeepSeek | 5 QPS | 更高 |
| Claude | 5 RPM | 50+ RPM |

**应对策略**：

```python
# 把 max_concurrency 设为低于 API 限制
chain.batch(inputs, config={"max_concurrency": 4})  # API 限制 5 QPS，设 4 留余量

# 加指数退避重试
from tenacity import retry, stop_after_attempt, wait_exponential
robust = chain.with_retry(
    stop_after_attempt=5,
    wait_exponential_multiplier=1, wait_exponential_max=60,
)
```

### 内存占用

```
stream  ← 逐 token，内存 O(1)
invoke  ← 完整结果，内存 O(response_size)
batch   ← N 个完整结果，内存 O(N × response_size)
```

**大 batch 时注意**：`batch(inputs=1000条, max_concurrency=10)` 同时持有 10 个完整响应，内存可控；`batch(inputs=1000条, max_concurrency=1000)` 同时持有 1000 个响应，可能 OOM。

### 异步提速的真实收益

```
场景：100 个问题，每个问题等待 LLM 响应 3 秒

同步 batch(max_concurrency=1):   100 × 3s = 300s (5 分钟)
同步 batch(max_concurrency=5):   100 / 5 × 3s = 60s   (1 分钟)
异步 asyncio.gather(限制5并发):   100 / 5 × 3s = 60s   (和同步 batch 一样)

结论：batch() 本身就并行了，大多数情况下不需要手写异步。
除非：你在 async web 框架（FastAPI）中，需要非阻塞地处理请求。

---

# 结构化输出解析：让 LLM 返回可编程的数据

