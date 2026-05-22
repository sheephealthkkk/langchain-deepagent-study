## 第十章：流式输出模式详解

### 10.1 三种核心流式模式

LangChain 1.0 提供三种流式调用，层级不同：

```
stream()           → 只看最终输出（用户视角）
astream_events()   → 看每步事件（开发者视角，调试用）
astream()          → stream 的异步版
```

```python
# === 模式1：stream() — 逐 Token 输出 ===
# 只返回链尾 LLM 的 token 流，中间步骤不可见
for chunk in chain.stream("What is RAG?"):
    print(chunk, end="", flush=True)
# 输出: RAG是检索增强生成...（一个字一个字蹦出来）

# === 模式2：astream_events() — 全链路事件流 ===
# 返回链上每个 Runnable 的 start/stream/end 事件
async for event in chain.astream_events("What is RAG?", version="v2"):
    print(f"[{event['event']}] {event['name']}")
# 输出:
# [on_chain_start] RunnableSequence
# [on_chat_model_start] ChatOpenAI
# [on_chat_model_stream] ChatOpenAI  ← 每个 token
# [on_chat_model_stream] ChatOpenAI
# ...
# [on_chat_model_end] ChatOpenAI
# [on_chain_end] RunnableSequence

# === 模式3：astream() — 异步版 stream ===
async for chunk in chain.astream("What is RAG?"):
    print(chunk, end="", flush=True)
```

### 10.2 stream() 的四种输出模式

`stream()` 返回的内容取决于链的最后一环是什么：

```python
# 模式 A：链尾是 LLM → 返回 token 块（AIMessageChunk）
chain = prompt | llm
for chunk in chain.stream(input):
    print(chunk.content, end="")  # → 逐 token 文本

# 模式 B：链尾是 StrOutputParser → 返回字符串块
chain = prompt | llm | StrOutputParser()
for chunk in chain.stream(input):
    print(chunk, end="")          # → 逐 token 字符串

# 模式 C：链尾是 PydanticOutputParser → 返回结构化块
chain = prompt | llm | PydanticOutputParser(pydantic_object=MyModel)
for chunk in chain.stream(input):
    print(chunk)                  # → 逐块构建的 Pydantic 对象

# 模式 D：链尾是 Retriever → 返回 Document 列表（一次性）
chain = retriever
for chunk in chain.stream(input):
    print(chunk)                  # → [Document, Document, ...]
```

### 10.3 astream_events 的事件类型与过滤

```
事件层级（从外到内）：
  on_chain_start    → on_chat_model_start → on_chat_model_stream (×N)
                    → on_chat_model_end
                    → on_tool_start → on_tool_end
                    → on_retriever_start → on_retriever_end
  on_chain_stream   → (每个 token 流经 chain 层)
  on_chain_end
```

```python
# 精确过滤：只看 LLM 的输出 token
async for event in chain.astream_events(input, version="v2",
    include_types=["chat_model"],
    include_names=["ChatOpenAI"],
):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk.content:
            print(chunk.content, end="", flush=True)

# 只看检索器
async for event in chain.astream_events(input, version="v2",
    include_types=["retriever"],
):
    if event["event"] == "on_retriever_end":
        docs = event["data"]["output"]
        print(f"检索到 {len(docs)} 条结果")
```

### 10.4 三种模式的对比

| 维度 | `stream()` | `astream()` | `astream_events()` |
|---|---|---|---|
| 返回内容 | 最终输出块 | 最终输出块 | 每步事件（start/stream/end） |
| 可见的步骤 | 只有链尾 | 只有链尾 | 链中每个 Runnable |
| 能否看到 Token 用量 | 否（只有最终） | 否 | 是（`on_chat_model_end` 中） |
| 能否看到检索结果 | 否 | 否 | 是（`on_retriever_end` 中） |
| 能否看到中间 Prompt | 否 | 否 | 是（`on_chat_model_start` 中） |
| 适用 | 聊天 UI | 异步聊天 UI | 调试、监控、日志 |
| 流式块内容 | 组件相关 | 组件相关 | 标准 `EventData {input,chunk,output,error}` |

### 10.5 stdin/stdout 的流式交互

```python
# 交互式流式终端
import sys

async def interactive_stream(chain):
    """逐 Token 打印的交互式终端。"""
    while True:
        try:
            user_input = input("👤 > ")
            if user_input.lower() in {"quit", "exit", "q"}:
                break

            print("🤖 ", end="", flush=True)
            async for event in chain.astream_events(user_input, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\n告辞！")
            break
```

### 10.6 常见误区

**误区 1："stream() 的 value 模式和 updates 模式差不多"**

`stream()` 返回的是链尾 Runnable 的输出 chunk——对 LLM 来说就是 token 块（即 "value"）。`astream_events()` 的 event 有 `data.chunk`（当前 token）和 `data.output`（完整输出，仅在 end 事件有）。

```python
# 区别：
# stream() → 只有 token 块本身（value）
for chunk in chain.stream("Hi"):
    print(type(chunk))  # → AIMessageChunk

# astream_events() → 事件包含 input/chunk/output
async for event in chain.astream_events("Hi"):
    # on_chat_model_stream → data.chunk 有 token
    # on_chat_model_end → data.output 有完整 AIMessage
    pass
```

**误区 2："value 模式只返回 LLM tokens"**

取决于链尾是什么。链尾是 `StrOutputParser` 时返回字符串 token，链尾是 `PydanticOutputParser` 时返回结构块，链尾是 `Retriever` 时返回文档列表。

```python
# 不同链尾，stream 返回完全不同
chain1 = prompt | llm                     # stream → AIMessageChunk
chain2 = prompt | llm | StrOutputParser() # stream → str chunk
chain3 = retriever                         # stream → Document 列表（一次性）
```

**误区 3："可以用 stream 看到 Prompt"**

`stream()` 看不到中间步骤——只能看链尾输出。要看 Prompt，必须用 `astream_events()` 监听 `on_chat_model_start`。

**误区 4："astream_events 的 chunk 累加等于 output"**

对 ChatModel 来说，`data.chunk` 累加确实等于 `data.output`。但对 Chain 来说，`on_chain_stream` 的 chunk 不一定等于 `on_chain_end` 的 output——因为 Chain 的 chunk 是子 Runnable 输出的透传，output 可能是最终处理后的结果。

**误区 5："stream() 一次返回所有 token"**

`stream()` 的默认行为是**逐 token 返回**。每个迭代是一个 token。如果链中有 `StrOutputParser`，每个迭代是新增的文本片段。

**误区 6："可以混用多种 stream 模式在同一个链上"**

这是**可以的**。`stream()` 给前端打字机效果，同时 `astream_events()` 给后端日志/监控。

```python
async def streaming_with_monitoring(chain, input):
    """前端流式 + 后端监控 同步进行。"""
    log_channel = asyncio.Queue()

    async def collect_logs():
        """后台收集所有事件用于监控。"""
        async for event in chain.astream_events(input, version="v2"):
            if event["event"].endswith("_end"):
                await log_channel.put(event)

    async def stream_to_frontend():
        """前端流式输出。"""
        async for chunk in chain.astream(input):
            yield chunk
            await asyncio.sleep(0)  # 让出控制权给 log 协程

    # 两个协程并发
    log_task = asyncio.create_task(collect_logs())
    async for chunk in stream_to_frontend():
        print(chunk, end="", flush=True)
    await log_task
```

**误区 7："流式输出一定比同步快"**

流式输出**总时间相同**，只是**用户感知更快**（看到第一个 token 的延迟更短）。不会减少总耗时，但显著改善体验。

---

