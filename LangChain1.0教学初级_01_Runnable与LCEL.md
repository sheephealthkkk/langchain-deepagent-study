# LangChain 1.0+ 核心概念教学：Runnable 与 LCEL

## 前置：为什么 LangChain 1.0 要重新设计？

LangChain 0.x 时代，不同组件用不同的调用方式：

```python
# 0.x 时代的混乱
llm(prompt)              # LLM 用 __call__
retriever.get_relevant_documents(query)   # 检索器有自己的方法
chain.run(input)         # Chain 用 run
tool.run(input)          # Tool 用 run
```

每个组件调用方式不同 → 组合困难 → 流式、异步需要各自实现 → 代码难以复用。

**LangChain 1.0 的核心思想：一切皆是 Runnable，一切用同一套协议。**

---

## 一、Runnable 是什么？

### 1.1 一句话定义

**Runnable 是 LangChain 的"万能接口"——任何可以"接收输入、返回输出"的组件都实现它。**

```python
from langchain_core.runnables import Runnable

# Runnable 的核心契约
class Runnable:
    def invoke(self, input) -> output:    ...    # 同步调用
    async def ainvoke(self, input) -> output: ... # 异步调用
    def batch(self, inputs) -> outputs:   ...    # 批量调用
    def stream(self, input) -> Iterator:  ...    # 流式输出
```

任何类只要实现了这个接口，就能被 LangChain 框架统一编排。

### 1.2 哪些东西是 Runnable？

**几乎所有组件都是 Runnable**，这就是为什么它们能用 `|` 串联：

| 组件 | 是 Runnable 吗？ | invoke(input) 做了什么？ |
|---|---|---|
| `ChatPromptTemplate` | ✅ | 输入 dict → 填入模板 → 输出 `ChatPromptValue` |
| `ChatOpenAI` | ✅ | 输入 `PromptValue` → LLM 推理 → 输出 `AIMessage` |
| `StrOutputParser` | ✅ | 输入 `AIMessage` → 提取文本 → 输出 `str` |
| `Chroma` (vectorstore) | ✅ | 输入 query → 相似度检索 → 输出 `List[Document]` |
| `retriever` | ✅ | 输入 query → 检索 → 输出 `List[Document]` |
| `RunnablePassthrough` | ✅ | 输入 x → 原样输出 x |
| `RunnableLambda` | ✅ | 输入 x → 执行自定义函数 → 输出 f(x) |
| `@tool` 装饰的函数 | ✅ | 输入 ToolCall → 执行函数 → 输出 ToolMessage |
| 你用 `|` 拼出来的 Chain | ✅ | 串联执行 → 输出最终结果 |

**关键思想：一个 Chain 本身也是 Runnable，所以 Chain 可以嵌套 Chain。**

---

### 1.3 为什么这样设计？—— Runnable 协议的四种能力

#### 能力 1：统一接口 → 任意组合

```python
# 所有组件用完全相同的方式调用
prompt.invoke({"topic": "AI"})       # → ChatPromptValue
llm.invoke(prompt_value)             # → AIMessage
parser.invoke(ai_message)            # → str
chain.invoke({"topic": "AI"})        # → str
retriever.invoke("What is RAG?")     # → List[Document]

# 因为是统一接口，所以可以任意串联
chain = prompt | llm | parser        # 管道的每一环都是 Runnable
```

#### 能力 2：自动获得流式输出

只要链上的 LLM 支持流式，整条链就自动支持：

```python
# 不用改任何代码，直接换 invoke 为 stream
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="")  # 逐 token 输出
```

**为什么能自动？** 因为 `|` 创建的 SequenceRunnable 内置了流式转发逻辑，前一环的输出逐块传给后一环。

#### 能力 3：自动获得批量处理

```python
# invoke 一次一个
result = chain.invoke({"topic": "AI"})

# batch 一次多个，自动并行
results = chain.batch([
    {"topic": "AI"},
    {"topic": "Python"},
    {"topic": "RAG"},
])
```

#### 能力 4：自动获得异步

```python
# 同步
result = chain.invoke({"topic": "AI"})

# 异步（同样的链，换 ainvoke）
result = await chain.ainvoke({"topic": "AI"})
```

---

### 1.4 Runnable 能调用工具吗？

**直接回答：Runnable 自身不"调用工具"，但 Tool 本身是 Runnable，可以被 Agent 调用。**

这里有个关键区分：

```python
# 情况1: Tool 自己是 Runnable — 可以独立被调用
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}天气晴朗"

# Tool 实现了 Runnable 接口，所以可以：
get_weather.invoke({"city": "北京"})   # → "北京天气晴朗"

# 情况2: LLM 通过 Function Calling 决定用哪个 Tool
# LLM 不直接调用 Tool，它输出一个 ToolCall 请求
# 然后框架（Agent/Executor）去执行这个 ToolCall

# 情况3: 把 Tool 绑定到 LLM 上
llm_with_tools = llm.bind_tools([get_weather, get_user_location])
# llm_with_tools 本身还是 Runnable，但它现在能输出 ToolCall 请求
```

**总结**：

```
Tool 是 Runnable（可以被 invoke）
  但 Tool 的调用者通常是 Agent/Chain，不是普通用户代码
    └─ Agent 决定"该调用哪个 Tool" → 执行 Tool → 结果返回 Agent → 继续推理
```

---

## 二、LCEL（LangChain Expression Language）

### 2.1 一句话定义

**LCEL 就是用 `|`（管道符）把多个 Runnable 串联成新的 Runnable。**

```python
chain = runnable_a | runnable_b | runnable_c
# 等价于：output = runnable_c.invoke(runnable_b.invoke(runnable_a.invoke(input)))
```

`|` 是 Python 的 `__or__` 运算符，LangChain 重载了它来实现 Runnable 的组合。

---

### 2.2 基础串联：线性管道

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 三个独立的 Runnable
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，用中文回答。"),
    ("user", "{question}"),
])  # Runnable: dict → ChatPromptValue

llm = ChatOpenAI(model="deepseek-v4-pro", ...)
   # Runnable: ChatPromptValue → AIMessage

parser = StrOutputParser()
   # Runnable: AIMessage → str

# LCEL：用 | 串起来
chain = prompt | llm | parser
# 得到一个新的 Runnable: dict → str

# 调用
result = chain.invoke({"role": "物理学家", "question": "什么是熵？"})
```

**每一步的数据类型变化**：

```
{"role": "物理学家", "question": "什么是熵？"}        # dict
        │ prompt.invoke()
        ▼
ChatPromptValue(messages=[SystemMessage(...), HumanMessage(...)])
        │ llm.invoke()
        ▼
AIMessage(content="熵是系统无序程度的度量...")
        │ parser.invoke()
        ▼
"熵是系统无序程度的度量..."                             # str
```

---

### 2.3 并行分支：RunnableParallel

当需要**同时输入多个数据**到 Prompt 时，用 dict + `|` 实现并行：

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 场景：检索文档 + 保留原始问题，同时送给 QA Prompt

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {
        "context": retriever | format_docs,      # 分支A：检索 → 格式化
        "question": RunnablePassthrough(),        # 分支B：原样透传
    }
    | qa_prompt                                   # 两支汇合 → 填入模板
    | llm
    | StrOutputParser()
)

# 调用
chain.invoke("What is LangChain?")
```

**数据流**：

```
输入: "What is LangChain?"
        │
        ├──→ 分支A: retriever.invoke("What is LangChain?") → [Doc, Doc]
        │         → format_docs → "LangChain is a framework..."
        │
        └──→ 分支B: RunnablePassthrough().invoke("...") → "What is LangChain?"
        │
        └──── 汇合 ────┘
                │
        {"context": "LangChain is...", "question": "What is LangChain?"}
                │
          qa_prompt.invoke(...) → ChatPromptValue
                │
          llm.invoke(...) → AIMessage → parser → str
```

**本质**：`{"key": runnable}` 这种 dict 语法创建的是 `RunnableParallel`——内部各 Runnable 并行执行，结果汇合成一个 dict。

---

### 2.4 条件分支：RunnableBranch

```python
from langchain_core.runnables import RunnableBranch

# 场景：根据用户意图路由到不同的链
positive_chain = prompt_positive | llm | parser
negative_chain = prompt_negative | llm | parser

branch = RunnableBranch(
    (lambda x: "正面" in x, positive_chain),
    (lambda x: "负面" in x, negative_chain),
    default_chain,   # 都不匹配时的默认链
)

branch.invoke("正面评价一下 LangChain")
```

---

### 2.5 自定义函数入链：RunnableLambda / RunnablePassthrough

```python
# RunnablePassthrough: 输入原样输出（常用于透传）
passthrough = RunnablePassthrough()
passthrough.invoke("hello")      # → "hello"

# RunnablePassthrough.assign: 在现有 dict 上追加字段
chain = RunnablePassthrough.assign(
    extra=lambda x: x["count"] * 2
)
chain.invoke({"count": 5})       # → {"count": 5, "extra": 10}

# RunnableLambda: 把普通函数包成 Runnable
def uppercase(s: str) -> str:
    return s.upper()

lambda_runnable = RunnableLambda(uppercase)
lambda_runnable.invoke("hello")  # → "HELLO"

# RunnableLambda 也可以放进管道
chain = prompt | llm | RunnableLambda(uppercase) | parser
```

---

### 2.6 链中链：Runnable 的嵌套组合

因为 Chain 本身是 Runnable，所以 Chain 可以嵌套 Chain：

```python
# 子链1：改写问题
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# 子链2：QA
qa_chain = qa_prompt | llm | StrOutputParser()

# 主链：把子链和一串起来
main_chain = (
    {
        "rewritten": rewrite_chain,    # 子链1 的结果
        "original": RunnablePassthrough(),
    }
    | some_prompt                       # 用 rewritten + original
    | retriever
    | qa_chain                         # 子链2
)
```

**这就是 "链中链" 的本质：Chain 是 Runnable，所以它可以被放进另一个 Chain。**

---

## 三、Runnable 协议速查表

| 方法 | 签名 | 用途 |
|---|---|---|
| `invoke` | `(input) → output` | 同步单次调用 |
| `ainvoke` | `async (input) → output` | 异步单次调用 |
| `batch` | `(inputs[]) → outputs[]` | 批量调用，自动并行 |
| `abatch` | `async (inputs[]) → outputs[]` | 异步批量 |
| `stream` | `(input) → Iterator[output]` | 同步流式 |
| `astream` | `async (input) → AsyncIterator` | 异步流式 |
| `astream_events` | `async (input) → AsyncIterator[event]` | 带事件的异步流式（可监控每步） |
| `bind` | `(**kwargs) → Runnable` | 绑定预设参数，生成新 Runnable |
| `with_config` | `(config) → Runnable` | 绑定运行时配置 |
| `with_fallbacks` | `(fallbacks[]) → Runnable` | 添加降级方案 |
| `with_retry` | `(retry_policy) → Runnable` | 添加重试策略 |
| `pipe` | `(other) → Runnable` | 即 `|`，串联另一个 Runnable |
| `pick` | `(keys[]) → Runnable` | 从输入中提取指定字段 |

---

## 四、LCEL 常用模式汇总

```python
# 1. 线性管道
chain = A | B | C

# 2. 并行分支
chain = {"x": A, "y": B} | C

# 3. 透传 + 追加
chain = RunnablePassthrough.assign(new_field=some_runnable)

# 4. 条件分支
chain = RunnableBranch((cond1, chain1), (cond2, chain2), default_chain)

# 5. 自定义函数
chain = A | RunnableLambda(my_func) | B

# 6. 链中链（嵌套）
sub_chain = X | Y | Z
main_chain = A | sub_chain | B

# 7. 绑定工具
llm_with_tools = llm.bind_tools([tool1, tool2])
chain = prompt | llm_with_tools | output_parser

# 8. 降级保护
safe_chain = primary_chain.with_fallbacks([backup_chain])

# 9. 重试
robust_chain = chain.with_retry(stop_after_attempt=3)

# 10. 提取字段
chain = A | B.pick(["answer"])  # 只取 B 输出中的 "answer" 字段
```

---

## 五、一个综合示例

把上述概念串起来：

```python
from langchain_core.runnables import (
    RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableBranch
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 子 Runnable 们
rewrite_prompt = ChatPromptTemplate.from_messages([...])
qa_prompt = ChatPromptTemplate.from_messages([...])
llm = ChatOpenAI(model="deepseek-v4-pro", ...)
parser = StrOutputParser()

# 子链：改写问题
rewrite_chain = rewrite_prompt | llm | parser

# 自定义函数包成 Runnable
def select_retriever(query: str):
    """根据问题语言选不同的检索器"""
    return cn_retriever if any('\u4e00' <= c <= '\u9fff' for c in query) else en_retriever

# 主链组装
rag_chain = (
    {
        # 分支1: 改写后的问题 → 选检索器 → 检索 → 格式化
        "context": rewrite_chain
                   | RunnableLambda(select_retriever)
                   | format_docs,
        # 分支2: 原始问题原样透传
        "question": RunnablePassthrough(),
    }
    | qa_prompt
    | llm
    | parser
    | RunnableLambda(lambda s: s.strip())   # 最后 trim 一下
)

# 使用：一行代码，六种能力自动获得
result = rag_chain.invoke("What is RAG?")
results = rag_chain.batch(["Q1", "Q2", "Q3"])    # 批量
async for chunk in rag_chain.astream("Hello"):   # 异步流式
    print(chunk)
```

---

## 六、核心要点回顾

| 问题 | 答案 |
|---|---|
| Runnable 是什么？ | 统一接口，任何组件都实现它 |
| 为什么这样设计？ | 统一 = 可组合 + 流式/异步/批量自动获得 |
| Tool 是 Runnable 吗？ | 是，可以被 invoke，但通常由 Agent 来调用 |
| LCEL 是什么？ | 用 `\|` 串联 Runnable 的声明式语法 |
| `RunnablePassthrough` 干什么？ | 原样透传数据，实现分支中的"什么都不做" |
| `RunnableLambda` 干什么？ | 把普通函数包装成 Runnable，嵌入管道 |
| Chain 是 Runnable 吗？ | 是，所以可以嵌套形成链中链 |
| LCEL 和手写代码的区别？ | LCEL 自动获得流式/异步/批量/可视化追踪 |

---

