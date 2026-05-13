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

# langchain-core 与 langchain 依赖包深度对比

## 一、一张图说清 LangChain 包体系

```
┌────────────────────────────────────────────────────────────────┐
│                        你的项目代码                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  langchain (编排层)                                             │
│  ┌──────────────────────────────────────────────────┐         │
│  │  init_chat_model()   Agent Factory   Middleware   │         │
│  │  ToolNode           ToolRuntime                   │         │
│  │  create_history_aware_retriever (in classic)      │         │
│  └────────────────────┬─────────────────────────────┘         │
│                       │ 依赖                                   │
│  langchain-core (协议层)                                       │
│  ┌──────────────────────────────────────────────────┐         │
│  │  Runnable       BaseChatModel    BasePromptTemplate│        │
│  │  BaseMessage    BaseRetriever    BaseTool          │        │
│  │  Document       Callbacks        Serializable      │        │
│  │  MessagesPlaceholder   StrOutputParser            │        │
│  └────────────────────┬─────────────────────────────┘         │
│                       │ 被实现                                 │
│  集成包 (实现层)                                                │
│  ┌──────────┐ ┌──────────────┐ ┌─────────────┐               │
│  │ openai   │ │  community   │ │  chroma     │  ...          │
│  │ChatOpenAI│ │WebBaseLoader │ │  Chroma     │               │
│  │OpenAIEmb │ │BS4, PDF, etc │ │             │               │
│  └──────────┘ └──────────────┘ └─────────────┘               │
└────────────────────────────────────────────────────────────────┘
```

**一句话总结**：

- `langchain-core` = 协议/接口/**"规定能做什么"**（纯抽象，零实现依赖）
- `langchain` = 编排/组装/**"教你怎么组合"**（工厂函数、高层封装）
- 集成包（`langchain-openai` 等）= 实现/**"真正干活的东西"**（具体模型、具体工具）

---

## 二、langchain-core：协议层

### 2.1 定位

> 类似 Java 的 Interface、C++ 的虚基类。定义"什么是 Runnable""什么是 ChatModel""什么是 Message"，但不提供具体实现。

### 2.2 核心模块一览

| 模块 | 关键类/函数 | 作用 |
|---|---|---|
| `langchain_core.runnables` | `Runnable`, `RunnablePassthrough`, `RunnableLambda`, `RunnableParallel`, `RunnableBranch`, `RunnableWithMessageHistory` | **整个框架的基石**，所有组件都实现这个接口 |
| `langchain_core.messages` | `BaseMessage`, `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage` | 消息的抽象定义，ChatModel 之间的共同语言 |
| `langchain_core.prompts` | `BasePromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`, `PromptTemplate` | Prompt 构建，模板 + 变量 = 最终 Prompt |
| `langchain_core.output_parsers` | `BaseOutputParser`, `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser` | LLM 输出 → 结构化数据 |
| `langchain_core.documents` | `Document` | 数据容器：`page_content` + `metadata` |
| `langchain_core.language_models` | `BaseChatModel`, `BaseLLM` | 语言模型的抽象接口 |
| `langchain_core.tools` | `BaseTool`, `tool` 装饰器 | 工具的定义 |
| `langchain_core.embeddings` | `Embeddings` | 嵌入模型的抽象 |
| `langchain_core.vectorstores` | `VectorStore`, `VectorStoreRetriever` | 向量数据库的抽象 |
| `langchain_core.chat_history` | `BaseChatMessageHistory`, `InMemoryChatMessageHistory` | 聊天历史存储的抽象 |
| `langchain_core.callbacks` | `BaseCallbackHandler`, `AsyncCallbackHandler` | 事件监听，LangSmith 追踪的基础 |
| `langchain_core.load` | `Serializable` | 序列化/反序列化，`.save()` / `.load()` |

### 2.3 举例：只看 core，能写什么？

```python
# 纯 langchain-core — 只有抽象定义，没有具体模型
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# ✅ 能做的事：定义数据结构
doc = Document(page_content="LangChain is a framework...",
               metadata={"source": "https://..."})
msg = HumanMessage(content="Hello")

# ✅ 能做的事：定义模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("user", "{question}"),
])

# ✅ 能做的事：编排流程（但没有具体 LLM，无法执行）
flow = {"question": RunnablePassthrough()} | prompt
# flow.invoke("Hi")  ← 报错！prompt 之后没有 LLM，无法执行

# ❌ 不能做的事：真正调用 LLM
# 因为 langchain-core 里没有 ChatOpenAI、没有 ChatDeepSeek
# 这些在 langchain-openai 包里
```

**core 是骨架，没有肉。它定义"Prompt 应该有 from_messages 方法""LLM 应该有 invoke 方法"，但真正的 HTTP 请求、模型推理在集成包里。**

---

## 三、langchain：编排层

### 3.1 定位

> 类似设计模式中的 **Facade / Factory**。它不做底层实现，而是把 core 的抽象 + 集成包的具体实现**按最佳实践组装好**。

### 3.2 核心模块一览

| 模块 | 关键函数/类 | 作用 |
|---|---|---|
| `langchain.chat_models` | `init_chat_model()` | **一站式模型初始化**：一行代码自动识别模型名、找到对应的集成包、实例化 |
| `langchain.agents` | `create_agent()`, Middleware 系列 | Agent 工厂 + 中间件（重试、限流、人机协作、摘要等） |
| `langchain.tools` | `ToolRuntime`, `tool` | 工具运行时上下文注入 |
| `langchain.messages` | 消息工具函数 | 消息的过滤、合并、截断 |
| `langchain.embeddings` | `init_embeddings()` | 一站式嵌入模型初始化 |
| `langchain.indexing` | 索引 API | 高效的文档索引（增量、去重） |
| `langchain.load` | 序列化工具 | 链的保存/加载 |

**另外 `langchain-classic` 包中的关键函数：**

| 模块 | 关键函数 | 我们项目中的用法 |
|---|---|---|
| `langchain_classic.chains` | `create_history_aware_retriever` | `05`/`06` 的子链1 |
| `langchain_classic.chains` | `create_retrieval_chain` | 组装 RAG 主链 |
| `langchain_classic.chains.combine_documents` | `create_stuff_documents_chain` | `05`/`06` 的子链2 |

### 3.3 举例：langchain 怎么省掉 boilerplate

**没有 `init_chat_model` 时（手写）：**

```python
# 需要自己判断模型名 → 找对应包 → 实例化
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek

def create_llm(model_name: str):
    if model_name.startswith("gpt"):
        return ChatOpenAI(model=model_name)
    elif model_name.startswith("claude"):
        return ChatAnthropic(model=model_name)
    elif model_name.startswith("deepseek"):
        return ChatDeepSeek(model=model_name)
    # ... 每个新模型都要加分支
```

**用 `init_chat_model` 后：**

```python
from langchain.chat_models import init_chat_model

# 一行，LangChain 自动匹配包
llm = init_chat_model("deepseek-v4-pro",
                       api_key="...",
                       base_url="...")

llm = init_chat_model("gpt-4o")           # 自动用 ChatOpenAI
llm = init_chat_model("claude-opus-4-6")  # 自动用 ChatAnthropic
```

`init_chat_model` 内部做的事：

```
"deepseek-v4-pro"
        │
        ▼ 查注册表
  model_provider = "deepseek"
        │
        ▼ 动态 import
  import langchain_deepseek
        │
        ▼ 实例化
  ChatDeepSeek(model="deepseek-v4-pro", ...)
```

---

## 四、对比总结

### 4.1 核心区别

| 维度 | `langchain-core` | `langchain` |
|---|---|---|
| **性质** | 协议层（Interface） | 编排层（Factory/Orchestrator） |
| **内容** | 抽象类 + 协议 + 数据结构 | 工厂函数 + 高层组装 + 最佳实践 |
| **能独立运行吗？** | 不能（没有 LLM 实现） | 能（依赖集成包） |
| **依赖方向** | 无外部依赖 | 依赖 langchain-core |
| **类比** | Java Interface / C++ ABC | Spring Boot AutoConfiguration |
| **安装大小** | 极小 | 中等 |
| **谁实现它？** | 集成包（openai/community 等） | 用户代码调用它 |

### 4.2 各包在项目中的分工（对应我们 01~06 的文件）

```
01_hello_langchain.py:
  langchain.chat_models.init_chat_model  ← langchain 的工厂函数
  ChatOpenAI                             ← langchain-openai 的具体实现
  ChatPromptTemplate                     ← langchain-core 的模板协议

03_rag_indexing.py:
  WebBaseLoader                          ← langchain-community 的具体加载器
  RecursiveCharacterTextSplitter         ← langchain-text-splitters 的切分实现
  HuggingFaceEmbeddings                  ← langchain-huggingface 的嵌入实现
  Chroma.from_documents()               ← langchain-chroma 的存储实现
  Document                              ← langchain-core 的数据容器

04_rag_retrieval.py:
  retriever \| format_docs \| prompt \| llm \| parser
    ↑                                    ↑
    LCEL 管道（langchain-core 的 Runnable 协议）
    每个 \\| 都是 Runnable.__or__()

05_conversational_rag.py:
  RunnableWithMessageHistory             ← langchain-core 的历史管理包装器
  create_history_aware_retriever         ← langchain-classic 的子链工厂
  create_stuff_documents_chain           ← langchain-classic 的子链工厂
  create_retrieval_chain                 ← langchain-classic 的组装工厂
  MessagesPlaceholder                    ← langchain-core 的模板占位符

06_persistent_rag.py:
  BaseChatMessageHistory                 ← langchain-core 的历史存储抽象
  SQLiteChatMessageHistory               ← 自己实现的持久化适配（实现 core 接口）
  Base (declarative_base)               ← SQLAlchemy 的 ORM 基类（外部库）
```

### 4.3 判断一个功能属于哪个包

**简单的判断原则**：

| 问题 | 答案 |
|---|---|
| 这是一个"标准"还是"做法"？ | 标准 → core；做法 → langchain |
| 能直接实例化一个可用的对象吗？ | 不能 → core（抽象）；能 → 集成包 |
| 是工厂/组装函数吗？ | 是 → langchain |
| 需要网络请求吗？ | 需要 → 集成包（openai/community 等） |

---

## 五、最核心使用模块速查

### langchain-core 的"必知"模块

```python
# 1. runnables — 一切的基础
from langchain_core.runnables import (
    Runnable,             # 接口本身
    RunnablePassthrough,  # 透传
    RunnableLambda,       # 函数 → Runnable
    RunnableBranch,       # 条件分支
)

# 2. messages — 对话的"原子单位"
from langchain_core.messages import (
    HumanMessage,         # 用户发的话
    AIMessage,            # AI 回的答
    SystemMessage,        # 系统指令
    ToolMessage,          # 工具返回的结果
)

# 3. prompts — 构造 Prompt
from langchain_core.prompts import (
    ChatPromptTemplate,         # 聊天模板
    MessagesPlaceholder,        # 历史消息占位符
    PromptTemplate,             # 字符串模板
)

# 4. documents — 数据容器
from langchain_core.documents import Document

# 5. output_parsers — 解析输出
from langchain_core.output_parsers import (
    StrOutputParser,      # 提取纯文本
    JsonOutputParser,     # 提取 JSON
)

# 6. chat_history — 历史存储
from langchain_core.chat_history import (
    BaseChatMessageHistory,       # 历史存储的抽象接口
    InMemoryChatMessageHistory,   # 内存实现
)
```

### langchain / langchain-classic 的"必知"函数

```python
# 模型初始化
from langchain.chat_models import init_chat_model

# Agent 创建
from langchain.agents import create_agent

# Tool 运行时
from langchain.tools import tool, ToolRuntime

# RAG 链组装（在 langchain-classic 中）
from langchain_classic.chains import (
    create_history_aware_retriever,   # 子链1：历史感知检索
    create_retrieval_chain,           # 主链：检索 + QA 组装
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,     # 子链2：文档填充 + LLM 生成
)
```

### 集成包的分工

```python
# langchain-openai — OpenAI / DeepSeek / 任何 OpenAI 兼容 API
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# langchain-community — 文档加载器、通用工具
from langchain_community.document_loaders import WebBaseLoader

# langchain-chroma — ChromaDB 向量库
from langchain_chroma import Chroma

# langchain-huggingface — HuggingFace 嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings
```

---

## 六、记忆口诀

```
core 定规矩（协议、接口、数据结构）
langchain 给套路（工厂函数、最佳实践、编排）
openai/community 干真活（网络请求、模型推理、文档解析）

写代码时：
  导入 core → 用它的类型和协议（Runnable, BaseMessage, Document）
  导入 langchain → 用它的工厂和工具（init_chat_model, create_agent）
  导入集成包 → 用具体实现（ChatOpenAI, HuggingFaceEmbeddings）
```

---

# LLM vs ChatModel：两类语言模型接口

## 一、一张表说清

| | `BaseLLM`（老，不推荐） | `BaseChatModel`（唯一推荐） |
|---|---|---|
| 输入 | `str` | `List[BaseMessage]` |
| 输出 | `str` | `BaseMessage`（含 token 用量等元数据） |
| 端点 | `/completions` | `/chat/completions` |
| System Prompt | 不支持 | `SystemMessage` |
| 多轮对话 | 手动拼字符串 | 消息列表，`RunnableWithMessageHistory` 自动管 |
| Tool Calling | 不支持 | `bind_tools()` → `AIMessage.tool_calls` |
| 结构化输出 | 不支持 | `with_structured_output(PydanticClass)` |
| 代表模型 | GPT-3（已淘汰） | GPT-4、Claude、DeepSeek |

**结论：2024 年后一律用 BaseChatModel。**

## 二、四种消息类型

```python
SystemMessage("你是助手")     # 系统指令
HumanMessage("什么是熵？")     # 用户发言
AIMessage("熵是...")          # AI 回复（含可选 .tool_calls）
ToolMessage("25°C", id="x")  # 工具返回值
```

一个多轮对话 = `List[BaseMessage]`：`[System, Human, AI, Human, AI, ...]`。消息不可变，修改用 `.copy()`。

## 三、核心三用法

```python
llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0.8)

# 用法1：基础调用（invoke/stream/batch/ainvoke）
r = llm.invoke("Hello")
for chunk in llm.stream("故事"): ...

# 用法2：Tool Calling（bind_tools — 模型只输出调用请求，执行由框架负责）
llm_tools = llm.bind_tools([get_weather])
r = llm_tools.invoke("北京天气")  # r.tool_calls ≠ None

# 用法3：结构化输出（强制 JSON → Pydantic）
llm_struct = llm.with_structured_output(WeatherReport)
report: WeatherReport = llm_struct.invoke("北京天气")
```

## 四、ChatOpenAI 关键参数

```python
ChatOpenAI(
    model="deepseek-v4-pro",          # 模型名
    temperature=0.8,                  # 0=确定，1=随机
    max_tokens=4096,                  # 最大输出长度
    timeout=60, max_retries=3,        # 可靠性
    base_url="https://api.deepseek.com",  # ← 核心：任何 OpenAI 兼容 API 都行
)
```

> **关键认知**：`ChatOpenAI` 不绑定 OpenAI。`base_url` 指向谁就是谁 —— DeepSeek、Moonshot、Ollama、vLLM 通用。

## 五、三个常见误区

1. **"bind_tools 后模型自动调工具"** → 错。模型只输出 `tool_calls` 请求，执行由 Agent/Chain 负责。
2. **"with_structured_output 不影响模型"** → 错。会强制 `response_format` JSON 模式。
3. **"多轮对话要手动管消息"** → 不需要。`RunnableWithMessageHistory` 自动注入 + 追加。

---

# 修改 langchain-core = 修改接口，所有实现自动跟随

## 一、核心原理

`langchain-core` 中的类定义了**方法实现**（不是只定义接口签名）。集成包的类**继承**这些方法，而不是重写它们。所以：

```
修改 core 中基类的方法
        ↓
所有继承这个基类的子类自动获得新行为
        ↓
除非子类显式重写了该方法
```

**类比**：你爸会开车，你继承了开车技能。你爸去学了漂移 → 你也会漂移了（因为你是照你爸的方法开的）。但如果你自己学过开车、重写了开车方式，那你爸的漂移你不会自动获得。

---

## 二、具体例子：给 `invoke` 加速率限制

### 场景

假设 LangChain 的 `Runnable.invoke()` 目前没有速率限制。你想给**所有** Runnable 加一个"每秒最多 5 次调用"的限制。

### 改 core 之前

```python
# langchain-core 中的原始代码（简化示意）
class Runnable:
    def invoke(self, input, config=None, **kwargs):
        # 直接调用，没有速率限制
        return self._invoke(input, config, **kwargs)
```

此时所有 Runnable（ChatOpenAI、retriever、chain、tool）调用 `invoke` 都没有速率限制。

### 改 core：加 5 QPS 速率限制

```python
# 你在 langchain-core 中修改 invoke，加入速率限制
import time
from collections import deque

class Runnable:
    _call_times: deque[float] = deque()  # 类级共享

    def invoke(self, input, config=None, **kwargs):
        # --- 新增：速率限制逻辑 ---
        now = time.monotonic()
        while len(self._call_times) >= 5:
            oldest = self._call_times[0]
            if now - oldest < 1.0:  # 1 秒内超过 5 次
                time.sleep(1.0 - (now - oldest))
                now = time.monotonic()
            else:
                self._call_times.popleft()
        self._call_times.append(now)
        # --- 速率限制结束 ---

        return self._invoke(input, config, **kwargs)
```

### 谁会自动跟着变？

**所有没有重写 `invoke` 的类全部自动获得速率限制**：

```python
# ChatOpenAI — 自动获得速率限制 ✅
# 因为它只重写了 _generate()，invoke() 是继承自 Runnable
llm = ChatOpenAI(model="deepseek-v4-pro")
llm.invoke("Hello")    # ← 自动受 5 QPS 限制

# Chroma retriever — 自动获得速率限制 ✅
retriever.invoke("What is RAG?")  # ← 自动受 5 QPS 限制

# 你用 | 拼出来的 Chain — 自动获得速率限制 ✅
chain.invoke({"topic": "AI"})     # ← 自动受 5 QPS 限制

# @tool 装饰的函数 — 自动获得速率限制 ✅
get_weather.invoke({"city": "北京"})  # ← 自动受 5 QPS 限制

# RunnablePassthrough — 自动获得速率限制 ✅
RunnablePassthrough().invoke("hi")    # ← 自动受 5 QPS 限制
```

### 谁不会自动变？

**只有显式重写了 `invoke` 的类不会**。但实践中几乎没有集成包会重写 `invoke`——它们只实现自己的业务逻辑（如 `_generate`、`_aget`），把 `invoke` 的通用逻辑留给基类。

---

## 三、继承链的真实结构

```python
# === langchain-core ===

class Runnable:
    def invoke(self, input, config=None, **kwargs):
        """你改这里 → 全局生效"""
        # 通用逻辑：回调、配置、速率限制...
        return self._invoke(input, config, **kwargs)

class BaseChatModel(Runnable):
    def invoke(self, input, stop=None, **kwargs):
        """BaseChatModel 重写了 invoke，但只加了 stop 处理，
        最终还是调 Runnable 的逻辑。如果你改 Runnable.invoke，
        这里的 super() 链会传递下去。"""
        ...

# === langchain-openai ===

class ChatOpenAI(BaseChatModel):
    # ChatOpenAI 只实现 _generate()，不重写 invoke()
    # → invoke() 的行为完全由 core 决定
    def _generate(self, messages, stop, **kwargs):
        """真正发 HTTP 请求"""
        response = self.client.chat.completions.create(...)
        return ChatResult(...)

# === langchain-chroma ===

class Chroma(VectorStore):
    # Chroma 只实现 _similarity_search()，不重写 invoke()
    # → invoke() 的行为完全由 core 决定
    def _similarity_search(self, query, k):
        ...

# === langchain_classic ===

# create_history_aware_retriever 返回的 Chain
# 是 SequenceRunnable(Runnable) 的实例
# → invoke() 由 core 统一控制
```

**关键图**：

```
Runnable.invoke()          ← 你改这里
   ├─ BaseChatModel.invoke()   ← 加了 stop 处理，最终 super() 回 Runnable
   │    ├─ ChatOpenAI          ← 只实现 _generate()，invoke 全继承
   │    ├─ ChatAnthropic       ← 同上
   │    └─ ChatDeepSeek        ← 同上
   │
   ├─ BaseRetriever.invoke()   ← 加了检索逻辑
   │    └─ Chroma / FAISS / ...  ← 只实现 _get_relevant_documents()
   │
   ├─ BaseTool.invoke()        ← 加了 ToolRuntime 注入
   │    └─ 所有 @tool 函数       ← 只实现工具逻辑本身
   │
   └─ SequenceRunnable.invoke() ← | 拼出来的 Chain
        └─ 你写的所有 chain      ← 自动继承
```

---

## 四、什么改什么不改：继承规则速查

| 你在 core 改了什么 | 影响范围 | 例外 |
|---|---|---|
| `Runnable.invoke()` | 所有 Runnable | 重写了 `invoke` 的类（极少） |
| `Runnable.batch()` | 所有 Runnable | 同上 |
| `Runnable.stream()` | 所有 Runnable | 同上 |
| `BaseChatModel.invoke()` | 所有 ChatModel | 同上 |
| `BaseChatModel._generate()` | 无 — 它是抽象方法 | 每个模型自己实现 |
| `BaseMessage.content` (属性) | 所有消息类型 | 无，属性被完全继承 |
| `BasePromptTemplate.format()` | 所有 Prompt 模板 | 重写了 `format` 的类 |
| `Document` 类 | 所有 Document 实例 | 无，数据类完全继承 |

---

## 五、实战建议

1. **改 core 的 `invoke/batch/stream`** → 全局加能力（限流、日志、监控），所有组件零成本获得
2. **不要改 core 的抽象方法**（如 `_generate`） → 改了没用，每个模型都自己实现了一套
3. **加新功能先想在 core 哪个层加** → 越底层（Runnable），覆盖面越大；越高层（BaseChatModel），越精准
4. **如果你的实现重写了基类方法** → core 的改动不会自动传递，需要手动同步

---

# Messages：消息体系的角色与执行顺序

## 一、Messages 是什么——一句话

**Messages 是 LangChain 对话系统的"原子单位"——所有组件间的信息传递都靠它，没有 Messages 就没有对话。**

## 二、Messages 的三大作用

### 作用 1：结构化对话——角色分离

没有 Messages，对话就是一个大字符串：

```python
# 没有 Messages — 角色混在一起，模型无法区分
"你是助手。用户：天气怎么样？助手：我查一下。工具结果：25°C。助手：今天25°C。"
```

有了 Messages，每个发言的角色、内容、元数据各自独立：

```python
[
    SystemMessage("你是助手，用中文回答。"),     # ← 系统知道自己是助手
    HumanMessage("北京天气怎么样？"),            # ← 模型知道这是用户说的
    AIMessage("我来查一下。", tool_calls=[...]), # ← 模型知道这是自己说的 + 想调工具
    ToolMessage("25°C, 晴天", id="call_1"),     # ← 模型知道这是工具返回的
    AIMessage("北京今天25°C，晴天！"),            # ← 模型继续回复
]
```

### 作用 2：统一传输协议——组件间的共同语言

LLM 吃 `List[BaseMessage]`，ChatPromptTemplate 产出 `List[BaseMessage]`，ChatHistory 存储 `List[BaseMessage]`，Tool 返回 `ToolMessage`。所有组件通过同一种数据类型沟通，用 `|` 串联时不需要做类型转换。

```
ChatPromptTemplate  ──List[BaseMessage]──→  ChatOpenAI  ──AIMessage──→  StrOutputParser
                                                       │
                                                   tool_calls
                                                       │
                                                       ▼
                                                  Tool.invoke()
                                                       │
                                                  ToolMessage
```

### 作用 3：承载元数据——不只是文本

```python
ai_msg = AIMessage(
    content="北京今天25°C，晴天！",     # 给人看的文本
    tool_calls=[                        # 工具调用请求
        {"name": "get_weather", "args": {"city": "北京"}, "id": "call_1"}
    ],
    response_metadata={                 # API 返回的原始信息
        "token_usage": {"prompt_tokens": 120, "completion_tokens": 30},
        "finish_reason": "stop",
        "model_name": "deepseek-v4-pro",
    },
    id="chatcmpl-xxx",                 # 唯一消息 ID
)
```

一条 AIMessage 同时携带：文本回复 + 工具调用意图 + Token 用量 + 模型信息，后续组件可以根据需要取不同的字段。

---

## 三、RunnableWithMessageHistory 用法详解

### 3.1 它解决什么问题

没有 `RunnableWithMessageHistory` 时，每次调用 Chain 都需要手动管理聊天历史：

```python
# ❌ 手动管理历史 — 繁琐且容易出错
history = []

# 第 1 轮
r1 = chain.invoke({"input": "What is LangChain?", "history": history})
history.append(HumanMessage("What is LangChain?"))
history.append(AIMessage(r1))

# 第 2 轮
r2 = chain.invoke({"input": "How is it different?", "history": history})
history.append(HumanMessage("How is it different?"))
history.append(AIMessage(r2))

# 每次都要记得传 history、记得追加、记得构造 HumanMessage/AIMessage
# 换一个 session_id 还要手动切换不同的 history 列表
```

`RunnableWithMessageHistory` 把这一切自动化：**调用前自动注入历史、调用后自动追加新消息、按 session_id 自动隔离不同会话**。

### 3.2 六个参数逐个讲解

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

runnable_with_history = RunnableWithMessageHistory(
    runnable,                     # ① 底层真正干活的链
    get_session_history,          # ② 根据 session_id 返回 BaseChatMessageHistory 对象
    input_messages_key="input",   # ③ 用户输入在 dict 中的 key 名（入口）
    history_messages_key="history", # ④ 历史消息注入到 dict 的哪个 key（桥梁）
    output_messages_key="output", # ⑤ 模型回复在输出 dict 中的 key 名（出口）
)
```

#### 参数 ①：`runnable` — 被包装的链

这是真正干活的 Runnable。可以是任何 Chain、Agent、LLM。`RunnableWithMessageHistory` 不关心它在做什么，只负责在调用它**之前**把历史注入 dict，在它返回**之后**从输出提取回答追加到历史。

```python
# runnable 可以是简单 LLM
runnable = prompt | llm | StrOutputParser()

# 也可以是复杂 RAG 链
runnable = create_retrieval_chain(retriever, qa_chain)

# 也可以是 Agent
runnable = create_agent(llm, tools)
```

#### 参数 ②：`get_session_history` — 历史存取的回调函数

**这是整个机制的核心**。它是一个函数，接收 `session_id: str`，返回一个实现了 `BaseChatMessageHistory` 接口的对象。

```python
# === 内存实现（开发用）===
from langchain_core.chat_history import InMemoryChatMessageHistory

store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """根据 session_id 返回对应的聊天历史。不存在则自动创建。"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# === SQLite 实现（生产用）===
# 见 06_persistent_rag.py 的 SQLiteChatMessageHistory
# 重启后历史仍然保留
```

**为什么是回调函数而不是直接传对象？** 因为 `RunnableWithMessageHistory` 每次调用都要根据 `config` 中的 `session_id` 动态获取对应会话的历史。传一个函数让它自己调，比传一个固定的对象更灵活——同一个链实例可以同时服务多个 `session_id`。

#### 参数 ③：`input_messages_key="input"` — 入口 Key

指定用户传入 dict 中的**哪个字段**是用户当前的输入。框架从 `config` 中读出来的 `session_id` → 调 `get_session_history(session_id)` → 拿到历史 → **把历史和用户输入拼成一个新的 dict**：

```python
# 用户调用时：
runnable_with_history.invoke(
    {"input": "How is it different?"},  # ← "input" 在用户 dict 中
    config={"configurable": {"session_id": "session_1"}},
)

# 框架内部自动拼装：
# {
#     "input": "How is it different?",          ← 用户传的，原样保留
#     "history": [HumanMessage(...), AIMessage(...)]  ← 框架自动注入！
# }
# ↑ "history" 就是 history_messages_key 指定的 key
```

**为什么用户只传 `input` 而不传 `history`？** 因为历史应该由框架管理，不应该让用户手动传——用户只需要关心当前要问什么。

#### 参数 ④：`history_messages_key="history"` — 桥梁 Key

指定**历史消息列表**在 dict 中的 key 名。框架将 `get_session_history` 返回的 `messages` 列表注入到 dict 的这个 key 下。

```python
# 这个 key 必须和 Chain 的 Prompt 模板中的 MessagesPlaceholder 名称对应！

# RunnableWithMessageHistory 侧：
history_messages_key = "history"               # ← 框架拼 dict 时用的 key

# Prompt 模板侧：
MessagesPlaceholder("history")                 # ← 模板中占位符的名字

# 两者必须一致！否则历史消息注入不了模板。
```

**为什么叫"桥梁 Key"？** 它是 RunnableWithMessageHistory（历史管理）和 Prompt 模板（消息构造）之间的**唯一耦合点**。改了这个 key，两头都要改。

#### 参数 ⑤：`output_messages_key="output"` — 出口 Key

指定链的输出 dict 中**哪个字段是 AI 的最终回答**。框架从输出中提取这个字段的值，追加到聊天历史。

```python
# rag_chain 的输出（由 create_retrieval_chain 构造）：
{
    "input": "How is it different from LangGraph?",
    "chat_history": [...],
    "context": [Document, ...],
    "answer": "LangChain 是一个高级框架..."  ← output_messages_key 指向这里
}

# 框架提取：output["answer"] → AIMessage("LangChain 是一个高级框架...")
# 然后追加到 history.add_message(AIMessage(...))
```

**如果链的输出是纯字符串（不是 dict）怎么办？** 那就不需要这个参数。框架直接把整个字符串当作回答追加到历史：

```python
# 链的输出是纯字符串
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "Hi"})
# → "Hello! How can I help you?"（纯字符串）

# RunnableWithMessageHistory 会直接把字符串包装为 AIMessage 追加
```

### 3.3 完整使用示例（从创建到调用）

```python
# ================================================================
# RunnableWithMessageHistory 完整用法
# ================================================================
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ---- 第 1 步：创建底层链 ----
# 这个链需要有 MessagesPlaceholder 来接收历史消息
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好助手。"),
    MessagesPlaceholder("history"),   # ← 历史消息注入位置
    ("human", "{input}"),             # ← 用户当前输入
])

llm = ChatOpenAI(model="deepseek-v4-pro")
chain = prompt | llm | StrOutputParser()
# 注意：此时 chain 还没有记忆功能。如果直接调 chain.invoke({"input": "Hi"})
# 会因为缺少 "history" 字段而报错——MessagesPlaceholder 需要这个字段。

# ---- 第 2 步：定义历史存取回调 ----
store = {}

def get_session_history(session_id: str):
    """根据 session_id 获取对应的聊天历史。"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ---- 第 3 步：包装为有记忆的链 ----
chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,                      # 底层链
    get_session_history=get_session_history,
    input_messages_key="input",          # 用户传 {"input": "..."}
    history_messages_key="history",      # 历史注入到 "history" 字段
    # output_messages_key 不指定——因为 chain 输出是纯字符串 (StrOutputParser)
)

# ---- 第 4 步：多轮对话 ----
config = {"configurable": {"session_id": "chat_1"}}

# 第 1 轮
r1 = chain_with_memory.invoke(
    {"input": "我叫 Alice，我喜欢 Python"},
    config=config,
)
print(f"AI: {r1}")  # → "你好 Alice！Python 很棒的..."

# 第 2 轮 — 不需要手动传历史！
r2 = chain_with_memory.invoke(
    {"input": "我叫什么？我喜欢什么语言？"},
    config=config,
)
print(f"AI: {r2}")  # → "你叫 Alice，你喜欢 Python！"
# RunnableWithMessageHistory 自动把第 1 轮的对话注入到了 Prompt 中

# 第 3 轮 — 换一个 session_id → 全新记忆
config_new = {"configurable": {"session_id": "chat_2"}}
r3 = chain_with_memory.invoke(
    {"input": "我叫什么？"},
    config=config_new,
)
print(f"AI: {r3}")  # → "我不知道你的名字，这是我们第一次对话。"
```

### 3.4 参数对应关系图

```
用户调用                      RunnableWithMessageHistory                底层 Chain
────────                    ──────────────────────────                ──────────

invoke(
  {"input": "Hi"},  ─────────→  提取 "input" 字段                      
  config={                       ↓                                      
    "session_id": "s1"         调 get_session_history("s1")            
  }                              ↓                                      
)                              拼接:                                   
                                {                                      
                                  "input": "Hi",       ← input_messages_key
                                  "history": [          ← history_messages_key
                                    HumanMsg("..."),                 
                                    AIMsg("..."),                    
                                  ],                                  
                                }                                      
                                  │                                    
                                  ▼                                    
                              chain.invoke(上面的dict) ────────────→  prompt 填模板
                                                                       │ {input}→"Hi"
                                                                       │ MessagesPlaceholder("history")
                                                                       │  ← 展开为 HumanMsg+AIMsg
                                                                       ▼
                                                                     LLM 生成回答
                                                                       │
                                                                       ▼
                              ←────────── "Hello Alice!" ──────────  返回值(纯字符串)
                                │
                                ▼
                              追加到历史:
                                HumanMsg("Hi")
                                AIMsg("Hello Alice!")
                                │
                                ▼
返回给用户: "Hello Alice!"
```

### 3.5 常见问题

**Q: 什么时候需要 `output_messages_key`？**

当底层链的输出是 **dict 格式**时。例如 `create_retrieval_chain` 返回 `{"input":..., "chat_history":..., "context":..., "answer":...}`。框架需要知道取哪个字段作为 AI 回答。

当底层链的输出是**纯字符串**时（用了 `StrOutputParser`），不需要指定——框架自动把整个字符串当作回答。

**Q: `input_messages_key` 和 `history_messages_key` 可以取任意名字吗？**

可以，但必须和 Prompt 模板中的对应字段名一致。常用的命名约定：
- `input_messages_key="input"` → 模板中 `{input}`
- `history_messages_key="history"` → 模板中 `MessagesPlaceholder("history")`
- 或者 `history_messages_key="chat_history"` → 模板中 `MessagesPlaceholder("chat_history")`

**Q: 如果底层链同时需要 `{input}` 和 `{context}`（如 RAG），怎么传？**

`RunnableWithMessageHistory` 只注入 `history_messages_key` 这一项。`{context}` 由 `create_retrieval_chain` 内部从 retriever 的结果注入，不经过 `RunnableWithMessageHistory`。

### 3.6 一句话概括

**`RunnableWithMessageHistory` 本质上是一个"自动追加聊天记录的代理"**：你在外层调用 `chain_with_memory.invoke({"input": "新消息"})`，它内部替你做了三件事——(1) 根据 `session_id` 取出历史消息列表，(2) 把历史消息通过 `MessagesPlaceholder` 注入底层 Runnable，(3) 底层 Runnable 返回后，把本轮对话自动追加回历史。底层那个 Runnable（`prompt | llm | parser`）通过 `MessagesPlaceholder` 接收历史，每次 `invoke` 看到的是"历史消息 + 当前输入"拼好的完整 Prompt，所以它自然就有了上下文记忆。

---

## 四、执行顺序：一条消息在 RAG 链中的完整旅程

本小节用 `05_conversational_rag.py` 的链为蓝本，逐行追踪 `"How is it different from LangGraph?"` 从用户输入到最终回答的**每一个函数调用、每一次数据变形、每一步为什么这样设计**。

### 3.1 先看清整条链的结构

在追踪执行之前，先回顾 `05_conversational_rag.py` 是怎么拼出这条链的。这决定了运行时数据如何流转。

```python
# ===== 05_conversational_rag.py 的链组装代码（去掉了无关细节）=====

# 步骤 ①：加载向量库 → 创建 retriever
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 步骤 ②：子链1 — 历史感知检索器
# 内部做的事：chat_history + input → LLM改写为独立问题 → 用改写后的问题检索
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history...formulate a standalone question..."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=contextualize_prompt
)

# 步骤 ③：子链2 — QA 链
# 内部做的事：context + chat_history + input → 拼接 → 填模板 → LLM 生成回答
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based STRICTLY on context...\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

# 步骤 ④：主 RAG Chain — 把子链1和子链2串起来
# create_retrieval_chain 做的事：
#   输入 {"input": ..., "chat_history": ...}
#     → history_aware_retriever 执行 → 拿到 [Document, ...]
#     → 拼成 {"input": ..., "chat_history": ..., "context": [Document, ...]}
#     → qa_chain 执行 → 拿到最终回答
rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=qa_chain,
)

# 步骤 ⑤：包装会话记忆 — 这是最外层，用户直接调用的入口
store: dict[str, InMemoryChatMessageHistory] = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    runnable=rag_chain,                     # ← 被包装的 RAG 链
    get_session_history=get_session_history, # ← 回调：根据 session_id 获取历史
    input_messages_key="input",             # ← 用户的输入从 dict 的哪个 key 取
    history_messages_key="chat_history",    # ← 历史消息注入到 dict 的哪个 key
    output_messages_key="answer",           # ← 输出的回答从 dict 的哪个 key 取（追加到历史）
)
```

**五个步骤的关系图**：

```
conversational_rag_chain  ← 用户代码调用这个
  = RunnableWithMessageHistory  ← 包装器：自动管理 chat_history
      └─ rag_chain  ← 主链
           = create_retrieval_chain  ← 组装器：先检索 → 再 QA
                ├─ history_aware_retriever  ← 子链1：改写问题 + 检索
                │    = create_history_aware_retriever(llm, retriever, prompt)
                └─ qa_chain  ← 子链2：拼接文档 + LLM 生成
                     = create_stuff_documents_chain(llm, prompt)
```

---

### 3.2 阶段 0：`RunnableWithMessageHistory` 接管请求

#### 它是什么

`RunnableWithMessageHistory` 是一个**包装器（Wrapper）**，接收任何 Runnable（这里是 `rag_chain`），在调用前后自动处理聊天历史的读写。它自己不参与 LLM 推理——它只负责 **"调用前注入历史，调用后追加新消息"**。

#### 谁调用了它

用户代码只和 `conversational_rag_chain` 交互，不直接碰 `rag_chain`：

```python
# 用户代码 — 这是整条链的触发点
response = conversational_rag_chain.invoke(
    {"input": "How is it different from LangGraph?"},   # ← 只传 input！
    config={"configurable": {"session_id": "session_1"}}, # ← 指定线程ID
)
# 注意：用户不需要传 chat_history —— RunnableWithMessageHistory 会自动处理。
```

#### 它内部做了什么

当 `conversational_rag_chain.invoke(...)` 被调用时，`RunnableWithMessageHistory` 在**真正执行 rag_chain 之前**先跑自己的逻辑：

```python
# === RunnableWithMessageHistory.invoke() 内部逻辑（框架代码，非用户写的）===

# 第 1 步：从 config 中提取 session_id
session_id = config["configurable"]["session_id"]     # → "session_1"

# 第 2 步：调用 get_session_history(session_id)
# 这个函数是用户在创建链时提供的回调（见步骤 ⑤）
# 它返回一个 BaseChatMessageHistory 对象（内存或 SQLite 实现）
history = get_session_history("session_1")
# → 返回 InMemoryChatMessageHistory 实例
#   如果 "session_1" 第一次用 → store 中创建新的，messages=[]
#   如果之前用过 → 返回已有的，messages=[HumanMessage("What is LangChain?"), AIMessage("没有相关信息")]

# 第 3 步：从 history 中读出所有历史消息
chat_history_messages = history.messages
# → [HumanMessage("What is LangChain?"), AIMessage("当前知识库中没有相关信息。")]

# 第 4 步：把用户输入和历史消息拼成 rag_chain 需要的 dict
chain_input = {
    "input": "How is it different from LangGraph?",  # ← 用户传入的 input
    "chat_history": chat_history_messages,            # ← 框架自动注入的历史
    # ↑ 这个 key 对应 history_messages_key="chat_history"
}
```

**关键理解**：用户传的是 `{"input": "..."}`，但 `rag_chain` 收到的是 `{"input": "...", "chat_history": [...]}`。`RunnableWithMessageHistory` 在中间做了**"自动拼装"**——用户不需要手动管理历史，框架替你做了。

---

### 3.3 阶段 1：`create_retrieval_chain` 把 dict 路由给子链1

`rag_chain`（由 `create_retrieval_chain` 创建）收到了上一步拼好的 dict：

```python
{
    "input": "How is it different from LangGraph?",
    "chat_history": [HumanMessage("What is LangChain?"), AIMessage("当前知识库中没有相关信息。")]
}
```

`create_retrieval_chain` 的设计是：**先调 retriever（子链1），把结果拼成 context，再调 combine_docs_chain（子链2）**。

```python
# === create_retrieval_chain 内部逻辑（框架代码，简化）===

# 第 1 步：把整个 dict 传给 history_aware_retriever
retrieved_docs = history_aware_retriever.invoke({
    "input": "How is it different from LangGraph?",
    "chat_history": [HumanMessage(...), AIMessage(...)],
})
# → [Document("LangChain vs. LangGraph..."), Document("..."), ...]

# 第 2 步：把检索结果拼到 dict 里（新增 context 字段）
# 这一步在框架内部自动完成，用户看不到
qa_input = {
    "input": "How is it different from LangGraph?",
    "chat_history": [HumanMessage(...), AIMessage(...)],
    "context": retrieved_docs,  # ← 框架自动追加的！
}

# 第 3 步：传给 qa_chain
final_answer = qa_chain.invoke(qa_input)
```

"拼接数据"不是用户写的代码——是 `create_retrieval_chain` 这个工厂函数内置的逻辑。

---

### 3.4 阶段 2：子链1 — `history_aware_retriever` 内部三步

子链1 收到 `{"input": "...", "chat_history": [...]}`。它由 `create_history_aware_retriever` 创建，内部做三件事。

#### Step 2a：`contextualize_prompt` 把 dict 填成 Prompt

```python
# contextualize_prompt 是一个 ChatPromptTemplate：
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest user question, "
               "formulate a standalone question which can be understood "
               "without the chat history. Do NOT answer the question, "
               "just reformulate it if needed..."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# .invoke() 时：
#   1. {input} → 替换为 "How is it different from LangGraph?"
#   2. MessagesPlaceholder("chat_history") → 展开为两条消息
#   3. 生成最终的 ChatPromptValue（包含 4 条消息）
filled_prompt = contextualize_prompt.invoke({
    "input": "How is it different from LangGraph?",
    "chat_history": [
        HumanMessage("What is LangChain?"),
        AIMessage("当前知识库中没有相关信息。"),
    ],
})
```

填入后 LLM 实际收到的 Prompt：

```
┌──────────────────────────────────────────────────────────────┐
│ SystemMessage:                                                │
│   "Given the chat history and the latest user question,      │
│    formulate a standalone question which can be understood    │
│    without the chat history. Do NOT answer the question..."   │
│                                                              │
│ HumanMessage: "What is LangChain?"          ← chat_history[0] │
│ AIMessage:    "当前知识库中没有相关信息。"     ← chat_history[1] │
│ HumanMessage: "How is it different from     ← {input} 填入    │
│                LangGraph?"                                   │
└──────────────────────────────────────────────────────────────┘
```

**为什么 MessagesPlaceholder 要用 HumanMessage/AIMessage 展开，而不是拼成字符串？**
因为 Chat Model 需要区分"谁说的"——System 是指令、Human 是用户历史、AI 是之前的回答。如果拼成纯文本，模型就看不到角色边界了。

#### Step 2b：LLM 改写问题

```python
# create_history_aware_retriever 内部（框架代码，简化）
# 把填好的 Prompt 发给 LLM
rewrite_result = llm.invoke(filled_prompt)
# → AIMessage(content="How is LangChain different from LangGraph?")

# LLM 做的事：看到对话历史中有 "What is LangChain?"，理解当前问题中的
# "it" 指的是 "LangChain"，于是把 "it" 替换掉，生成一个独立的检索查询。
```

**为什么需要改写？** 用户说的 "it" 在向量库中搜不到任何东西——向量库里存的是 "LangChain" 不是 "it"。改写后的 "How is LangChain different from LangGraph?" 才能命中相关文档。

#### Step 2c：用改写后的查询去检索

```python
# create_history_aware_retriever 内部（框架代码，简化）
standalone_query = rewrite_result.content
# → "How is LangChain different from LangGraph?"

# 用改写后的文本去向量库做语义检索
retrieved_docs = retriever.invoke(standalone_query)
# → [Document("LangChain vs. LangGraph..."),
#     Document("..."),
#     Document("..."),
#     Document("...")]
```

retriever 内部又分三步（对中间件不可见，但帮助你理解）：
1. `embeddings.embed_query(standalone_query)` → 查询向量 `[0.023, -0.451, ...]`
2. 与 Chroma 库中所有文档向量做余弦相似度计算
3. 返回最相似的 4 个 Document

---

### 3.5 阶段 3：`create_retrieval_chain` 拼接数据

```python
# create_retrieval_chain 拿到子链1的结果后，自动拼装：
qa_input = {
    "input": "How is it different from LangGraph?",   # 原始用户问题（未改写！）
    "chat_history": [HumanMessage(...), AIMessage(...)], # 原始历史
    "context": [                                       # ★ 子链1的输出，被放到 context 字段
        Document("LangChain vs. LangGraph vs. Deep Agents\nUse LangGraph..."),
        Document("..."),
        Document("..."),
        Document("..."),
    ],
}
# 注意：input 是原始用户问题，不是改写后的查询。
# 改写后的查询只用于检索（retriever.invoke），用户看到的还是原始问题。
```

---

### 3.6 阶段 4：子链2 — QA 链把文档"喂"给 LLM

子链2 由 `create_stuff_documents_chain` 创建，内部做三件事。

#### Step 4a：拼接 Document 列表为字符串

```python
# create_stuff_documents_chain 内部（框架代码，简化）
# 把 List[Document] 变成一段连续的文本

# 每个 Document 转成文本：page_content
# 多个 Document 之间用 "\n\n" 分隔（默认 separator）
context_str = "\n\n".join(doc.page_content for doc in qa_input["context"])
# → "LangChain vs. LangGraph vs. Deep Agents\n\nUse LangGraph, our low-level
#    orchestration framework, for advanced needs combining deterministic and
#    agentic workflows. Deep Agents build on LangChain's agents..."

# 然后填入 {context} 变量
```

**为什么用 `create_stuff_documents_chain` 而不是手写拼接？** 因为它在拼接之前还做了 Token 限制检查、文档排序、分隔符统一——手写容易漏掉边界条件。

#### Step 4b：填 QA Prompt

```python
# qa_prompt 是一个 ChatPromptTemplate：
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledgeable AI assistant. Answer based STRICTLY "
               "on the following retrieved context.\n\n"
               "Rules:\n"
               "1. If the context contains the answer, answer based on it.\n"
               "2. If it does NOT, say: '当前知识库中没有相关信息。'\n\n"
               "## Retrieved Context\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# .invoke() 时：
#   1. {context} → 替换为 Step 4a 拼接好的文档字符串
#   2. {input} → 替换为原始用户问题
#   3. MessagesPlaceholder("chat_history") → 展开为历史消息
filled_qa_prompt = qa_prompt.invoke(qa_input)
```

填入后 LLM 实际收到的 Prompt：

```
┌──────────────────────────────────────────────────────────────┐
│ SystemMessage:                                                │
│   "You are a knowledgeable AI assistant. Answer based         │
│    STRICTLY on the following retrieved context.               │
│    ## Retrieved Context                                       │
│    LangChain vs. LangGraph vs. Deep Agents                    │
│    Use LangGraph, our low-level orchestration framework..."   │
│                                                              │
│ HumanMessage: "What is LangChain?"          ← chat_history[0] │
│ AIMessage:    "当前知识库中没有相关信息。"     ← chat_history[1] │
│ HumanMessage: "How is it different from     ← {input} 填入    │
│                LangGraph?"                                   │
└──────────────────────────────────────────────────────────────┘
```

**注意**：子链2 的 Prompt 中有 `{context}`（检索到的文档），子链1 的 Prompt 中没有——这就是两个子链的核心区别。子链1 只负责"找文档"，子链2 负责"基于文档回答"。

#### Step 4c：LLM 生成最终回答

```python
# qa_chain 内部（框架代码，简化）
final_response = llm.invoke(filled_qa_prompt)
# → AIMessage(content="LangChain 是一个高级框架，LangGraph 是低级编排框架。
#        LangChain 的代理实际上构建在 LangGraph 之上，继承了其持久化执行、
#        人机协同等特性。简而言之，LangGraph 更底层更强大，LangChain 是
#        在其之上构建的更简洁的接口。")

# qa_chain 返回的是纯字符串（内部已经过 StrOutputParser）
# 但 rag_chain（create_retrieval_chain）返回的是完整的 dict：
```

---

### 3.7 阶段 5：`RunnableWithMessageHistory` 追加新消息

`rag_chain` 执行完毕，返回 dict 给 `RunnableWithMessageHistory`：

```python
# rag_chain 的返回值（由 create_retrieval_chain 包装后）：
rag_output = {
    "input": "How is it different from LangGraph?",
    "chat_history": [HumanMessage(...), AIMessage(...)],
    "context": [Document(...), Document(...), Document(...), Document(...)],
    "answer": "LangChain 是一个高级框架，LangGraph 是低级编排框架..."
    # ↑ answer 字段是 create_retrieval_chain 从 qa_chain 的输出中提取的
}
```

`RunnableWithMessageHistory` 拿到这个 dict 后，执行"调用后"逻辑：

```python
# === RunnableWithMessageHistory.invoke() 的"调用后"逻辑（框架代码，简化）===

# 第 1 步：从输出中提取用户问题和 AI 回答
user_input = rag_output["input"]
# → "How is it different from LangGraph?"
ai_answer = rag_output["answer"]  # ← output_messages_key="answer" 指定的字段
# → "LangChain 是一个高级框架..."

# 第 2 步：把本轮对话追加到历史
history.add_message(HumanMessage(user_input))   # 追加用户问题
history.add_message(AIMessage(ai_answer))       # 追加 AI 回答

# 第 3 步：返回给用户（rag_output 原样透传）
return rag_output
```

**此时 `store["session_1"]` 的内容**：

```python
# 两轮对话后，store["session_1"].messages 变成 4 条：
[
    HumanMessage("What is LangChain?"),                    # 第1轮用户
    AIMessage("当前知识库中没有相关信息。"),                   # 第1轮AI
    HumanMessage("How is it different from LangGraph?"),   # 第2轮用户
    AIMessage("LangChain 是一个高级框架，LangGraph 是低级编排框架..."),  # 第2轮AI
]
```

**下一次**同一 `session_id` 的调用会从 Step 3.2 开始，`history.messages` 就有 4 条了——LLM 能看到完整的对话上下文。

---

### 3.8 完整调用栈（一次 `invoke` 内发生了什么）

```
用户代码：
  conversational_rag_chain.invoke(
      {"input": "How is it different from LangGraph?"},
      config={"configurable": {"session_id": "session_1"}}
  )
        │
        ▼
┌─ RunnableWithMessageHistory.invoke() ──────────────────────┐
│  [Before] get_session_history("session_1")                  │
│           → history.messages = [HumanMsg, AIMsg]           │
│  [Before] 拼装 chain_input = {input, chat_history}         │
│                                                            │
│  ┌─ rag_chain.invoke(chain_input) ─────────────────────┐  │
│  │                                                      │  │
│  │  ┌─ history_aware_retriever.invoke() ────────────┐  │  │
│  │  │  ① contextualize_prompt.invoke() 填模板        │  │  │
│  │  │  ② llm.invoke(填好的Prompt) → "How is         │  │  │
│  │  │     LangChain different from LangGraph?"       │  │  │
│  │  │  ③ retriever.invoke(改写后的查询)              │  │  │
│  │  │     → [Document, Document, Document, Document] │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │                      │                                 │  │
│  │  create_retrieval_chain 拼接 context 字段              │  │
│  │                      │                                 │  │
│  │  ┌─ qa_chain.invoke() ────────────────────────────┐  │  │
│  │  │  ① 拼接 Document → 字符串                       │  │  │
│  │  │  ② qa_prompt.invoke() 填 {context}+{input}     │  │  │
│  │  │  ③ llm.invoke(填好的Prompt)                    │  │  │
│  │  │     → AIMessage("LangChain 是一个高级框架...")  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │                      │                                 │  │
│  │  返回 {input, chat_history, context, answer}          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  [After] history.add_message(HumanMsg)                     │
│  [After] history.add_message(AIMsg)                        │
│  返回 rag_output 给用户                                     │
└────────────────────────────────────────────────────────────┘
```

---

## 五、完整时序图

```
时间 ──────────────────────────────────────────────────────────→

RunnableWithMessageHistory
  │
  ├─[Before] 读历史
  │   chat_history = [HumanMsg("What is LangChain?"), AIMsg("没有")]
  │   chain_input = {"input": "...", "chat_history": [...]}
  │
  ├─[子链1] 历史感知检索器
  │   │
  │   ├─ MessagesPlaceholder 展开 chat_history → 插入模板
  │   │   Template = System + [HumanMsg, AIMsg] + HumanMsg("{input}")
  │   │
  │   ├─ LLM.invoke(Template) → AIMsg("How is LangChain different...")
  │   │   │      ↑
  │   │   │   此时消息类型还是 AIMessage，content 是改写后的问题
  │   │   │
  │   ├─ 提取 content: "How is LangChain different from LangGraph?"
  │   │
  │   └─ retriever.invoke("How is LangChain different...")
  │       → [Document, Document, Document, Document]
  │
  ├─[拼接] create_retrieval_chain
  │   chain_input["context"] = [Document, ...]
  │
  ├─[子链2] QA 链
  │   │
  │   ├─ create_stuff_documents_chain
  │   │   ├─ 拼接 Document → 字符串 → 填入 {context}
  │   │   ├─ MessagesPlaceholder 展开 chat_history → 插入模板
  │   │   └─ 填入 {input}
  │   │
  │   └─ LLM.invoke(完整Prompt) → AIMsg("LangChain 是一个高级框架...")
  │
  └─[After] 写历史
      chat_history.append(HumanMsg("How is it different..."))
      chat_history.append(AIMsg("LangChain 是一个高级框架..."))
```

---

## 六、MessagesPlaceholder 的执行细节

这是最容易困惑的地方，展开说：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手。"),
    MessagesPlaceholder("chat_history"),   # ← 占位符，不是一条具体消息
    ("human", "{input}"),
])
```

**运行时，`MessagesPlaceholder` 被替换为实际的消息列表**：

```python
# 第一次调用（chat_history = []）
# 实际发给 LLM 的是：
[
    SystemMessage("你是助手。"),
    HumanMessage("{input} 的值"),
]
# MessagesPlaceholder 为空 → 不插入任何东西

# 第二次调用（chat_history = [HumanMessage("Hi"), AIMessage("Hello!")]）
# 实际发给 LLM 的是：
[
    SystemMessage("你是助手。"),
    HumanMessage("Hi"),        # ← 展开的第1条
    AIMessage("Hello!"),       # ← 展开的第2条
    HumanMessage("{input} 的值"),
]
# MessagesPlaceholder 被两条消息替代
```

**关键点**：`MessagesPlaceholder` 不是转化成字符串插入，而是**逐条展开为原始消息对象**，保留每条消息的类型（Human/AI/Tool）。这样 LLM 能正确区分对话中不同角色的发言。

---

## 七、Messages 在设计中的位置

```
输入/输出层          Messages 是这里的主角
══════════════════════════════════════════════
User Input   →   HumanMessage    ┐
AI Output    →   AIMessage       │  这些都是 Messages
Tool Result  →   ToolMessage     ┘
System Rule  →   SystemMessage
                                  │
Prompt 层                          ▼
══════════════════════════════════════════════
ChatPromptTemplate  把 Messages + 变量 →  最终 Message 列表
MessagesPlaceholder 把历史 Message 列表 → 展开插入

                                  │
模型层                             ▼
══════════════════════════════════════════════
BaseChatModel.invoke(List[Message]) → AIMessage
  └─ 内部调用 _generate(messages: List[Message])

                                  │
解析层                             ▼
══════════════════════════════════════════════
StrOutputParser    AIMessage → str
JsonOutputParser   AIMessage → dict/Pydantic
```

Messages 贯穿了从输入到输出的每一层——这就是为什么它是 LangChain 的"原子单位"。

---

# 提示词模板（Prompt Template）：从定义到使用

## 一、模板是什么——一句话

**Prompt Template = Messages 的"模具"**。定义好结构和占位符 `{variable}`，运行时灌入数据，产出一条条具体的 Message。

```python
# 模板（模具）                        # 填充后（成品）
SystemMessage("你是{role}。")    →   SystemMessage("你是翻译官。")
HumanMessage("翻译：{text}")     →   HumanMessage("翻译：Hello World")
```

## 二、模板在 core 定义，在 langchain 使用

### 定义在哪里（langchain-core）

全部在 `langchain_core.prompts`：

```python
from langchain_core.prompts import (
    PromptTemplate,              # 字符串模板
    ChatPromptTemplate,          # 聊天消息模板 ← 最常用
    MessagesPlaceholder,         # 消息列表占位符
    PipelinePromptTemplate,      # 多模板流水线
    FewShotPromptTemplate,       # 少样本字符串模板
    FewShotChatMessagePromptTemplate,  # 少样本聊天模板
)
```

这些都是 core 的**抽象 + 实现**，不依赖任何集成包。所以拿到任何项目里都能用。

### 怎么用（在 langchain 和各集成包中）

langchain 层的 `init_chat_model`、`create_agent` 等工厂函数**内部使用这些模板类型作为参数**：

```python
# langchain 层接受 prompt 参数，类型定义来自 core
from langchain_classic.chains import create_history_aware_retriever
#                                      ↓ prompt 参数类型是 langchain_core 的 ChatPromptTemplate
retriever = create_history_aware_retriever(llm, retriever, prompt=some_chat_prompt)
```

你自己写 Chain 时也是将 core 的模板类型作为零件使用：

```python
# 你在项目中写的
prompt = ChatPromptTemplate.from_messages([...])  # ← core 的类型
chain = prompt | llm | parser                     # ← 放进 langchain 的 LCEL 管道
```

---

## 三、核心模板类型详解

### 类型 1：PromptTemplate — 字符串模板

最原始的形式，`{变量}` 占位，返回纯字符串。**用于老式 LLM（completion API），新项目少用。**

```python
from langchain_core.prompts import PromptTemplate

t = PromptTemplate.from_template("用{language}解释：{topic}")
t.invoke({"language": "中文", "topic": "熵"})
# → "用中文解释：熵"
```

### 类型 2：ChatPromptTemplate — 聊天模板 ★ 最常用

返回 `ChatPromptValue`（一个 Messages 列表），用于 ChatModel。

**四种构造方式**：

```python
from langchain_core.prompts import ChatPromptTemplate

# 方式1：from_messages — 最常用
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，用中文回答。"),
    ("user", "{question}"),
])

# 方式2：from_template — 快捷方式（只有一条 user 消息）
prompt = ChatPromptTemplate.from_template("翻译：{text}")

# 方式3：手动指定每条消息类型
from langchain_core.prompts import (
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
)
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是{role}"),
    HumanMessagePromptTemplate.from_template("{question}"),
])

# 方式4：混合静态消息 + 模板消息
from langchain_core.messages import SystemMessage
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个数学家。"),   # ← 静态，无变量
    ("user", "证明：{theorem}"),              # ← 模板，有变量
])
```

### 类型 3：MessagesPlaceholder — 消息列表占位符

已经在项目中大量使用。核心特点：**占的是一段"消息列表"，不只是一个字符串**。

```python
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手。"),
    MessagesPlaceholder("history"),   # ← 运行时展开为多条消息
    ("user", "{input}"),
])

# 调用时传入消息列表
prompt.invoke({
    "history": [
        HumanMessage("Hi"),
        AIMessage("Hello!"),
    ],
    "input": "天气怎么样？",
})
# 结果：SystemMessage + HumanMessage("Hi") + AIMessage("Hello!") + HumanMessage("天气怎么样？")
```

**为什么不用 `{history}` 字符串变量？**

| | `{history}` | `MessagesPlaceholder` |
|---|---|---|
| 插入的是 | 一个字符串 | 多条原始 Message 对象 |
| 角色区分 | 无，全混在一起 | 保留每条消息的 Human/AI/Tool 类型 |
| LLM 看到的 | 一段文本 | 结构化的对话记录 |

---

## 四、三种进阶操作

### 操作 1：partial — 预填变量

把一部分变量提前填好，返回一个新模板（剩下的变量调用时再填）。

```python
# 基础模板：两个变量
base = ChatPromptTemplate.from_messages([
    ("system", "你是{role}。"),       # ← 预填
    ("user", "{question}"),          # ← 调用时填
])

# partial：预填 role
math_prompt = base.partial(role="数学家")
coder_prompt = base.partial(role="程序员")

# 现在两个子模板调用时只需要填 question
math_prompt.invoke({"question": "什么是群论？"})
coder_prompt.invoke({"question": "什么是闭包？"})
```

**实际场景**：一个基础模板 → 多个专用模板，不用重复写 system 规则。

### 操作 2：Few-Shot — 给模型示例

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

# 定义示例
examples = [
    {"input": "Hello", "output": "你好"},
    {"input": "Thank you", "output": "谢谢"},
    {"input": "Goodbye", "output": "再见"},
]

# 把每个示例转成 Human + AI 消息对
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 构造 Few-Shot 模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# 拼入主模板
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是翻译官，按照以下示例翻译："),
    few_shot_prompt,                           # ← 示例自动展开
    ("human", "{input}"),
])

final_prompt.invoke({"input": "Good morning"})
# LLM 看到：
#   System: 你是翻译官，按照以下示例翻译：
#   Human: Hello       →  AI: 你好
#   Human: Thank you   →  AI: 谢谢
#   Human: Goodbye     →  AI: 再见
#   Human: Good morning
# → AI: 早上好
```

### 操作 3：PipelinePromptTemplate — 多模板流水线

把一个模板的**输出**作为另一个模板的**变量**。

```python
from langchain_core.prompts import PipelinePromptTemplate

# 子模板1：生成角色描述
role_prompt = PromptTemplate.from_template("你是一个{field}领域的专家。")

# 子模板2：生成任务描述
task_prompt = PromptTemplate.from_template("请解释{topic}。")

# 主模板：拼装子模板的输出
full_prompt = PromptTemplate.from_template("{role_desc}\n\n{task_desc}")

# 流水线
pipeline = PipelinePromptTemplate(
    final_prompt=full_prompt,
    pipeline_prompts=[
        ("role_desc", role_prompt),   # role_prompt 的输出 → 填入 {role_desc}
        ("task_desc", task_prompt),   # task_prompt 的输出 → 填入 {task_desc}
    ],
)

pipeline.invoke({"field": "物理", "topic": "熵"})
# → "你是一个物理领域的专家。\n\n请解释熵。"
```

**场景**：复杂 Prompt 由多个独立模板拼装，各自维护、各自复用。

---

## 五、常用操作速查

```python
prompt = ChatPromptTemplate.from_messages([...])

# 调用
prompt.invoke({"key": "val"})         # → ChatPromptValue（Runnable 接口）
prompt.ainvoke({"key": "val"})        # 异步版
prompt.format(key="val")              # → str（旧式，不推荐）
prompt.format_messages(key="val")     # → List[BaseMessage]

# 信息
prompt.input_variables               # → ["role", "question"]  有哪些占位符
prompt.messages                      # 模板中的消息列表
prompt.partial(key="fixed")          # 预填变量 → 新模板
prompt.pretty_print()                # 打印模板结构
prompt.invoke(...).to_messages()     # 获取 List[BaseMessage]

# 组合
prompt | llm                         # 放进 LCEL 管道
prompt.pipe(llm)                     # 等价于 |
```

---

## 六、我们项目中的模板用法总结

| 文件 | 用了什么 | 做什么 |
|---|---|---|
| `01` | `ChatPromptTemplate.from_messages([("system",...),("user","{topic}")])` | 最简单的两消息模板 |
| `02` | 同上 | 两消息 + 加入输出解析器 |
| `03` | 无（只做索引） | — |
| `04` | `ChatPromptTemplate.from_messages([("system","{context}..."),("user","{question}")])` | 含 `{context}` 的 RAG Prompt |
| `05` | + `MessagesPlaceholder("chat_history")` | 在 04 基础上加历史占位符 |
| `06` | 同 05 | — |

演进路线一目了然：**基础模板 → 加 context → 加历史 → 加工具调用 → 加少样本**。

---

## 七、模板设计的最佳实践

1. **System 放规则、约束、角色** → 不参与变量变化，通常用 `partial` 预填
2. **User 放任务、问题** → 用 `{variable}` 留给每次调用时填
3. **MessagesPlaceholder 放历史** → 动态长度、运行时才确定的消息列表
4. **`{context}` 放检索结果** → 由 retriever 自动注入，不要手动传
5. **Few-Shot 示例放中间** → 在 System 之后、User 之前，给模型做参考
6. **模板 = Runnable** → 可以直接 `|` 进入任何 Chain，可以被嵌套

---

# ContentBlock：多模态消息的标准格式

## 一、本质是什么

**ContentBlock = LLM I/O 的"世界语"。** 一套统一的数据结构，描述模型输入/输出中各种类型的内容，屏蔽厂商之间的 API 差异。

传统上，消息内容是一个字符串：

```python
# 旧范式：content 就是纯字符串
HumanMessage(content="今天天气怎么样？")
AIMessage(content="北京今天25°C，晴天。")
```

但 `str` 装不下以下需求：

- **思考/推理过程**（DeepSeek-R1 的 `<think>`、Claude 的 extended thinking）
- **多模态**（图片、视频、音频、文件）
- **工具调用请求**（tool_call name + args + id）
- **引用/注释**（citations）
- **流式块**（streaming 时 content 是 token 片段）
- **不同厂商**返回同一语义但字段名完全不同

所以 LangChain 1.0 在 `langchain_core.messages.content` 里定义了 **ContentBlock —— 一套 TypedDict 规则，把所有内容类型标准化。**

## 二、它解决了什么

三个不同的 LLM 返回"思考过程 + 文本回答 + 工具调用"，原始格式完全不同：

```json
// OpenAI 格式
{
  "choices": [{
    "message": {
      "content": "我来查一下天气。",
      "tool_calls": [{ "id": "x", "function": { "name": "get_weather", "arguments": "{}" } }]
    }
  }]
}

// Anthropic 格式
{
  "content": [
    { "type": "thinking", "thinking": "用户想知道天气..." },
    { "type": "text", "text": "我来查一下天气。" },
    { "type": "tool_use", "id": "x", "name": "get_weather", "input": {} }
  ]
}

// Google Gemini 格式
{
  "candidates": [{
    "content": {
      "parts": [
        { "thought": "用户想知道天气..." },
        { "text": "我来查一下天气。" },
        { "functionCall": { "name": "get_weather", "args": {} } }
      ]
    }
  }]
}
```

**三个厂商，字段名、嵌套层级、数据结构全不一样。** 如果你直接解析原始 JSON，写三套代码。

**ContentBlock 的解法**：每个厂商有一个 `block_translator`，把原始响应翻译为统一的 `List[ContentBlock]`：

```python
# 同一套标准格式，无论底层是什么厂商
[
    ReasoningContentBlock(type="reasoning", reasoning="用户想知道天气..."),
    TextContentBlock(type="text", text="我来查一下天气。"),
    ToolCall(type="tool_call", id="x", name="get_weather", args={}),
]
```

## 三、所有 Block 类型一览

| Block 类型 | `type` 值 | 用途 | 方向 |
|---|---|---|---|
| `TextContentBlock` | `"text"` | 纯文本内容 | 输入 + 输出 |
| `ReasoningContentBlock` | `"reasoning"` | 模型推理/思考过程 | 输出 |
| `ToolCall` | `"tool_call"` | 工具调用请求 | 输出 |
| `InvalidToolCall` | `"invalid_tool_call"` | 格式错误的工具调用 | 输出 |
| `ToolCallChunk` | `"tool_call_chunk"` | 流式工具调用片段 | 输出 |
| `ServerToolCall` | `"server_tool_call"` | 服务端工具调用 | 输出 |
| `ServerToolResult` | `"server_tool_result"` | 服务端工具结果 | 输入 |
| `ImageContentBlock` | `"image"` | 图片（url/base64/file_id） | 输入 |
| `VideoContentBlock` | `"video"` | 视频 | 输入 |
| `AudioContentBlock` | `"audio"` | 音频 | 输入 |
| `FileContentBlock` | `"file"` | 文件（PDF/Word 等） | 输入 |
| `PlainTextContentBlock` | `"text-plain"` | 纯文本文件 | 输入 |
| `Citation` | `"citation"` | 引用标记 | 输出 |
| `NonStandardContentBlock` | `"non_standard"` | 尚未标准化的厂商特有数据 | 输出 |

## 四、如何翻译成各厂商能识别的格式

### 架构

```
你的代码 (使用 ContentBlock)
        │
        ▼
┌───────────────────────────────────────────────┐
│  AIMessage                                    │
│    content = [                                │
│      TextContentBlock("描述这张图"),            │
│      ImageContentBlock(url="https://..."),    │
│    ]                                          │
│                                               │
│  .content_blocks 属性 → 自动转标准格式          │
└───────────────────────────────────────────────┘
        │
        ▼ chat_model._generate(messages)
┌───────────────────────────────────────────────┐
│  block_translators/<provider>.py              │
│                                               │
│  输入：标准 ContentBlock                        │
│  输出：厂商 API 要求的 JSON 格式                 │
│                                               │
│  OpenAI:   content → [{type, text}, {type,    │
│                        image_url, ...}]       │
│  Anthropic: content → [{type, text}, {type,   │
│                         image, source, ...}]  │
│  Google:   content → [{parts: [{text},        │
│                        {inlineData, ...}]}]   │
└───────────────────────────────────────────────┘
        │
        ▼
    厂商 API 调用
        │
        ▼ 响应返回
┌───────────────────────────────────────────────┐
│  block_translators/<provider>.py              │
│                                               │
│  输入：厂商 API 的原始 JSON 响应                 │
│  输出：标准化 ContentBlock 列表                 │
│                                               │
│  translate_content(ai_message) → [            │
│    ReasoningContentBlock(...),                │
│    TextContentBlock(...),                     │
│    ToolCall(...),                             │
│  ]                                            │
└───────────────────────────────────────────────┘
        │
        ▼
  AIMessage.content_blocks → 你的代码可直接用
```

### 代码层的翻译调用链

```python
# 在 AIMessage 上调用 .content_blocks 属性
ai_msg = llm.invoke("分析这张图片")

# .content_blocks 内部逻辑：
# 1. 如果 content 已经是 List[dict] → 直接返回
# 2. 如果 content 是 str → 检测 additional_kwargs 中的厂商标记
#    → 调用对应厂商的 translate_content()
#    → 返回标准 ContentBlock 列表
blocks = ai_msg.content_blocks

for block in blocks:
    match block["type"]:
        case "text":
            print(f"文本: {block['text']}")
        case "reasoning":
            print(f"思考: {block['reasoning']}")
        case "tool_call":
            print(f"调用工具: {block['name']}({block['args']})")
```

## 五、典型输出场景

这一节用四个真实场景展示 ContentBlock 能做什么——从 AI 回复中精确提取"思考过程""图片""工具调用""引用来源"。

### 场景 1：带思考过程的回答

**背景**：DeepSeek-R1、Claude Thinking 等推理模型在给出最终回答前，会先在内部推理（写在 `<think>` 标签或 `thinking` 字段中）。你通常只关心最终回答——但调试时需要看到思考过程。

**模型返回的原始数据**（厂商各自不同）：

```
# DeepSeek-R1 的原始响应（简化）：
"content": "好的，用户问的是量子纠缠。\n\n量子纠缠是指..."   
"additional_kwargs": {"reasoning_content": "嗯，量子纠缠是量子力学的核心概念之一，用户可能想要一个通俗的解释..."}

# Claude Thinking 的原始响应（简化）：
"content": [
    {"type": "thinking", "thinking": "这是量子力学核心概念，需要先铺垫..."}, 
    {"type": "text", "text": "量子纠缠是指..."}
]

# ↑ 两个厂商的字段名、嵌套方式完全不同！
# 如果直接解析原始 JSON，每个厂商要写一套代码。
```

**`content_blocks` 统一后的结果**（你的代码只处理一种格式）：

```python
# 调用模型
ai_msg = llm.invoke("用通俗的语言解释量子纠缠")

# 不管底层是 DeepSeek 还是 Claude，content_blocks 都是同一套格式：
for block in ai_msg.content_blocks:
    print(f"[{block['type']}]")

# 输出（DeepSeek-R1 或 Claude Thinking 都这样）：
# [reasoning]   ← 模型的思考过程（自动从 <think> 或 thinking 字段提取）
# [text]        ← 模型给用户的最终回答
```

**data 长什么样**——遍历每一个 block，看看里面的实际字段：

```python
for block in ai_msg.content_blocks:
    match block["type"]:
        case "reasoning":
            # ReasoningContentBlock 的结构：
            # {
            #     "type": "reasoning",
            #     "id": "lc_abc123...",          ← 框架自动生成的唯一 ID
            #     "reasoning": "嗯，用户想要通俗解释。量子纠缠的核心是..."  ← 思考全文
            # }
            print(f"🧠 思考过程: {block['reasoning'][:80]}...")

        case "text":
            # TextContentBlock 的结构：
            # {
            #     "type": "text",
            #     "id": "lc_def456...",
            #     "text": "量子纠缠指的是两个粒子无论相隔多远..."  ← 最终回答全文
            #     "annotations": [...]  ← 可能有引用标记（见场景4）
            # }
            print(f"💬 最终回答: {block['text'][:80]}...")
```

**为什么思考过程被单独抽出来？** 因为 `content` 字段直接给用户看（聊天 UI），而 `reasoning` 你已经通过 `content_blocks` 拿到了——不需要去解析 DeepSeek 的 `additional_kwargs` 或 Claude 的 `thinking` 块。

**如果你只想要文本，一行就够了**：

```python
# 提取纯文本回答（跳过思考过程）
text = "".join(b["text"] for b in ai_msg.content_blocks if b["type"] == "text")

# 提取思考过程（调试用）
reasoning = "".join(b["reasoning"] for b in ai_msg.content_blocks if b["type"] == "reasoning")

# 提取工具调用请求
tool_calls = [b for b in ai_msg.content_blocks if b["type"] == "tool_call"]
```

---

### 场景 2：多模态输入 — 让 LLM"看图说话"

**背景**：你想让 LLM 分析一张图片，需要把图片和文字说明一起发给模型。不同厂商传图片的方式完全不同。

**你用 ContentBlock 构造一条消息**（厂商无关）：

```python
from langchain_core.messages import HumanMessage
from langchain_core.messages.content import create_text_block, create_image_block

# 一条消息 = 文字块 + 图片块，顺序就是你写的顺序
msg = HumanMessage(content=[
    create_text_block("描述这张图片里的内容："),
    create_image_block(
        url="https://example.com/architecture_diagram.png",
        mime_type="image/png",
    ),
])

# 这条消息在代码里的实际结构：
# [
#     {"type": "text",  "id": "lc_001", "text": "描述这张图片里的内容："},
#     {"type": "image", "id": "lc_002", "url": "https://...", "mime_type": "image/png"},
# ]
```

**block_translator 把统一格式转为厂商格式**（你不需要写这段，框架自动做）：

```python
# 当 llm.invoke([msg]) 时，ChatOpenAI 内部调用：
# _convert_from_v1_to_chat_completions(msg)

# 你的 ContentBlock（统一格式）→ 转为 OpenAI 格式：
# [
#     {"type": "text", "text": "描述这张图片里的内容："},
#     {"type": "image_url", "image_url": {"url": "https://...", "detail": "auto"}},
# ]

# 如果是 Anthropic 模型，同样的 ContentBlock 转为 Anthropic 格式：
# [
#     {"type": "text", "text": "描述这张图片里的内容："},
#     {"type": "image", "source": {"type": "url", "url": "https://...", "media_type": "image/png"}},
# ]
```

**三种传图方式一视同仁**：

```python
# 方式 1：URL（远程图片）
create_image_block(url="https://example.com/photo.jpg", mime_type="image/jpeg")

# 方式 2：Base64（本地图片，把文件读成 base64 字符串）
import base64
with open("local_photo.jpg", "rb") as f:
    b64_data = base64.b64encode(f.read()).decode()
create_image_block(base64=b64_data, mime_type="image/jpeg")

# 方式 3：File ID（先上传到 OpenAI/Anthropic 的 Files API，用返回的 file_id）
create_image_block(file_id="file-abc123")
```

**完整调用**：

```python
response = llm.invoke([msg])
# → AIMessage(content="这张架构图展示了微服务之间的调用关系：API Gateway 连接了...")

# 如果 LLM 返回的回复中也包含图片（如生成图表），同样可以从 content_blocks 读取：
for block in response.content_blocks:
    if block["type"] == "image":
        save_image(block["url"])  # 保存生成的图片
    elif block["type"] == "text":
        print(block["text"])      # 打印文字说明
```

---

### 场景 3：工具调用 + 流式 — 边生成边识别 tool_calls

**背景**：LLM 在流式输出时，可能在中途决定调用工具。`content_blocks` 能让你在流式过程中区分"这是文字 token"还是"这是工具调用的参数片段"。

**先理解：流式中 ContentBlock 的两种类型**

流式输出时，content_blocks 中的类型不是 `text` 就是 `tool_call_chunk`：

```python
llm_with_tools = llm.bind_tools([get_weather])

# chunk 是 AIMessageChunk（流式片段），每个 chunk 包含若干 content_blocks
for chunk in llm_with_tools.stream("北京今天天气怎么样？适合户外运动吗？"):
    for block in chunk.content_blocks:
        match block["type"]:
            case "text":
                # 正常的文字 token。流式过程中逐字返回。
                # block = {"type": "text", "text": "北"}  ← 第一个 token
                # block = {"type": "text", "text": "京"}  ← 第二个 token
                # block = {"type": "text", "text": "今"}  ← ...
                print(block["text"], end="", flush=True)

            case "tool_call_chunk":
                # 工具调用的参数片段。流式过程中逐字段返回。
                # block = {"type": "tool_call_chunk", "name": "get_weather", "id": "call_1", "args": ""}
                # block = {"type": "tool_call_chunk", "args": "{\"city\": \"北京\"}"}
                # 所有 chunk 累加后 → 完整 ToolCall
                print(f"\n 🔧 正在准备工具调用: {block.get('name', '?')}...")
```

**流式输出的完整时间线**（某个真实时刻的状态）：

```
时间 →

chunk 1:  AIMessageChunk(content_blocks=[{"type": "text", "text": "我来"}])
chunk 2:  AIMessageChunk(content_blocks=[{"type": "text", "text": "查一下"}])
chunk 3:  AIMessageChunk(content_blocks=[{"type": "text", "text": "天气"}])
  ↓ LLM 意识到需要调工具了
chunk 4:  AIMessageChunk(content_blocks=[{"type": "tool_call_chunk", "name": "get_weather", "id": "call_1", "args": ""}])
chunk 5:  AIMessageChunk(content_blocks=[{"type": "tool_call_chunk", "args": "{\"city\": \"北京\"}"}])
  ↓ 工具调用参数传输完毕
chunk 6:  AIMessageChunk(content_blocks=[])  ← 空块，流式结束
```

**`tool_call` vs `tool_call_chunk`**：

| | `tool_call`（完整） | `tool_call_chunk`（流式片段） |
|---|---|---|
| 出现时机 | 非流式调用 `invoke()` 后 | 流式调用 `stream()` 过程中 |
| 内容 | 完整的 `{"name": ..., "args": {...}, "id": "..."}` | 逐个字段累加的片段 |
| 可执行？ | 是，直接传给 Tool | 否，需要等待全部 chunk 合并 |

**合并规则**：同名 + 同 index 的 chunk 自动累加——`name="get"` + `name="_weather"` = `name="get_weather"`。这是 AIMessageChunk 的 `+` 运算符内置的。

---

### 场景 4：引用与注释 — 知道 AI 的回答来自哪里

**背景**：LLM（特别是 Anthropic Claude + 网页检索）可以标注回答的每个段落引用了哪个来源。`content_blocks` 把这些引用以 annotation 形式附在 text 块上。

**一条带引用的回复长什么样**：

```python
ai_msg = llm.invoke("根据 LangChain 文档，框架的核心组件有哪些？")

for block in ai_msg.content_blocks:
    if block["type"] != "text":
        continue

    # 打印文本
    print(f"💬 {block['text']}")

    # 检查这段文本有没有引用标记
    annotations = block.get("annotations", [])
    for ann in annotations:
        if ann["type"] == "citation":
            # Citation 的结构：
            # {
            #     "type": "citation",
            #     "id": "lc_cite001",
            #     "url": "https://docs.langchain.com/oss/python/overview",
            #     "title": "LangChain Overview",
            #     "start_index": 12,      ← 引用从回复文本的第 12 个字符开始
            #     "end_index": 35,        ← 到第 35 个字符结束
            #     "cited_text": "LangChain provides modular core components"
            # }
            cited_part = block["text"][ann["start_index"]:ann["end_index"]]
            print(f"  ↑「{cited_part}」引用自: {ann.get('url', 'N/A')}")
```

**实际输出效果**（在聊天 UI 中的渲染）：

```
💬 LangChain 的核心组件包括 Models、Messages、Tools、Agents 和 Middleware。

  ↑「Models、Messages、Tools、Agents 和 Middleware」引用自: https://docs.langchain.com/oss/python/overview

💬 其中 Middleware 是 1.0 版本引入的新特性...

  ↑「Middleware 是 1.0 版本引入」引用自: https://docs.langchain.com/oss/python/middleware
```

**`start_index` / `end_index` 指的是什么？** 它们指向的是 **LLM 的回复文本**，不是原始文档。所以 `start_index=12, end_index=35` 表示"当前 text 块的第 12~35 个字符引用了那个来源"。这样前端可以精确高亮被引用的文字。

**哪些模型支持引用？** Anthropic Claude（原生 citations）、OpenAI（通过 response_format + web_search 产生）、Google Gemini（grounding metadata → 转为 citation）。不同厂商的引用格式各异，但 `content_blocks` 把它们统一为 `Citation`。

## 六、`extras` 字段 — 厂商特性不丢失

标准 ContentBlock 可能没有某个厂商的特有字段。`extras` 字典保留这些数据：

```python
# Google Gemini 的 thought signature
TextContentBlock(
    type="text",
    text="J'adore la programmation.",
    extras={"signature": "EpoWCpc..."},  # ← Google 特有字段
)

# 翻译回 Google 格式时，extras 会被带上
# 其他厂商则忽略 extras
```

## 七、什么时候用 ContentBlock

| 场景 | 用法 |
|---|---|
| 你写普通文本对话 | 不需要 — `content="字符串"` 即可 |
| 你处理模型的思考过程 | `ai_msg.content_blocks` → 找 `"reasoning"` |
| 你处理多模态（图片/音频） | `create_image_block()` / `create_audio_block()` |
| 你处理工具调用 | `ai_msg.content_blocks` → 找 `"tool_call"` |
| 你处理多个厂商的响应 | 统一用 `content_blocks`，不用关心底层格式 |
| 你要拿 token 用量 | `ai_msg.usage_metadata`，不在 ContentBlock 里 |

**一句话总结**：日常聊天 `content="string"` 足够；一旦涉及多模态、推理过程、工具调用、跨厂商兼容，用 `content_blocks` 统一处理。

---

# 批处理、流式处理、事件监听与异步并发

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

## 一、为什么需要结构化输出

当你需要把 LLM 的回答交给代码处理时，**字符串不够用**：

```python
# 不结构化：你需要手动正则、split、strip
raw = "温度：25°C\n天气：晴天\n湿度：45%"
temp = raw.split("\n")[0].split("：")[1]  # ← 脆如纸，换个表述就崩

# 结构化：直接用对象的属性
report.temperature  # → 25.0
report.condition    # → "晴天"
report.humidity     # → 45.0
```

---

## 二、方式一：`with_structured_output`（推荐）

### 2.1 原理

告诉底层 API 直接使用 JSON Schema 约束模型输出，由 API 保证结构正确，而不是靠 Prompt 文本去"恳求"模型。

```python
# OpenAI/DeepSeek 底层等价于：
# response_format={"type": "json_schema", "json_schema": {...}, ...}
```

**优点是可靠**：API 层面强约束，不会出现 JSON 格式错误、缺字段、多余文本。

### 2.2 定义 Pydantic Schema

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="deepseek-v4-pro", temperature=0)

# ===== 定义一个结构化输出 =====
class WeatherReport(BaseModel):
    """天气查询结果"""
    city: str = Field(description="查询的城市名称")
    temperature: float = Field(description="当前温度，单位摄氏度")
    condition: Literal["晴天", "多云", "下雨", "下雪", "雾霾"] = Field(
        description="天气状况"
    )
    humidity: float = Field(description="湿度百分比，范围 0~100")
    summary: str = Field(description="一句话天气总结和建议")
```

### 2.3 `Field` 的完整用法

```python
from pydantic import BaseModel, Field
from typing import Optional

class ProductReview(BaseModel):
    product_name: str = Field(
        description="被评价的产品名称",
        min_length=2,               # 最少 2 个字符
        max_length=100,             # 最多 100 个字符
    )
    rating: float = Field(
        description="评分，1~5 分",
        ge=1.0,                     # ≥ 1（greater than or equal）
        le=5.0,                     # ≤ 5（less than or equal）
    )
    pros: list[str] = Field(
        description="优点列表，最多 3 条",
        max_length=3,               # 列表最多 3 项
    )
    cons: Optional[list[str]] = Field(
        default=None,
        description="缺点列表，没有则为空",
    )
    sentiment: str = Field(
        default="中性",
        description="整体情感倾向：正面/负面/中性",
        pattern=r"^(正面|负面|中性)$",  # 精确正则约束
    )

# 嵌套模型
class AnalysisResult(BaseModel):
    """完整分析结果"""
    summary: str = Field(description="分析摘要")
    sentiment_score: float = Field(description="情感得分，0~1", ge=0, le=1)
    key_points: list[str] = Field(description="关键要点列表")
```

### 2.4 绑定并使用

```python
# 绑定 schema
structured_llm = llm.with_structured_output(WeatherReport)

# 直接调用 → 返回 Pydantic 对象！
report = structured_llm.invoke("北京今天天气怎么样？")
# → WeatherReport(
#     city="北京",
#     temperature=25.0,
#     condition="晴天",
#     humidity=45.0,
#     summary="今天北京天气晴朗，温度舒适，适合户外活动。"
# )

# 像普通对象一样使用
print(report.temperature)    # 25.0
print(report.condition)      # "晴天"
print(report.model_dump())   # {"city":"北京", ...}  → dict
print(report.model_dump_json())  # '{"city":"北京",...}' → JSON str
```

### 2.5 三种 method 选项

```python
# method=json_schema（默认）— API 传 JSON Schema 约束
#   → 最可靠，支持字段校验
llm.with_structured_output(MyModel, method="json_schema")

# method=function_calling — 利用 function calling 机制
#   → 模型"假装"调用一个函数，参数就是你的 schema
llm.with_structured_output(MyModel, method="function_calling")

# method=json_mode — 简单 JSON 模式
#   → 只保证是 JSON，不保证符合 schema（弱约束）
llm.with_structured_output(MyModel, method="json_mode")

# include_raw=True — 同时返回解析后的对象和原始响应
result = llm.with_structured_output(MyModel, include_raw=True).invoke(...)
# result["raw"]    → 原始 AIMessage
# result["parsed"] → Pydantic 对象（解析失败时为 None）
# result["parsing_error"] → 解析错误信息
```

### 2.6 放入 Chain 中

```python
# 直接放进 LCEL 管道
chain = prompt | llm.with_structured_output(WeatherReport)

report: WeatherReport = chain.invoke({"city": "北京"})
# 返回的就是 Pydantic 对象，不需要 StrOutputParser
```

---

## 三、方式二：`JsonOutputParser`（模板驱动）

### 3.1 原理

`JsonOutputParser` **不靠 API 约束**，而是把格式要求写进 Prompt 文本，请 LLM 照做。适合不支持 `response_format` 的旧模型。

### 3.2 定义并获取格式指令

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 定义 schema
class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    director: str = Field(description="导演姓名")
    genre: str = Field(description="电影类型")
    score: float = Field(description="评分，1~10", ge=1, le=10)
    review: str = Field(description="简短影评，不超过 100 字", max_length=100)

# 创建解析器
parser = JsonOutputParser(pydantic_object=MovieReview)

# 获取格式指令 → 注入到 Prompt 中
format_instructions = parser.get_format_instructions()
# → '{"title": "str", "director": "str", "genre": "str", "score": "float", "review": "str"}'
```

### 3.3 构造带格式指令的 Prompt

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "你是电影评论专家。\n"
        "请严格按照以下 JSON 格式回复，只输出 JSON，不要有任何其他文字。\n\n"
        "{format_instructions}"
    )),
    ("user", "评价电影：{movie_name}"),
])

# 调用时填入格式指令
chain = prompt | llm | parser

result = chain.invoke({
    "movie_name": "星际穿越",
    "format_instructions": format_instructions,
})
# → {"title": "星际穿越", "director": "克里斯托弗·诺兰", ...}
```

---

## 四、方式三：`PydanticOutputParser`

功能和 `JsonOutputParser` 类似，但返回的是 Pydantic 对象而不是 dict：

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=MovieReview)
format_instructions = parser.get_format_instructions()

chain = prompt | llm | parser
result: MovieReview = chain.invoke({...})
# → MovieReview(title="星际穿越", director="克里斯托弗·诺兰", ...)
```

---

## 五、各类解析器对比

| 解析器 | 原理 | 返回值 | 可靠性 | 适用 |
|---|---|---|---|---|
| `StrOutputParser` | 提取 `.content` | `str` | 100% | 聊天、文本生成 |
| `JsonOutputParser` | Prompt 文本要求 JSON | `dict` | 中，可能格式错 | 旧模型、不支持 json_schema |
| `PydanticOutputParser` | Prompt 文本要求 JSON → Pydantic | `PydanticModel` | 中，可能格式错 | 同上 + 需要类型校验 |
| `with_structured_output` | API 底层 `response_format` | `PydanticModel` | **高**，API 强约束 | 现代模型（GPT-4+, Claude, DeepSeek） |

**选择决策**：

```
你需要结构化？
├─ 只要纯文本 → StrOutputParser
├─ 模型支持 json_schema ？
│   ├─ 是 → with_structured_output(PydanticModel)   ← 推荐
│   └─ 否 → JsonOutputParser / PydanticOutputParser
│           └─ 注意：Prompt 里必须强调"只输出 JSON，不要其他"
└─ 要同时拿原始响应？
    └─ with_structured_output(schema, include_raw=True)
```

---

## 六、结构化输出要点总结

1. **`Field(description=...)` 是给 LLM 的语义锚点** — 不写 description 时模型可能填任意值；写了描述后模型按语义填充，准确率显著提升。

2. **`Literal` 枚举优于自由文本** — `condition: Literal["晴天","多云","下雨"]` 比 `condition: str` 更可靠，模型被强制选一个。

3. **`ge`/`le`/`min_length`/`pattern` 是附加约束** — 它们限制值域，模型输出不符合时解析报错 → 触发重试。

4. **`with_structured_output` > Prompt 文本要求** — API 约束 100% 是 JSON，Prompt 文本约束 ≈ 95%。生产环境选前者。

5. **`temperature=0` 配合结构化输出** — 结构化解析不需要创意，温度越低越稳定。

6. **嵌套模型不要过深** — 2~3 层嵌套就够了，太深模型容易丢字段。

7. **`include_raw=True` 用于容错** — 解析失败的原始响应仍可拿到，做降级处理。

8. **流式 + 结构化不冲突** — `stream()` 返回 Pydantic 对象的流式块，框架自动组装。

9. **`JsonOutputParser` 必须配合 Prompt** — 不传 `format_instructions` 到 Prompt，模型不知道要输出 JSON。

10. **JSON 修复** — `JsonOutputParser` 内部有容错逻辑（自动补逗号、引号等），但仍可能失败，生产环境建议 `with_structured_output`。

---

## 七、非 JSON 类型的结构化解析

除了返回 dict/Pydantic 对象，还有很多场景只需要 LLM 输出一个**简单类型**——布尔、枚举、列表等。

### 7.1 布尔值解析

**场景**：判断用户意图是否属于某类、文本是否违规、情感是否正面等。

```python
from pydantic import BaseModel, Field

# === 方式1：with_structured_output + Pydantic（推荐）===

class BooleanJudgment(BaseModel):
    """布尔判断结果"""
    result: bool = Field(description="判断结果：true 或 false")
    reason: str = Field(description="判断理由，一句话")

chain = prompt | llm.with_structured_output(BooleanJudgment)

r = chain.invoke("用户: 我想取消订单，太难用了")
print(r.result)   # True
print(r.reason)   # "用户表达了负面情绪和取消意图"

# === 方式2：轻量 RunnableLambda（不需要理由时）===

from langchain_core.runnables import RunnableLambda

bool_chain = (
    prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(lambda s: s.strip().lower().startswith("true"))
)

result = bool_chain.invoke("Is the following text spam? ...")
# → True / False
```

### 7.2 枚举分类解析

**场景**：情感分类、意图识别、优先级分级、领域分类。

```python
from typing import Literal
from pydantic import BaseModel, Field

# === 方式1：Literal 枚举（with_structured_output，推荐）===

class SentimentResult(BaseModel):
    sentiment: Literal["正面", "负面", "中性"] = Field(
        description="情感倾向"
    )
    confidence: float = Field(
        description="置信度，0~1", ge=0, le=1
    )

chain = prompt | llm.with_structured_output(SentimentResult)
r = chain.invoke("分析这条评论的感情：产品很好，但不值得这个价。")
print(r.sentiment)   # "中性"
print(r.confidence)  # 0.85

# === 方式2：IntEnum 用数字分类 ===

from enum import IntEnum

class Priority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TriageResult(BaseModel):
    priority: Priority = Field(description="工单优先级")
    reason: str = Field(description="判断原因")

chain = prompt | llm.with_structured_output(TriageResult)
r = chain.invoke("帮我分析一下这个 Bug：登录页面崩溃，所有用户无法登录")
print(r.priority)     # Priority.CRITICAL
print(r.priority.value)  # 4
```

### 7.3 多标签分类

一个对象可以同时属于多个类别：

```python
class TagResult(BaseModel):
    categories: list[str] = Field(
        description="匹配的分类标签列表，可多选",
        # LLM 会从上下文理解可选范围
    )
    primary: str = Field(description="主要分类")

# 使用
class NewsClassifier(BaseModel):
    topics: list[Literal["科技", "金融", "体育", "娱乐", "教育", "医疗"]] = Field(
        description="新闻涉及的主题，可多选"
    )
    is_breaking: bool = Field(description="是否突发新闻")
    difficulty: Literal["通俗", "专业", "学术"] = Field(description="阅读难度")

chain = prompt | llm.with_structured_output(NewsClassifier)
r = chain.invoke("苹果发布新一代 M8 芯片，性能提升 50%，股价上涨 3%")
print(r.topics)       # ["科技", "金融"]
print(r.is_breaking)  # True
print(r.difficulty)   # "通俗"
```

### 7.4 列表解析器

LangChain 内置了三种列表解析器，直接从文本中提取列表。

```python
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    NumberedListOutputParser,
    MarkdownListOutputParser,
)

# === CommaSeparatedListOutputParser ===
parser = CommaSeparatedListOutputParser()
format_instructions = parser.get_format_instructions()
# → 'Your response should be a comma separated list, eg: `foo, bar, baz`'

chain = prompt | llm | parser
chain.invoke("列出 5 种编程语言")
# → ['Python', 'Java', 'JavaScript', 'C++', 'Go']

# === NumberedListOutputParser ===
parser = NumberedListOutputParser()
# 解析：1. 苹果\n2. 香蕉\n3. 橘子 → ['苹果', '香蕉', '橘子']

# === MarkdownListOutputParser ===
parser = MarkdownListOutputParser()
# 解析：- 苹果\n- 香蕉\n- 橘子 → ['苹果', '香蕉', '橘子']
```

这三个解析器的本质都是 `StrOutputParser` + 正则匹配，适合简单列表提取。

### 7.5 XML 解析器

```python
from langchain_core.output_parsers import XMLOutputParser

parser = XMLOutputParser()
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages([
    ("system", "用 XML 格式回复。\n{format_instructions}"),
    ("user", "{input}"),
])

chain = prompt | llm | parser
result = chain.invoke({
    "input": "描述 LangChain 和 LangGraph 的区别",
    "format_instructions": format_instructions,
})
# → {"output": {"tool": {...}, "description": "..."}}
```

| 解析器 | 输入格式 | 输出 | 适用 |
|---|---|---|---|
| `CommaSeparatedListOutputParser` | `a, b, c` | `list[str]` | 简单关键词提取 |
| `NumberedListOutputParser` | `1. a\n2. b` | `list[str]` | 有序列表 |
| `MarkdownListOutputParser` | `- a\n- b` | `list[str]` | Markdown 文档解析 |
| `XMLOutputParser` | `<tag>...</tag>` | `dict` | 嵌套结构化数据 |

### 7.6 混合结果（字典 + 嵌套结构）

```python
class ComplexResult(BaseModel):
    """混合类型示例：单个 schema 同时包含 bool/enum/list/嵌套对象"""
    is_valid: bool = Field(description="输入是否合法")
    category: Literal["投诉", "咨询", "建议", "闲聊"] = Field(description="分类")
    keywords: list[str] = Field(description="关键词列表，最多 5 个", max_length=5)
    confidence: float = Field(description="分类置信度", ge=0, le=1)

    class FollowUpQA(BaseModel):
        """嵌套：补充问答对"""
        question: str = Field(description="需要进一步确认的问题")
        expected_answer: Literal["是", "否", "不确定"] = Field(description="预期答案")

    follow_up: FollowUpQA = Field(description="如果不确定，需要追问的问题")

chain = prompt | llm.with_structured_output(ComplexResult)
r = chain.invoke("用户: 你们的产品什么时候发货？我上周就下单了。")
# → ComplexResult(
#     is_valid=True,
#     category="咨询",
#     keywords=["发货", "订单", "物流"],
#     confidence=0.92,
#     follow_up=FollowUpQA(question="你的订单号是多少？", expected_answer="不确定")
# )
```

---

## 八、结构化输出全类型速查

| 你需要 | 方案 | 代码 |
|---|---|---|
| 纯文本 | `StrOutputParser` | `chain = prompt \| llm \| StrOutputParser()` |
| 布尔值 | `with_structured_output(BoolModel)` | `class M(BaseModel): result: bool` |
| 枚举单选 | `Literal["A","B","C"]` | `class M(BaseModel): choice: Literal["A","B","C"]` |
| 数字枚举 | `IntEnum` | `class P(IntEnum): LOW=1; HIGH=2` |
| 多标签 | `list[Literal[...]]` | `class M(BaseModel): tags: list[Literal["A","B","C"]]` |
| 简单列表 | `CommaSeparatedListOutputParser` | `chain \| parser` |
| JSON dict | `JsonOutputParser` | `parser = JsonOutputParser()` |
| Pydantic 对象 | `PydanticOutputParser` / `with_structured_output` | `parser = PydanticOutputParser(pydantic_object=MyModel)` |
| 嵌套结构 | `with_structured_output(ComplexModel)` | 模型嵌套模型 |
| XML | `XMLOutputParser` | `parser = XMLOutputParser()` |

**结论**：`with_structured_output` + Pydantic 是万能方案——bool、enum、list、嵌套对象一个 Schema 全部覆盖。只有简单列表提取和 XML 场景才需要专用解析器。

---

# LangChain 1.0 设计总结：五大支柱

## 一、统一 Runnable 抽象 → 一切皆可组合

### 问题

0.x 时代，Prompt 用 `format()`、LLM 用 `__call__()`、Chain 用 `run()`、Tool 用 `_run()`——每个组件调用方式不同，组合需要胶水代码。

### 1.0 解法

**所有组件实现同一个接口：`Runnable`。**

```python
# 五个核心方法，所有组件通用
runnable.invoke(input)       # 同步
runnable.ainvoke(input)      # 异步
runnable.stream(input)       # 流式
runnable.batch(inputs)       # 批量并行
runnable.astream_events(input)  # 带事件的流式
```

### 带来的连锁收益

```
统一接口
  ├─→ | 管道符串联（LCEL）→ 声明式编程
  ├─→ 流式/异步/批量 自动获得 → 零额外代码
  ├─→ Chain = Runnable → 链中链嵌套
  ├─→ 任何 Runnable 可被 LangSmith 追踪
  ├─→ 任何 Runnable 可被 LangGraph 编排
  └─→ 自定义函数用 RunnableLambda 无缝嵌入
```

**一句话：Runnable 是整个 LangChain 生态的"通用语言"，所有上层能力都建立在它之上。**

---

## 二、标准化模型接口 → 厂商无关

### 问题

OpenAI、Anthropic、Google、DeepSeek 的 API 格式各不相同——字段名、嵌套层级、端点路径全不一样。

### 1.0 解法

**三层抽象解耦**：

```
langchain-core      BaseChatModel（定义 invoke/stream/batch/_generate）
                         ↑ 继承
langchain-openai    ChatOpenAI（任何 OpenAI 兼容 API → DeepSeek/Moonshot/Ollama 通用）
langchain-anthropic ChatAnthropic（Anthropic 原生协议）
langchain-deepseek  ChatDeepSeek（DeepSeek 原生协议）
                         ↑ 工厂函数
langchain           init_chat_model("模型名") → 自动识别厂商、加载对应包
```

**ContentBlock 统一所有模型的输入输出**：

```
厂商原始格式 → block_translators/<provider>.py → 标准 ContentBlock 列表
  OpenAI tool_calls     ──→  ToolCall(type="tool_call")
  Anthropic tool_use    ──→  ToolCall(type="tool_call")   ← 你的代码只处理这个
  Google functionCall   ──→  ToolCall(type="tool_call")

  DeepSeek <think>      ──→  ReasoningContentBlock(type="reasoning")
  Claude thinking       ──→  ReasoningContentBlock(type="reasoning")
  Gemini thought        ──→  ReasoningContentBlock(type="reasoning")
```

**你的代码不碰原始 API JSON，只和标准 ContentBlock 打交道。**

---

## 三、强化结构化输出 → LLM 真正可编程

### 问题

LLM 返回字符串，下游代码需要正则/字符串切割/`json.loads` + try/except，脆弱且不可靠。

### 1.0 解法

**`with_structured_output` 从 API 层面约束模型输出**：

```
你的 Pydantic Schema
        │
        ▼
API 原生 json_schema / function_calling / json_mode
        │
        ▼
模型严格遵守 Schema 生成 JSON
        │
        ▼
框架自动解析为 Pydantic 对象 → 类型安全、IDE 补全、字段校验
```

**覆盖全部类型**：

```python
bool      → class M(BaseModel): result: bool
enum      → class M(BaseModel): choice: Literal["A","B","C"]
int/float → class M(BaseModel): score: float = Field(ge=0, le=10)
list      → class M(BaseModel): tags: list[str]
嵌套      → class M(BaseModel): child: ChildModel
混合      → 一个 Schema 同时含 bool/enum/list/嵌套
```

**从"恳求模型好好输出"变成"约束模型必须这样输出"。**

---

## 四、完整事件体系与回调 → 全链路可观测

### 问题

黑盒调用 `chain.invoke(input) → output`，内部每一步发生了什么完全不知道。

### 1.0 解法

**`astream_events` — 7 种组件类型 × 3 个生命周期阶段 = 21 种标准事件**：

```
on_XXX_start     → data.input    (组件收到什么)
on_XXX_stream    → data.chunk    (流式中间片段)
on_XXX_end       → data.output   (组件产出什么) + 错误时 data.error
```

**5 种实战用途**：

| 用途 | 实现 |
|---|---|
| **调试** | 逐事件打印每步的输入/输出，定位哪个环节出错 |
| **性能剖析** | start 记时间 → end 算耗时，找出瓶颈 |
| **Token 监控** | `on_chat_model_end` 提取 `usage_metadata`，累计成本 |
| **流式 UI** | `on_retriever_start` → "检索中..."，`on_chat_model_stream` → 打字机效果 |
| **日志审计** | 记录每一步的完整 data，支持事后回溯 |

**`RunnableConfig` 提供运行时控制**：`tags`（环境标签）、`metadata`（用户 ID、请求来源）、`max_concurrency`（并发控制）、`timeout`（超时保护）、`callbacks`（自定义处理器）。

---

## 五、与 LangSmith + LangGraph 的生态协同

### 5.1 LangSmith — 可观测性平台

LangSmith 是 LangChain 的官方监控/调试/评估平台，**与 Runnable 协议零配置集成**。

```
你的 Chain 一行不改
        │
        ▼
设置环境变量 LANGCHAIN_TRACING_V2=true
        │
        ▼
LangSmith 自动捕获：
  ├─ 每个 Runnable 的调用链（嵌套层级、父子关系）
  ├─ 每步的输入/输出（完整快照）
  ├─ 每步的耗时（性能瓶颈一眼可见）
  ├─ 每步的 token 用量（成本可视化）
  ├─ 错误栈（失败点精准定位）
  └─ 用户反馈（thumbs up/down → 持续优化）
```

**关键**：不需要在代码里写 `log()` 或 `report()`，追踪是 Runnable 协议的**原生能力**。

### 5.2 LangGraph — 有状态编排引擎

LangGraph 是 LangChain 的兄弟项目，专注**复杂多步 Agent 编排**。Runnable 可以直接作为 LangGraph 的节点。

```
LangGraph 节点 = 任意 Runnable（chain / llm / tool / ...）

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  rewrite_   │────→│  retrieve   │────→│  generate   │
│  chain      │     │  (retriever)│     │  chain      │
└─────────────┘     └─────────────┘     └─────────────┘
       ↑                                       │
       │         ┌─────────────┐               │
       └─────────│   verify    │←──────────────┘
                 │   chain     │  (有环 → Agent)
                 └─────────────┘
```

| | LangChain | LangGraph |
|---|---|---|
| 做什么 | 单次调用（invoke/stream/batch） | 多步编排（有环、有条件、有持久化） |
| 状态 | 无状态 | 有状态（checkpoint，支持暂停/恢复） |
| 结构 | 线性管道 | 有向图（可分支、循环、条件路由） |
| 关系 | 提供 Runnable 零件 | 编排 Runnable 零件 |

**什么时候用 LangGraph**：链中链不够用的时候——需要循环（Agent 反复调用工具）、需要条件分支（检索不够时重新生成）、需要人工审批（Human-in-the-loop）、需要持久化断点续跑。

---

## 六、LangChain 1.0 设计理念

一张图总结：

```
┌───────────────────────────────────────────────────────────────┐
│                      LangChain 1.0                            │
│                                                               │
│  Runnable 协议 ─── 一切皆可 invoke / stream / batch            │
│       │                                                       │
│       ├─→ LCEL (|) ─── 声明式组合                              │
│       ├─→ ContentBlock ─── 厂商无关的 I/O                      │
│       ├─→ with_structured_output ─── LLM 可编程                │
│       ├─→ astream_events ─── 全链路可观测                      │
│       └─→ RunnableConfig ─── 运行时控制                        │
│                                                               │
│  langchain-core ─── 协议定义（Interface）                      │
│  langchain ─── 编排工厂（Factory / Best Practice）             │
│  集成包 ─── 具体实现（openai / anthropic / chroma / ...）       │
│                                                               │
│  LangSmith ─── 可观测性（追踪 / 监控 / 评估）                   │
│  LangGraph ─── 有状态编排（Agent / 循环 / 人机协作）            │
└───────────────────────────────────────────────────────────────┘
```

**LangChain 1.0 的核心哲学**：**让 LLM 应用开发像搭积木一样——每个模块都是 Runnable，用 `|` 串联，用 `with_structured_output` 确保输出可靠，用 `astream_events` 看清一切，用 LangSmith 监控，用 LangGraph 编排复杂流程。**

---

# 补充专题：未覆盖的核心知识点

## 一、Tool / Function Calling 完整生命周期

之前的章节提到了 `bind_tools` 和 `@tool`，但没有串起来讲完整的 Tool 生命周期。

### 1.1 定义工具

```python
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass

# === 简单工具：无上下文 ===
@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气。返回温度、天气状况、湿度。"""
    return f"{city}：晴，25°C，湿度 45%"

# === 带上下文的工具：注入运行时信息 ===
@dataclass
class UserContext:
    user_id: str
    location: str

@tool
def get_user_location(runtime: ToolRuntime[UserContext]) -> str:
    """获取当前用户的地理位置（无需参数，从上下文自动获取）。"""
    return runtime.context.location

# === 异步工具 ===
@tool
async def search_database(query: str) -> str:
    """搜索内部数据库。"""
    await asyncio.sleep(0.5)
    return f"搜索结果：关于 {query} 的 3 条记录"
```

### 1.2 Tool 需要什么才能被模型识别

**只有三样：函数名 + 参数类型 + docstring。**

```python
@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气。"""
    # ↑ docstring  → 模型判断"什么时候该用这个工具"
    # city: str   → 模型知道这个工具需要什么参数
    # 函数名       → 模型调用时指定 name="get_weather"
```

### 1.3 完整执行循环

```
用户: "北京天气怎么样？"
        │
        ▼
  llm.bind_tools([get_weather, get_user_location])
        │
        ▼
  LLM 输出: AIMessage(content=None, tool_calls=[{
      "name": "get_weather",
      "args": {"city": "北京"},
      "id": "call_001"
  }])
        │
        ▼
  框架执行: get_weather.invoke({"city": "北京"})
        │
        ▼
  ToolMessage(content="北京：晴，25°C", tool_call_id="call_001")
        │
        ▼
  把 ToolMessage 追加回消息列表，再次发给 LLM
        │
        ▼
  LLM 输出: AIMessage(content="北京今天晴天，25°C，适合户外活动。")
```

### 1.4 Agent — 自动完成循环

手写上面的循环很繁琐。`create_agent` 是 LangChain 提供的一站式 Agent 工厂：

```python
from langchain.agents import create_agent

agent = create_agent(
    llm=llm,
    tools=[get_weather, get_user_location, search_database],
    system_prompt="你是生活助手，用中文回复。需要查询时主动调用工具。",
)

# Agent 自动处理：LLM → 判断 → 调用工具 → 接收结果 → 再次 LLM → ... → 最终回答
result = agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
```

**Agent 与普通 Chain 的区别**：

| | Chain | Agent |
|---|---|---|
| 流程 | 固定线性 | 动态循环（LLM 决定调哪个工具、调几次） |
| 工具调用 | 手动处理 | 自动循环直到 LLM 不再要求调工具 |
| 中间件 | 无 | 支持 ToolRetry、HumanInTheLoop 等 |
| 适用 | 确定性流程（RAG） | 需要推理+行动的开放任务 |

### 1.5 Agent 中间件

LangChain 1.0 Agent 提供可插拔的中间件，影响每次工具调用前后的行为：

```python
from langchain.agents.middleware import (
    ToolRetry,          # 工具调用失败后自动重试
    ToolCallLimit,      # 限制工具调用次数防止死循环
    HumanInTheLoop,     # 关键操作前需要人工审批
    Summarization,      # 对话过去时自动摘要上下文
    TodoMiddleware,     # 自动写 TODO 列表规划多步任务
)

agent = create_agent(
    llm=llm,
    tools=[...],
    middleware=[
        ToolRetry(max_retries=2),
        ToolCallLimit(max_tool_calls=10),
        Summarization(trigger_token_count=4000),
    ],
)
```

| 中间件 | 解决什么问题 |
|---|---|
| `ToolRetry` | 工具偶发失败时自动重试 |
| `ToolCallLimit` | 防止 Agent 陷入无限工具调用循环 |
| `HumanInTheLoop` | 危险操作（删除、支付）需要人工确认 |
| `Summarization` | 对话历史太长时自动压缩，节省 Token |
| `TodoMiddleware` | 多步任务自动拆解为 TODO 列表 |

---

## 二、降级、重试与容错

### 2.1 降级链（Fallbacks）

当一个 LLM 调用失败时，自动切换到备用方案：

```python
# 主模型 + 备用模型
primary_llm = ChatOpenAI(model="gpt-4", timeout=10)
backup_llm = ChatOpenAI(model="gpt-3.5-turbo", timeout=10)

robust_llm = primary_llm.with_fallbacks([backup_llm])

# 当 primary_llm 失败（超时/限流/错误）→ 自动 fallback 到 backup_llm
response = robust_llm.invoke("Hello")

# 多级降级
multi_fallback = (
    ChatOpenAI(model="gpt-4")          # 首选
    .with_fallbacks([
        ChatOpenAI(model="deepseek-v4-pro"),  # 降级1
        ChatOpenAI(model="gpt-3.5-turbo"),    # 降级2
    ])
)
```

**不仅 LLM 可以降级，任何 Runnable 都可以**：

```python
# 检索器降级：先查 Chroma → 失败 → 返回空结果
safe_retriever = chroma_retriever.with_fallbacks([
    RunnableLambda(lambda q: [Document(page_content="无匹配结果")])
])
```

### 2.2 重试

```python
from tenacity import retry, stop_after_attempt, wait_exponential

# 方式1：Runnable 级别重试
retry_chain = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_multiplier=1,
    wait_exponential_max=60,
)

# 方式2：LLM 初始化时配置
llm = ChatOpenAI(
    model="deepseek-v4-pro",
    max_retries=3,                     # ← 内置，失败自动重试
    timeout=30,
)
```

### 2.3 批量容错

```python
# batch 时某个请求失败了不中断整个 batch
results = chain.batch(
    ["Q1", "Q2", "Q3", "Q4", "Q5"],
    return_exceptions=True,  # ← 失败返回异常对象而非抛异常
)

for i, r in enumerate(results):
    if isinstance(r, Exception):
        print(f"Q{i+1} 失败: {r}")
    else:
        print(f"Q{i+1}: {r}")
```

---

## 三、Document + Embedding + VectorStore 体系

之前的章节在具体代码中用过，但没有作为概念体系讲解。

### 3.1 三个类的关系

```
Document             →  文本 + 元数据的容器
     │
     ▼
Embedding            →  把 Document 文本转为向量
     │
     ▼
VectorStore          →  存储向量 + 提供相似度检索
     │
     ▼
Retriever            →  标准检索接口（Runnable！可进 LCEL）
```

### 3.2 Document

```python
from langchain_core.documents import Document

# Document 只有两个字段
doc = Document(
    page_content="LangChain is a framework for building LLM applications.",
    metadata={
        "source": "https://docs.langchain.com/",
        "title": "LangChain Overview",
        "page": 1,
        "author": "LangChain Team",
    },
)

print(doc.page_content)   # 文本内容
print(doc.metadata)        # 来源、作者、页码等
```

**Document 贯穿整个 RAG 流程**：Loader → Document → Splitter → Document(小) → Embedding → VectorStore → Retriever → Document(检索结果)。

### 3.3 Embeddings — 文本到向量的桥梁

```python
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# 云端嵌入（需要网络）
cloud_emb = OpenAIEmbeddings(model="text-embedding-3-small")

# 本地嵌入（离线，CPU）
local_emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 嵌入文档列表（批量）
vectors = local_emb.embed_documents([
    "今天天气真好",
    "怎样修电脑",
    "Python 编程入门",
])

# 嵌入单条查询
query_vec = local_emb.embed_query("天气怎么样？")

# vectors 是 List[List[float]]，每个子列表是 512~1536 维的向量
```

**关键参数**：

| 参数 | 含义 | 建议值 |
|---|---|---|
| `model_name` | 模型名 | BGE 中文选 `bge-small-zh-v1.5`，英文选 `all-MiniLM-L6-v2` |
| `model_kwargs["device"]` | 设备 | `"cpu"` 无论什么机器都行，`"cuda"` 有 GPU 时用 |
| `encode_kwargs["normalize_embeddings"]` | 归一化 | `True`：提升余弦相似度的精度 |
| `encode_kwargs["batch_size"]` | 嵌入批次大小 | CPU 设小（8~16），GPU 可设大（64~256） |

### 3.4 VectorStore — 存储 + 检索

```python
from langchain_chroma import Chroma

# 从零创建
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=local_emb,
    persist_directory="./my_db",
    collection_name="my_knowledge",
)

# 加载已有
vectorstore = Chroma(
    persist_directory="./my_db",
    embedding_function=local_emb,
    collection_name="my_knowledge",
)

# 三种检索方式
# 1. 直接检索
docs = vectorstore.similarity_search("query", k=4)

# 2. 带分数检索
docs_with_scores = vectorstore.similarity_search_with_score("query", k=4)
for doc, score in docs_with_scores:
    print(f"score={score:.4f}  {doc.page_content[:50]}")

# 3. MMR 检索（多样性优先，避免内容重复）
docs = vectorstore.max_marginal_relevance_search("query", k=4, fetch_k=10)

# 转成 Retriever（Runnable，进 LCEL）
retriever = vectorstore.as_retriever(
    search_type="similarity",        # 或 "mmr", "similarity_score_threshold"
    search_kwargs={"k": 4},
)
```

### 3.5 全文检索链

```python
# VectorStore → Retriever → Chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**三步搞定一个 RAG**：`VectorStore` 存 → `as_retriever()` 取 → `|` 进 Chain。

---

## 四、自定义 Runnable

当内置组件不满足需求时，可以写自己的 Runnable。

### 4.1 方式1：RunnableLambda（简单）

```python
from langchain_core.runnables import RunnableLambda

# 一行包成 Runnable
clean = RunnableLambda(lambda text: text.strip().replace("\n\n\n", "\n"))
chain = prompt | llm | clean | StrOutputParser()
```

### 4.2 方式2：继承 Runnable（完整控制）

```python
from langchain_core.runnables import Runnable
from typing import Iterator, AsyncIterator
from langchain_core.runnables.config import RunnableConfig

class TranslateAndSummarize(Runnable):
    """自定义 Runnable：翻译 + 摘要，有内部状态（计数器）"""

    def __init__(self, target_lang: str = "中文"):
        self.target_lang = target_lang
        self.call_count = 0  # 状态

    def invoke(self, input: str, config: RunnableConfig | None = None) -> str:
        self.call_count += 1
        # ... 调用 LLM 做翻译和摘要
        result = f"[{self.target_lang}] 第{self.call_count}次调用: {input}"
        return result

    def stream(self, input: str, config=None) -> Iterator[str]:
        for word in input.split():
            yield f"[{self.target_lang}] {word} "

    async def astream(self, input: str, config=None) -> AsyncIterator[str]:
        for word in input.split():
            yield f"[{self.target_lang}] {word} "

# 可以放进 LCEL
custom = TranslateAndSummarize("中文")
chain = prompt | llm | custom | StrOutputParser()
```

**需要实现的三个方法**：`invoke`（同步）、`stream`（同步流式）、`astream`（异步流式）。`batch`/`abatch`/`ainvoke` 由基类自动推导。

---

## 五、补充总结：LangChain 1.0 知识图谱

```
                     Runnable 协议
                    （一切皆 Runnable）
                     /    |    \
                    /     |     \
        LCEL 声明式组合  事件体系  结构化输出
       (| 管道 + dict)  (21种事件)  (Pydantic)
              |            |          |
              └────────────┼──────────┘
                           │
               ┌───────────┼───────────┐
               │           │           │
           Tool 体系    RAG 体系    容错体系
        (@tool + Agent) (Document +  (fallback +
          bind_tools     Embedding +  retry +
          create_agent   VectorStore   timeout)
          middleware      Retriever)
               │           │           │
               └───────────┼───────────┘
                           │
                    LangSmith（监控）
                    LangGraph（编排）
```

**学习路线建议**：

```
1. Runnable + LCEL ──→ 能写基础 Chain
2. Messages + Prompt ──→ 能构造对话
3. ChatModel + bind_tools ──→ 能调工具
4. with_structured_output ──→ 输出结构化
5. Document + Embedding + VectorStore ──→ 能做 RAG
6. astream_events ──→ 能调试和监控
7. Agent + create_agent + middleware ──→ 能做自主 Agent
8. Fallback + Retry ──→ 生产级可靠
9. LangGraph ──→ 复杂多步编排
10. LangSmith ──→ 全链路可观测
```
```
