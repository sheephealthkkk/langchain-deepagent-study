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
