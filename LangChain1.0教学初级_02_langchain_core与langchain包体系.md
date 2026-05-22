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

