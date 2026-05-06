# RAG 完整流程详解

> 包含 `03_rag_indexing.py`（第一阶段：索引）和 `04_rag_retrieval.py`（第二阶段：检索+生成）

## RAG 是什么？

**RAG（Retrieval-Augmented Generation）** = 检索增强生成。让大模型在回答问题时，先从外部知识库中检索相关信息，再把检索结果作为"参考资料"一起送给模型，从而减少幻觉、获得更准确的答案。

RAG 分为两大阶段，这个文件完成的是**第一阶段：索引（Indexing）**。

```
┌───────────── 第一阶段：Indexing（离线）─────────────┐
│                                                    │
│  网页 → 加载 → 切分小块 → 向量化 → 存入向量数据库    │
│                                                    │
└────────────────────────────────────────────────────┘
                         ↓
┌───────────── 第二阶段：Retrieval + Generation（在线）─┐
│                                                    │
│  用户提问 → 向量检索 → 找到相关片段 → 拼入 Prompt → LLM 回答 │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 逐块详解

### 1. 初始化大模型（temperature=0.8）

```python
llm = ChatOpenAI(
    model="deepseek-v4-pro",
    temperature=0.8,
)
```

- **`temperature=0.8`**：控制模型输出的随机性
  - `0` = 完全确定性（同样输入 → 同样输出），适合事实型任务
  - `1` = 最大随机性，适合创意写作
  - `0.8` = 偏创意但仍保持一定可控性，是一个常见的平衡值
- 这里初始化模型是**为后续的生成（Generation）阶段做准备**，Indexing 阶段本身不调用 LLM

---

### 2. 加载网页（WebBaseLoader + BS4）

```python
loader = WebBaseLoader(
    web_path=WEB_URL,
    default_parser="html.parser",
    bs_kwargs={"features": "lxml"},
    bs_get_text_kwargs={"separator": "\n", "strip": True},
    header_template={"User-Agent": "Mozilla/5.0 ..."},
    requests_per_second=2,
)
```

#### 为什么用 WebBaseLoader？

它是 LangChain 内置的网页加载器，底层用 `requests` 发 HTTP 请求 + `BeautifulSoup4` 解析 HTML，一行代码搞定"抓取网页 → 提取纯文本"。

#### 各参数的作用

| 参数 | 作用 | 为什么这样设置 |
|---|---|---|
| `web_path` | 要抓取的网页 URL | 这里用 LangChain 官方文档作为知识库 |
| `default_parser="html.parser"` | Python 标准库的 HTML 解析器 | 无需额外安装，兼容性好 |
| `bs_kwargs={"features": "lxml"}` | 让 BS4 用 lxml 引擎解析 | lxml 比 html.parser 快很多，适合大网页 |
| `bs_get_text_kwargs` | 控制提取文本的格式 | `separator="\n"` 让元素间换行；`strip=True` 去掉多余空白 |
| `header_template` | 自定义 HTTP 请求头 | 加 User-Agent 防止被网站当作爬虫拦截 |
| `requests_per_second=2` | 每秒最多发 2 个请求 | 礼貌爬取，不给目标网站造成压力 |

#### `loader.load()` vs `loader.lazy_load()`

- **`load()`** — 一次性把所有网页加载到内存，返回 `List[Document]`。适合小网页
- **`lazy_load()`** — 延迟加载，返回迭代器，逐个 yield Document。适合大量网页，节省内存

---

### 3. 文档切分（RecursiveCharacterTextSplitter）

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", ".", " ", ""],
)
```

#### 为什么需要切分？

1. **大模型的上下文窗口有限** — 不能一次性塞入整本百科全书
2. **检索精度** — 小块更容易精准匹配到用户问题
3. **嵌入质量** — 嵌入模型对短文本的效果更好

#### 核心参数

| 参数 | 含义 | 为什么这样设置 |
|---|---|---|
| `chunk_size=500` | 每块最多 500 字符 | 太小丢失上下文，太大检索精度下降。500 是实践中常用的平衡点 |
| `chunk_overlap=100` | 相邻块之间重叠 100 字符 | 防止关键信息刚好被切在两块边界上丢失 |
| `separators` | 切分优先级列表 | 递归尝试：先按段落切 → 按行切 → 按句号切 → 按空格切 → 字符级切 |

#### "递归"的含义

不是一次性切完，而是**按 separators 优先级逐级尝试**：

```
1. 先用 "\n\n" 切 → 如果某块仍 > 500 字符
2. 再用 "\n" 切   → 如果某块仍 > 500 字符
3. 再用 "。" 切   → ...
4. 最后逐字符切
```

这样保证切出来的每个块既不超过 `chunk_size`，又尽量在**自然的语义边界**（段落、句子）处断开。

---

### 4. 向量化 + 存入向量库

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
```

#### 什么是向量化（Embedding）？

把一段文本映射成一组数字（向量），语义相近的文本在向量空间中距离更近：

```
"今天天气真好"  →  [0.12, -0.34, 0.78, ...]  (通常 1536 维)
"今天阳光明媚"  →  [0.13, -0.31, 0.75, ...]  ← 向量很近
"怎样修电脑"    →  [-0.89, 0.45, -0.12, ...] ← 向量很远
```

#### 为什么用 Chroma？

| 特性 | 说明 |
|---|---|
| **本地运行** | 不需要额外服务，数据存在本地磁盘 |
| **轻量级** | Python 原生实现，pip install 即用 |
| **持久化** | `persist_directory` 指定路径，数据不会丢失 |
| **相似度检索** | 内置余弦相似度等算法，一行 `similarity_search()` 即可 |

#### `Chroma.from_documents()` 做了什么？

```
chunks (文本块列表)
  ↓
embeddings.embed_documents(chunks)  ← 每个块 → 向量
  ↓
存入 Chroma 本地数据库（chroma_db/ 目录）
  ↓
后续可通过 similarity_search("问题") 检索最相关的块
```

---

### 5. 验证检索

```python
retrieved = vectorstore.similarity_search("What is LangChain?", k=2)
```

- **`similarity_search`**：将查询文本向量化，在库中找到最相似的 k 个文档块
- **`k=2`**：返回最相关的 2 个块
- 这一步只是验证 Indexing 是否成功，真正的 RAG 问答在第二阶段

---

## 关键概念速查

| 概念 | 一句话解释 |
|---|---|
| **Document** | LangChain 的数据容器，有 `page_content`（文本）和 `metadata`（来源等） |
| **Chunk** | 切分后的小文本块，每个仍是 Document 对象 |
| **Embedding** | 文本 → 向量的转换，语义相近的向量距离近 |
| **Vector Store** | 存储向量+原文的数据库，支持相似度检索 |
| **Chroma** | 轻量级本地向量库，适合开发和小规模使用 |

## Indexing 阶段的完整数据流

```
https://xxx.com (网页)
      │
      ▼ WebBaseLoader (requests + BeautifulSoup)
┌─────────────┐
│  Document   │  page_content="LangChain is a framework for..."
│             │  metadata={'source': 'https://...', 'title': '...'}
└─────────────┘
      │
      ▼ RecursiveCharacterTextSplitter
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Chunk 1    │  │  Chunk 2    │  │  Chunk 3    │  ...
└─────────────┘  └─────────────┘  └─────────────┘
      │                │                │
      ▼ HuggingFaceEmbeddings (BAAI/bge-small-zh-v1.5)
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Vector 1   │  │  Vector 2   │  │  Vector 3   │  ...  (1536维浮点数)
└─────────────┘  └─────────────┘  └─────────────┘
      │                │                │
      └────────────────┼────────────────┘
                       ▼
              ┌─────────────────┐
              │    Chroma DB    │  ← 持久化到 chroma_db/
              │  (向量 + 原文)   │
              └─────────────────┘
```

---

# 第二阶段：检索 + 增强生成（04_rag_retrieval.py）

## 整体流程

```
用户提问 "LangChain是什么？"
      │
      ▼ HuggingFaceEmbeddings（同一个模型）
┌─────────────┐
│  Query Vec  │  问题的向量表示
└─────────────┘
      │
      ▼ Chroma.similarity_search()
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Chunk 1    │  │  Chunk 3    │  │  Chunk 7    │  │  Chunk 5    │  ← 最相关的4块
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
      │                │                │                │
      └────────────────┼────────────────┼────────────────┘
                       ▼ format_docs()
              ┌─────────────────┐
              │  拼接后的上下文    │  ← {context}
              └─────────────────┘
                       │
                       ▼ + 用户问题 {question}
              ┌─────────────────┐
              │  RAG_PROMPT     │  ← 填充好的完整 Prompt
              └─────────────────┘
                       │
                       ▼ ChatOpenAI (deepseek-v4-pro, temperature=0.8)
              ┌─────────────────┐
              │  最终回答        │
              └─────────────────┘
```

---

## 逐块详解

### 1. 加载已持久化的向量库（而非重建）

```python
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="langchain_docs",
)
```

#### 和 03 的区别

| | `03_rag_indexing.py` | `04_rag_retrieval.py` |
|---|---|---|
| 操作 | `Chroma.from_documents()` **创建** | `Chroma(...)` **加载已存在** |
| 何时用 | 首次建立知识库 | 后续每次问答 |
| 数据流向 | 文本 → 向量 → 写入磁盘 | 从磁盘读取 → 就绪 |

#### 为什么分开？

**Indexing 是一次性的离线工作**（像建索引），**Retrieval 是每次问答都要执行的在线工作**。分开设计可以：
- 问答时不需要重新加载网页、重新切分、重新向量化
- 知识库可以增量更新而不影响问答服务

---

### 2. 将用户问题向量化 + 检索

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)
```

#### `as_retriever()` 的作用

`vectorstore` 是向量数据库本身，`retriever` 是 LangChain 的标准检索接口。`as_retriever()` 做了一层包装：

```
vectorstore (数据库)          retriever (标准接口)
┌──────────────┐            ┌──────────────────┐
│ add_documents │            │  invoke(query)    │  ← 统一入口
│ similarity_   │  ──包装──→ │  get_relevant_    │
│   search      │            │    documents()    │
│ delete        │            │                   │
│ ...           │            └──────────────────┘
└──────────────┘
```

#### 为什么用 Retriever 而不是直接调 `similarity_search`？

因为 **LangChain 的 LCEL Chain 只能串联 Runnable 对象**。`retriever` 是 Runnable，可以直接用 `|` 连接；而 `vectorstore.similarity_search` 是普通方法，不能放入 Chain。

#### 用户问题的向量化是自动的

`retriever.invoke("LangChain是什么？")` 内部自动完成：
1. `embeddings.embed_query("LangChain是什么？")` → 问题向量
2. 与向量库中的所有文档向量做余弦相似度计算
3. 返回最相似的 k 个 Document

这一过程对开发者透明，不需要手动调用 `embed_query`。

#### `search_type` 选项

| 类型 | 含义 | 适用场景 |
|---|---|---|
| `"similarity"` | 纯余弦相似度 | 通用，默认选择 |
| `"mmr"` (Max Marginal Relevance) | 既相关又多样，避免重复 | 知识库内容重复度高时 |
| `"similarity_score_threshold"` | 只返回相似度≥阈值的 | 需要最低置信度过滤 |

---

### 3. 构建增强提示词模板

```python
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一个知识渊博的 AI 助手。请严格根据以下「参考资料」回答用户的问题。\n\n"
        "## 规则\n"
        "1. 如果参考资料中包含答案，请基于资料内容回答，并注明引用来源。\n"
        "2. 如果参考资料不包含答案，请明确告知用户「当前知识库中没有相关信息」。\n"
        "3. 回答要简洁、准确，用中文回复。\n\n"
        "## 参考资料\n"
        "{context}"
    )),
    ("user", "{question}"),
])
```

#### 为什么这样设计？

| 设计要点 | 原因 |
|---|---|
| **占位符 `{context}` 和 `{question}`** | Chain 运行时自动填充，context 来自检索，question 来自用户 |
| **角色分离**（system 放约束，user 放问题） | Chat Model 天然理解这种结构，比纯拼字符串效果好 |
| **明确规则**（"严格根据参考资料"） | 防止模型无视检索结果、自己编造（减少幻觉） |
| **"不知道就说不知道"** | 避免模型在没有相关信息时胡编乱造 |
| **要求注明来源** | 可追溯性，用户能验证回答的准确性 |

#### 这个模板的实际效果（运行时）

```
System: 你是一个知识渊博的 AI 助手。请严格根据以下「参考资料」回答...

## 参考资料
[来源1] https://docs.langchain.com/...
LangChain is a framework for building LLM-powered applications...

[来源2] https://docs.langchain.com/...
LangChain vs. LangGraph vs. Deep Agents...

User: LangChain 和 LangGraph 有什么区别？
```

---

### 4. 拼接检索文档为上下文文本

```python
def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[来源{i+1}] {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
```

#### 为什么需要 `format_docs`？

检索返回的是 `List[Document]` 对象，但 `{context}` 占位符需要的是一个**字符串**。`format_docs` 负责这个转换：

```
[Document("LangChain is..."), Document("LangGraph is...")]
                        │ format_docs()
                        ▼
"[来源1] https://...\nLangChain is...\n\n---\n\n[来源2] https://...\nLangGraph is..."
```

#### 为什么标注来源编号？

模型可以在回答中引用 `[来源1]`、`[来源2]`，让用户知道信息出自哪里。

---

### 5. LCEL Chain 完整串联（核心）

```python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
```

#### 这是整个 RAG 流程的"总调度"，逐步拆解：

**第一步：并行构建字典**

```python
{
    "context": retriever | format_docs,    # 分支A
    "question": RunnablePassthrough(),      # 分支B
}
```

用户输入 `question` 字符串后，**同时**执行两条路径：

```
                    "LangChain是什么？"（用户输入）
                       /              \
                      /                \
         分支A       /                  \     分支B
      retriever                          RunnablePassthrough()
          │                                    │
      向量化 + 检索                             原样透传
          │                                    │
      [Doc1, Doc2, Doc3, Doc4]                 "LangChain是什么？"
          │                                    │
      format_docs()                            │
          │                                    │
      "[来源1]..."                             │
          │                                    │
          └──────────────┬─────────────────────┘
                         ▼
              {"context": "[来源1]...",
               "question": "LangChain是什么？"}
```

**`RunnablePassthrough()` 的作用**：把输入原封不动传下去。这里就是保持用户问题不变，让它填入 `{question}`。

**第二步：填入模板**

```python
| RAG_PROMPT
```

上一步的字典被解包填入模板：
- `dict["context"]` → `{context}`
- `dict["question"]` → `{question}`

结果是完整的 ChatPromptValue（含 system 和 user 两条消息）。

**第三步：LLM 生成**

```python
| llm
```

把完整的 Prompt 发给 DeepSeek，模型根据参考资料生成回答。

**第四步：输出解析**

```python
| StrOutputParser()
```

LLM 返回的是 `AIMessage` 对象，`StrOutputParser` 提取出纯文本字符串。

#### 为什么用 LCEL（`|` 管道）而不是手写代码？

| 对比 | 手写代码 | LCEL Chain |
|---|---|---|
| 流式输出 | 需要自己处理 | 自动支持 `.stream()` |
| 异步 | 需要 `async/await` 改造 | 自动支持 `.ainvoke()` |
| 可观测性 | 需要手动加日志 | LangSmith 自动追踪每一步 |
| 组合性 | 嵌套函数调用 | 声明式串联，结构清晰 |

---

### 6. 运行

```python
answer = rag_chain.invoke("LangChain 和 LangGraph 有什么区别？")
```

一行代码触发整条链路：检索 → 格式化 → 填 Prompt → LLM 生成 → 解析输出。

---

## 完整 RAG 链路总览

```
┌────────── 03_rag_indexing.py（离线，执行一次）──────────┐
│                                                        │
│   WebBaseLoader   →   RecursiveCharTextSplitter        │
│        ↓                         ↓                     │
│   抓取网页纯文本     切分为 500 字符的块                 │
│                              ↓                         │
│                     HuggingFaceEmbeddings               │
│                              ↓                         │
│                      Chroma.from_documents()            │
│                              ↓                         │
│                      chroma_db/（持久化磁盘）            │
└────────────────────────────────────────────────────────┘
                               │
                               │ 读取
                               ▼
┌────────── 04_rag_retrieval.py（在线，每次问答）─────────┐
│                                                        │
│   用户问题                                              │
│      │                                                 │
│      ├──→ retriever (向量化+检索) → format_docs()      │
│      │         ↓                         ↓             │
│      │    查 chroma_db              拼接为上下文字符串   │
│      │                              ↓                  │
│      │                          {context}              │
│      │                              │                  │
│      └──→ RunnablePassthrough ───→ {question}          │
│                     │                                  │
│                RAG_PROMPT（注入模板）                    │
│                     │                                  │
│                ChatOpenAI（生成回答）                    │
│                     │                                  │
│                StrOutputParser（提取文本）               │
│                     │                                  │
│                 最终回答                                │
└────────────────────────────────────────────────────────┘
```

## 关键设计决策

| 决策 | 做法 | 原因 |
|---|---|---|
| 检索数量 `k=4` | 返回 4 个最相关块 | 太少信息不足，太多会稀释关键信息、增加 token 消耗 |
| 嵌入模型选 BGE 中文 | `BAAI/bge-small-zh-v1.5` | 中英文双语，体积小（~100MB），CPU 可跑，适合本地开发 |
| Temperature 0.8 vs 0 | indexing 无关，retrieval 用 0.8 | RAG 场景建议偏低（0~0.3）以减少幻觉。0.8 是用户指定值 |
| 分离 Indexing 和 Retrieval | 两个独立文件 | Indexing 是一次性离线任务，Retrieval 是高频在线任务 |

---

# 第三阶段：带记忆的对话式 RAG（05_conversational_rag.py）

## 为什么需要记忆？

之前的 `04_rag_retrieval.py` 是**无状态**的——每次问答独立，不记得上一轮说了什么。真实对话中，用户会这样问：

```
用户第1轮: What is LangChain?        ← 首次提问
用户第2轮: How is it different?       ← "it" 指代上一轮的 LangChain
用户第3轮: Summarize what we discussed ← 要求总结前两轮的讨论
```

没有记忆时，模型无法理解 "it" 指什么。这就是 `05_conversational_rag.py` 要解决的问题。

---

## 整体架构：链中链（Chain in Chain）

```
┌─ RunnableWithMessageHistory（最外层）────────────────────┐
│  职责：管理会话级聊天历史的存取                            │
│                                                          │
│  ┌─ 主 RAG Chain ────────────────────────────────────┐  │
│  │                                                    │  │
│  │  ┌─ 子链1: 历史感知检索器 ─────────────────────┐  │  │
│  │  │  chat_history + 当前问题                     │  │  │
│  │  │       ↓                                      │  │  │
│  │  │  LLM 改写为独立检索查询                      │  │  │
│  │  │       ↓                                      │  │  │
│  │  │  retriever 检索                              │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                         ↓                           │  │
│  │  ┌─ 子链2: QA 链（Stuff Documents）───────────┐   │  │
│  │  │  context + chat_history + input             │   │  │
│  │  │       ↓                                      │   │  │
│  │  │  LLM 生成回答                                │   │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  调用前：get_session_history() → 注入 chat_history        │
│  调用后：input + answer → 追加到 chat_history             │
└──────────────────────────────────────────────────────────┘
```

**链中链**的含义：外链（RAG Chain）内部包含两个子链（历史感知检索器 + QA 链），每个子链又是独立的 LLM 调用链。

---

## 逐块详解

### 1. MessagesPlaceholder — 统一处理"有/无历史"

```python
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest user question..."),
    MessagesPlaceholder("chat_history"),   # ← 关键组件
    ("human", "{input}"),
])
```

#### 为什么不用 `{chat_history}` 普通变量？

| 方式 | 写法 | 效果 |
|---|---|---|
| 普通变量 | `{chat_history}` | 只能填充一个**字符串** |
| `MessagesPlaceholder` | `MessagesPlaceholder("chat_history")` | 展开为**多条消息**（HumanMessage + AIMessage 对） |

`MessagesPlaceholder` 的本质：在运行时根据 key 找到对应的消息列表，逐条插入模板。

#### 同一个模板，两种场景自动适配

**场景 A：第一轮对话（无历史）**
```
实际 Prompt：
  System: Given the chat history...
  Human: What is LangChain?
```
`MessagesPlaceholder` 展开为空，什么都不插入。

**场景 B：第三轮对话（有历史）**
```
实际 Prompt：
  System: Given the chat history...
  Human: What is LangChain?
  AI: LangChain is a framework for...
  Human: How is it different from LangGraph?
  AI: LangGraph is a low-level...
  Human: Summarize what we discussed
```
`MessagesPlaceholder` 展开为前面所有的 Human/AI 消息对。

#### 与 04 版本的区别

| | `04_rag_retrieval.py` | `05_conversational_rag.py` |
|---|---|---|
| Prompt 模板 | `{context}` + `{question}` 两个占位符 | 新增 `MessagesPlaceholder("chat_history")` |
| 历史处理 | 不支持 | 自动展开完整对话历史 |
| 指代理解 | 无法理解 "it" 等代词 | LLM 结合上下文改写为独立查询 |

---

### 2. 子链1：历史感知检索器（History-Aware Retriever）

```python
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_prompt,
)
```

#### 它做了什么？

```
用户输入: "How is it different from LangGraph?"
历史记录: 上一轮 "What is LangChain?" + 回答
                │
                ▼
      LLM 改写（contextualize_prompt + 历史）
                │
                ▼
      改写后: "How is LangChain different from LangGraph?"
                │
                ▼
      retriever 用改写后的查询去向量库检索
                │
                ▼
      [Doc1, Doc2, Doc3, Doc4]
```

**核心价值**：用户说 "it" 时，LLM 结合历史把 "it" 替换为 "LangChain"，再拿这个明确的问题去检索，命中率远高于直接用 "How is it different" 检索。

#### 为什么用 `create_history_aware_retriever` 而不是手写？

它是一个**工厂函数**，内部封装了完整流程：

1. 接收 `{"input": "...", "chat_history": [...]}`
2. 将 `chat_history` 和 `input` 填入 `contextualize_prompt`
3. 调用 LLM 生成改写后的独立问题
4. 提取 LLM 输出的纯文本
5. 传给 `retriever.invoke(rewritten_query)`

手写需要 20+ 行，工厂函数一行搞定。

#### 提示词设计要点

```python
contextualize_system = (
    "Given the chat history and the latest user question, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed. If the question is already "
    "standalone, return it as is. Respond in the SAME LANGUAGE "
    "as the original question."
)
```

| 指令 | 目的 |
|---|---|
| "formulate a standalone question" | 消除指代，生成独立查询 |
| "Do NOT answer the question" | 防止 LLM 在此步直接回答（只改写） |
| "If already standalone, return it as is" | 避免对完整问题做无意义改写 |
| "Respond in the SAME LANGUAGE" | 保持中/英文一致，不做翻译 |

---

### 3. 子链2：QA 链（Stuff Documents Chain）

```python
qa_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt,
    document_variable_name="context",
)
```

#### `create_stuff_documents_chain` 做了什么？

"Stuff" 的意思是**把所有检索到的文档"塞"进一个 Prompt**（而不是逐个处理）：

```
输入: {"input": "How is it different...", "context": [Doc1, Doc2, Doc3, Doc4], "chat_history": [...]}

步骤1: 把 [Doc1, Doc2, Doc3, Doc4] 拼接为字符串
        ↓
      Doc1.page_content + "\n\n" + Doc2.page_content + ...

步骤2: 把拼接后的字符串填入 {context}

步骤3: 把 chat_history 展开为消息列表，把 {input} 填入模板

步骤4: 调用 llm 生成回答
```

#### 与 04 中手写 `format_docs` 的区别

| | `04` 手写 | `05` `create_stuff_documents_chain` |
|---|---|---|
| 文档拼接 | 手动写 `format_docs` 函数 | 内置，通过 `document_separator` 参数控制分隔符 |
| 模板填入 | 手动 `RunnablePassthrough` 构建 dict | 自动解包，识别 `document_variable_name` |
| 复杂度 | 需要理解 LCEL 并行字典语法 | 一行函数调用 |
| 灵活性 | 完全自定义 | 标准流程，够用 |

**为什么 05 改用 `create_stuff_documents_chain`？** 因为 05 的链更复杂（已有子链1 + 历史管理），用工厂函数降低组合复杂度。04 是入门示例，手写更直观。

---

### 4. 组装主 RAG Chain

```python
rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=qa_chain,
)
```

`create_retrieval_chain` 把两个子链串在一起：

```
输入 {"input": "How is it different...", "chat_history": [...]}
        │
        ▼
  history_aware_retriever (改写问题 → 检索)
        │
        ▼
  {"input": "How is it different...", "chat_history": [...],
   "context": [Doc1, Doc2, Doc3, Doc4]}
        │
        ▼
  qa_chain (填充模板 → LLM 生成)
        │
        ▼
  {"input": ..., "chat_history": ..., "context": [...], "answer": "LangGraph is..."}
```

---

### 5. RunnableWithMessageHistory — 会话记忆管理

```python
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
```

#### 工作流程

```
调用 chain.invoke({"input": "How is it different..."}, 
                  config={"configurable": {"session_id": "session_1"}})

        ↓  Before invoke（调用前钩子）
  1. get_session_history("session_1") → 获取或创建该会话的历史
  2. 把历史消息列表注入到 chain 的 "chat_history" key
  3. chain 内部各 MessagesPlaceholder 自动展开历史

        ↓  Chain 执行
  RAG chain 完整运行（改写→检索→生成）

        ↓  After invoke（调用后钩子）
  4. 从输出中提取 "answer" key 的值
  5. 把 (用户input, AI answer) 追加到 session_1 的历史
  6. 下次该会话再调用时，历史自动包含上一轮内容
```

#### 为什么用字典 `store` 而不是全局变量？

```
store = {
    "session_1": [msg1, msg2, msg3, ...],  ← 用户A的对话
    "session_2": [msg1, msg2, ...],        ← 用户B的对话（隔离）
}
```

每个 `session_id` 对应独立的对话历史，不同用户/会话互不干扰。

#### `input_messages_key` / `history_messages_key` / `output_messages_key` 的作用

| 参数 | 含义 | 取值 | 为什么 |
|---|---|---|---|
| `input_messages_key` | 用户输入从 dict 的哪个 key 取 | `"input"` | chain 的输入是 `{"input": "..."}` |
| `history_messages_key` | 历史注入到 dict 的哪个 key | `"chat_history"` | 对应模板中 `MessagesPlaceholder("chat_history")` |
| `output_messages_key` | 输出中哪个 key 的值要追加到历史 | `"answer"` | `create_retrieval_chain` 返回 `{"answer": "..."}` |

---

### 6. 调用方式

```python
response = conversational_rag_chain.invoke(
    {"input": question},
    config={"configurable": {"session_id": session_id}},
)
```

与 04 的调用有两点不同：

| | `04` | `05` |
|---|---|---|
| 输入 | `rag_chain.invoke("问题字符串")` | `conversational_rag_chain.invoke({"input": "..."})` |
| config | 无 | 必须传 `session_id` 以区分会话 |
| 输出 | 纯文本 `str` | `{"input": ..., "answer": ..., "context": [...]}` |

需要 `config` 是因为 `RunnableWithMessageHistory` 要通过 `session_id` 找到对应的历史。

---

## 完整运行流程演示

对照实际运行的 3 轮对话：

```
Round 1: session_1, "What is LangChain?"
  → store["session_1"] 为空
  → MessagesPlaceholder 展开为空
  → 上下文改写：已经是独立问题，不改写
  → 检索结果中无明确定义 → 回答"知识库中没有相关信息"
  → 追加 (Human: "...", AI: "...") 到历史

Round 2: session_1, "How is it different from LangGraph?"
  → store["session_1"] 已有 Round 1 的对话
  → MessagesPlaceholder 展开为 Round 1 的 Human/AI 消息
  → LLM 看到历史，理解 "it" = "LangChain"
  → 改写为: "How is LangChain different from LangGraph?"
  → 用改写后的问题检索，命中 LangChain vs LangGraph 相关内容
  → 生成详细对比回答
  → 追加到历史

Round 3: session_1, "Summarize what we just discussed about"
  → store["session_1"] 已有 Round 1 + Round 2
  → LLM 看到完整历史，理解 "what we just discussed"
  → 改写的查询包含 LangChain 和 LangGraph 关键词
  → 检索命中 → 生成涵盖两轮讨论的总结
  → 追加到历史
```

---

## 对话记忆选项对比

| 类型 | 类 | 适用场景 |
|---|---|---|
| **内存** | `InMemoryChatMessageHistory` | 开发测试，重启丢失 |
| **数据库** | `CassandraChatMessageHistory` 等 | 生产环境，持久化 |
| **Redis** | `RedisChatMessageHistory` | 分布式，高并发 |
| **文件** | `FileChatMessageHistory` | 单机持久化 |

当前用 `InMemoryChatMessageHistory` 是因为它在内存中、零配置。切换到其他后端只需替换 `get_session_history` 中的返回值，链代码不用改。

---

## 关键概念速查（新增）

| 概念 | 一句话解释 |
|---|---|
| `MessagesPlaceholder` | 模板占位符，运行时展开为多条消息（而非单一字符串） |
| `create_history_aware_retriever` | 工厂函数，包装"LLM 改写问题 → 检索"的完整流程 |
| `create_stuff_documents_chain` | 工厂函数，包装"拼接文档 → 填模板 → LLM 生成"的 QA 流程 |
| `create_retrieval_chain` | 工厂函数，串起检索器和 QA 链 |
| `RunnableWithMessageHistory` | 链包装器，自动管理会话历史的存取 |
| `InMemoryChatMessageHistory` | 内存中的聊天历史存储 |
| `get_session_history` | 回调函数，根据 session_id 返回对应的历史对象 |

---

# 附录：两个子链到底干了什么 — 用一个完整对话走一遍

## 示例场景

假设用户进行两轮对话：

```
第1轮: "What is LangChain?"
第2轮: "How is it different from LangGraph?"
```

下面是每一步**数据长什么样**、**谁在干活**、**产出什么**的完整追踪。

---

## 第1轮: "What is LangChain?"（无历史）

### Step 0: 进入链路之前

```
用户输入:
  {"input": "What is LangChain?"}

config:
  {"configurable": {"session_id": "session_1"}}
```

`RunnableWithMessageHistory` 检查 `store["session_1"]`，不存在 → 创建空的 `InMemoryChatMessageHistory()`。

此时 `chat_history = []`（空列表）。

然后把 `chat_history` 注入到输入 dict：

```
传入子链的数据:
  {
    "input": "What is LangChain?",
    "chat_history": []       ← 空！
  }
```

---

### Step 1: 子链1（历史感知检索器）干活

子链1收到的就是上一步的数据。它的内部逻辑分三步：

**1a. 填模板 → 生成 Prompt**

```
Prompt 模板:
┌────────────────────────────────────────────┐
│ System: Given the chat history and the     │
│   latest user question, formulate a        │
│   standalone question...                   │
│                                            │
│ MessagesPlaceholder("chat_history")  ← 空!  │
│                                            │
│ Human: {input}                             │
└────────────────────────────────────────────┘
```

`chat_history=[]`，`MessagesPlaceholder` 展开为**空**：

```
实际发给 LLM 的 Prompt:
┌────────────────────────────────────────────┐
│ System: Given the chat history and the     │
│   latest user question, formulate a        │
│   standalone question...                   │
│                                            │
│ Human: What is LangChain?                  │
└────────────────────────────────────────────┘
```

**1b. LLM 改写问题**

LLM 看到 "What is LangChain?" → 已经是独立问题，不需要改写 → 原样返回：

```
LLM 输出: "What is LangChain?"
```

**1c. 用改写结果去检索**

```
retriever.invoke("What is LangChain?")
  → 向量化 "What is LangChain?"
  → 在 chroma_db 中做相似度搜索
  → 返回最相似的 4 个 Document
```

```
子链1 最终输出:
[
  Document("LangChain overview - Docs by LangChain\n..."),
  Document("LangChain vs. LangGraph vs. Deep Agents\n..."),
  Document("Start with Deep Agents for a batteries-included..."),
  Document("Use LangGraph, our low-level orchestration..."),
]
```

**子链1总结一句话：把"用户怎么说"翻译成"向量库能搜到什么"，有历史就用历史来消除指代。**

---

### Step 2: 子链2（QA 链）干活

子链2 收到主链传入的 dict：

```
输入:
{
  "input": "What is LangChain?",
  "chat_history": [],
  "context": [Document("..."), Document("..."), Document("..."), Document("...")]
}
↑ 这个 "context" 就是子链1的4个Document，由 create_retrieval_chain 自动拼上去的
```

内部流程：

**2a. 拼接文档**

```
create_stuff_documents_chain 先把 context 里的 4 个 Document 拼成字符串:

"[来源1] https://docs.langchain.com/...
 LangChain overview - Docs by LangChain...

 ---

 [来源2] https://docs.langchain.com/...
 LangChain vs. LangGraph vs. Deep Agents...

 ---

 [来源3] ...
"
```

**2b. 填入 QA Prompt**

```
QA Prompt 模板:
┌────────────────────────────────────────────┐
│ System: You are a knowledgeable AI         │
│   assistant. Answer based STRICTLY on      │
│   the following retrieved context.         │
│                                            │
│ ## Retrieved Context                       │
│ {context}   ← 填入上面拼接好的文档字符串      │
│                                            │
│ MessagesPlaceholder("chat_history")  ← 空!  │
│                                            │
│ Human: {input}  ← "What is LangChain?"     │
└────────────────────────────────────────────┘
```

**2c. LLM 生成回答**

LLM 看到：系统指令 + 检索资料 + 用户问题 → 生成回答。

由于知识库中 LangChain 文档的 overview 页面主要是目录/导航，没有明确定义段落，LLM 遵守规则：

```
子链2 输出 → "当前知识库中没有相关信息。"
```

> **为什么会这样？—— 根因分析**
>
> 抓取的网页 URL 是 `https://docs.langchain.com/oss/python/langchain/overview`，
> 这是一个**导航/目录页**，不是概念介绍页。检索到的 4 个 Document 实际内容是：
>
> ```
> 结果1: "LangChain overview - Docs by LangChain
>         Skip to main content / Search... / Navigation /
>         LangChain overview / Deep Agents / ..."
>                                ↑ 全是导航菜单
>
> 结果2: 和结果1基本相同（chunk_overlap=100 导致的重复）
>
> 结果3: "LangChain vs. LangGraph vs. Deep Agents
>         Start with Deep Agents... use LangChain directly..."
>                                 ↑ 是对比段落，不是定义段落
>
> 结果4: 和结果3基本相同
> ```
>
> **没有一条检索结果包含** "LangChain is a framework for..." 之类的定义句。
> 而 QA Prompt 规则明确要求：*"如果参考资料不包含答案，请明确告知用户"*。
> LLM 遵守了规则 → 如实返回「没有相关信息」。
>
> **这也解释了为什么第2轮能回答**：第2轮问的是 "LangChain 和 LangGraph 的区别"，
> 结果3/4 恰好有 `LangChain vs. LangGraph vs. Deep Agents` 对比段落，命中到了。
>
> **解决方式**：换一个包含实质内容的网页重建知识库，例如：
> `https://python.langchain.com/docs/get_started/introduction`

**子链2总结一句话：把检索到的文档 + 历史 + 问题 揉进一个 Prompt，让 LLM 基于资料回答。**

---

### Step 3: 离开链路之后

`RunnableWithMessageHistory` 拿到完整输出：

```python
{
  "input": "What is LangChain?",
  "chat_history": [],
  "context": [Document, Document, Document, Document],
  "answer": "当前知识库中没有相关信息。"
}
```

从 `output_messages_key="answer"` 提取值，把这一轮对话追加到历史：

```python
store["session_1"].add_message(HumanMessage("What is LangChain?"))
store["session_1"].add_message(AIMessage("当前知识库中没有相关信息。"))
```

现在 `store["session_1"]` 里有 2 条消息了。

---

## 第2轮: "How is it different from LangGraph?"（有历史）

### Step 0: 进入链路之前

```
用户输入:
  {"input": "How is it different from LangGraph?"}

config:
  {"configurable": {"session_id": "session_1"}}
```

`RunnableWithMessageHistory` 找到 `store["session_1"]`，取出历史，注入到 dict：

```
传入子链的数据:
{
  "input": "How is it different from LangGraph?",
  "chat_history": [
    HumanMessage("What is LangChain?"),
    AIMessage("当前知识库中没有相关信息。")
  ]
}
```

---

### Step 1: 子链1（历史感知检索器）干活

**1a. 填模板**

`chat_history` 这次不为空！`MessagesPlaceholder` 展开为 2 条消息：

```
实际发给 LLM 的 Prompt:
┌────────────────────────────────────────────┐
│ System: Given the chat history and the     │
│   latest user question, formulate a        │
│   standalone question which can be         │
│   understood without the chat history...   │
│                                            │
│ Human: What is LangChain?                  │  ← 历史1
│ AI: 当前知识库中没有相关信息。                 │  ← 历史2
│ Human: How is it different from LangGraph? │  ← 当前问题
└────────────────────────────────────────────┘
```

**关键点**：LLM 看到了前面的 "What is LangChain?" → 理解了用户这一轮说的 **"it" = "LangChain"**。

**1b. LLM 改写**

```
LLM 输出:
"How is LangChain different from LangGraph?"
```

"it" 被替换成了 "LangChain"。

**1c. 用改写结果检索**

```
retriever.invoke("How is LangChain different from LangGraph?")
  → 向量化 → 相似度搜索 → 4 个 Document
```

与第1轮用原始问题检索相比，这次的命中率显著提升——因为 "LangChain different from LangGraph" 比 "it different from LangGraph" 的语义更加明确。

```
子链1 最终输出:
[
  Document("LangChain vs. LangGraph vs. Deep Agents
           Use LangGraph, our low-level orchestration framework,
           for advanced needs combining deterministic and
           agentic workflows. Deep Agents build on LangChain's
           agents and inherit LangGraph's persistence..."),
  Document("..."),
  Document("..."),
  Document("..."),
]
```

**对比两次子链1的输入输出：**

| | 第1轮 | 第2轮 |
|---|---|---|
| 输入 chat_history | `[]` | `[Human, AI]` |
| LLM 看到的 Prompt | 只有当前问题 | 历史 + 当前问题 |
| LLM 做了什么 | 问题已独立，不改写 | 把 "it" 替换为 "LangChain" |
| 改写后查询 | `"What is LangChain?"` | `"How is LangChain different from LangGraph?"` |
| 检索命中 | 低（无明确定义） | 高（精准命中对比段落） |

---

### Step 2: 子链2（QA 链）干活

```
输入:
{
  "input": "How is it different from LangGraph?",
  "chat_history": [HumanMessage("What is LangChain?"), AIMessage("...")],
  "context": [Document("LangChain vs. LangGraph..."), ...]
}
```

**2a. 拼接文档** → 4 个 Document 拼成一个字符串

**2b. 填入 QA Prompt**

```
实际 Prompt:
┌────────────────────────────────────────────┐
│ System: You are a knowledgeable AI         │
│   assistant. Answer based STRICTLY on      │
│   the following retrieved context.         │
│                                            │
│ ## Retrieved Context                       │
│ LangChain vs. LangGraph vs. Deep Agents    │
│ Use LangGraph, our low-level orchestration │
│ framework...                                │
│                                            │
│ Human: What is LangChain?                  │  ← 历史
│ AI: 当前知识库中没有相关信息。                 │
│ Human: How is it different from LangGraph? │  ← 当前
└────────────────────────────────────────────┘
```

**2c. LLM 生成**

这次检索资料中包含了 LangChain vs LangGraph 的对比信息，LLM 基于此生成：

```
子链2 输出:
"LangChain 提供了一个易于使用、高度灵活的代理抽象。
 LangGraph 是一个低级编排框架，适用于高级场景。
 LangChain 的代理构建在 LangGraph 之上..."
```

---

### Step 3: 离开链路

新一轮对话追加到 `store["session_1"]`：

```
现在 store["session_1"]:
  HumanMessage("What is LangChain?")
  AIMessage("当前知识库中没有相关信息。")
  HumanMessage("How is it different from LangGraph?")
  AIMessage("LangChain 提供了一个易于使用...")
```

---

## 总结：两个子链的分工

用一个类比来理解：

```
你是一个研究员（用户），向图书管理员（系统）提问。

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  子链1 = 翻译官 + 检索员                                      │
│                                                             │
│  你: "How is it different from LangGraph?"                   │
│                                                             │
│  翻译官 翻看之前的对话记录，发现你刚才问了"LangChain"，         │
│  于是把"it"替换成"LangChain"：                                │
│    "How is LangChain different from LangGraph?"              │
│                                                             │
│  检索员 拿这个明确的问题去图书馆（chroma_db）找书，            │
│  抱回来 4 本最相关的。                                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  子链2 = 分析师                                               │
│                                                             │
│  分析师收到：                                                 │
│    - 检索员抱来的 4 本书（{context}）                         │
│    - 你们之前的聊天记录（{chat_history}）                      │
│    - 你当前的问题（{input}）                                  │
│                                                             │
│  分析师翻开 4 本书，对照聊天记录理解你的意图，                  │
│  从书中找出答案，写成一段回复。                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**子链1 只管"找对资料"，子链2 只管"基于资料回答"。**

没有子链1，子链2 拿到的就是一堆不相关的文档（因为 "it" 没法检索）。
没有子链2，用户拿到的就是一堆原始文档而不是一个整合好的答案。

| 子链 | 输入 | 核心动作 | 输出 |
|---|---|---|---|
| 子链1: history_aware_retriever | 历史 + 当前问题 | LLM 改写 → 向量检索 | `List[Document]`（4个） |
| 子链2: qa_chain | 检索文档 + 历史 + 问题 | 拼接文档 → 填 Prompt → LLM 生成 | `str`（最终回答） |
