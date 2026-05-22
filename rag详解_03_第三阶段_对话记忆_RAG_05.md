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

---

