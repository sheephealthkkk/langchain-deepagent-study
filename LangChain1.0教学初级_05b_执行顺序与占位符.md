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

