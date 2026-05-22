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

