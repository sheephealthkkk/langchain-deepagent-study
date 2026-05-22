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

