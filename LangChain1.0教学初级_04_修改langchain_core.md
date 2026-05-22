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

