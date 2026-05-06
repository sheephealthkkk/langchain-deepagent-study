# usetools.py 说明

## 这个文件在做什么？

向 AI 大模型注册"工具函数"，让模型在需要时可以调用这些函数来获取外部信息——这就是 LangChain 的 **Tool（工具）** 机制。

类比：就像你给 ChatGPT 装了两个"插件"，一个是查天气的，一个是查用户位置的。

---

## 逐块解释

### 1. 定义一个简单工具

```python
@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"
```

- `@tool` 装饰器把普通函数变成 LangChain 能识别的"工具"
- **文档字符串（docstring）是关键**：`"""获取指定城市的天气。"""` 会被自动提取为工具描述，模型根据这个描述来决定什么时候调用这个工具
- 参数类型提示 `city: str` 也会被自动解析成工具的输入 schema，传给模型
- 这里返回的是假数据（"总是阳光明媚"），真实场景会调用天气 API

### 2. 定义运行时上下文

```python
@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str
```

- 这是一个数据类，存放当前请求的**上下文信息**（比如当前用户是谁）
- 作用是让工具能拿到"请求级别的信息"，而不是每次都让模型传参
- 比如 `user_id` 不需要模型每次猜或传，而是框架自动注入

### 3. 定义带上下文注入的工具

```python
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"
```

- 参数不是普通的 `str`，而是 `ToolRuntime[Context]` —— 这是 LangChain 的**运行时注入机制**
- 框架会自动将 `Context` 实例注入到 `runtime.context`，工具直接从中取值
- 模型调用这个工具时**不需要传 user_id**，只需要触发调用即可
- 这里的逻辑：user_id=1 → 佛罗里达，其他 → 旧金山

---

## 核心概念总结

| 概念 | 作用 | 类比 |
|---|---|---|
| `@tool` | 把函数注册为模型可调用的工具 | 给 AI 装插件 |
| docstring | 工具的描述信息，模型据此判断何时调用 | 插件说明书 |
| 类型提示 | 自动生成工具的输入参数 schema | 插件接口定义 |
| `ToolRuntime` | 框架在运行时自动注入上下文，无需模型传参 | 后台自动获取的会话信息 |

## 工作流程

```
用户提问 → 模型分析 → 需要调用工具？
                          ↓ 是
                    框架注入 Context
                          ↓
                    执行工具函数（如 get_user_location）
                          ↓
                    返回结果给模型 → 模型整合答案 → 回复用户
```

**关键点**：模型自身不执行这些函数，它只是"请求调用"，实际执行由 LangChain 框架完成，结果再传回给模型。
