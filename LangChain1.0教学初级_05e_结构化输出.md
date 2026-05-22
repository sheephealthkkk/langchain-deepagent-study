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

