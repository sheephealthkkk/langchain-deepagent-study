## 第十二章：长期记忆（Long-Term Memory）

### 12.1 什么是长期记忆

**短期记忆 = 跟着 thread_id 走**。换一个 thread_id，一切归零。  
**长期记忆 = 跟着 user_id 走**。不管多少个 thread_id，同一用户的所有对话共享记忆。

```
短期记忆（第11章）              长期记忆（本章）

  Thread A: 有历史               ┌─────────────────────┐
  Thread B: 全新开始             │   用户 Alice 的记忆库  │ ← 跨会话共享
  Thread C: 全新开始             │ ┌─────────────────┐ │
       ↑                         │ │ Alice 喜欢 Python │ │
  每条线程独立                   │ │ Alice 住在北京    │ │
                                 │ │ Alice 是 VIP 用户 │ │
                                 │ └─────────────────┘ │
                                 └─────────────────────┘
                                   ↑ 所有 Thread 都能检索
```

### 12.2 实现方式总览

| 存储类型 | 技术选型 | 适用记忆内容 | 检索方式 |
|---|---|---|---|
| **向量数据库** | Chroma / Milvus / Pinecone / Qdrant / Weaviate | 用户偏好、历史事实、语义记忆 | 语义相似度搜索 |
| **键值数据库** | Redis / DynamoDB / MongoDB | 用户配置、登录信息、简单偏好 | Key 精确查找 |
| **图数据库** | Neo4j / NebulaGraph | 实体关系、知识图谱 | 图遍历 + 推理 |
| **全文检索引擎** | Elasticsearch | 对话日志、关键词记忆 | 关键词 + 语义混合搜索 |

**本章重点**：向量数据库（最常用的长期记忆方案）。

### 12.3 语义检索与向量数据库

**核心思路**：把用户相关的信息（偏好、事实、历史）编码为向量，存入向量库。新对话时，用当前问题去检索最相关的记忆。

```
存储（写入记忆）：
  用户偏好文本 "Alice 喜欢 Python，常用 VS Code"
       │
       ▼ Embedding 模型编码
  [0.12, -0.45, 0.78, ...]   ← 向量（浮点数列表）
       │
       ▼ 存入向量数据库
  Chroma / Milvus / Pinecone

检索（查询记忆）：
  用户当前问题 "推荐一个开发工具"
       │
       ▼ Embedding 模型编码为向量
  [0.13, -0.42, 0.75, ...]   ← 与上面的向量相近！
       │
       ▼ 向量相似度检索 → 找到 "Alice 喜欢 Python，常用 VS Code"
       │
       ▼ 注入 Prompt
  "根据你的偏好：你喜欢 Python，常用 VS Code。推荐：PyCharm..."
```

### 12.4 主流向量数据库对比

| | Chroma | Milvus | Pinecone | Qdrant | Weaviate |
|---|---|---|---|---|---|
| **部署** | 本地嵌入 / 轻量 | 本地 / 集群 | 云服务（SaaS） | 本地 / 云 | 本地 / 云 |
| **安装难度** | `pip install` | Docker/K8s | 注册即用 | Docker | Docker |
| **适用规模** | 小~中（<100万条） | 大~超大（十亿级） | 中~大 | 中~大 | 中~大 |
| **是否需要 GPU** | 否 | 推荐 | 不需要（云端） | 否 | 否 |
| **多模态** | 否（文本为主） | 是 | 是 | 是 | 是 |
| **成本** | 免费 | 免费（开源） | 按量付费 | 免费（开源） | 免费（开源） |
| **适用场景** | 个人开发、原型 | 企业级、海量数据 | 快速启动、免运维 | 高性能、过滤查询 | 知识图谱、混合搜索 |

**选择指南**：

```
开发/学习/小项目       → Chroma（零配置）
企业级/海量数据         → Milvus（开源高性能）
不想管运维              → Pinecone（SaaS，付钱就行）
需要复杂过滤 + 高性能   → Qdrant（Rust 实现，极快）
需要 GraphQL + 混合搜索 → Weaviate
```

### 12.5 多模态长期记忆

向量数据库不仅支持文本，还可以存图片、音频的向量。这意味着**用户上传的图片、语音也能被检索**：

```python
# 文本记忆
text_memory = "Alice 的宠物是一只橘猫"

# 图片记忆 — 用户上传了猫的照片
image_vector = multimodal_embedding.embed_image("cat_photo.jpg")
# → [0.34, -0.12, 0.89, ...]  ← 和文本 "橘猫" 的向量空间相近

# 检索时：用户问 "我的猫长什么样？"
# → 文本向量 "猫" 与图片向量匹配 → 返回猫的照片
```

### 12.6 实战：ChromaDB 长期记忆模块

#### 模块 1：记忆存储（写入）

```python
# ================================================================
# long_term_memory_store.py — 长期记忆存储模块
# ================================================================
import os
from datetime import datetime, timezone
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === 1. 初始化嵌入模型 ===
# 同一个模型同时用于"写入"和"查询"两个阶段
# 写入时：encode(记忆文本) → 向量
# 查询时：encode(查询文本) → 向量 → 与库中向量做相似度计算
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",    # BGE 中文小模型，CPU 可跑
    model_kwargs={"device": "cpu"},          # 用 CPU（没 GPU 也能跑）
    encode_kwargs={
        "normalize_embeddings": True,        # 归一化：向量模长=1，余弦相似度变内积
        "batch_size": 16,                    # 每批处理 16 条，CPU 不宜设太大
    },
)

# === 2. 初始化向量数据库 ===
# Chroma 文件型存储，数据持久化到 ./long_term_memory/ 目录
vector_store = Chroma(
    persist_directory="./long_term_memory",   # 持久化路径（重启不丢失）
    embedding_function=embeddings,            # 用什么模型做向量化
    collection_name="user_memories",          # 集合名（类似 MySQL 的 table）
)

# === 3. 记忆存储函数 ===
def store_memory(
    user_id: str,           # 用户唯一标识（区分不同用户的记忆）
    memory_text: str,       # 记忆的文本内容（会被向量化）
    memory_type: str,       # 记忆类型：preference / fact / event / preference
    importance: int = 1,    # 重要程度 1~5（检索时可以加权）
    source_conversation_id: str = "",  # 来源对话 ID（用于追溯）
) -> str:
    """
    将一条记忆写入长期记忆库。

    执行流程：
      1. 构造元数据（metadata）—— 这些字段不参与向量化，但用于过滤和追溯
      2. 调用 add_texts —— 内部自动：文本 → embeddings.embed_documents → 写入 Chroma
      3. 返回确认信息
    """
    # 构造元数据：这些字段不会变成向量，但检索时可以用于过滤
    metadata = {
        "user_id": user_id,                                   # 谁的记忆
        "memory_type": memory_type,                           # 什么类型
        "importance": importance,                             # 重要程度
        "source_conversation_id": source_conversation_id,     # 来自哪次对话
        "created_at": datetime.now(timezone.utc).isoformat(), # 创建时间
    }

    # 核心操作：文本 + 元数据 → 入向量库
    # add_texts 内部自动完成：
    #   texts[i] → embeddings.embed_documents(texts) → 向量 → Chroma 存储
    #   metadatas[i] → 与向量绑定存储（不影响向量位置，但检索时可过滤）
    vector_store.add_texts(
        texts=[memory_text],      # 要向量化的文本（可一次传多条，这里是单条）
        metadatas=[metadata],     # 与每条文本一一对应的元数据
    )

    return f"✅ 记忆已存储：[{memory_type}] {memory_text[:50]}...（用户：{user_id}，重要度：{importance}）"


# === 4. 批量记忆写入 —— 演示存储多条不同类别的记忆 ===
def seed_memories_for_alice():
    """为 Alice 写入初始记忆数据。"""
    memories = [
        # (user_id, 记忆内容, 记忆类型, 重要度, 来源对话ID)
        ("alice", "Alice 最喜欢的编程语言是 Python，其次是 Go",
         "preference", 4, "chat_001"),
        ("alice", "Alice 目前住在北京朝阳区，在国贸上班",
         "fact", 3, "chat_001"),
        ("alice", "Alice 上个月买了 M3 MacBook Pro，对性能很满意",
         "event", 2, "chat_002"),
        ("alice", "Alice 是素食主义者，不吃任何肉类和海鲜",
         "preference", 5, "chat_003"),           # ← 重要度最高！
        ("alice", "Alice 的猫叫 Luna，是一只 3 岁的橘猫",
         "fact", 3, "chat_001"),
        ("alice", "Alice 习惯在晚上 10 点后工作，白天开会",
         "preference", 3, "chat_004"),
        # Bob 的记忆 —— 不同用户，检索时会自动过滤
        ("bob", "Bob 是前端开发，主要用 React 和 TypeScript",
         "fact", 3, "chat_005"),
        ("bob", "Bob 住在上海浦东，在张江工作",
         "fact", 2, "chat_005"),
    ]

    for user_id, text, mtype, imp, src in memories:
        store_memory(
            user_id=user_id,
            memory_text=text,
            memory_type=mtype,
            importance=imp,
            source_conversation_id=src,
        )

    print(f"✅ 已为 Alice 和 Bob 写入 {len(memories)} 条初始记忆")


if __name__ == "__main__":
    seed_memories_for_alice()
```

#### 模块 2：记忆查询（检索）

```python
# ================================================================
# long_term_memory_query.py — 长期记忆查询模块
# ================================================================
from long_term_memory_store import vector_store, embeddings


def retrieve_memories(
    user_id: str,           # 要查询哪个用户的记忆（跨用户隔离）
    query_text: str,        # 查询文本（会被向量化后与记忆做相似度匹配）
    top_k: int = 2,         # 返回最相似的 top_k 条记忆
    memory_type: str = "",  # 可选过滤：只查某类记忆（preference/fact/event）
    min_importance: int = 0,  # 可选过滤：只查重要度 ≥ 此值的记忆
) -> list[dict]:
    """
    从长期记忆库中检索与当前查询最相关的记忆。

    执行流程：
      1. 构造查询向量 —— 和存储时用的是同一个 embedding 模型
      2. 在向量空间中找最相似的 top_k 条记录
      3. 返回每条记忆的文本 + 元数据

    参数详解：
      user_id:      必传。每个用户的记忆在向量空间中是隔离的（靠 metadata 过滤）
      query_text:   必传。当前用户的问题/上下文，会被向量化后去匹配历史记忆
      top_k:        返回几条。2 是经验值——太少可能漏，太多会稀释关键信息
      memory_type:  可选过滤。"preference" = 只查偏好, "" = 全部类型
      min_importance: 可选过滤。3 = 只查重要度 ≥3 的记忆
    """
    # === 构建过滤条件 ===
    # Chroma 的 where 条件类似 SQL 的 WHERE 子句，在检索时过滤 metadata
    where_filter = {"user_id": user_id}  # 必须按用户隔离

    if memory_type:
        where_filter["memory_type"] = memory_type  # 按类型过滤

    if min_importance > 0:
        # Chroma 支持比较运算符：$gte (≥), $lte (≤), $eq (=)
        where_filter["importance"] = {"$gte": min_importance}

    # === 执行检索 ===
    # similarity_search 内部流程：
    #   1. query_text → embeddings.embed_query(query_text) → 查询向量
    #   2. 与库中所有向量的余弦相似度计算（因为 normalize=True，等价于内积）
    #   3. 按相似度从高到低排序 → 取前 top_k 条
    #   4. 返回对应的 Document 对象（含 page_content 和 metadata）
    results = vector_store.similarity_search(
        query=query_text,         # 查询文本（会被自动向量化）
        k=top_k,                  # 返回最相似的几条
        filter=where_filter,      # metadata 过滤条件（用户隔离在此实现）
    )

    # === 格式化返回结果 ===
    memories = []
    for i, doc in enumerate(results):
        memories.append({
            "rank": i + 1,                                     # 排名（1 = 最相关）
            "score": "相似度最高",                               # similarity_search 不返回分数
            "content": doc.page_content,                        # 记忆文本
            "memory_type": doc.metadata.get("memory_type", ""), # 什么类型的记忆
            "importance": doc.metadata.get("importance", 0),    # 重要程度
            "created_at": doc.metadata.get("created_at", ""),   # 什么时候记的
            "source_conversation": doc.metadata.get(
                "source_conversation_id", ""
            ),                                                 # 来源对话
        })

    return memories


def retrieve_with_score(user_id: str, query_text: str, top_k: int = 2) -> list[dict]:
    """
    带相似度分数的检索。

    similarity_search_with_score 返回 (Document, score) 元组列表。
    score 是 L2 距离（越小越相似），因为 normalize=True 时可以近似理解为
    score = sqrt(2 - 2*cos_sim)，score≈0 表示几乎一样，score≈2 表示完全不同。
    """
    where_filter = {"user_id": user_id}
    results_with_scores = vector_store.similarity_search_with_score(
        query=query_text,
        k=top_k,
        filter=where_filter,
    )

    memories = []
    for i, (doc, score) in enumerate(results_with_scores):
        # 将 L2 距离转换为相似度百分比（近似，方便人类理解）
        # L2 ∈ [0, 2]（归一化情况下）→ 映射到 [100%, 0%]
        similarity_percent = max(0, (1 - score / 2) * 100)
        memories.append({
            "rank": i + 1,
            "score": round(score, 4),
            "similarity": f"{similarity_percent:.1f}%",
            "content": doc.page_content,
            "memory_type": doc.metadata.get("memory_type", ""),
            "importance": doc.metadata.get("importance", 0),
        })

    return memories


# === 演示检索 ===
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 长期记忆检索演示")
    print("=" * 60)

    # 场景1：Alice 问晚饭推荐 → 检索到素食偏好 + 居住区域
    print('\n📋 场景1：Alice 问 "推荐一个今晚吃饭的地方"')
    results = retrieve_memories(
        user_id="alice",
        query_text="推荐一个今晚吃饭的地方，什么类型都可以",
        top_k=2,
    )
    for r in results:
        print(f"  排名{r['rank']}: [{r['memory_type']}] {r['content'][:80]}...")
    # → 会检索到：
    #   排名1: [preference] Alice 是素食主义者，不吃任何肉类和海鲜
    #   排名2: [fact] Alice 目前住在北京朝阳区...

    # 场景2：跨用户隔离验证 → Bob 查 Alice 的记忆 → 返回空
    print('\n📋 场景2：Bob 问 "推荐一个今晚吃饭的地方"（Bob 的记忆库不同）')
    results = retrieve_memories(
        user_id="bob",
        query_text="推荐一个今晚吃饭的地方",
        top_k=2,
    )
    for r in results:
        print(f"  排名{r['rank']}: [{r['memory_type']}] {r['content'][:80]}...")
    # → Bob 没有饮食偏好相关记忆，返回的是 Bob 的其他记忆

    # 场景3：类型过滤 → 只看 Alice 的偏好类记忆
    print('\n📋 场景3：只看 Alice 的 preference 类型记忆')
    results = retrieve_memories(
        user_id="alice",
        query_text="工作习惯",
        memory_type="preference",  # ← 只查偏好类型
        top_k=2,
    )
    for r in results:
        print(f"  排名{r['rank']}: [{r['memory_type']}] {r['content'][:80]}...")

    # 场景4：带分数的检索
    print('\n📋 场景4（带分数）：查询 Alice 的工作相关信息')
    results = retrieve_with_score("alice", "工作工具和习惯", top_k=2)
    for r in results:
        print(f"  排名{r['rank']}: 相似度{r['similarity']} [{r['memory_type']}] {r['content'][:80]}...")
```

### 12.7 记忆检索 → 注入 Agent Prompt

```python
# ================================================================
# 将长期记忆注入 Agent 的 System Prompt
# ================================================================
def build_prompt_with_memory(user_id: str, user_input: str) -> str:
    """
    构建包含长期记忆的 System Prompt。

    流程：
      1. 根据用户当前输入，检索相关记忆
      2. 将记忆拼入 System Prompt
      3. LLM 基于记忆生成个性化回答
    """
    # 检索最相关的记忆
    memories = retrieve_memories(
        user_id=user_id,
        query_text=user_input,
        top_k=3,
    )

    # 格式化记忆为 Prompt 片段
    if memories:
        memory_lines = []
        for m in memories:
            memory_lines.append(
                f"- [{m['memory_type']}]（重要度 {m['importance']}/5）"
                f" {m['content']}"
            )
        memory_section = (
            "## 用户长期记忆（来自历史对话）\n"
            + "\n".join(memory_lines)
            + "\n\n请根据以上记忆提供个性化建议。如果记忆中没有相关信息，忽略即可。\n"
        )
    else:
        memory_section = ""

    return (
        "你是智能助手。当前用户信息如下：\n"
        + memory_section
        + f"\n用户问题：{user_input}"
    )


# === 使用示例 ===
user_input = "推荐今晚吃饭的地方"
personalized_prompt = build_prompt_with_memory("alice", user_input)
# → LLM 看到：
#   ## 用户长期记忆（来自历史对话）
#   - [preference]（重要度 5/5） Alice 是素食主义者，不吃任何肉类和海鲜
#   - [fact]（重要度 3/5） Alice 目前住在北京朝阳区，在国贸上班
#   - [preference]（重要度 3/5） Alice 习惯在晚上 10 点后工作，白天开会
#
#   请根据以上记忆提供个性化建议。
#   用户问题：推荐今晚吃饭的地方
# → LLM 回答："考虑到你是素食主义者，推荐国贸附近的「莲花素食」...
```

### 12.8 长期记忆的更新策略

| 策略 | 做法 | 适用 |
|---|---|---|
| **追加写入** | 新信息直接 `add_texts` | 事实型信息（居住地可以变，但历史事实保留） |
| **覆盖更新** | 先删旧的再写新的 | 配置型信息（用户改了偏好） |
| **合并更新** | 检索到旧记忆 → LLM 合并新旧 → 写入 | 需要增量更新的信息 |
| **TTR（过期）** | 写入时加 `ttl` 字段，到期自动清理 | 临时记忆（本周的出行计划） |
| **重要度过滤** | 只存 importance ≥ 3 的记忆 | 海量对话时控制记忆库大小 |

### 12.9 长期记忆架构总结

```
用户提问 ──→ 短期记忆（Checkpointer: 当前会话历史）
                  │
                  ├──→ 长期记忆检索（向量库: 用户所有历史记忆）
                  │         │
                  │         ▼
                  │    检索到相关记忆
                  │         │
                  ├─────────┤
                  │  合并   │
                  ▼         ▼
            个性化 System Prompt
                  │
                  ▼
            LLM 生成个性化回答
                  │
                  ▼
            关键信息提取 → 写入长期记忆库（为下次对话准备）
```

### 11.11 TypedDict vs Pydantic — State 定义的选择

| 维度 | `TypedDict` | `Pydantic BaseModel` |
|---|---|---|
| **类型检查** | 静态（mypy），运行时不校验 | 运行时自动校验 + 类型转换 |
| **默认值** | 不支持 | `Field(default=...)` |
| **数据校验** | 无 | `Field(ge=0, max_length=100)` 等 |
| **序列化** | 需手动 | `.model_dump()` / `.model_dump_json()` |
| **性能** | 极快（纯类型标注） | 有校验开销 |
| **复杂嵌套** | 可嵌套 TypedDict，但写起来啰嗦 | 自然嵌套 |
| **LangGraph 兼容** | 原生，最常用 | 支持，需 `model_config` |
| **适用** | 简单 State（如只有 messages） | 复杂 State（多字段 + 校验） |

```python
# TypedDict 方式 — 简单 State
class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]

# Pydantic 方式 — 复杂 State
from pydantic import BaseModel, Field

class ComplexState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    user_id: str = Field(default="anonymous")
    turn_count: Annotated[int, add] = Field(default=0, ge=0)
    tags: list[str] = Field(default_factory=list, max_length=20)

    class Config:
        arbitrary_types_allowed = True  # 允许 BaseMessage 等非标准类型
```

**选择指南**：

```
只有 messages 字段            → TypedDict
多个简单字段（无校验）          → TypedDict
需要默认值/校验/序列化          → Pydantic BaseModel
复杂嵌套结构                   → Pydantic BaseModel
追求极致性能                   → TypedDict
```

### 11.12 完整示例：生产级短期记忆 Agent

```python
# ================================================================
# memory_agent_full.py — 完整的短期记忆 Agent
# ================================================================
import tiktoken
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import trim_messages
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass

# ---- 1. 定义扩展 State ----
class MemoryAgentState(TypedDict):
    """含记忆管理的 Agent State。"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    turn_count: Annotated[int, add]       # 自增计数器
    trimmed_at_turn: int                   # 上次裁剪发生在第几轮
    important_facts: Annotated[list, add]  # 累积重要事实

# ---- 2. 带状态的 Tool ----
@dataclass
class MemoryToolContext:
    user_id: str
    turn_count: int
    important_facts: list[str]

@tool
def remember_fact(
    fact: str,
    runtime: ToolRuntime[MemoryAgentState],
) -> str:
    """记住一个重要事实。用户明确说「记住」时调用。"""
    user = runtime.state.get("user_id", "unknown")
    facts = runtime.state.get("important_facts", [])
    return f"已记住：{fact}（用户：{user}，累计 {len(facts)+1} 条）"

@tool
def recall_facts(runtime: ToolRuntime[MemoryAgentState]) -> str:
    """回顾已记住的所有事实。用户问「我告诉过你什么」时调用。"""
    facts = runtime.state.get("important_facts", [])
    return "\n".join(f"• {f}" for f in facts) if facts else "没有记住任何事实。"

# ---- 3. 上下文裁剪函数 ----
tokenizer = tiktoken.get_encoding("cl100k_base")

def auto_trim(state: MemoryAgentState, max_tokens: int = 4000) -> MemoryAgentState:
    """自动裁剪：超过 max_tokens 时保留最近的。"""
    current_tokens = sum(len(tokenizer.encode(m.content or "")) for m in state["messages"])
    if current_tokens <= max_tokens:
        return state

    state["messages"] = trim_messages(
        messages=state["messages"],
        max_tokens=max_tokens,
        token_counter=tokenizer,
        strategy="last",
        include_system=True,
        start_on="human",
    )
    return state

# ---- 4. 创建 Agent ----
async def create_memory_agent(user_id: str):
    """为每个用户创建记忆 Agent。"""
    # Postgres 持久化
    checkpointer = PostgresSaver.from_conn_string(
        "postgresql://user:pass@localhost:5432/memory_db"
    )
    await checkpointer.setup()

    tools = [get_weather, search_web, remember_fact, recall_facts]

    agent = create_react_agent(
        model=ChatOpenAI(model="deepseek-v4-pro", temperature=0.7),
        tools=tools,
        checkpointer=checkpointer,
        state_schema=MemoryAgentState,   # ← 用扩展 State
    )

    return agent, checkpointer

# ---- 5. 使用 ----
async def chat(agent, user_id: str, thread_id: str, message: str):
    """一次对话。"""
    config = {"configurable": {
        "thread_id": thread_id,
        "user_id": user_id,          # ← user_id 注入了 State
    }}

    # 调用前可插入裁剪
    state = agent.get_state(config)
    if state.values:
        trimmed = auto_trim(state.values)
        if trimmed["turn_count"] > state.values["turn_count"]:
            print(f"(对话在第 {trimmed['turn_count']} 轮被裁剪)")

    response = agent.invoke(
        {"messages": [HumanMessage(message)], "user_id": user_id},
        config=config,
    )
    return response["messages"][-1].content

# ---- 6. 多用户多会话演示 ----
async def demo():
    agent, cp = await create_memory_agent("system")

    # Alice 的对话
    print(await chat(agent, "alice", "chat_1", "我叫 Alice，记住我喜欢 Python"))
    print(await chat(agent, "alice", "chat_1", "我告诉过你什么？"))
    # → "• 你喜欢 Python"

    # Bob 的对话 — 隔离于 Alice
    print(await chat(agent, "bob", "chat_2", "我告诉过你什么？"))
    # → "没有记住任何事实。" ← Bob 看不到 Alice 的数据

    # Alice 开新会话 — 短期记忆不跨 thread_id
    print(await chat(agent, "alice", "chat_3", "我告诉过你什么？"))
    # → "没有记住任何事实。" ← 新 thread_id = 新 State

    await cp.close()
```

---

### 11.13 记忆管理总结

| 维度 | 实现 | 说明 |
|---|---|---|
| **隔离性** | `thread_id` | 不同 thread_id = 不同 State = 独立记忆，天然支持多租户 |
| **持久化** | `Checkpointer` | InMemorySaver（开发）/ PostgresSaver（生产）/ SqliteSaver（单机） |
| **效率** | `trim_messages` | 按 token/轮次自动裁剪，防止上下文溢出 |
| **可控性** | 自定义 State | 增加计数器、事实累积、摘要等字段，精确控制记忆内容 |
| **扩展性** | `ToolRuntime` | 工具无需 LLM 传上下文参数，直接从 State 读取 |
| **线程安全** | `thread_id` + Checkpointer | 每个 thread_id 有自己的 checkpoint 链，互不影响 |
| **可回溯** | Checkpointer 的版本链 | Postgres 存储所有历史 checkpoint，支持回退到任意版本 |
| **成本控制** | `trim_messages` + `token_counter` | 裁剪后减少 Prompt Token，直接降成本 |

---

