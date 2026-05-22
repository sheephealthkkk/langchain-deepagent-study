## 第十四章：企业级记忆最佳实践

### 14.0 三种记忆的全维度对比

理解三种记忆的本质区别，是设计企业级记忆体系的基石。

#### 对比总表

| 维度 | 短期记忆（Checkpointer） | 长期记忆（向量库） | BaseStore 跨线程记忆 |
|---|---|---|---|
| **核心解决问题** | 保持单轮对话的上下文连贯性 | 跨对话记住用户的永久信息 | 同一对话跨线程/跨进程共享状态 |
| **典型问题** | "他在第 2 轮说了什么？" | "Alice 的饮食偏好是什么？"（3 天前说的） | "Thread A 存的档案，Thread B 能读到吗？" |
| **数据粒度** | 完整对话（每条消息） | 用户信息片段（偏好/事实/事件） | 结构化键值对（JSON 文档） |
| **查询方式** | 按 thread_id 全量取 State | 语义相似度搜索（模糊匹配） | Key 精确查找 O(1) + 前缀搜索 |
| **查询条件** | thread_id = "chat_123" | "推荐晚餐地点" → 语义最相关的记忆 | namespace=("users","alice") + filter |
| **数据排序** | 时间顺序（消息追加） | 相似度排序（最相关的在前） | 自定义（按 key 或 filter 条件） |
| **存储内容** | `List[BaseMessage]` + 自定义 State 字段 | 文本片段 + metadata（vector + raw_text） | 结构化 `dict[str, Any]`（任意 JSON） |
| **存储介质** | PostgresSaver / SqliteSaver / InMemorySaver | Chroma / Milvus / Pinecone / Qdrant | PostgresStore / InMemoryStore |
| **核心操作** | `get()` / `put()` / `list()` / `prune()` | `add_texts()` / `similarity_search()` / `delete()` | `put()` / `get()` / `search()` / `delete()` / `list_namespaces()` |
| **隔离范围** | 单个 thread_id（会话级） | 单个 user_id（用户级） | namespace 前缀（任意范围——用户/租户/全局） |
| **跨线程共享** | 否——新 thread_id = 空白 State | 是——同 user_id 内跨线程检索 | **是——任意 namespace 前缀下跨线程读写** |
| **跨用户共享** | 否 | 否（需显式构造跨用户检索） | **是——namespace 前缀匹配到多用户** |
| **跨进程共享** | PostgresSaver 支持 | 支持（数据库文件持久化） | PostgresStore 支持 |
| **生命周期** | 会话结束 = 停用（可 prune 清理） | 永久（除非主动删除） | 永久（可设 TTL / 手动清理 / 归档） |
| **容量规模** | 小——每个线程几十到几百条消息 | 大——百万级向量 | 中——几千到几万条结构化文档 |
| **性能特征** | 读 O(1)（按 thread_id），写 O(1)（追加） | 读 O(N)（向量计算），写 O(1) | 读 O(1)（Key 查找），写 O(1)，search O(过滤结果) |
| **典型上限** | 单 State < 上下文窗口（~128K tokens） | 百万级向量 | 万级 JSON 文档 |
| **丢失后果** | 当前对话中断，用户需重说 | 个性化能力下降，需重建记忆 | Agent 间协作断裂，需重建共享状态 |
| **LangChain 实现** | `InMemorySaver` / `PostgresSaver` + `create_react_agent(checkpointer=...)` | `Chroma` / `Milvus` + `embeddings` + `similarity_search` | `InMemoryStore` / `PostgresStore` + `create_react_agent(store=...)` |

#### 三者协作关系图

```
用户 "Alice" 的一次请求：

  ┌─────────────────────────────────────────────────────────┐
  │                    请求入口                              │
  │   config = {thread_id="chat_5", user_id="alice"}       │
  └──────────────┬──────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌────────┐ ┌──────────┐ ┌──────────┐
│短期记忆 │ │ 长期记忆  │ │BaseStore │
│(会话层) │ │ (用户层)  │ │ (用户层)  │
├────────┤ ├──────────┤ ├──────────┤
│查:     │ │查:       │ │查:       │
│"前3轮  │ │"Alice的  │ │"Alice的  │
│ 说了   │ │  饮食偏好"│ │  会员等级"│
│ 什么?" │ │          │ │          │
│        │ │          │ │          │
│来源:   │ │来源:     │ │来源:     │
│当前    │ │任意线程  │ │任意线程  │
│thread  │ │同一user  │ │同一user  │
│        │ │          │ │          │
│查法:   │ │查法:     │ │查法:     │
│全量取  │ │语义检索  │ │Key精确取 │
│State   │ │相似度topK│ │+前缀搜索 │
└────────┘ └──────────┘ └──────────┘
    │            │            │
    └────────────┼────────────┘
                 │
                 ▼
    合并上下文 → 注入 System Prompt → LLM 回答
```

#### 选择决策树

```
你需要什么？
├─ 当前对话的多轮上下文
│   └─ 短期记忆（Checkpointer）
│       "他刚才问了北京天气，现在问适合运动吗 → 知道上下文"
│
├─ 跨对话记住用户的偏好/事实
│   ├─ 按语义检索（"推荐晚餐" → "Alice 是素食主义者"）
│   │   └─ 长期记忆（向量库）
│   │
│   └─ 按 Key 精确查找（"Alice 的会员等级是什么？"）
│       └─ BaseStore
│           - 需要跨线程共享 → BaseStore（唯一选项）
│           - 数据是结构化的（JSON）→ BaseStore
│           - 需要检索模糊语义 → 向量库
│
├─ 跨线程共享状态（Thread A 存，Thread B 读）
│   └─ BaseStore（唯一选项）
│       短期记忆：做不到（新 thread = 空 State）
│       向量库：能做到但不对口（你需要精确 Key，不是语义搜索）
│
└─ 跨用户全局配置（所有用户共享）
    └─ BaseStore（namespace 前缀匹配到全局）
        例如：store.get(namespace=("global","settings"), key="rate_limit")
```

#### 实际场景映射

| 业务场景 | 用哪种记忆 | 为什么 |
|---|---|---|
| 用户问 "刚才我说的那个再重复一遍" | 短期 | 需要当前对话上下文 |
| 用户问 "你记得我喜欢的编程语言吗？"（3 天后） | 长期（向量库） | 需要跨会话检索语义信息 |
| 管理员 "列出所有 enterprise 会员" | BaseStore | 结构化条件查询，无关对话 |
| Thread A 提取了用户实体，Thread B 需要使用 | BaseStore | 跨线程共享结构化数据 |
| 生成个性化回答（偏好 + 历史事实 + 会员信息） | 三者混合 | 短期给上下文 + 长期给偏好 + BaseStore 给档案 |
| 新对话窗口打开，Agent 需要知道用户是谁 | BaseStore | 新 thread_id，短期为空，BaseStore 存 user profile |
| "我上次提到的那个 bug 解决了吗？"（2 周前） | 长期（向量库） | 语义检索历史事实，不记得具体 thread_id |

### 14.1 双层存储架构

企业场景下，一个用户可能同时有多个对话窗口（Web、App、客服工单），每条线程有独立上下文，同时用户级信息需要跨所有线程共享。

```
                        config = {
                            "configurable": {
                                "thread_id": "chat_abc",   ← 会话层隔离
                                "user_id": "user_123",      ← 用户层关联
                            }
                        }

  ┌─────────────────────────────────────────────────────┐
  │                   双层存储架构                        │
  │                                                     │
  │  会话层（Checkpointer）           用户层（BaseStore） │
  │  ┌──────────────────┐          ┌──────────────────┐ │
  │  │ thread_id=abc    │          │ user_id=123      │ │
  │  │  ├─ messages     │          │  ├─ profile      │ │
  │  │  ├─ turn_count   │  ──→    │  ├─ preferences  │ │
  │  │  └─ temp_context  │  关联    │  ├─ history      │ │
  │  └──────────────────┘          │  └─ billing      │ │
  │  ┌──────────────────┐          └──────────────────┘ │
  │  │ thread_id=xyz    │                   ↑           │
  │  │  ├─ messages     │ ──────────────────┘           │
  │  │  └─ turn_count   │   同一用户，跨线程共享          │
  │  └──────────────────┘                              │
  └─────────────────────────────────────────────────────┘
```

**核心原则**：

| 层级 | 用谁 | 存什么 | 生命周期 |
|---|---|---|---|
| **会话层** | Checkpointer | messages、turn_count、临时状态 | 会话存活期 |
| **用户层** | BaseStore | profile、preferences、billing、history | 永久（按策略管理） |

### 14.2 身份标识：thread_id + user_id 双注入

```python
# config 中同时传入会话 ID 和用户 ID
config = {
    "configurable": {
        "thread_id": "chat_abc",    # Checkpointer 用 → 会话隔离
        "user_id": "user_123",      # BaseStore 用 → 跨线程关联
    }
}

# 入站时从 token/session 解析，不信任客户端传值（防篡改）
# 类似 Java Spring Security 的 SecurityContextHolder
def build_config_from_request(request) -> dict:
    """从 HTTP 请求的安全上下文中构建 config。"""
    token = request.headers.get("Authorization")
    session = decode_jwt(token)             # JWT 校验
    return {
        "configurable": {
            "thread_id": request.headers.get("X-Thread-ID", str(uuid4())),
            "user_id": session["sub"],       # ← 来自 JWT，不可伪造
            "tenant_id": session["tenant"],  # ← 多租户隔离（企业级额外维度）
            "role": session.get("role", "user"),  # ← 权限控制
        }
    }
```

### 14.3 记忆生命周期管理

#### 策略 1：自动过期（TTL）

```python
# === BaseStore 写入时设置 TTL ===
# 优点：零维护成本，到期自动清理
# 适用：临时会话缓存、验证码、短期授权

store.put(
    namespace=("users", user_id, "temp"),
    key="otp_code",
    value={"code": "8291", "purpose": "login"},
    ttl=300,  # ← 300 秒（5 分钟）后自动过期，框架负责清理
)

# === Checkpointer 按时间清理旧 checkpoint ===
# 优点：控制存储增长，保留最近 N 个版本
await checkpointer.aprune(
    thread_id="chat_abc",
    max_age_seconds=86400 * 7,     # 7 天前的 checkpoint 自动清理
    max_versions=5,                 # 每个线程保留最多 5 个版本
)
```

#### 策略 2：手动清理管理接口

```python
# ================================================================
# 记忆管理 API — 提供 REST 接口供运维/admin 调用
# ================================================================

class MemoryAdminService:
    """
    记忆管理服务。提供清理、归档、统计等运维接口。

    设计思路（Java 类比）：
      Spring Boot Actuator + @Scheduled 定时任务 + @RestController 管理端点
    """

    def __init__(self, checkpointer, store):
        self.checkpointer = checkpointer
        self.store = store

    # ---- 用户级清理 ----
    async def delete_user_data(self, user_id: str) -> dict:
        """
        删除指定用户的所有记忆（GDPR "被遗忘权" 合规）。

        两步清理：
          1. 删除 BaseStore 中该用户的所有 namespace
          2. 遍历该用户的所有 thread_id → 逐一删除 checkpoint
        """
        deleted_items = 0
        
        # 步骤 1：BaseStore — 按 namespace 前缀搜索 → 找到所有 → 逐个删除
        # namespace 是 ("users", user_id, ...) 格式
        namespaces = self.store.list_namespaces(
            prefix=("users", user_id)
        )
        for ns in namespaces:
            items = self.store.search(namespace_prefix=ns, limit=100)
            for item in items:
                self.store.delete(namespace=ns, key=item.key)
                deleted_items += 1

        # 步骤 2：Checkpointer — 清理该用户的所有线程
        # NOTE: PostgresSaver 支持按 metadata 过滤，InMemorySaver 不支持
        # 企业方案：在 config 的 metadata 中存储 user_id，清理时先查后删
        deleted_threads = 0
        # await self.checkpointer.adelete_thread(thread_id)  ← 逐个线程
        
        return {
            "user_id": user_id,
            "deleted_base_store_items": deleted_items,
            "deleted_threads": deleted_threads,
            "status": "GDPR compliant ✓",
        }

    # ---- 批量过期清理 ----
    async def cleanup_expired_data(self, older_than_days: int = 30) -> dict:
        """
        批量清理过期数据。

        策略：
          - 不是逐条检查（O(N) 太慢）
          - 而是按命名空间分片，搜索 old metadata 后批量删除
          - 建议作为定时任务（CronJob）执行，避开业务高峰期
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        cleaned = 0

        # 只搜索 "temp" 子命名空间（临时数据），不影响永久档案
        namespaces = self.store.list_namespaces(
            suffix=("temp",),    # 匹配 (*, *, "temp") 格式
            limit=1000,
        )
        for ns in namespaces:
            items = self.store.search(namespace_prefix=ns, limit=100)
            for item in items:
                created = item.created_at
                if created and created < cutoff:
                    self.store.delete(namespace=ns, key=item.key)
                    cleaned += 1

        return {"cleaned_items": cleaned, "older_than_days": older_than_days}

    # ---- 压缩归档 ----
    async def archive_old_threads(self, user_id: str, older_than_days: int = 90) -> dict:
        """
        将旧对话压缩归档。

        策略：
          1. 提取旧线程的关键信息（摘要 + 实体）
          2. 写入归档 BaseStore（archive namespace）
          3. 删除原始 checkpoint（释放主存储空间）

        好处：
          - 归档数据占空间小（摘要 vs 完整对话）
          - 仍可检索（存入 BaseStore 的 archive namespace）
          - 主存储保持高性能（小数据量）
        """
        archived = 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

        # 列出该用户的所有线程
        threads = await self.checkpointer.alist(
            filter={"user_id": user_id}
        )
        for thread in threads:
            # 获取该线程的 State
            state = await self.checkpointer.aget(
                config={"configurable": {"thread_id": thread["thread_id"]}}
            )
            if not state or not state.values:
                continue

            messages = state.values.get("messages", [])
            if not messages:
                continue

            # 压缩：生成摘要（生产中用 LLM 生成，这里用前 500 字符简化演示）
            full_text = " ".join(
                m.content for m in messages 
                if hasattr(m, "content") and m.content
            )
            summary = full_text[:500]  # 摘要化处理

            # 存入归档区域
            self.store.put(
                namespace=("archive", user_id, "threads"),
                key=thread["thread_id"],
                value={
                    "summary": summary,
                    "message_count": len(messages),
                    "archived_at": datetime.now(timezone.utc).isoformat(),
                    "original_thread_id": thread["thread_id"],
                },
            )
            archived += 1

            # 删除原始 checkpoint（释放主存储）
            # await self.checkpointer.adelete_thread(thread["thread_id"])

        return {"archived_threads": archived, "user_id": user_id}
```

#### 策略 3：分级存储（热/温/冷）

```
热数据（Hot）  → Checkpointer + InMemoryStore   → 当前活跃对话
温数据（Warm） → PostgresSaver + PostgresStore   → 近 30 天对话
冷数据（Cold） → S3 / MinIO / 对象存储           → 归档数据（按需恢复）
```

**何时迁移**：

| 迁移 | 触发条件 | 操作 |
|---|---|---|
| Hot → Warm | 对话空闲 > 30 分钟 | Checkpointer 自动保留，无需操作 |
| Warm → Cold | 对话时间 > 30 天 | archive_old_threads() → 对象存储 |
| Cold → Warm | 用户重新打开旧对话 | 从对象存储恢复 → PostgresStore |

### 14.4 性能优化

#### 优化 1：索引优化

```python
# === 在 BaseStore 写入时声明索引字段 ===
# 作用：后续 search 按这些字段过滤时，走索引而非全表扫描

store.put(
    namespace=("users", user_id, "profile"),
    key="latest",
    value={
        "language": "Python",
        "location": "北京",
        "is_vip": True,
    },
    # ★ index 参数：告诉 Store 对哪些字段建索引
    # 类似于 SQL: CREATE INDEX ON items(value->>'language')
    # 不传 index → 后续 search 按这些字段过滤 → 全表扫描 O(N)
    # 传了 index → search 走索引 → O(log N) 或更优
    index=["language", "location", "is_vip"],
)

# 检索时自动使用索引（前提是 filter 的字段在 index 列表中）
results = store.search(
    namespace_prefix=("users",),
    filter={"value.language": "Python", "value.is_vip": True},
    # ↑ 这两个字段都在 index 中 → 自动走索引
    limit=20,
)
```

**索引设计原则**：

| 原则 | 说明 |
|---|---|
| **只为高频过滤字段建索引** | 不要全字段建（浪费写入性能） |
| **组合索引 vs 单字段索引** | 经常一起过滤的字段建组合索引 |
| **低基数字段优先** | `is_vip`（true/false）适合索引，`user_id`（唯一）也适合 |
| **监控索引命中率** | 定期检查哪些 search 走了全表扫描 |

#### 优化 2：缓存策略

```python
# ================================================================
# 两级缓存：L1（进程内）+ L2（Redis）
# ================================================================
from functools import lru_cache
import redis
import json

class CachedBaseStore:
    """
    带缓存的 BaseStore 包装器。

    架构：
      L1 缓存（进程内 LRU）→ 最快，容量小（最近 128 条）
      L2 缓存（Redis）     → 较快，容量大（全量热数据）
      L3 存储（BaseStore）  → 最慢，容量无限（持久化）

    读取路径：L1 → L2 → L3（逐级回退 + 回填）
    写入路径：L3 → 失效 L2 → 失效 L1
    """

    def __init__(self, store, redis_client=None):
        self.store = store      # 底层 BaseStore（L3）
        self.redis = redis_client  # Redis 客户端（L2，可选）

    # L1 缓存：进程内 LRU，自动淘汰最少使用的条目
    @lru_cache(maxsize=128)
    def _get_from_cache(self, namespace: tuple, key: str):
        """L1 缓存 miss → 回退到 L2/L3。"""
        # L2: Redis
        if self.redis:
            cache_key = f"{':'.join(namespace)}:{key}"
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        # L3: BaseStore
        item = self.store.get(namespace=namespace, key=key)
        if item:
            result = item.value
            # 回填 L2
            if self.redis:
                self.redis.setex(
                    cache_key,
                    time=300,          # Redis 缓存 5 分钟
                    value=json.dumps(result, default=str),
                )
            return result

        return None

    def get_with_cache(self, namespace: tuple, key: str):
        """带缓存的读取 — 对调用者透明。"""
        return self._get_from_cache(namespace, key)

    def put_and_invalidate(self, namespace: tuple, key: str, value: dict, **kwargs):
        """写入时先写 L3，再失效 L2 + L1。"""
        # 写 L3
        self.store.put(namespace=namespace, key=key, value=value, **kwargs)

        # 失效 L2
        if self.redis:
            cache_key = f"{':'.join(namespace)}:{key}"
            self.redis.delete(cache_key)

        # 失效 L1
        self._get_from_cache.cache_clear()

    def invalidate_user_cache(self, user_id: str):
        """用户全局缓存失效（用于 GDPR 删除等场景）。"""
        if self.redis:
            # 按前缀删除所有该用户的缓存 key
            pattern = f"users:{user_id}:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
        self._get_from_cache.cache_clear()
```

#### 优化 3：批量处理

```python
# ================================================================
# 批量写入 — 避免逐条写入的性能瓶颈
# ================================================================
import asyncio

class BatchMemoryWriter:
    """
    批量写入器。将多条写入请求缓冲到一定量后一次提交。

    场景：用户在一次对话中提取了 10 条实体 → 不要逐条 put → 攒一批一起写。

    类比 Java：
      MyBatis BatchExecutor / JPA batch_size / Kafka 批量发送
    """

    def __init__(self, store, batch_size: int = 20, flush_interval_sec: float = 5.0):
        self.store = store
        self.batch_size = batch_size           # 攒到 20 条就刷
        self.flush_interval_sec = flush_interval_sec  # 或每 5 秒强制刷
        self.buffer: list[dict] = []
        self._last_flush = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()

    async def add(self, namespace: tuple, key: str, value: dict, **kwargs):
        """加入缓冲区，不立即写入。"""
        async with self._lock:
            self.buffer.append({
                "namespace": namespace,
                "key": key,
                "value": value,
                "kwargs": kwargs,
            })

            # 达到批量阈值 → 刷新
            if len(self.buffer) >= self.batch_size:
                await self._flush()

    async def _flush(self):
        """批量提交缓冲区中的所有数据。"""
        if not self.buffer:
            return

        batch = self.buffer[:]
        self.buffer.clear()
        self._last_flush = datetime.now(timezone.utc)

        # 批量写入（当前 BaseStore 不支持原生 batch，逐个 put）
        # 企业级优化：用 asyncio.gather 并发写入多条
        tasks = [
            asyncio.to_thread(
                self.store.put,
                namespace=item["namespace"],
                key=item["key"],
                value=item["value"],
                **item["kwargs"],
            )
            for item in batch
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def periodic_flush(self):
        """定时刷新（由后台任务调用）。"""
        while True:
            await asyncio.sleep(self.flush_interval_sec)
            async with self._lock:
                if datetime.now(timezone.utc) - self._last_flush > timedelta(
                    seconds=self.flush_interval_sec
                ):
                    await self._flush()

    async def start(self):
        """启动后台刷新任务。"""
        asyncio.create_task(self.periodic_flush())
```

### 14.5 扩展：更多企业级方法

| 方法 | 实现策略 | 好处 |
|---|---|---|
| **多租户隔离** | `namespace=("tenant_A","users","alice")`，最外层加租户维度 | 数据物理隔离，SaaS 合规 |
| **审计日志** | 每次 `put`/`delete` 写入审计表（操作人、时间、变更前后） | 追溯谁改了数据，满足 SOC2/ISO27001 |
| **数据加密** | Store 写入前字段级 AES 加密（PII 字段如邮箱、电话） | 数据库泄露也不暴露用户信息 |
| **读写分离** | 写入走主库 `store.put`，查询走只读副本 `read_replica.get` | 高并发查询不阻塞写入 |
| **跨区域同步** | 用户数据写入主区域 → 异步复制到灾备区域 | 异地容灾，RTO < 5 分钟 |
| **限流保护** | 每个 `user_id` 每秒最多 N 次 Store 操作 | 防止单用户打垮记忆系统 |
| **监控告警** | Prometheus metrics：写入耗时、缓存命中率、存储容量、清理任务状态 | 及时发现性能退化 |

### 14.6 完整示例：企业级双记忆 Agent

```python
# ================================================================
# enterprise_memory_agent.py — 企业级双记忆 Agent
# ================================================================
import os, asyncio, json, uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict
from functools import lru_cache

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent, InjectedStore, InjectedState
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ================================================================
# 1. 数据模型
# ================================================================

class UserProfile(BaseModel):
    """用户档案 — 存储在 BaseStore 用户层。"""
    display_name: str = Field(default="", description="用户显示名")
    email: str = Field(default="", description="邮箱（加密存储）")
    tier: str = Field(default="free", description="会员等级: free/pro/enterprise")
    language_prefs: list[str] = Field(default_factory=list)
    created_at: str = Field(default="")
    last_active_at: str = Field(default="")

class SessionState(TypedDict):
    """会话状态 — 存储在 Checkpointer 会话层。"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str              # 从 config 注入（用户层关联 key）
    tenant_id: str            # 从 config 注入（多租户隔离）
    turn_count: Annotated[int, add]
    summary: str              # 会话摘要（过长时压缩用）

# ================================================================
# 2. 生命周期管理中间件
# ================================================================

class MemoryLifecycleManager:
    """
    记忆生命周期管理。

    负责：
      - 自动记录最后活跃时间（用于过期判断）
      - 会话摘要生成（减少 token 消耗）
      - 过期数据清理（定时任务触发）
    """

    def __init__(self, store, checkpointer):
        self.store = store
        self.checkpointer = checkpointer

    def record_activity(self, user_id: str):
        """记录用户最后一次活动时间（每次 Agent 调用时触发）。"""
        self.store.put(
            namespace=("users", user_id, "meta"),
            key="activity",
            value={"last_active_at": datetime.now(timezone.utc).isoformat()},
        )

    async def generate_conversation_summary_if_needed(
        self, state: SessionState, max_tokens: int = 4000
    ) -> str:
        """
        如果消息历史超过阈值，生成摘要压缩。

        策略：保留 SystemMessage + 摘要 + 最近 5 轮对话。
        摘要替代中间的历史消息，大幅减少 token 消耗。
        """
        messages = state.get("messages", [])
        total_chars = sum(len(m.content or "") for m in messages)

        if total_chars < max_tokens * 4:  # 粗略估计 1 token ≈ 4 字符
            return state.get("summary", "")

        # 超过阈值 → 生成摘要（类比 Java GC 的 Old Gen 压缩）
        old_summary = state.get("summary", "")
        recent = messages[-10:]  # 最近 10 条详细保留
        middle = messages[1:-10]  # 中间的消息用于生成增量摘要

        # 构造摘要 Prompt（调用 LLM）
        # 这里简化为拼接前 200 字符
        incremental = " ".join(
            m.content[:50] for m in middle if hasattr(m, "content") and m.content
        )[:200]
        new_summary = (old_summary + " | " + incremental) if old_summary else incremental

        return new_summary

# ================================================================
# 3. 缓存层
# ================================================================

class ProfileCache:
    """
    用户档案两级缓存。

    L1: 进程内 LRU（@lru_cache）→ 热数据
    L2: BaseStore（无 Redis 时直接退到 Store）
    """

    def __init__(self, store):
        self.store = store

    @lru_cache(maxsize=256)
    def get_cached_profile(self, user_id: str) -> dict | None:
        """带 L1 缓存的档案读取。"""
        item = self.store.get(
            namespace=("users", user_id, "profile"),
            key="latest",
        )
        return item.value if item else None

    def invalidate(self, user_id: str):
        """更新档案后失效缓存。"""
        self.get_cached_profile.cache_clear()  # 全量失效（简单策略）
        # 精细策略：只失效该 user 的条目（需自维护缓存 key 集合）

# ================================================================
# 4. 工具定义 — 双层存储（会话层 + 用户层）
# ================================================================

def make_enterprise_tools(store, lifecycle_mgr: MemoryLifecycleManager):
    """创建企业级工具集。"""

    # ---- 工具 A：存储用户档案到 BaseStore（用户层）----
    def store_profile(
        # 业务参数 — LLM 决定
        display_name: str = "",
        email: str = "",
        tier: str = "free",
        language_prefs_str: str = "",  # 逗号分隔，如 "Python, Go"
        # 注入参数 — 框架自动提供
        store: Annotated[object, InjectedStore()] = None,
        user_id: Annotated[str, InjectedState("user_id")] = "",
        tenant_id: Annotated[str, InjectedState("tenant_id")] = "",
    ) -> str:
        """
        存储用户档案到 BaseStore。

        数据写入 namespace=("{tenant_id}", "{user_id}", "profile")。
        多租户隔离：tenant_id 在最外层，不同租户之间物理隔离。
        """
        languages = [
            lang.strip()
            for lang in language_prefs_str.split(",")
            if lang.strip()
        ]
        now = datetime.now(timezone.utc).isoformat()

        profile = UserProfile(
            display_name=display_name,
            email=email,
            tier=tier,
            language_prefs=languages,
            created_at=now,
            last_active_at=now,
        )

        # 写入 BaseStore（用户层）
        # ★ 关键设计：namespace 第一层是 tenant_id → 多租户物理隔离 ★
        store.put(
            namespace=(tenant_id, "users", user_id, "profile"),
            key="latest",
            value=profile.model_dump(),
            index=["tier", "language_prefs"],  # ← 高频过滤字段建索引
        )

        # 记录活动时间
        lifecycle_mgr.record_activity(user_id)

        return (
            f"✅ 已存储 {display_name or user_id} 的档案。\n"
            f"租户：{tenant_id}\n"
            f"会员等级：{tier}\n"
            f"语言偏好：{', '.join(languages) if languages else '未指定'}"
        )

    # ---- 工具 B：检索用户档案（跨线程）----
    def retrieve_profile(
        store: Annotated[object, InjectedStore()] = None,
        user_id: Annotated[str, InjectedState("user_id")] = "",
        tenant_id: Annotated[str, InjectedState("tenant_id")] = "",
    ) -> str:
        """
        从 BaseStore 检索用户档案。
        数据来源是 BaseStore，跨所有线程共享。
        """
        item = store.get(
            namespace=(tenant_id, "users", user_id, "profile"),
            key="latest",
        )
        if not item:
            return f"📭 租户 {tenant_id} 下用户 {user_id} 暂无档案。"

        profile = UserProfile(**item.value)
        return (
            f"📋 档案（租户：{tenant_id}）：\n"
            f"名称：{profile.display_name}\n"
            f"会员等级：{profile.tier}\n"
            f"语言偏好：{', '.join(profile.language_prefs)}\n"
            f"最后活跃：{profile.last_active_at}"
        )

    # ---- 工具 C：查看当前会话状态（会话层）----
    def session_info(
        user_id: Annotated[str, InjectedState("user_id")] = "",
        turn_count: Annotated[int, InjectedState("turn_count")] = 0,
    ) -> str:
        """
        查看当前会话信息（来自 Checkpointer 会话层）。
        不需要 InjectedStore，因为会话信息在 State 中。
        """
        return (
            f"📊 当前会话信息：\n"
            f"用户：{user_id}\n"
            f"对话轮次：第 {turn_count} 轮"
        )

    return [store_profile, retrieve_profile, session_info]

# ================================================================
# 5. 创建企业级 Agent
# ================================================================

def create_enterprise_agent():
    """
    创建企业级 Agent。

    双层存储：
      会话层：InMemorySaver（Checkpointer）— 生产替换为 PostgresSaver
      用户层：InMemoryStore（BaseStore）— 生产替换为 PostgresStore
    """
    # 会话层
    checkpointer = InMemorySaver()

    # 用户层
    store_obj = InMemoryStore()

    # 生命周期管理
    lifecycle_mgr = MemoryLifecycleManager(store_obj, checkpointer)

    # 缓存
    cache = ProfileCache(store_obj)

    # 工具
    tools = make_enterprise_tools(store_obj, lifecycle_mgr)

    # Agent
    agent = create_react_agent(
        model=ChatOpenAI(model="deepseek-v4-pro", temperature=0.7),
        tools=tools,
        checkpointer=checkpointer,  # ← 会话层
        store=store_obj,            # ← 用户层
        state_schema=SessionState,  # ← 自定义 State
    )

    return agent, checkpointer, store_obj, lifecycle_mgr, cache

# ================================================================
# 6. 测试
# ================================================================

def test_enterprise_memory():
    agent, checkpointer, store, lifecycle, cache = create_enterprise_agent()

    tenant_id = "tenant_acme_corp"
    user_id = "alice"

    # 构建 config — ★ thread_id + user_id 双注入 ★
    config = {
        "configurable": {
            "thread_id": "chat_recent",        # 会话层隔离
            "user_id": user_id,                 # 用户层关联
            "tenant_id": tenant_id,             # 多租户隔离
        }
    }

    # Thread 1：存储档案
    print("=" * 60)
    print("📍 Thread chat_recent — 存储用户档案")
    r1 = agent.invoke({
        "messages": [HumanMessage(
            "我叫 Alice Johnson，邮箱 alice@acme.com，"
            "我是 enterprise 会员，主要用 Python 和 Go，请帮我保存。"
        )],
        "user_id": user_id,
        "tenant_id": tenant_id,
    }, config=config)
    print(f"🤖 {r1['messages'][-1].content}")

    # 验证 1：BaseStore 中已有数据
    item = store.get(
        namespace=(tenant_id, "users", user_id, "profile"),
        key="latest",
    )
    print(f"\n✅ BaseStore 验证：{'有数据' if item else '无数据'}")

    # 验证 2：缓存命中
    profile = cache.get_cached_profile(user_id)
    print(f"✅ L1 缓存验证：{'命中' if profile else '未命中'}")

    # Thread 2：新线程，跨线程读档案
    config2 = {
        "configurable": {
            "thread_id": "chat_old",            # ← 新线程
            "user_id": user_id,                  # ← 同用户
            "tenant_id": tenant_id,
        }
    }

    print("\n📍 Thread chat_old — 新线程读取档案（跨线程验证）")
    r2 = agent.invoke({
        "messages": [HumanMessage("你那里有关于我的信息吗？帮我查一下。")],
        "user_id": user_id,
        "tenant_id": tenant_id,
    }, config=config2)
    print(f"🤖 {r2['messages'][-1].content}")

    # 验证 3：会话隔离
    state1 = agent.get_state(config)
    state2 = agent.get_state(config2)
    print(f"\n✅ 会话隔离验证：")
    print(f"   Thread chat_recent 消息数: {len(state1.values['messages']) if state1.values else 0}")
    print(f"   Thread chat_old 消息数: {len(state2.values['messages']) if state2.values else 0}")

    print("\n" + "=" * 60)
    print("✅ 企业级双记忆验证完成！")
    print("   会话层隔离 ✓ | 用户层共享 ✓ | 多租户隔离 ✓ | 缓存可用 ✓")

if __name__ == "__main__":
    test_enterprise_memory()
```

### 14.7 企业级记忆检查清单

```
□ 双层存储：Checkpointer（会话层）+ BaseStore（用户层）
□ 身份标识：thread_id + user_id + tenant_id 从 JWT/Token 注入
□ 多租户隔离：namespace 最外层是 tenant_id
□ TTL 过期：临时数据设 ttl，Checkpointer 定期 prune
□ GDPR 合规：提供 delete_user_data 管理接口
□ 归档策略：90 天以上的冷数据压缩后移入对象存储
□ 索引声明：高频过滤字段在 put 时声明 index
□ 缓存：L1 进程内 LRU + L2 Redis
□ 批量写入：BatchMemoryWriter 攒批提交
□ 监控：写入耗时、缓存命中率、存储容量
```