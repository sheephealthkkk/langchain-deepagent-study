## 第五章：完整配置速查

```bash
# ===== 必须设置 =====
LANGCHAIN_TRACING_V2=true                     # 启用追踪
LANGCHAIN_API_KEY=lsv2_pt_your_key             # API Key
LANGCHAIN_PROJECT=my-project                   # Project 名称

# ===== 可选设置 =====
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  # API 端点（SaaS 默认）
LANGCHAIN_TRACING_SAMPLING_RATE=0.1                # 采样率 0~1

# ===== 自托管（私有化部署）=====
LANGCHAIN_ENDPOINT=https://langsmith.internal.company.com  # 自托管地址
```

```python
# ===== 代码配置速查 =====
from langchain_core.runnables import RunnableConfig

# invoke 时带配置
chain.invoke(input, config=RunnableConfig(
    tags=["tag1", "tag2"],
    metadata={"key": "value"},
    run_name="my_custom_trace_name",
))

# 绑定到 Chain
chain.with_config(
    tags=["production"],
    metadata={"env": "prod"},
)

# 自定义追踪
from langsmith import traceable
@traceable(run_type="tool")
def my_tool(x): return x

# 手动上报 Feedback
from langsmith import Client
Client().create_feedback(run_id, key="user_feedback", score=1)

# 按条件查询
from langsmith import Client
client = Client()
runs = client.list_runs(
    project_name="my-project",
    filter='eq(tags, "production")',
)
```

---

## 第六章：核心要点总结

```
LangSmith 做了什么：
  你的 Chain (零改动) ──→ 自动上报 ──→ LangSmith 平台
                                             │
                              ┌──────────────┼──────────────┐
                              │              │              │
                           追踪日志       实时分析      监控调试
                              │              │              │
                        Project/Trace   延迟/Token/    每步输入输出
                        /Run/Feedback   成本趋势      可视可查
                              │              │              │
                              └──────────────┴──────────────┘
                                             │
                                    让 LLM 应用从黑盒
                                    变成透明玻璃盒

核心对象层级：
  Project > Trace > Run (可嵌套) > Feedback
             │        │
           Tags    Metadata

三行启用：
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=...
  LANGCHAIN_PROJECT=...

典型工作流：
  发现问题 → 找到 Trace → 展开 Run → 看每步数据 → 定位根因 → 修复 → 验证
```
