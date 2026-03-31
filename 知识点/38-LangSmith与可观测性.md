# 第 38 课：LangSmith 与可观测性

## 1. 本节目标

在上一课“本地日志系统”的基础上，再给主项目补上一层：

- LangSmith tracing

这样你的项目就同时具备：

- 本地 JSONL 日志
- 外部可视化调用链

这一课的目标不是替换本地日志，而是形成：

**本地日志 + LangSmith tracing 的双层可观测体系。**

## 2. 是什么

这一课新增的是：

- 可选开启的 LangSmith tracing

也就是说：

- 默认不开启时，项目仍然像之前一样正常运行
- 配好环境变量后，CLI 和网页端的请求链路会自动进入 LangSmith

这比“一上来强依赖外部平台”更适合当前项目阶段。

## 3. 为什么

上一课的本地日志已经能告诉你：

- 用户输入了什么
- 哪个工具调用了
- 哪次知识问答命中了哪些文件

但它不擅长解决：

- 一次请求里到底发生了哪些 LangChain 调用
- 哪一步模型调用最慢
- 某次请求从入口到输出的完整链路长什么样

所以这一课的价值在于：

**把“本地结构化事件”升级成“可视化调用轨迹”。**

## 4. 怎么做

### 第一步：增加 LangSmith 相关配置

文件：

- [.env.example](/mnt/d/AIcodes/Agent/.env.example)
- [src/config.py](/mnt/d/AIcodes/Agent/src/config.py)

新增：

```env
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=personal-learning-agent
```

这里最关键的是：

- `LANGSMITH_TRACING`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`

### 第二步：单独封装 LangSmith 观察器

文件：

- [src/langsmith_observer.py](/mnt/d/AIcodes/Agent/src/langsmith_observer.py)

不要把 LangSmith 相关代码直接散落在：

- `main.py`
- `web_app.py`
- `app_runtime.py`

更合理的做法是抽成：

- `LangSmithObserver`

这样后面继续扩展 tracing 时，主项目结构会更稳。

### 第三步：在统一运行层包 tracing 上下文

文件：

- [src/app_runtime.py](/mnt/d/AIcodes/Agent/src/app_runtime.py)

因为现在：

- CLI
- Web

都已经走同一个 `handle_user_input(...)`

所以最适合在这一层外面再包一层：

- `observer.request_context(...)`

### 第四步：CLI 和 Web 端都共用同一个 observer

文件：

- [src/main.py](/mnt/d/AIcodes/Agent/src/main.py)
- [src/web_app.py](/mnt/d/AIcodes/Agent/src/web_app.py)

这样你无论在：

- 终端里测试
- 浏览器里测试

都能进入同一个 LangSmith 项目里看 trace。

## 5. 关键代码

### 代码 1：根据配置创建观察器

文件：

- [src/langsmith_observer.py](/mnt/d/AIcodes/Agent/src/langsmith_observer.py)

```python
@classmethod
def from_settings(cls, settings: Settings) -> "LangSmithObserver":
    return cls(
        enabled=settings.langsmith_tracing,
        project_name=settings.langsmith_project,
    )
```

这里的作用是：

- 把配置读取和观察器初始化统一起来

这样主程序只需要写：

```python
observer = LangSmithObserver.from_settings(settings)
```

不需要自己到处读环境变量。

### 代码 2：不开 tracing 时自动降级

```python
if not self.enabled or ls is None:
    with nullcontext():
        yield
    return
```

这一段非常重要。

因为当前项目不能要求：

- 每个环境都必须装好 LangSmith
- 每次运行都必须连 LangSmith

所以这里的工程原则是：

**可观测能力要可选开启，不能影响主链路可用性。**

### 代码 3：真正建立 tracing 上下文

```python
with ls.tracing_context(
    enabled=True,
    project_name=self.project_name,
):
    yield
```

这里的含义是：

- 当前这次请求内部的 LangChain 调用
- 都会被记到指定项目里

### 代码 4：统一运行层包住单次请求

文件：

- [src/app_runtime.py](/mnt/d/AIcodes/Agent/src/app_runtime.py)

```python
context = (
    observer.request_context(session_id=session_id, user_input=cleaned)
    if observer is not None
    else nullcontext()
)
with context:
    return _handle_user_input_core(...)
```

这段代码的关键理解是：

- 不是给每个能力单独接 LangSmith
- 而是把“整个请求处理过程”包起来

这样更接近真实项目里的“单请求 tracing”思路。

### 代码 5：本地日志和 LangSmith 是并行关系

当前项目现在同时有：

- [src/runtime_logger.py](/mnt/d/AIcodes/Agent/src/runtime_logger.py)
- [src/langsmith_observer.py](/mnt/d/AIcodes/Agent/src/langsmith_observer.py)

这不是重复，而是分工：

- 本地日志：结构化事件、排错、成本低、离线可看
- LangSmith：可视化 trace、调用链分析、链路观察

## 6. 常见错误

### 错误 1：以为接了 LangSmith 就不需要本地日志

不对。

本地日志和 LangSmith 解决的问题不完全一样。

当前更合理的工程结构是：

- 本地日志保底
- LangSmith 增强观察能力

### 错误 2：把 tracing 代码写散在各个入口里

这样后面会非常乱。

当前项目更合理的结构是：

- 观察器单独封装
- 在统一运行层包 tracing

### 错误 3：默认强制依赖 LangSmith

这样会让：

- 本地开发更重
- 环境要求更高
- 演示时更容易出问题

所以这里必须保证：

- 不开 tracing 也能正常跑

## 7. 常见面试问题

### 问题 1：为什么有了日志还要接 LangSmith？

回答要点：

- 日志适合看事件
- LangSmith 更适合看调用链
- 二者结合后，既能做本地排错，也能做链路可视化分析

### 问题 2：为什么这里做成“可选开启”？

回答要点：

- tracing 属于增强能力，不应阻塞主链路
- 本地开发、演示环境、测试环境不一定都需要联网 tracing
- 设计成可选开启，更符合真实工程实践

### 问题 3：LangSmith 在当前项目里的价值是什么？

回答要点：

- 帮你观察每次请求的完整模型调用链
- 为后续评测、性能分析、问题回放提供基础
- 让项目更贴近真实 Agent 工程形态

## 8. 本节验收

先在 `.env` 里配置：

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的LangSmith API Key
LANGSMITH_PROJECT=personal-learning-agent
```

然后运行：

```bash
python src/main.py
```

或：

```bash
uvicorn web_app:app --app-dir src --reload
```

执行：

```text
/v1 我要学习通信原理
/qa 什么是输出解析器？
```

如果你能在 LangSmith 项目里看到这些请求对应的 trace，说明这节已经跑通。
