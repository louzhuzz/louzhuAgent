# Agent 边做边学

这个项目用于一边实现 Agent，一边系统学习 Agent 的核心知识。

我为你选定的第一个项目是：

**个人学习助理 Agent**

它的目标是：

- 接收你的问题
- 读取你的本地学习笔记
- 必要时调用搜索工具补充信息
- 给出带来源依据的回答
- 帮你拆解学习任务并生成下一步行动

为什么选它：

- 简单：先从单轮问答开始就能跑通
- 全面：后续能自然扩展到 Prompt、Tools、RAG、Memory、Workflow、Multi-Agent
- 实用：做完后你自己就能真的拿来学习和整理知识
- 可分步：每一步都能独立验证，不容易做着做着失控

建议的渐进式版本：

1. `v0`：纯聊天 Agent
2. `v1`：支持 Prompt 模板
3. `v2`：支持工具调用
4. `v3`：支持本地知识库 RAG
5. `v4`：支持记忆与学习记录
6. `v5`：支持多 Agent 分工

学习资料在 [`知识点`](./知识点) 目录。

## 目录说明

- `src/`：主项目核心代码，只保留当前学习助理真正运行所需的文件
- `examples/`：旧示例与教学脚本，包含 LangChain、RAG、Rerank 等独立演示
- `prompts/`：主项目与示例共用的 Prompt 模板
- `知识点/`：配套教学文档

## 当前已完成

- 已确定项目主题：个人学习助理 Agent
- 已搭建最小项目骨架
- 已拆出配置、Prompt、主程序、Agent 类
- 已支持基础多轮对话记忆
- 已支持结构化学习计划输出
- 已支持学习计划 Prompt 模板化
- 已完成多消息 Prompt 组织层重构
- 已新增独立的 LangChain 入门示例
- 已将主 Agent 的模型调用迁移到 LangChain
- 已新增独立的输出解析层
- 已新增独立的 LCEL/Runnable 入门示例
- 已新增 RunnableParallel 并行链示例
- 已新增 RAG 文档加载示例
- 已新增 RAG 文本切分示例
- 已新增 Embedding 与向量化示例
- 已新增内存版向量存储示例
- 已新增 Retriever 示例
- 已新增 RAG v1 检索增强生成示例
- 已新增 RAG v2 引用与可追溯回答示例
- 已新增 RAG 参数调优与误差分析示例
- 已新增真实 Embedding 与更真实向量检索示例
- 已新增第一个本地工具函数示例

## 运行方式

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python src/main.py
```

如果你要运行旧示例，请改用：

```bash
python examples/langchain_intro.py
python examples/rag_v1_intro.py
python examples/rerank_intro.py
```

你需要先在 `.env` 中填入可用的火山方舟配置。

当前默认接入方式：

- SDK：`openai` Python SDK
- 平台：火山方舟
- 默认 Base URL：`https://ark.cn-beijing.volces.com/api/v3`
- 模型字段：填写你可调用的 `Endpoint ID` 或模型名

当前命令行支持：

- `exit`：退出程序
- `/clear`：清空当前对话记忆
- `/plan 主题 | 当前基础 | 学习天数 | 学习目标`：输出项目化学习计划 JSON
- `/breakdown 目标 | 当前基础 | 可用天数 | 输出风格`：输出项目化任务拆解 JSON
- `/qa 问题`：运行主项目版知识点问答（少量候选文件 + embedding 缓存 + ChromaDB 持久化）
- `/v1 问题`：运行个人学习助理 Agent v1 统一入口，让系统自动判断该走哪种能力
- `/notes`：列出 `知识点/` 目录下的 Markdown 文件
- `/read 文件名`：读取指定知识点文件内容
- `/tools`：查看当前可用工具的 schema
- `/tool 工具名 JSON参数`：手动按 schema 执行工具
- `/study 问题`：自动选择一个知识点文件并基于该文件回答
- `/agent 问题`：让模型自己决定是否调用工具
- `/react 问题`：运行显式 Thought/Action/Observation 的 ReAct Agent

当前运行日志默认写到：

```text
data/logs/agent_runtime.jsonl
```

如果你想开启 LangSmith tracing，请在 `.env` 中补充：

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的LangSmith API Key
LANGSMITH_PROJECT=personal-learning-agent
```

开启后，CLI 和网页端的请求链路都会进入 LangSmith。

## 运行评测

如果你想运行主项目当前内置的最小评测体系：

```bash
python src/run_evals.py
```

默认样例文件在：

```text
evals/agent_eval_cases.json
```

结果会写到：

```text
data/evals/latest_eval_report.json
```

## 导出项目指标摘要

如果你想把“评测结果 + 运行日志”整理成更适合简历和项目包装的指标摘要：

```bash
python src/export_project_metrics.py
```

结果会写到：

```text
data/metrics/latest_project_metrics.json
data/metrics/latest_project_metrics.md
```

## 网页对话端

如果你想启动网页对话端，请先安装依赖：

```bash
pip install -r requirements.txt
```

然后运行：

```bash
uvicorn web_app:app --app-dir src --reload
```

浏览器打开：

```text
http://127.0.0.1:8000
```

网页端和命令行端共用同一套主项目运行逻辑，支持：

- 普通聊天
- `/qa`
- `/plan`
- `/breakdown`
- `/v1`
- `/agent`
- `/react`
