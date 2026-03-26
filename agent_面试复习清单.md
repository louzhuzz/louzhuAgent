# Agent 面试复习清单

目标：面向 `AI Agent / Java后端 / AI应用开发` 岗位。  
原则：`LLM 基础弱化`，把时间优先放在 `Agent 设计`、`RAG`、`工具调用`、`后端工程化`、`项目表达`。

## 1. 必会

- `Agent` 和 `Workflow` 的区别
- `ReAct` 的基本流程
- `Tool Calling / Function Calling` 的实现思路
- `MCP` 是什么，解决什么问题
- `RAG` 的完整链路：切分、召回、重排、生成、评估
- `Memory` 的类型：短期记忆、长期记忆、对话状态
- `Multi-Agent` 的协作方式：分工、路由、handoff、协商
- `Prompt` 设计的基本方法：角色、目标、约束、输出格式、few-shot
- `Agent` 失败处理：重试、超时、降级、人工接管
- `Agent` 评估：任务成功率、准确率、耗时、成本、稳定性

## 2. 后端工程化

- `Python` 后端：`FastAPI` / `Flask`
- `Java` 后端：`Spring Boot` / `MyBatis`
- 数据存储：`MySQL` / `PostgreSQL` / `Redis`
- 消息和异步：`MQ`、任务队列、定时任务
- 向量数据库：`Milvus` / `pgvector` / `FAISS` / `Chroma`
- 容器化：`Docker`
- 部署与运维：`Linux`、日志、监控、告警、CI/CD
- 接口设计：`RESTful API`、鉴权、限流、幂等、错误码
- 并发与性能：线程池、异步 IO、缓存、超时控制

## 3. 常见框架

- `LangChain`
- `LangGraph`
- `LlamaIndex`
- `AutoGen`
- `OpenAI Agents SDK`
- `Dify`
- `Coze`

建议掌握方式：
- 至少精通其中 `1-2` 个
- 其他框架知道定位、优势、缺点、适合场景即可

## 4. 项目必讲点

每个项目都要能讲清楚下面这些：

- 业务目标是什么
- 为什么要用 `Agent`，不用纯规则或纯 RAG
- 整体架构怎么分层
- 工具是怎么接入的
- 检索是怎么做的
- 记忆是怎么存的
- 如何控制输出格式
- 如何降低幻觉
- 如何做失败重试和降级
- 如何评估效果
- 上线后有没有实际效果

## 5. 简历写法

- 项目名要具体，不要写成“AI项目”
- 技术栈要分层写清楚
- 一段经历只写最关键的 3-5 个动作
- 尽量写量化结果
- 如果没有大规模指标，就写工程结果
- 例子：
  - 提升响应速度 `X%`
  - 降低人工介入 `X%`
  - 支持 `X` 类工具调用
  - 覆盖 `X` 个业务场景
  - 日请求量 `X`

## 6. 面试高频题

- 什么是 Agent
- Agent 和 Workflow 有什么区别
- ReAct 为什么有效
- 工具调用怎么做
- 如何让模型稳定输出 JSON
- RAG 为什么会幻觉
- 怎么提升召回质量
- 怎么设计记忆模块
- 如何做多轮对话状态管理
- 如何做多 Agent 协作
- 如何防止 Agent 乱调用工具
- 如何评估一个 Agent
- Agent 项目里遇到的最大问题是什么

## 7. `LLM` 基础只保留这些

- 知道 Transformer 是什么
- 知道 token、上下文窗口、KV Cache 的作用
- 知道指令微调和基础预训练的区别
- 知道 temperature、top-p、top-k 大概影响什么
- 知道幻觉是什么

不用深挖：
- 训练细节
- 数学推导
- 各种预训练损失函数
- 大量模型架构比较

## 8. 复习顺序

1. 先把 `Agent / RAG / Tool Calling / Memory` 讲明白
2. 再补 `FastAPI / Spring Boot / Docker / Redis / MySQL`
3. 再补 `LangChain / LangGraph / LlamaIndex / AutoGen / Dify`
4. 最后只补一点 `LLM 基础`
5. 每个项目都准备一版 `2 分钟介绍`

## 9. 2 分钟项目模板

- 这是什么场景
- 你做了什么
- 核心技术是什么
- 最难的问题是什么
- 结果是什么

## 10. 面试时优先说的关键词

- `Agent`
- `workflow`
- `tool calling`
- `RAG`
- `memory`
- `state`
- `orchestration`
- `eval`
- `fallback`
- `retry`
- `observability`
- `FastAPI` / `Spring Boot`
- `Docker`

