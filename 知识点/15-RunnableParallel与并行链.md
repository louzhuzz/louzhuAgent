# 第十五步：RunnableParallel 与并行链

## 1. 本节目标

这一节要让你理解：

- LCEL 不只是串行链
- LangChain 也支持并行分支执行

本节新增一个独立示例：

- [examples/runnable_parallel_intro.py](/mnt/d/AIcodes/Agent/examples/runnable_parallel_intro.py)

它会对同一个主题并行生成两类结果：

- 核心概念总结
- 常见误区提醒

## 2. 是什么

`RunnableParallel` 可以理解成：

**把同一个输入并行送到多个 Runnable 分支里，再把结果聚合成一个字典。**

如果前一课的 LCEL 是：

`Prompt -> Model -> Parser`

那么这一课看到的是：

`同一个输入 -> 多条链并行执行 -> 结果汇总`

## 3. 为什么

很多时候我们面对同一个主题，不只需要一种输出。

比如学习一个概念时，你可能同时想得到：

- 核心总结
- 常见误区
- 例子
- 面试角度

如果都串行做，当然也能完成。

但从抽象上说，这其实是：

**同一个输入，产生多个互相独立的结果。**

这种场景就很适合并行链。

所以 RunnableParallel 的价值是：

- 让多分支结果组织更清楚
- 为后面做多视角分析、多路检索、结果汇总打基础

## 4. 怎么做

### 第一步：先定义一条最小串行链

这一课没有换模型，也没有换解析器。

我们仍然使用：

- `ChatPromptTemplate`
- `ChatOpenAI`
- `StrOutputParser`

先得到一条最小串行链：

```python
core_chain = prompt | model | parser
```

### 第二步：为不同分支准备不同 Prompt

虽然两条分支都处理同一个主题，但目标不同：

- 一个输出核心概念
- 一个输出常见误区

所以我们分别准备了两个 Prompt 模板：

- [prompts/parallel_core_prompt.txt](/mnt/d/AIcodes/Agent/prompts/parallel_core_prompt.txt)
- [prompts/parallel_pitfalls_prompt.txt](/mnt/d/AIcodes/Agent/prompts/parallel_pitfalls_prompt.txt)

### 第三步：用 `RunnableParallel` 把分支装起来

本节最核心的代码是：

```python
parallel_chain = RunnableParallel(
    core_summary=...,
    common_pitfalls=...,
)
```

这里的两个键：

- `core_summary`
- `common_pitfalls`

就是最终结果字典里的两个字段。

### 第四步：统一执行并拿到聚合结果

最后直接调用：

```python
result = parallel_chain.invoke({"topic": topic})
```

得到的就是一个字典对象，里面包含两个分支的结果。

## 5. 关键代码

### 代码 1：并行链定义

文件：[examples/runnable_parallel_intro.py](/mnt/d/AIcodes/Agent/examples/runnable_parallel_intro.py)

```python
parallel_chain = RunnableParallel(
    core_summary=lambda data: core_chain.invoke(
        {"user_input": render_parallel_core_prompt(data["topic"])}
    ),
    common_pitfalls=lambda data: pitfalls_chain.invoke(
        {"user_input": render_parallel_pitfalls_prompt(data["topic"])}
    ),
)
```

这段代码的作用：

- 同一个 `topic` 输入
- 同时送到两个不同分支
- 最终聚合成一个结果字典

这是本节最关键的代码。

### 代码 2：并行结果执行

文件：[examples/runnable_parallel_intro.py](/mnt/d/AIcodes/Agent/examples/runnable_parallel_intro.py)

```python
result = parallel_chain.invoke({"topic": topic})
```

你要把这句理解成：

- 输入只传一次
- 内部由并行链自动分发给多个分支

### 代码 3：结果为什么适合 JSON 打印

文件：[examples/runnable_parallel_intro.py](/mnt/d/AIcodes/Agent/examples/runnable_parallel_intro.py)

```python
print(json.dumps(result, ensure_ascii=False, indent=2))
```

因为 `RunnableParallel` 的结果天然就是字典结构，所以非常适合：

- 直接打印
- 直接传给前端
- 直接继续处理

## 6. 常见错误

### 错误 1：把并行链理解成“为了更快”

这不完整。

并行链首先是一种：

- 组织多分支结果的方式

性能只是其中一个潜在收益。

### 错误 2：分支之间其实互相依赖，却硬要并行

如果 B 分支必须依赖 A 分支结果，那就不应该并行。

### 错误 3：并行结果没有清晰字段名

如果输出字段命名混乱，后面很难消费结果。

所以像：

- `core_summary`
- `common_pitfalls`

这种命名是有价值的。

## 7. 常见面试问题

### 问题 1：RunnableParallel 是什么？

回答要点：

- 它用于把同一个输入并行送到多个 Runnable 分支
- 输出通常会聚合成一个字典
- 适合多视角生成、多路检索、多结果汇总场景

深入追问：

- 它和普通串行链最大的思维差别是什么？

### 问题 2：什么时候适合用并行链？

回答要点：

- 同一个输入需要生成多个互相独立的结果
- 各分支没有前后依赖关系
- 最终需要统一聚合输出

深入追问：

- 如果分支之间有依赖，应该怎么改成串行或路由链？

### 问题 3：为什么并行链输出通常适合结构化处理？

回答要点：

- 结果天然是多字段字典
- 更适合程序消费
- 更适合传给后续节点或前端

深入追问：

- 如果并行链后面还要接一个总结节点，应该怎么组织？

### 问题 4：并行链和多 Agent 有什么区别？

回答要点：

- 并行链是同一条工作流中的多个分支
- 多 Agent 更强调职责独立和更复杂交互
- 并行链通常是更轻量的组织方式

深入追问：

- 什么场景下不需要多 Agent，只要并行链就够了？

## 8. 本节验收

运行：

```bash
python examples/runnable_parallel_intro.py
```

或者：

```bash
python examples/runnable_parallel_intro.py LangChain
```

如果输出是一个包含这两个字段的 JSON：

- `core_summary`
- `common_pitfalls`

就说明这节已经跑通了。

## 9. 这一节和后面有什么关系

这一步是后面这些能力的基础：

- 多路检索
- 多视角分析
- 汇总链
- 更复杂的工作流

所以这一节会帮你建立一个重要意识：

**不是所有任务都只有一条线性链，有些任务天然就是多分支。**

