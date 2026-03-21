# 第十步：多消息 Prompt 设计

## 1. 本节目标

这一节不新增用户命令，而是重构消息组织方式。

目标是让你从代码层真正理解：

- `system` 消息负责什么
- `history` 消息负责什么
- `user` 消息负责什么

本节完成后，你的项目会从“把消息拼在一起能跑”升级成“消息分层清楚、职责明确”。

## 2. 是什么

多消息 Prompt 设计，指的是把一次模型请求拆成多类消息，而不是把所有内容混成一段长文本。

在当前项目里，一次请求的消息结构是：

1. `system`
2. `history`
3. `user`

也就是：

```text
系统角色说明 -> 历史对话 -> 当前用户任务
```

这就是现代聊天模型最常见的输入组织方式。

## 3. 为什么

如果你把这些信息全部混成一个大字符串，会有几个问题：

- 角色边界不清楚
- 历史消息不容易管理
- 后面接 LangChain `ChatPromptTemplate` 时不容易理解

而分层之后，好处很明显：

- `system` 负责长期规则
- `history` 负责上下文连续性
- `user` 负责本轮任务

所以多消息 Prompt 设计的本质是：

**不是让 Prompt 变长，而是让消息职责更清晰。**

## 4. 怎么做

### 第一步：给消息定义统一类型

我们先在 [src/prompts.py](/mnt/d/AIcodes/Agent/src/prompts.py) 里定义：

```python
Message: TypeAlias = dict[str, str]
```

这不是高级技巧，它的主要价值是：

- 让代码读起来更明确
- 让你知道项目里“消息对象”长什么样

### 第二步：把不同角色消息拆成独立构造函数

例如：

- `build_system_message(...)`
- `build_user_message(...)`

这样你不会在业务代码里反复手写：

```python
{"role": "system", "content": ...}
{"role": "user", "content": ...}
```

### 第三步：把历史消息复制出来再拼接

我们用了：

```python
clone_history_messages(history)
```

它的意义是：

- 构造请求消息时，不直接在原始历史列表上做不必要操作
- 保持消息拼装逻辑更干净

### 第四步：统一由一个函数生成完整消息列表

最终由：

```python
build_chat_messages(system_prompt, history, user_input)
```

来返回本次完整请求。

这样以后无论是普通聊天，还是结构化输出，都能走同一套消息组织逻辑。

## 5. 关键代码

### 代码 1：消息类型定义

文件：[src/prompts.py](/mnt/d/AIcodes/Agent/src/prompts.py)

```python
from typing import TypeAlias

Message: TypeAlias = dict[str, str]
```

这段代码的作用：

- 给“消息对象”一个统一名字
- 让后续函数签名更清楚

你要理解的点：

- 这主要是可读性增强
- 它不是必须的，但很适合教学和工程维护

### 代码 2：单条消息构造函数

文件：[src/prompts.py](/mnt/d/AIcodes/Agent/src/prompts.py)

```python
def build_system_message(system_prompt: str) -> Message:
    return {"role": "system", "content": system_prompt}


def build_user_message(user_input: str) -> Message:
    return {"role": "user", "content": user_input}
```

这段代码的作用：

- 明确不同角色消息的构造方式
- 减少业务代码里的字典重复拼写

### 代码 3：完整消息拼装

文件：[src/prompts.py](/mnt/d/AIcodes/Agent/src/prompts.py)

```python
def build_chat_messages(
    system_prompt: str,
    history: list[Message],
    user_input: str,
) -> list[Message]:
    return [
        build_system_message(system_prompt),
        *clone_history_messages(history),
        build_user_message(user_input),
    ]
```

这就是本节最关键的函数。

它明确表达了当前项目的消息顺序：

1. 系统提示词
2. 历史消息
3. 当前用户输入

### 代码 4：Agent 中复用消息构造层

文件：[src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)

```python
def _build_messages(self, user_input: str) -> list[Message]:
    return build_chat_messages(
        system_prompt=self.system_prompt,
        history=self.history,
        user_input=user_input,
    )
```

这一改动的意义在于：

- `agent.py` 不再负责拼消息细节
- `agent.py` 只负责“我要一组可发给模型的消息”

这就是职责分层。

## 6. 常见错误

### 错误 1：把 system、history、user 混成一个大字符串

后果：

- 很难管理上下文
- 很难看清每一层作用

### 错误 2：在多个地方重复写消息字典

后果：

- 重复代码增多
- 后面改角色结构容易漏改

### 错误 3：直接在原始 history 上随意改动

后果：

- 容易引入隐性副作用
- 调试时不容易看出问题

## 7. 常见面试问题

### 问题 1：为什么聊天模型要区分 system、user、assistant？

回答要点：

- `system` 负责全局规则和角色设定
- `user` 负责本轮需求
- `assistant` 负责历史回复上下文

深入追问：

- 如果没有 `system`，只靠 user 提示可以吗？

### 问题 2：为什么要把消息构造逻辑单独抽出来？

回答要点：

- 避免重复拼装
- 提高可读性
- 为后续 LangChain / ChatPromptTemplate 做准备

深入追问：

- 什么时候这个抽象会过度？

### 问题 3：history 为什么要作为独立层存在？

回答要点：

- history 是上下文状态
- system 是规则，user 是当前输入
- 把 history 独立出来更利于裁剪和总结

深入追问：

- 长对话里应该怎么裁剪 history？

### 问题 4：多消息 Prompt 和单字符串 Prompt 的本质区别是什么？

回答要点：

- 前者强调角色和结构
- 后者更像拼接文本
- 多消息方式更适合聊天模型和复杂 Agent

深入追问：

- LangChain `ChatPromptTemplate` 本质上帮你做了什么？

## 8. 本节验收

这节最简单的验收方式有两个：

1. 正常聊天一次，比如输入：

```text
你好
```

2. 再跑一次结构化任务：

```text
/plan RAG
```

如果两个功能都正常，说明这次消息层重构没有破坏行为。

然后你再回答下面这个问题：

**当前项目里，一次请求为什么要按 `system -> history -> user` 的顺序组织？**

如果你能自己解释清楚，这一节就算掌握了。

## 9. 这一节和后面有什么关系

这一节是在给后面这些内容打底：

- ChatPromptTemplate
- 多消息模板
- Tool Calling 消息组织
- LangChain 聊天链

所以这一步虽然看起来像“重构”，其实是在建立你对现代聊天模型输入结构的正确理解。

