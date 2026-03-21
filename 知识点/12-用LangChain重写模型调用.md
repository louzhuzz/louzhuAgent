# 第十二步：用 LangChain 重写模型调用

## 1. 本节目标

这一节的目标非常明确：

**保留现有项目结构不变，只把模型调用层从 `openai` SDK 切换到 `ChatOpenAI`。**

也就是说：

- `main.py` 不重写
- Prompt 不重写
- 历史消息机制不重写
- 只替换真正发请求的那一层

## 2. 是什么

这一节做的事情，本质上是一次“调用层迁移”。

迁移前：

- 你手写 `OpenAI(...)`
- 你手写 `chat.completions.create(...)`

迁移后：

- 你使用 `ChatOpenAI(...)`
- 你使用 `model.invoke(...)`

但注意：

- 平台没变
- API Key 没变
- Base URL 没变
- Model 没变

变的是：

**你组织模型调用的抽象层。**

## 3. 为什么

这一课很重要，因为它会让你真正意识到：

LangChain 的价值不在于“多厉害”，而在于：

- 给模型调用统一抽象
- 让后续 PromptTemplate、Parser、Retriever、Tools 能自然接上

如果你不做这一步，后面学：

- Output Parser
- LCEL
- Retriever
- Agent

会始终停留在“看懂文档，但和自己项目接不上”的状态。

## 4. 怎么做

### 第一步：把模型客户端从 `OpenAI` 改成 `ChatOpenAI`

原来你的 `LearningAgent` 用的是：

```python
OpenAI(...)
```

现在改成：

```python
ChatOpenAI(...)
```

这一步说明：

- 你开始进入 LangChain 统一模型接口

### 第二步：保留现有消息构造逻辑

你之前已经把消息组织成：

- `system`
- `history`
- `user`

这一层不用推倒重来。

我们只是把这些字典消息，转换成 LangChain 的消息对象。

### 第三步：增加“消息转换层”

我们新增了：

- [src/langchain_helpers.py](/mnt/d/AIcodes/Agent/src/langchain_helpers.py)

其中最关键的函数是：

- `to_langchain_messages(...)`

它负责把你的消息字典转换成：

- `SystemMessage`
- `HumanMessage`
- `AIMessage`

### 第四步：统一抽一个 `_invoke_text(...)`

普通对话和结构化输出都要调用模型，所以这一节顺手把调用逻辑抽成了公共函数。

这一步非常重要，因为它意味着：

- 项目开始有“模型调用层”
- 后面改 Parser、加重试、加日志会更方便

## 5. 关键代码

### 代码 1：模型初始化迁移

文件：[src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)

```python
self.model = ChatOpenAI(
    model=settings.model,
    api_key=settings.api_key,
    base_url=settings.base_url,
    temperature=0.7,
)
```

你要对照以前的写法看：

- 以前是 `OpenAI(...)`
- 现在是 `ChatOpenAI(...)`

但配置参数本质没变。

这就是“抽象层变了，平台没变”。

### 代码 2：消息转换层

文件：[src/langchain_helpers.py](/mnt/d/AIcodes/Agent/src/langchain_helpers.py)

```python
def to_langchain_messages(messages: list[Message]) -> list[BaseMessage]: 
    converted: list[BaseMessage] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
```

这段代码的作用：

- 把你自己的消息结构
- 映射成 LangChain 标准消息对象

这是这一课最关键的“桥接层”。

### 代码 3：统一模型调用入口

文件：[src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)

```python
def _invoke_text(self, user_input: str, temperature: float) -> str:
    response = self.model.bind(temperature=temperature).invoke(
        to_langchain_messages(self._build_messages(user_input))
    )
    return content_to_text(response.content)
```

这段代码做了三件事：

1. 复用已有消息构造逻辑
2. 转成 LangChain 消息对象
3. 统一返回文本结果

这里你要理解一个很重要的工程点：

**调用模型的地方应该尽量收敛，而不是散落在多个函数里。**

### 代码 4：普通聊天和结构化输出复用同一调用层

文件：[src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)

```python
answer = self._invoke_text(user_input, temperature=0.7)
raw_answer = self._invoke_text(user_input, temperature=0.3)
```

这表示：

- 普通聊天
- 结构化输出

都已经走同一层 LangChain 模型调用逻辑，只是温度不同。

这就叫：

**复用调用层，区分任务参数。**

## 6. 常见错误

### 错误 1：切到 LangChain 后把项目所有层都一起推翻

后果：

- 不知道到底是哪一层出问题
- 丢掉和旧实现的对照关系

### 错误 2：不做消息转换层，直接在业务逻辑里到处手写 LangChain 消息对象

后果：

- 重复代码多
- 后面维护困难

### 错误 3：以为用了 LangChain，底层平台配置就失效了

其实不会。

你这里仍然在用：

- `api_key`
- `base_url`
- `model`

只是调用入口换成了 LangChain。

## 7. 常见面试问题

### 问题 1：用 LangChain 重写模型调用，核心变化是什么？

回答要点：

- 从底层 SDK 调用切到统一抽象层
- 不是换模型平台
- 是为后续组件化能力铺路

深入追问：

- 为什么说这是一种“抽象迁移”，不是“平台迁移”？

### 问题 2：为什么要保留原来的消息构造层？

回答要点：

- 消息组织逻辑本来就是业务层的一部分
- LangChain 替换的是调用接口，不必强行推翻上层结构
- 这样更利于渐进迁移

深入追问：

- 什么情况下你会连消息组织层也交给框架？

### 问题 3：为什么要单独做 `to_langchain_messages(...)`？

回答要点：

- 做桥接层
- 减少业务代码耦合
- 保持自有消息结构和框架消息结构的边界

深入追问：

- 如果以后框架换掉，这一层能帮你减少什么成本？

### 问题 4：为什么 `_invoke_text(...)` 要抽出来？

回答要点：

- 统一模型调用入口
- 便于后续加日志、重试、监控、Parser
- 减少重复代码

深入追问：

- 如果以后支持多个模型，调用层应该如何继续抽象？

## 8. 本节验收

你现在可以直接做两个验证：

1. 普通聊天：

```text
你好
```

2. 结构化任务：

```text
/plan LangChain
```

如果这两个功能都正常，说明主程序已经成功迁移到 LangChain 调用层。

## 9. 这一节和后面有什么关系

这一步完成后，后面的这些能力才更自然：

- Output Parser
- LCEL / Runnable
- Retriever
- Tool Calling Agent

因为从现在开始，你的主项目已经真正进入 LangChain 体系了。

