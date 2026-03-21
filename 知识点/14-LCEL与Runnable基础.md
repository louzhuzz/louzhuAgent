# 第十四步：LCEL 与 Runnable 基础

## 1. 本节目标

这一节的目标是让你第一次真正用上 LangChain 的链式表达方式。

本节不直接改主 Agent，而是新增一个独立示例：

- [src/lcel_intro.py](/mnt/d/AIcodes/Agent/src/lcel_intro.py)

你会看到一条最小的链：

`Prompt -> Model -> Parser`

## 2. 是什么

LCEL 是 LangChain Expression Language。

你现在可以先把它理解成：

**一种把多个 LangChain 组件用统一方式连接起来的表达方式。**

而 Runnable 则可以理解成：

**能被统一调用、统一组合的执行单元。**

例如这些都可以是 Runnable：

- Prompt
- Model
- Parser

所以当你写：

```python
chain = prompt | model | parser
```

本质上就是把多个 Runnable 串成一条链。

## 3. 为什么

你前面已经学了：

- Prompt
- 模型调用
- 输出解析器

如果继续手写，代码通常会变成：

1. 先拼 Prompt
2. 再调模型
3. 再解析输出

这个流程本身没错，但不够统一。

LCEL 的价值在于：

- 让执行链表达得更直接
- 让组件组合更自然
- 为后续并行链、路由链、检索链打基础

所以这一步是你从“会调用 LangChain”进入“会组合 LangChain” 的关键门槛。

## 4. 怎么做

### 第一步：用 `ChatPromptTemplate` 定义 Prompt

这一节我们第一次正式用到：

- `ChatPromptTemplate`

它比你之前手写模板更进一步，因为它本身就是 LangChain 体系里的 Runnable。

### 第二步：保留 `ChatOpenAI` 作为模型层

模型层仍然是：

- `ChatOpenAI`

这说明 LCEL 不是替换模型，而是定义组件之间怎么连接。

### 第三步：加入 `StrOutputParser`

我们这次不再用自己手动拿 `response.content`，而是用：

- `StrOutputParser`

它负责把模型结果解析成字符串。

### 第四步：用 `|` 把它们串起来

这是本节最核心的动作：

```python
chain = prompt | model | parser
```

你要把这句理解成：

`prompt.invoke(...) -> model.invoke(...) -> parser.invoke(...)`

只是 LCEL 帮你把它写成了统一链式表达。

## 5. 关键代码

### 代码 1：Prompt 作为 Runnable

文件：[src/lcel_intro.py](/mnt/d/AIcodes/Agent/src/lcel_intro.py)

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个严谨的 AI 学习教练，回答要简洁、结构化。"),
        ("human", "请围绕主题“{topic}”输出一个简洁总结 ..."),
    ]
)
```

这段代码说明：

- Prompt 不只是字符串模板
- 在 LangChain 里，它本身就是可以参与链式执行的组件

### 代码 2：模型层

文件：[src/lcel_intro.py](/mnt/d/AIcodes/Agent/src/lcel_intro.py)

```python
model = ChatOpenAI(
    model=settings.model,
    api_key=settings.api_key,
    base_url=settings.base_url,
    temperature=0.3,
)
```

这和你之前学的 LangChain 模型调用是一致的。

所以你要看到的是：

- 变化不在模型
- 变化在“怎么把组件串起来”

### 代码 3：解析器层

文件：[src/lcel_intro.py](/mnt/d/AIcodes/Agent/src/lcel_intro.py)

```python
parser = StrOutputParser()
```

它的作用很直接：

- 把模型输出转成字符串

这就是标准解析器的最小例子。

### 代码 4：链式组合

文件：[src/lcel_intro.py](/mnt/d/AIcodes/Agent/src/lcel_intro.py)

```python
chain = prompt | model | parser
result = chain.invoke({"topic": topic})
```

这两行就是本节最关键的代码。

第一行定义链。

第二行执行链。

这就是 LCEL 的核心体验。

## 6. 常见错误

### 错误 1：把 LCEL 理解成一种“新模型调用接口”

不准确。

LCEL 更准确的定位是：

- 组件组合语言

### 错误 2：只会写 `chain = ...`，但不知道每一段在做什么

后果：

- 一旦链复杂一点，就完全看不懂

所以你必须始终能把：

```python
prompt | model | parser
```

还原成：

- 先 Prompt
- 再 Model
- 再 Parser

### 错误 3：一开始就把主项目全改成 LCEL

这一步不建议。

当前阶段更适合先用独立示例理解 LCEL，再决定怎么迁移主项目。

## 7. 常见面试问题

### 问题 1：LCEL 是什么？

回答要点：

- 是 LangChain 的组件表达与组合方式
- 用于把 Prompt、Model、Parser、Retriever 等 Runnable 串起来
- 它的价值是统一执行链的描述方式

深入追问：

- 为什么说 LCEL 是“组合层”，而不是“能力层”？

### 问题 2：Runnable 是什么？

回答要点：

- 是 LangChain 里可被统一调用和组合的执行单元
- Prompt、Model、Parser 都可以是 Runnable
- 它们共享统一调用与组合语义

深入追问：

- 为什么统一抽象 Runnable 对工程扩展有价值？

### 问题 3：`prompt | model | parser` 背后的本质是什么？

回答要点：

- 是把多个执行节点按顺序串起来
- 前一个节点输出作为后一个节点输入
- 只是用更统一的写法描述流程

深入追问：

- 如果中间某一层失败，调试应该怎么看？

### 问题 4：为什么这一步先做独立示例，而不是直接重构主 Agent？

回答要点：

- 为了保持教学可对照性
- 避免一次改太多层
- 先建立 LCEL 心智模型，再做工程迁移

深入追问：

- 真正适合把主项目改成 LCEL 的时机是什么？

## 8. 本节验收

先运行：

```bash
python src/lcel_intro.py
```

或者：

```bash
python src/lcel_intro.py LangChain 输出解析器的作用
```

如果能稳定输出 3 条 `- ` 开头的总结，这一节就算跑通了。

## 9. 这一节和后面有什么关系

这一步是后面这些内容的前置：

- RunnableParallel
- 检索链
- 路由链
- 更复杂的 Agent 工作流

所以 LCEL 不是一个“语法糖小技巧”，而是你后面学习 LangChain 组合能力的地基。

