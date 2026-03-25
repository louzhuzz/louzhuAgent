# 第十一步：LangChain 入门与整体架构

## 1. 本节目标

这一节不直接重构主程序，而是先新增一个独立的 LangChain 入门示例。

本节目标是让你理解三件事：

1. LangChain 到底是什么
2. 它和你现在手写的 `openai` SDK 代码是什么关系
3. 为什么现在开始学它，时机才是合适的

本节完成后，你会多一个可运行脚本：

- [examples/langchain_intro.py](/mnt/d/AIcodes/Agent/examples/langchain_intro.py)

## 2. 是什么

LangChain 不是模型，也不是平台。

它更准确的定位是：

**围绕大模型应用开发的一层抽象与组件化框架。**

它主要帮你做这些事：

- 统一模型调用接口
- 统一 Prompt 组织方式
- 统一输出解析方式
- 统一工具、检索器、链、Agent 的组合方式

所以你可以把它理解成：

**它不是替代模型，而是在模型之上帮你组织应用逻辑。**

## 3. 为什么

你现在已经手写过这些东西：

- 模型调用
- 多轮记忆
- 结构化输出
- Prompt 模板化
- 多消息 Prompt 组织

这时候开始学 LangChain，收益最大。

因为你已经知道底层是什么，所以不会把 LangChain 当成“黑盒魔法”。

你现在看 LangChain，会更容易建立这种映射：

- 手写 `OpenAI(...)` 对应 `ChatOpenAI`
- 手写消息列表对应 `SystemMessage / HumanMessage`
- 手写 Prompt 模板对应后续的 `PromptTemplate`

## 4. 怎么做

### 第一步：安装 LangChain 最小依赖

这一节只需要这两个包：

- `langchain-core`
- `langchain-openai`

它们已经被加入 [requirements.txt](/mnt/d/AIcodes/Agent/requirements.txt)。

安装命令：

```bash
pip install -r requirements.txt
```

### 第二步：写一个最小 LangChain 示例

我们没有直接改 [src/main.py](/mnt/d/AIcodes/Agent/src/main.py)，而是新增一个独立文件：

- [examples/langchain_intro.py](/mnt/d/AIcodes/Agent/examples/langchain_intro.py)

这样做的好处是：

- 不会打断你当前的主项目
- 你能更清楚地对照两种写法

### 第三步：继续复用你已有的配置和系统 Prompt

这一节没有另起炉灶，而是继续复用：

- [src/config.py](/mnt/d/AIcodes/Agent/src/config.py)
- [prompts/system_prompt.txt](/mnt/d/AIcodes/Agent/prompts/system_prompt.txt)

这点很重要，因为它说明：

**LangChain 不是让你抛弃已有工程结构，而是可以叠加在现有结构上。**

### 第四步：用 `ChatOpenAI` 替代手写客户端调用

本节的核心动作是：

- 从 `openai.OpenAI(...)`
- 切到 `langchain_openai.ChatOpenAI(...)`

但底层仍然是：

- 同一个 `api_key`
- 同一个 `base_url`
- 同一个 `model`

这就是为什么我一直强调：

**先学底层，再学框架。**

## 5. 关键代码

### 代码 1：最小 LangChain 模型初始化

文件：[examples/langchain_intro.py](/mnt/d/AIcodes/Agent/examples/langchain_intro.py)

```python
model = ChatOpenAI(
    model=settings.model,
    api_key=settings.api_key,
    base_url=settings.base_url,
    temperature=0.7,
)
```

这一段和你之前手写版最值得对照：

- 之前是 `OpenAI(...)`
- 现在是 `ChatOpenAI(...)`

你要理解的点：

- 它不是换了平台
- 它是换了一层抽象

### 代码 2：LangChain 的消息对象

文件：[examples/langchain_intro.py](/mnt/d/AIcodes/Agent/examples/langchain_intro.py)

```python
messages = [
    SystemMessage(content=load_system_prompt()),
    HumanMessage(content=user_input),
]
```

这段代码对应你之前手写的：

```python
{"role": "system", "content": ...}
{"role": "user", "content": ...}
```

区别是：

- 手写版是字典
- LangChain 版是消息类对象

本质上它们表达的是同一个概念。

### 代码 3：调用模型

文件：[examples/langchain_intro.py](/mnt/d/AIcodes/Agent/examples/langchain_intro.py)

```python
response = model.invoke(messages)
```

这就是 LangChain 初学阶段最核心的调用方式。

你可以先把它理解成：

**把消息送进去，拿一个统一的响应对象出来。**

### 代码 4：为什么要做响应文本提取

文件：[examples/langchain_intro.py](/mnt/d/AIcodes/Agent/examples/langchain_intro.py)

```python
def _message_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    ...
```

我这里专门写了一个小函数，是为了让你建立一个工程意识：

- LangChain 返回的不一定永远是最简单的纯字符串
- 稍复杂场景下，响应内容可能是结构化块

所以程序端不要想当然地假设“永远是一个字符串”。

## 6. 手写版和 LangChain 版怎么对照

### 手写版

- 你自己构造消息字典
- 你自己调用 SDK
- 你自己处理返回结构

### LangChain 版

- 用标准消息对象
- 用统一模型接口
- 后续可以自然接 PromptTemplate、Parser、Retriever、Tools

所以这一节真正要学的，不是“换一行导入”，而是：

**开始理解 LangChain 的统一抽象。**

## 7. 常见错误

### 错误 1：一学 LangChain 就把整个项目全重写

后果：

- 很容易失去对底层的理解
- 出问题时不知道是框架问题还是业务问题

### 错误 2：以为用了 LangChain 就不用理解消息结构

后果：

- 工具调用、Prompt 设计、调试时还是会懵

### 错误 3：不知道 `base_url` 仍然可以复用

你当前接的是火山方舟兼容 OpenAI 接口。

所以这里依然可以通过：

- `api_key`
- `base_url`
- `model`

来初始化 LangChain 的 `ChatOpenAI`。

## 8. 常见面试问题

### 问题 1：LangChain 是什么？

回答要点：

- 它是大模型应用开发框架，不是模型本身
- 它提供统一抽象来组织 Prompt、Model、Parser、Retriever、Tools、Agent
- 它的价值是降低组合复杂度

深入追问：

- 为什么说它是“抽象层”而不是“能力来源”？

### 问题 2：为什么要先手写，再学 LangChain？

回答要点：

- 先理解底层，后理解抽象
- 否则容易会用但不懂
- 调试复杂问题时必须知道底层发生了什么

深入追问：

- 如果一开始就直接学 LangChain，最容易丢掉哪些理解？

### 问题 3：LangChain 和 OpenAI SDK 的关系是什么？

回答要点：

- OpenAI SDK 是底层调用工具
- LangChain 是更高层的应用组织框架
- LangChain 仍然会依赖底层模型提供商接口

深入追问：

- 为什么换成 LangChain 并不意味着“换模型平台”？

### 问题 4：为什么这节不直接重构主程序？

回答要点：

- 为了教学上的可对照性
- 先建立映射，再做替换
- 降低同时改太多东西的风险

深入追问：

- 真正适合把主程序迁到 LangChain 的时机是什么？

## 9. 本节验收

先安装新依赖：

```bash
pip install -r requirements.txt
```

然后运行：

```bash
python examples/langchain_intro.py
```

或者自定义输入：

```bash
python examples/langchain_intro.py LangChain 和普通 SDK 的区别是什么？
```

如果能正常输出结果，说明你已经成功跑通了第一个 LangChain 示例。

## 10. 这一节和后面有什么关系

这一节是后面这些内容的入口：

- 用 LangChain 重写模型调用
- Output Parser
- LCEL / Runnable
- Retriever
- Tool Calling Agent

所以这节课的关键不是“功能很多”，而是：

**你终于开始把手写 Agent 和 LangChain 抽象层对应起来。**

## 11. 参考方向

这一节的设计主要参考 LangChain 官方资料中关于：

- `ChatOpenAI`
- 消息对象
- 统一模型调用接口

你后面如果继续往下学，最值得看的官方方向是：

- LangChain Python `ChatOpenAI`
- LangChain Core 的 Runnable 抽象

