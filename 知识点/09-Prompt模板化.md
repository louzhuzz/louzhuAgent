# 第九步：Prompt 模板化

## 1. 本节目标

这一节要把项目从：

- Prompt 直接硬编码在 Python 里

升级成：

- Prompt 放到独立模板文件里
- 代码只负责给模板填参数

本节完成后，你会得到一个更清晰的结构：

- `system_prompt.txt` 管系统角色
- `study_plan_prompt.txt` 管学习计划任务
- `prompts.py` 负责读取和渲染模板

## 2. 是什么

Prompt 模板化，指的是把 Prompt 写成可复用的模板，而不是每次都在代码里拼长字符串。

比如我们现在把：

```python
f"请围绕主题“{topic}”生成一个 3 天学习计划..."
```

升级成：

- 模板文件里写 `{topic}`
- 代码里再把真实参数填进去

## 3. 为什么

如果 Prompt 一直写死在代码里，后面会越来越难维护：

- 改一个词也要改 Python 文件
- 很难比较多个 Prompt 版本
- 很难让 Prompt 和代码职责分离

模板化之后的好处是：

- Prompt 更容易维护
- 更适合后续做 Prompt 实验
- 更容易过渡到 LangChain 的 `PromptTemplate`

所以这一节本质上是在建立你对 Prompt 工程化的第一层理解。

## 4. 怎么做

### 第一步：把任务 Prompt 单独放到文件

我们新建：

- `prompts/study_plan_prompt.txt`

里面保留模板变量：

- `{topic}`

### 第二步：在 `prompts.py` 里统一读取 Prompt 文件

这样所有 Prompt 的读取逻辑就不会散落在各处。

### 第三步：写一个模板渲染函数

例如：

```python
def render_study_plan_prompt(topic: str) -> str:
    template = _load_prompt_file("study_plan_prompt.txt")
    return template.format(topic=topic)
```

这样代码调用时不需要再拼长字符串。

### 第四步：在 Agent 里改用模板函数

把原来 `create_study_plan()` 里的大段 f-string 删除，改成：

```python
user_input = render_study_plan_prompt(topic)
```

这就是 Prompt 模板化的落地动作。

## 5. 关键代码

### 代码 1：统一 Prompt 文件读取

文件：[src/prompts.py](/mnt/d/AIcodes/Agent/src/prompts.py)

```python
def _load_prompt_file(file_name: str) -> str:
    prompt_path = PROMPTS_DIR / file_name
    return prompt_path.read_text(encoding="utf-8").strip()
```

作用：

- 避免每个 Prompt 都各写一遍读取逻辑
- 让 Prompt 文件读取集中管理

这是一个典型的小型抽象。

### 代码 2：渲染学习计划模板

文件：[src/prompts.py](/mnt/d/AIcodes/Agent/src/prompts.py)

```python
def render_study_plan_prompt(topic: str) -> str:
    template = _load_prompt_file("study_plan_prompt.txt")
    return template.format(topic=topic)
```

这里的核心就是：

- 读取模板
- 把 `{topic}` 替换成真实值

你要理解：

**模板负责结构，代码负责填值。**

### 代码 3：模板文件本体

文件：[prompts/study_plan_prompt.txt](/mnt/d/AIcodes/Agent/prompts/study_plan_prompt.txt)

```text
请围绕主题“{topic}”生成一个 3 天学习计划，并严格只返回 JSON。
```

这表示 `topic` 是一个变量槽位。

模板文件的意义在于：

- Prompt 改动不用动业务逻辑
- Prompt 更像配置资产，而不是代码实现

### 代码 4：Agent 里使用模板

文件：[src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)

```python
def create_study_plan(self, topic: str) -> dict:
    user_input = render_study_plan_prompt(topic)
```

这一改动看起来很小，但很关键。

它意味着：

- Agent 不再关心 Prompt 的具体长文本
- Agent 只关心“我要一个学习计划 Prompt”

这就是职责分离。

## 6. 常见错误

### 错误 1：模板里用了花括号，但没考虑 `format()` 语法

比如 JSON 模板本身也有 `{}`。

这时候如果你直接用 `format()`，需要把字面量大括号写成：

```text
{{
}}
```

否则会报格式化错误。

### 错误 2：模板变量名和代码传参名不一致

比如模板写的是：

```text
{subject}
```

但代码传的是：

```python
format(topic=topic)
```

这样就会报错。

### 错误 3：每个 Prompt 都各写一套读取逻辑

后果：

- 重复代码变多
- 后面很难统一维护

## 7. 常见面试问题

### 问题 1：为什么要做 Prompt 模板化？

回答要点：

- Prompt 会频繁迭代
- 模板化有利于维护和版本管理
- 模板化是 Prompt 工程化的起点

深入追问：

- Prompt 应该算代码、配置，还是内容资产？

### 问题 2：Prompt 模板化和 LangChain `PromptTemplate` 有什么关系？

回答要点：

- 本质思想一样，都是“模板 + 变量替换”
- 现在是手写版，帮助理解底层
- 后面迁移到 LangChain 会更自然

深入追问：

- 手写模板和框架模板的边界在哪里？

### 问题 3：为什么不把所有 Prompt 都写在 Python 常量里？

回答要点：

- 不利于独立维护
- 不利于 A/B 对比
- 不利于非核心逻辑和业务逻辑分层

深入追问：

- 真实项目里 Prompt 应该怎么做版本管理？

## 8. 本节验收

你现在可以做两个验证：

1. 运行：

```text
/plan LangChain
```

确认功能仍然正常。

2. 打开 [prompts/study_plan_prompt.txt](/mnt/d/AIcodes/Agent/prompts/study_plan_prompt.txt)，把“3 天学习计划”改成“5 天学习计划”，再重新运行 `/plan LangChain`。

如果输出明显跟着模板变化，说明你已经理解了：

- Prompt 的行为确实可以通过模板文件控制
- 代码和 Prompt 已经解耦

## 9. 这一节和后面有什么关系

这一节是后面这些内容的前置基础：

- 多消息 Prompt 模板
- LangChain `PromptTemplate`
- Prompt 版本管理
- Prompt A/B 测试

所以它不是一个简单重构，而是你开始进入 Prompt Engineering 的工程化阶段。

