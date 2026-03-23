# 第三十步：Tool Calling Agent

## 1. 本节目标

这一节终于进入真正意义上的：

- 模型决定是否调用工具
- 程序负责真实执行工具
- 再把工具结果回传给模型

也就是说，这一节不再是上一节那种“程序先写死工具流程”，而是开始进入最小版 Tool Calling Agent。

对应文件：

- [src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)
- [src/main.py](/mnt/d/AIcodes/Agent/src/main.py)
- [src/prompts.py](/mnt/d/AIcodes/Agent/src/prompts.py)
- [prompts/tool_agent_decision_prompt.txt](/mnt/d/AIcodes/Agent/prompts/tool_agent_decision_prompt.txt)

## 2. 是什么

Tool Calling Agent 的核心不是“多了工具”，而是：

**模型开始参与工具决策。**

当前这节的最小循环是：

1. 用户提出问题
2. 模型看到可用工具 schema 和已有执行记录
3. 模型决定：
   - 是直接回答
   - 还是先调用某个工具
4. 程序真实执行工具
5. 程序把工具结果追加到执行记录
6. 模型再做下一步决策
7. 最终输出答案

这就是很多框架里 `create_tool_calling_agent` 背后的基本流程。

## 3. 为什么

你前面已经完成了三层能力：

- 工具函数
- 工具 schema
- 工具型学习助手

但这些能力里，真正决定“调不调工具”的还是程序。

Tool Calling Agent 再往前走一步的关键是：

**把“要不要用工具”这件事交给模型判断。**

这会让系统更接近真实应用，因为真实业务里用户问题变化很大：

- 有些问题直接回答就够了
- 有些问题必须先读文件
- 有些问题要先列文件，再决定读哪个

如果这些判断全写死在程序里，扩展性会很差。

## 4. 怎么做

### 第一步：给模型一个“决策提示词”

我们新增了：

- [prompts/tool_agent_decision_prompt.txt](/mnt/d/AIcodes/Agent/prompts/tool_agent_decision_prompt.txt)

这个提示词会把三类信息一起给模型：

- 用户问题
- 当前可用工具 schema
- 到目前为止的执行记录

然后要求模型只返回一个 JSON，对下一步动作做决策。

### 第二步：给 Agent 增加 JSON 决策能力

在 [src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py) 中新增了：

- `_invoke_json(...)`

它的作用是：

- 先调用模型
- 再把返回内容解析成 JSON 对象

这里你要特别注意一个工程点：

- 普通聊天输出是文本
- Tool Calling 决策输出必须是结构化对象

### 第三步：把工具执行记录组织成 scratchpad

我们新增了：

- `_build_tool_agent_scratchpad(steps)`

它负责把已经发生过的工具调用整理成：

- 调了什么工具
- 传了什么参数
- 得到了什么结果

这样下一轮模型决策时，就能“看到前面发生过什么”。

### 第四步：实现最小工具调用循环

最核心的方法是：

- `run_tool_calling_agent(question, max_steps=3)`

它的执行流程是：

1. 让模型做决策
2. 如果模型选择 `tool_call`
   - 程序执行工具
   - 保存工具结果
   - 继续下一轮
3. 如果模型选择 `final_answer`
   - 程序返回最终答案

这就是一个最小可运行的 Agent 循环。

### 第五步：给命令行增加 `/agent`

现在你可以直接测试：

```text
/agent 什么是输出解析器？
```

程序会把每一步工具调用打印出来。

## 5. 关键代码

### 代码 1：模型的动作输出格式

文件：[prompts/tool_agent_decision_prompt.txt](/mnt/d/AIcodes/Agent/prompts/tool_agent_decision_prompt.txt)

```json
{
  "reason": "一句话说明为什么这样决策",
  "action": "tool_call 或 final_answer",
  "tool_name": "当 action=tool_call 时填写工具名，否则填空字符串",
  "arguments": {},
  "answer": "当 action=final_answer 时填写最终回答，否则填空字符串"
}
```

这一段非常关键，因为它把“模型的下一步动作”变成了程序可执行的数据结构。

你要把它理解成：

- 这不是最终业务结果
- 这是 Agent 控制流的一部分

### 代码 2：把文本输出解析成 JSON

文件：[src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)

```python
def _invoke_json(self, user_input: str, temperature: float) -> dict[str, Any]:
    """调用模型并把返回结果解析成 JSON 对象。"""
    raw_text = self._invoke_text(user_input, temperature=temperature)
    return parse_json_output(raw_text)
```

这里的关键点是：

- 模型虽然底层还是输出文本
- 但在 Agent 层，我们把它解释成“动作对象”

这就是结构化控制流。

### 代码 3：工具执行记录 scratchpad

文件：[src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)

```python
def _build_tool_agent_scratchpad(self, steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "暂无执行记录。"
```

这个 scratchpad 的意义是：

- 让模型知道前面已经做了哪些动作
- 避免重复调用同一个工具
- 给模型提供观察结果

这和 ReAct 里的：

- Thought
- Action
- Observation

在工程上是同一类东西，只是当前这节做的是更结构化的版本。

### 代码 4：最小工具调用循环

文件：[src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)

```python
for step_number in range(1, max_steps + 1):
    scratchpad = self._build_tool_agent_scratchpad(steps)
    prompt = render_tool_agent_decision_prompt(...)
    decision = self._invoke_json(prompt, temperature=0.1)

    action = decision.get("action", "")

    if action == "final_answer":
        ...

    if action != "tool_call":
        raise ValueError(...)

    result = self.execute_tool(tool_name, arguments)
    steps.append(...)
```

这段代码是这一课最重要的部分。

按顺序理解：

1. 每轮先把“历史执行记录”整理好
2. 交给模型做下一步决策
3. 如果模型说“已经够了”，就返回答案
4. 如果模型说“还需要工具”，程序就去执行
5. 执行结果再进入下一轮

这就是 Agent loop。

### 代码 5：命令行测试入口

文件：[src/main.py](/mnt/d/AIcodes/Agent/src/main.py)

```python
if user_input.startswith("/agent "):
    question = user_input[7:].strip()
    ...
    result = agent.run_tool_calling_agent(question)
```

这段代码的价值是：

- 你可以直接看到每一步工具调用
- 不只是看到最终回答

这对调试 Agent 非常重要。

## 6. 常见错误

### 错误 1：以为 Tool Calling Agent 就是“模型自己真的执行了工具”

不是。

模型只是返回：

- 想调哪个工具
- 想传什么参数

真正执行工具的仍然是程序。

### 错误 2：不把工具执行结果回传给模型

如果不回传，模型就不知道工具做完后拿到了什么，也就没法继续决策。

### 错误 3：没有最大步数限制

这样模型可能会陷入循环调用。

所以当前代码里加了：

- `max_steps=3`

### 错误 4：把 Tool Calling 和普通聊天混成一层

Tool Calling Agent 不只是“问一次答一次”，它是：

- 决策
- 执行
- 观察
- 再决策

## 7. 常见面试问题

### 问题 1：Tool Calling Agent 和上一节工具型学习助手的区别是什么？

回答要点：

- 上一节是程序先写死工具流程
- 这一节是模型开始决定是否调用工具
- 程序只负责执行和控制循环

### 问题 2：为什么 `create_tool_calling_agent` 这一类框架抽象本质上还是一个循环？

回答要点：

- 因为 Agent 不是一次调用就结束
- 它要根据工具结果继续判断下一步
- 所以底层一定有“决策 -> 执行 -> 观察 -> 再决策”的循环

### 问题 3：为什么工具调用 Agent 比纯聊天更接近真实应用？

回答要点：

- 因为真实业务问题往往需要外部数据和外部动作
- 纯聊天只能依赖模型已有知识
- Tool Calling Agent 才能接数据库、文件系统、搜索、内部系统接口

## 8. 本节验收

运行：

```bash
python src/main.py
```

然后测试：

```text
/agent 什么是输出解析器？
/agent 请读取和工具 schema 相关的知识点并总结
/agent 现在有哪些知识点文件？
```

你应该观察到：

- 系统会打印每一步工具调用
- 某些问题会先调用 `list_notes`
- 某些问题会继续调用 `read_note`
- 最后才输出答案

如果你已经能清楚区分：

- 模型负责决策
- 程序负责执行
- 工具结果必须回传

这一节就完成了。
