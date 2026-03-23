# 第三十一步：ReAct Agent

## 1. 本节目标

这一节的目标不是再做一个“会调工具的 Agent”，而是要让你看清：

- ReAct 和普通 Tool Calling Agent 到底差在哪
- 为什么很多教程会单独讲 `Thought -> Action -> Observation`

本节新增：

- [src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py) 中的 `run_react_agent(...)`
- [prompts/react_agent_prompt.txt](/mnt/d/AIcodes/Agent/prompts/react_agent_prompt.txt)
- [src/main.py](/mnt/d/AIcodes/Agent/src/main.py) 中的 `/react`

## 2. 是什么

`ReAct` 是：

- `Reasoning + Acting`

最常见的理解方式是：

1. 先思考
2. 再行动
3. 看观察结果
4. 再继续思考

所以 ReAct 的核心不是“多一个工具调用”，而是：

**把中间推理过程显式暴露出来。**

当前这节的最小循环就是：

1. Thought
2. Action
3. Observation
4. 再进入下一轮 Thought

## 3. 为什么

上一节的 Tool Calling Agent 已经能做：

- 模型决定要不要调用工具
- 程序真实执行工具
- 再把结果回传给模型

但上一节更偏“结构化决策流”：

- `reason`
- `action`
- `tool_name`

ReAct 更强调的是：

- 模型当前为什么要这么做
- 拿到观察结果后它准备怎么继续

这在调试复杂 Agent 时很有价值，因为你能看到：

- 它为什么先去列文件
- 为什么接着去读文件
- 为什么觉得现在已经够回答了

## 4. 怎么做

### 第一步：新增 ReAct 专用 Prompt

我们新增了：

- [prompts/react_agent_prompt.txt](/mnt/d/AIcodes/Agent/prompts/react_agent_prompt.txt)

这个 Prompt 的重点不是简单要求模型“返回动作”，而是要求它同时返回：

- `thought`
- `action`
- `tool_name`
- `arguments`
- `answer`

这就把 ReAct 的“思考 + 行动”显式结构化了。

### 第二步：为 ReAct 单独组织 scratchpad

我们新增了：

- `_build_react_scratchpad(steps)`

它会把每一步整理成：

- Thought
- Action
- Arguments
- Observation

这样下一轮模型决策时，就能看到一条很清晰的推理链。

### 第三步：实现 `run_react_agent(...)`

这个方法和上一节的 `run_tool_calling_agent(...)` 很像，但表达方式不一样：

- Tool Calling Agent 更像“控制流 JSON”
- ReAct Agent 更像“显式思考链 + 工具调用”

### 第四步：增加 `/react`

现在你可以直接在命令行里看 ReAct 的每一步：

```text
/react 什么是输出解析器？
```

## 5. 关键代码

### 代码 1：ReAct 输出格式

文件：[prompts/react_agent_prompt.txt](/mnt/d/AIcodes/Agent/prompts/react_agent_prompt.txt)

```json
{
  "thought": "当前这一步你是怎么想的，要简洁具体",
  "action": "tool_call 或 final_answer",
  "tool_name": "当 action=tool_call 时填写工具名，否则填空字符串",
  "arguments": {},
  "answer": "当 action=final_answer 时填写最终回答，否则填空字符串"
}
```

这里最关键的字段是：

- `thought`

上一节也有 `reason`，但这节把它明确命名为 `thought`，是为了更贴近 ReAct 的经典表达。

### 代码 2：ReAct scratchpad

文件：[src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)

```python
def _build_react_scratchpad(self, steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "暂无思考与观察记录。"
```

这段代码的作用是：

- 把前面发生过的 Thought / Action / Observation 串起来

这和单纯保存“工具结果列表”不一样。

它保留的是：

- 为什么这么做
- 做了什么
- 看到了什么

### 代码 3：ReAct 循环

文件：[src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)

```python
for step_number in range(1, max_steps + 1):
    scratchpad = self._build_react_scratchpad(steps)
    prompt = render_react_agent_prompt(...)
    decision = self._invoke_json(prompt, temperature=0.1)

    thought = str(decision.get("thought", "")).strip()
    action = str(decision.get("action", "")).strip()
```

这里你要重点理解：

- ReAct 并不是“随便想一想”
- 它仍然是结构化控制流
- 只是我们把“想法”显式保留下来了

### 代码 4：Observation 的来源

文件：[src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)

```python
observation = self.execute_tool(tool_name, arguments)
steps.append(
    {
        "step": step_number,
        "thought": thought,
        "tool_name": tool_name,
        "arguments": arguments,
        "observation": observation,
    }
)
```

这段代码非常关键。

你要清楚：

- `Observation` 不是模型自己编出来的
- `Observation` 是程序执行工具后拿到的真实结果

也就是说：

- Thought：模型生成
- Action：模型决定
- Observation：程序执行后返回

### 代码 5：命令行输出

文件：[src/main.py](/mnt/d/AIcodes/Agent/src/main.py)

```python
if user_input.startswith("/react "):
    question = user_input[7:].strip()
    ...
    result = agent.run_react_agent(question)
```

然后打印：

```python
print(f"  Thought: {step['thought']}")
print(f"  Action: {step['tool_name']} {step['arguments']}")
```

这一步的意义是：

- 让你在终端里直接看到 Agent 的思考链

## 6. 常见错误

### 错误 1：把 ReAct 理解成“把模型思维链完整暴露出来”

当前这节不是在追求长篇隐式推理，而是在做：

- 工程上可控的短 thought

目的是帮助理解流程，而不是追求无穷展开的思维链。

### 错误 2：把 Thought 和 Observation 混在一起

这两者来源不同：

- Thought 来自模型
- Observation 来自工具执行结果

### 错误 3：以为 ReAct 一定比普通 Tool Calling 更高级

不一定。

很多场景里，结构化 Tool Calling 已经够用。

ReAct 的优势主要体现在：

- 多步工具调用
- 调试过程可解释
- 需要显式观察中间推理时

## 7. 常见面试问题

### 问题 1：ReAct 和 Tool Calling Agent 的核心差别是什么？

回答要点：

- 两者都可以调用工具
- Tool Calling Agent 更偏结构化决策流
- ReAct 更强调显式 Thought / Action / Observation 链

### 问题 2：为什么 ReAct 有助于调试复杂 Agent？

回答要点：

- 因为你能看到每一步为什么这样做
- 能区分是“思考错了”还是“工具结果不够”
- 对多步工具链尤其有帮助

### 问题 3：Observation 为什么必须来自程序执行结果？

回答要点：

- 因为如果 Observation 只是模型自己编的，就失去了工具调用的意义
- ReAct 的价值就在于把真实外部反馈接进推理链

## 8. 本节验收

运行：

```bash
python src/main.py
```

然后测试：

```text
/react 什么是输出解析器？
/react 现在有哪些知识点文件？
/react 请读取和工具 schema 相关的知识点并总结
```

你应该能看到：

- 每一步的 Thought
- 每一步的 Action
- 最终回答

如果你已经能清楚区分：

- Tool Calling Agent 偏结构化动作控制
- ReAct Agent 偏显式 Thought / Action / Observation

这一节就完成了。
