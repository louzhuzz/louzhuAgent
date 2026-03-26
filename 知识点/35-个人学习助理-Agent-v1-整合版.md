# 第 35 课：个人学习助理 Agent v1（整合版）

## 1. 本节目标

把前面已经项目化的三个主能力：

- 学习计划生成器
- 知识点问答系统
- 任务拆解器

再加上：

- Tool Calling Agent
- 普通对话

统一收敛到一个主项目入口里，形成真正的 **个人学习助理 Agent v1**。

这一步完成后，你的主项目不再只是“很多独立命令的集合”，而是开始具备：

- 统一入口
- 统一意图判断
- 统一能力路由

## 2. 是什么

Agent v1 的核心不是“功能更多”，而是：

**先判断用户现在想解决哪一类问题，再把请求路由到最合适的能力模块。**

也就是说，`Agent v1` 干的是“统筹和分发”：

- 想做学习计划，路由到 `study_plan.py`
- 想问知识点，路由到 `knowledge_qa.py`
- 想拆任务，路由到 `task_breakdown.py`
- 想查文件或走多步工具，路由到 Tool Calling Agent
- 其他情况，回落到普通聊天

## 3. 为什么

前面的项目化模块已经能独立工作，但还像这样：

- `/plan` 做计划
- `/qa` 做问答
- `/breakdown` 做拆解

这对开发者很清楚，但对最终用户并不自然。

真正的产品形态通常不是“你先知道该敲哪个命令”，而是：

**你直接描述需求，系统自己判断该走哪条能力链。**

所以这一课的价值在于：

1. 让主项目从“能力集合”走向“统一入口”
2. 让你开始理解真正 Agent 产品里的“意图路由层”
3. 为后面接 LangChain 1.x 的更正式 Agent 抽象做准备

## 4. 怎么做

这一课的实现分三层：

### 第一步：单独抽一个路由服务层

新增文件：

- [src/agent_v1.py](/mnt/d/AIcodes/Agent/src/agent_v1.py)

不要把统一路由逻辑直接塞回 [src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)。

原因很简单：

- `agent.py` 已经承担了模型调用、历史记录、工具能力
- 如果继续把“统一入口决策”塞进去，后面会越来越难读

所以这里要新增一个专门的 `AgentV1Service`。

### 第二步：先做“意图判断”

新增 Prompt：

- [prompts/agent_v1_router_prompt.txt](/mnt/d/AIcodes/Agent/prompts/agent_v1_router_prompt.txt)

这个 Prompt 的任务不是回答问题，而是做一个小型分类器：

- `study_plan`
- `knowledge_qa`
- `task_breakdown`
- `tool_agent`
- `general_chat`

### 第三步：按意图路由到已有主项目能力

路由目标如下：

- `study_plan` -> `create_study_plan(...)`
- `knowledge_qa` -> `answer_knowledge_question(...)`
- `task_breakdown` -> `create_task_breakdown(...)`
- `tool_agent` -> `run_tool_calling_agent(...)`
- `general_chat` -> `reply(...)`

### 第四步：在命令行里暴露统一入口

在 [src/main.py](/mnt/d/AIcodes/Agent/src/main.py) 增加：

- `/v1 问题`

这样你就能直接说：

```text
/v1 给我一个 5 天 LangChain 学习计划
/v1 什么是输出解析器？
/v1 把做一个个人学习助理拆成 14 天执行步骤
```

## 5. 关键代码

### 代码 1：Agent v1 的路由决策对象

文件：

- [src/agent_v1.py](/mnt/d/AIcodes/Agent/src/agent_v1.py)

```python
@dataclass
class AgentV1Decision:
    """描述 Agent v1 在统一入口中做出的路由决策。"""

    intent: str
    reason: str
    rewritten_input: str
```

这里不是随便返回一个字典，而是先把“路由结果”抽成一个明确对象。

三个字段分别表示：

- `intent`：最终决定走哪个能力模块
- `reason`：为什么这样路由
- `rewritten_input`：给目标模块使用的规范化输入

为什么要有 `rewritten_input`？

因为统一入口接收的是自然语言，但下面很多模块希望拿到更稳定的输入格式。

例如：

- 用户说：`给我一个 5 天 LangChain 学习计划`
- 路由层可以把它规范成：
  `LangChain | 零基础 | 5 | 做出一个可运行 Demo`

这样下面的 `parse_study_plan_request(...)` 就更容易处理。

### 代码 2：先让模型做路由，而不是直接回答

```python
def _make_decision(self, user_input: str) -> AgentV1Decision:
    """先让模型判断当前问题最适合路由到哪个主项目能力。"""
    prompt = self.render_router_prompt(user_input)
    result = self.invoke_json(prompt, 0.1)
```

这段代码是整个 `Agent v1` 的核心起点。

注意这里不是：

- `invoke_text(...)`

而是：

- `invoke_json(...)`

原因是路由决策必须稳定。

如果你让模型自由文本回答：

- “我觉得这更像学习计划”
- “也可能是任务拆解”

那程序就不好稳定消费。

所以这里强制模型返回结构化 JSON，程序才能明确知道该怎么分发。

### 代码 3：对路由结果做程序校验

```python
if intent not in self.allowed_intents:
    raise ValueError(f"Agent v1 收到了未知意图：{intent}")
if not rewritten_input:
    raise ValueError("Agent v1 路由结果缺少 rewritten_input。")
if not reason:
    raise ValueError("Agent v1 路由结果缺少 reason。")
```

这一步非常关键。

很多初学者写 Agent 时会犯一个错误：

- 只相信模型输出
- 不做程序层校验

这样一旦模型输出：

- 拼错意图名
- 漏字段
- 返回空字符串

整个主项目路由就会直接失控。

所以这里你要记住一个工程原则：

**模型可以参与决策，但决策结果必须经过程序校验。**

### 代码 4：真正的统一路由

```python
if decision.intent == "study_plan":
    request = parse_study_plan_request(decision.rewritten_input)
    result = self.create_study_plan(request)
    ...

if decision.intent == "knowledge_qa":
    result = self.answer_knowledge_question(
        KnowledgeQARequest(question=decision.rewritten_input)
    )
    ...
```

这段逻辑的本质是：

**统一入口不自己实现所有能力，而是把请求分发给已有服务层。**

这正是项目化的重要特征：

- 路由层负责判断和分发
- 各业务服务层负责真正执行

这样后面继续升级时，你才能做到：

- 替换问答系统，不影响学习计划模块
- 替换工具 Agent，不影响任务拆解模块

### 代码 5：在 `LearningAgent` 里接入 Agent v1

你会在 [src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py) 里看到类似这样的初始化：

```python
self.agent_v1_service = AgentV1Service(
    invoke_json=self._invoke_json,
    render_router_prompt=render_agent_v1_router_prompt,
    create_study_plan=self.create_study_plan,
    answer_knowledge_question=self.answer_knowledge_question,
    create_task_breakdown=self.create_task_breakdown,
    run_tool_calling_agent=self.run_tool_calling_agent,
    reply=self.reply,
)
```

这里要重点理解：

- `AgentV1Service` 不直接依赖底层模型 SDK
- 它依赖的是上层已经整理好的能力函数

这叫：

**面向能力编排，而不是面向底层实现编排。**

## 6. 常见错误

### 错误 1：把所有判断逻辑都写回 `agent.py`

短期看省事，长期会让主项目越来越难维护。

更合理的做法是：

- 模型调用层保留在 `agent.py`
- 统一路由抽到 `agent_v1.py`

### 错误 2：路由层直接生成最终答案

如果路由层一边分类、一边回答，就会重新变回“大而全函数”。

这一课的重点不是“再写一个万能回答器”，而是：

**先做统一决策，再调用已有能力。**

### 错误 3：不校验路由结果

只要你让模型输出 JSON，就要做字段校验。

否则：

- 拼错 `intent`
- 漏掉 `rewritten_input`
- 返回空字符串

都会导致主项目行为不稳定。

### 错误 4：把 `knowledge_qa` 和 `general_chat` 混在一起

二者的差别在于：

- `knowledge_qa`：尽量基于本地知识点资料回答
- `general_chat`：直接普通聊天

如果不分开，问答系统就容易“看起来像查资料，实际上又在自由发挥”。

## 7. 常见面试问题

### 问题 1：为什么 Agent v1 需要单独的路由层？

回答要点：

- 统一入口能把多个业务能力收敛成一个产品形态
- 路由层负责“分发”，业务层负责“执行”
- 这样后面扩展能力时不会把所有逻辑塞进一个大类里

深入追问：

- 如果后面能力越来越多，路由层还怎么扩展？
- 是继续分类式路由，还是进入 LangGraph/Workflow？

### 问题 2：为什么路由层也要结构化输出？

回答要点：

- 路由结果本质上是程序决策，不是给人看的自然语言
- 程序需要稳定消费 `intent / reason / rewritten_input`
- 所以必须用 JSON，而不是自由文本

深入追问：

- 如果模型路由不稳定，怎么提高稳定性？
- 是继续靠 Prompt，还是引入规则兜底？

### 问题 3：Agent v1 和真正的 LangChain 1.x `create_agent` 有什么关系？

回答要点：

- 当前这节是项目化整合版，目的是先吃透“统一入口 + 路由 + 分发”
- LangChain 1.x 的高层 agent 抽象，本质上也是在做能力编排
- 只是官方框架会把状态、工具、循环、追踪封装得更完整

深入追问：

- 什么时候该自己写路由，什么时候该直接上 LangChain/LangGraph？

## 8. 本节验收

运行：

```bash
python src/main.py
```

然后测试：

```text
/v1 给我一个 5 天 LangChain 学习计划
/v1 什么是输出解析器？
/v1 把做一个个人学习助理拆成 14 天执行步骤
/v1 先帮我看看现在有哪些知识点文件
```

如果表现分别是：

- 学习计划 -> 返回结构化 JSON
- 知识问答 -> 返回带资料来源倾向的回答
- 任务拆解 -> 返回结构化 JSON
- 查文件 -> 走工具 Agent

这一课就算真正跑通了。
