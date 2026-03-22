# 第二十八步：工具 Schema 与参数约束

## 1. 本节目标

这一节要把上一节的“工具函数”升级成：

- 工具函数
- 参数结构定义
- 统一校验入口
- 统一执行入口

也就是说，这一节之后你不再只是“有两个 Python 函数”，而是开始拥有一个最小可控的工具系统。

对应文件：

- [src/tools.py](/mnt/d/AIcodes/Agent/src/tools.py)
- [src/agent.py](/mnt/d/AIcodes/Agent/src/agent.py)
- [src/main.py](/mnt/d/AIcodes/Agent/src/main.py)

## 2. 是什么

`Tool Schema` 可以先简单理解成：

**工具的参数说明书。**

它要描述的核心信息包括：

- 工具叫什么
- 工具做什么
- 参数有哪些
- 哪些参数必填
- 参数类型是什么
- 是否允许多余字段

这一层为什么重要？

因为一旦开始让模型调用工具，模型给你的不再是“人类手工输入”，而是一份结构化参数。

如果没有 schema，你就很难判断：

- 参数是不是少了
- 参数类型对不对
- 有没有夹带奇怪字段

## 3. 为什么

上一节你已经有：

- `list_notes()`
- `read_note(file_name)`

但那还只是“可以调用的函数”。

真实 Agent 系统还需要一层：

**在执行之前，先检查这次调用是否合法。**

这层检查就是参数约束。

它的价值有三个：

1. 防止模型传错参数
2. 防止工具行为失控
3. 让后面的 Tool Calling 更容易接框架

## 4. 怎么做

### 第一步：给每个工具定义 schema

我们在 [src/tools.py](/mnt/d/AIcodes/Agent/src/tools.py) 里新增了 `TOOL_SCHEMAS`。

它是一个字典，里面定义了两个工具：

- `list_notes`
- `read_note`

其中 `read_note` 的 schema 明确要求：

- 参数必须是对象
- 必填字段是 `file_name`
- `file_name` 必须是字符串
- 不允许传额外字段

### 第二步：增加统一的参数校验逻辑

这节没有直接引入第三方校验库，而是先手写一个最小版校验器。

这样做的目的不是“手写比框架强”，而是为了让你先看懂：

- schema 到底在校验什么
- 校验错误是怎么抛出来的

### 第三步：增加统一的工具执行入口

我们新增了：

- `execute_tool_call(tool_name, arguments)`

它负责两件事：

1. 先校验
2. 再执行

这样以后无论是：

- 命令行手动调用
- 模型自动发起工具调用

都会走同一个受控入口。

### 第四步：给命令行增加调试入口

为了让你能直接观察 schema 的效果，这一节还加了两个命令：

- `/tools`
- `/tool 工具名 JSON参数`

例如：

```text
/tools
/tool list_notes {}
/tool read_note {"file_name":"13-输出解析器.md"}
```

## 5. 关键代码

### 代码 1：工具 schema 定义

文件：[src/tools.py](/mnt/d/AIcodes/Agent/src/tools.py)

```python
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_notes": {
        "name": "list_notes",
        "description": "列出知识点目录下所有 Markdown 文件名。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    "read_note": {
        "name": "read_note",
        "description": "读取指定知识点 Markdown 文件内容。",
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "知识点目录下的 Markdown 文件名，例如 13-输出解析器.md",
                }
            },
            "required": ["file_name"],
            "additionalProperties": False,
        },
    },
}
```

这段代码要重点看 `read_note`。

它明确规定了：

- 只能传 `file_name`
- 这个字段必填
- 类型必须是字符串
- 不能多传别的字段

这就是工具约束的雏形。

### 代码 2：参数对象校验

文件：[src/tools.py](/mnt/d/AIcodes/Agent/src/tools.py)

```python
def _expect_object(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是 JSON 对象。")
    return arguments
```

这一小段虽然简单，但意义很大。

因为工具调用参数在工程里最常见的结构就是：

- JSON object

如果这里不先卡住类型，后面就可能出现：

- 传字符串
- 传列表
- 传数字

导致执行层变得混乱。

### 代码 3：按 schema 做字段校验

文件：[src/tools.py](/mnt/d/AIcodes/Agent/src/tools.py)

```python
missing = required - set(arguments)
if missing:
    missing_fields = ", ".join(sorted(missing))
    raise ValueError(f"工具 {tool_name} 缺少必填参数：{missing_fields}")

if not additional_properties:
    extra_fields = set(arguments) - set(properties)
    if extra_fields:
        extra_field_names = ", ".join(sorted(extra_fields))
        raise ValueError(f"工具 {tool_name} 存在未定义参数：{extra_field_names}")
```

这一段在检查两类最常见错误：

1. 缺参数
2. 多参数

为什么“多参数”也要拦？

因为工具系统里，多余字段往往意味着：

- 模型理解错了
- 参数拼错了
- 或调用方在夹带额外信息

如果你默默忽略，多半会给后续排错带来麻烦。

### 代码 4：统一执行入口

文件：[src/tools.py](/mnt/d/AIcodes/Agent/src/tools.py)

```python
def execute_tool_call(tool_name: str, arguments: Any) -> Any:
    validated_arguments = _validate_against_schema(
        tool_name,
        _expect_object(arguments),
    )

    if tool_name == "list_notes":
        return list_notes()

    if tool_name == "read_note":
        return read_note(validated_arguments["file_name"])
```

你要把这段代码理解成：

- 入口层先做校验
- 业务层再做执行

这就是为什么我们一直强调：

**工具调用不是“直接运行函数”，而是“受控地运行函数”。**

### 代码 5：命令行里的工具调试入口

文件：[src/main.py](/mnt/d/AIcodes/Agent/src/main.py)

```python
if user_input == "/tools":
    print("\n工具 Schema:")
    print(json.dumps(agent.get_tool_schemas(), ensure_ascii=False, indent=2))
    continue
```

这个命令的作用是：

- 先让你看见“当前有哪些工具、参数长什么样”

再看另一个命令：

```python
if user_input.startswith("/tool "):
    ...
    arguments = json.loads(raw_arguments)
    result = agent.execute_tool(tool_name, arguments)
```

这一段的数据流是：

1. 用户输入字符串
2. 先解析成 JSON
3. 再走 Agent 的统一工具入口
4. Agent 再调用 `execute_tool_call(...)`
5. 执行前先做 schema 校验

这已经非常接近后面真正 Tool Calling 的数据流了。

## 6. 常见错误

### 错误 1：以为工具 schema 只是“写给人看的文档”

不是。

它首先是程序要使用的约束规则。

### 错误 2：只校验必填，不校验多余字段

这样会让工具边界变得模糊，后面很难判断调用方到底传了什么。

### 错误 3：执行工具时绕过统一入口

如果有的人直接调 `read_note()`，有的人走 `execute_tool_call()`，系统就会出现两套标准，不利于维护。

### 错误 4：以为这一步已经等于“模型自动调用工具”

还没有。

这一节只是把基础设施搭好：

- schema
- 校验
- 统一执行入口

下一步才会进入：

- 模型决定调用哪个工具

## 7. 常见面试问题

### 问题 1：为什么工具系统需要 schema？

回答要点：

- 让调用方知道参数结构
- 让执行层有校验依据
- 降低调用错误和安全风险

### 问题 2：为什么还要拦截“多余字段”？

回答要点：

- 多余字段常常意味着调用方理解错了
- 忽略它们会隐藏错误
- 工具越受控，系统越稳定

### 问题 3：为什么要设计统一的 `execute_tool_call()`？

回答要点：

- 统一校验和执行流程
- 降低重复代码
- 后续接模型自动工具调用时更容易复用

深入追问：

- 如果后面要接入权限控制、日志记录、重试机制，应该放在哪一层最合适？

## 8. 本节验收

运行：

```bash
python src/main.py
```

然后测试：

```text
/tools
/tool list_notes {}
/tool read_note {"file_name":"13-输出解析器.md"}
/tool read_note {"wrong":"13-输出解析器.md"}
/tool read_note {"file_name":123}
```

你应该能观察到：

- `/tools` 能打印工具 schema
- 合法参数可以执行成功
- 缺少 `file_name` 会报错
- `file_name` 类型不对会报错

如果这些都正常，这一节就完成了。
