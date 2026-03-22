from pathlib import Path
from typing import TypeAlias

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
Message: TypeAlias = dict[str, str]

# 前面加一个_表示这是一个私有函数，不应该被外部直接调用
# 通过文件名加载提示模板内容
def _load_prompt_file(file_name: str) -> str:
    prompt_path = PROMPTS_DIR / file_name
    return prompt_path.read_text(encoding="utf-8").strip()

# 加载系统提示语
def load_system_prompt() -> str:
    return _load_prompt_file("system_prompt.txt")

# 生成结构化学习计划的提示
def render_study_plan_prompt(topic: str) -> str:
    template = _load_prompt_file("study_plan_prompt.txt")
    return template.format(topic=topic)


def render_lcel_summary_prompt(topic: str) -> str:
    template = _load_prompt_file("lcel_summary_prompt.txt")
    return template.format(topic=topic)


def render_parallel_core_prompt(topic: str) -> str:
    template = _load_prompt_file("parallel_core_prompt.txt")
    return template.format(topic=topic)


def render_parallel_pitfalls_prompt(topic: str) -> str:
    template = _load_prompt_file("parallel_pitfalls_prompt.txt")
    return template.format(topic=topic)


def render_tool_learning_prompt(file_name: str, note_content: str, question: str) -> str:
    """生成“基于单个知识点工具结果回答问题”的提示词。"""
    template = _load_prompt_file("tool_learning_assistant_prompt.txt")
    return template.format(
        file_name=file_name,
        note_content=note_content,
        question=question,
    )

# 其他提示模板的加载函数可以在这里添加
def build_system_message(system_prompt: str) -> Message:
    return {"role": "system", "content": system_prompt}

# 构建用户消息
def build_user_message(user_input: str) -> Message:
    return {"role": "user", "content": user_input}

# 构建历史消息
def clone_history_messages(history: list[Message]) -> list[Message]:
    return [message.copy() for message in history]

# 构建完整的聊天消息列表，包括系统提示、历史消息和当前用户输入
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
