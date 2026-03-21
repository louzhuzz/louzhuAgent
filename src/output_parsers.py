import json
from typing import Any


class OutputParserError(ValueError):
    pass


def parse_text_output(content: object) -> str: # 这个函数的作用是将模型返回的内容解析成纯文本字符串，适用于模型可能返回字符串或者包含文本片段的列表的情况
    if isinstance(content, str): # isinstance 用于检查 content 是否是字符串类型，如果是，则直接返回 content
        return content

    if isinstance(content, list): # 如果 content 是一个列表，遍历列表中的每个元素，如果元素是一个字典且包含 "type" 键且值为 "text"，则提取该字典中的 "text" 键的值，并将这些文本片段连接成一个字符串返回
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()

    return str(content)


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"): # 如果以 ```json 开头，去掉前 7 个字符 是为了去掉可能的代码块开始标记
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"): # 如果以 ``` 开头，去掉前 3 个字符
        cleaned = cleaned[3:]

    if cleaned.endswith("```"): # 如果以 ``` 结尾，去掉后 3 个字符 是为了去掉可能的代码块结尾标记
        cleaned = cleaned[:-3]

    return cleaned.strip()


def parse_json_output(raw_text: str) -> dict[str, Any]: # -> dict[str, Any] 表示函数返回一个字典，键是字符串，值可以是任意类型
    cleaned = strip_code_fence(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise OutputParserError(f"模型返回的内容不是合法 JSON: {cleaned}") from exc

    if not isinstance(parsed, dict):
        raise OutputParserError("当前结构化输出要求返回 JSON 对象。")

    return parsed

