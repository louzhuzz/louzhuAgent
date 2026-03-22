from pathlib import Path
from typing import Any


KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "知识点"

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


def list_notes() -> list[str]:
    """列出知识点目录下所有 Markdown 笔记文件名。"""
    return sorted(path.name for path in KNOWLEDGE_DIR.glob("*.md"))


def read_note(file_name: str) -> str:
    """读取指定知识点笔记内容，并限制只能访问知识点目录下的 Markdown 文件。"""
    cleaned_name = file_name.strip()
    if not cleaned_name:
        raise ValueError("文件名不能为空。")

    if "/" in cleaned_name or "\\" in cleaned_name:
        raise ValueError("只允许传入文件名，不允许传入路径。")

    if not cleaned_name.endswith(".md"):
        raise ValueError("当前工具只允许读取 .md 文件。")

    note_path = KNOWLEDGE_DIR / cleaned_name
    if not note_path.exists():
        raise FileNotFoundError(f"未找到知识点文件：{cleaned_name}")

    return note_path.read_text(encoding="utf-8").strip()


def get_tool_schemas() -> dict[str, dict[str, Any]]:
    """返回当前项目暴露给 Agent 的工具 schema 定义。"""
    return TOOL_SCHEMAS


def _expect_object(arguments: Any) -> dict[str, Any]:
    """校验工具调用参数必须是对象结构。"""
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是 JSON 对象。")
    return arguments


def _validate_against_schema(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """按照工具 schema 校验参数字段、必填项和多余字段。"""
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        raise ValueError(f"未知工具：{tool_name}")

    parameters = schema["parameters"]
    required = set(parameters.get("required", []))
    properties = parameters.get("properties", {})
    additional_properties = parameters.get("additionalProperties", True)

    missing = required - set(arguments)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"工具 {tool_name} 缺少必填参数：{missing_fields}")

    if not additional_properties:
        extra_fields = set(arguments) - set(properties)
        if extra_fields:
            extra_field_names = ", ".join(sorted(extra_fields))
            raise ValueError(f"工具 {tool_name} 存在未定义参数：{extra_field_names}")

    for field_name, field_schema in properties.items():
        if field_name not in arguments:
            continue

        expected_type = field_schema.get("type")
        value = arguments[field_name]

        if expected_type == "string" and not isinstance(value, str):
            raise ValueError(f"工具 {tool_name} 的参数 {field_name} 必须是字符串。")

    return arguments


def execute_tool_call(tool_name: str, arguments: Any) -> Any:
    """统一执行工具调用，并在执行前做 schema 校验。"""
    validated_arguments = _validate_against_schema(
        tool_name,
        _expect_object(arguments),
    )

    if tool_name == "list_notes":
        return list_notes()

    if tool_name == "read_note":
        return read_note(validated_arguments["file_name"])

    raise ValueError(f"未知工具：{tool_name}")
