from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TaskBreakdownRequest:
    """描述一次任务拆解请求。"""

    goal: str
    current_level: str = "零基础"
    available_days: int = 7
    output_style: str = "可执行步骤"


class TaskBreakdownService:
    """封装任务拆解功能的项目化服务层。"""

    def __init__(
        self,
        invoke_json: Callable[[str, float], dict[str, Any]],
        render_prompt: Callable[[TaskBreakdownRequest], str],
    ) -> None:
        """保存模型调用函数和 Prompt 渲染函数。"""
        self.invoke_json = invoke_json
        self.render_prompt = render_prompt

    def _validate_request(self, request: TaskBreakdownRequest) -> TaskBreakdownRequest:
        """校验任务拆解请求。"""
        request.goal = request.goal.strip()
        request.current_level = request.current_level.strip()
        request.output_style = request.output_style.strip()

        if not request.goal:
            raise ValueError("任务目标不能为空。")
        if not request.current_level:
            raise ValueError("当前基础不能为空。")
        if not request.output_style:
            raise ValueError("输出风格不能为空。")
        if request.available_days <= 0:
            raise ValueError("可用天数必须大于 0。")
        if request.available_days > 60:
            raise ValueError("当前项目版任务拆解最多支持 60 天。")

        return request

    def _validate_result(
        self,
        result: dict[str, Any],
        request: TaskBreakdownRequest,
    ) -> dict[str, Any]:
        """校验模型返回的任务拆解结果。"""
        steps = result.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("任务拆解结果必须包含非空的 steps 列表。")

        if len(steps) > request.available_days:
            raise ValueError(
                f"模型返回了 {len(steps)} 个步骤，但可用天数只有 {request.available_days}。"
            )

        return result

    def generate(self, request: TaskBreakdownRequest) -> dict[str, Any]:
        """生成结构化任务拆解结果。"""
        validated_request = self._validate_request(request)
        prompt = self.render_prompt(validated_request)
        result = self.invoke_json(prompt, 0.3)
        return self._validate_result(result, validated_request)


def parse_task_breakdown_request(raw_text: str) -> TaskBreakdownRequest:
    """把 `/breakdown` 命令后的文本解析成任务拆解请求对象。

    支持两种形式：
    1. `/breakdown 做一个个人学习助理`
    2. `/breakdown 做一个个人学习助理 | 零基础 | 14 | 可执行步骤`
    """
    parts = [part.strip() for part in raw_text.split("|")]
    if not parts or not parts[0]:
        raise ValueError("请至少提供任务目标，例如 /breakdown 做一个个人学习助理")

    goal = parts[0]
    current_level = parts[1] if len(parts) > 1 and parts[1] else "零基础"

    available_days = 7
    if len(parts) > 2 and parts[2]:
        try:
            available_days = int(parts[2])
        except ValueError as exc:
            raise ValueError("可用天数必须是整数，例如 7 或 14。") from exc

    output_style = parts[3] if len(parts) > 3 and parts[3] else "可执行步骤"
    return TaskBreakdownRequest(
        goal=goal,
        current_level=current_level,
        available_days=available_days,
        output_style=output_style,
    )
