from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class StudyPlanRequest:
    """描述一次学习计划生成请求的输入参数。"""

    topic: str
    current_level: str = "零基础"
    days: int = 3
    goal: str = "完成系统入门并输出可验证的学习成果"


class StudyPlanService:
    """封装学习计划生成功能的项目化服务层。"""

    def __init__(
        self,
        invoke_json: Callable[[str, float], dict[str, Any]],
        render_prompt: Callable[[StudyPlanRequest], str],
    ) -> None:
        """保存模型调用函数和 Prompt 渲染函数。"""
        self.invoke_json = invoke_json
        self.render_prompt = render_prompt

    def _validate_request(self, request: StudyPlanRequest) -> StudyPlanRequest:
        """校验学习计划请求，避免把明显非法的数据交给模型。"""
        request.topic = request.topic.strip()
        request.current_level = request.current_level.strip()
        request.goal = request.goal.strip()

        if not request.topic:
            raise ValueError("学习主题不能为空。")
        if not request.current_level:
            raise ValueError("当前基础不能为空。")
        if not request.goal:
            raise ValueError("学习目标不能为空。")
        if request.days <= 0:
            raise ValueError("学习天数必须大于 0。")
        if request.days > 30:
            raise ValueError("当前项目版学习计划最多支持 30 天。")

        return request

    def _validate_plan(self, plan: dict[str, Any], request: StudyPlanRequest) -> dict[str, Any]:
        """校验模型返回的计划结构，确保核心字段符合项目要求。"""
        days = plan.get("days")
        if not isinstance(days, list):
            raise ValueError("学习计划返回结果缺少合法的 days 列表。")

        if len(days) != request.days:
            raise ValueError(
                f"模型返回的 days 数量是 {len(days)}，但请求要求 {request.days} 天。"
            )

        return plan

    def generate(self, request: StudyPlanRequest) -> dict[str, Any]:
        """根据请求生成学习计划，并在返回前做结构校验。"""
        validated_request = self._validate_request(request)
        prompt = self.render_prompt(validated_request)
        plan = self.invoke_json(prompt, 0.3)
        return self._validate_plan(plan, validated_request)


def parse_study_plan_request(raw_text: str) -> StudyPlanRequest:
    """把 `/plan` 命令后的文本解析成项目化的学习计划请求对象。

    支持两种形式：
    1. `/plan LangChain`
    2. `/plan LangChain | 零基础 | 5 | 做出一个可运行 Demo`
    """
    parts = [part.strip() for part in raw_text.split("|")]
    if not parts or not parts[0]:
        raise ValueError("请至少提供学习主题，例如 /plan LangChain")

    topic = parts[0]
    current_level = parts[1] if len(parts) > 1 and parts[1] else "零基础"

    days = 3
    if len(parts) > 2 and parts[2]:
        try:
            days = int(parts[2])
        except ValueError as exc:
            raise ValueError("学习天数必须是整数，例如 3 或 7。") from exc

    goal = parts[3] if len(parts) > 3 and parts[3] else "完成系统入门并输出可验证的学习成果"
    return StudyPlanRequest(
        topic=topic,
        current_level=current_level,
        days=days,
        goal=goal,
    )
