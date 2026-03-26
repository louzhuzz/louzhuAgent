from dataclasses import dataclass
from typing import Any, Callable

from knowledge_qa import KnowledgeQARequest
from study_plan import StudyPlanRequest, parse_study_plan_request
from task_breakdown import TaskBreakdownRequest, parse_task_breakdown_request


@dataclass
class AgentV1Decision:
    """描述 Agent v1 在统一入口中做出的路由决策。"""

    intent: str
    reason: str
    rewritten_input: str


class AgentV1Service:
    """把多个主项目能力整合成统一入口的项目化服务层。"""

    def __init__(
        self,
        invoke_json: Callable[[str, float], dict[str, Any]],
        render_router_prompt: Callable[[str], str],
        create_study_plan: Callable[[StudyPlanRequest], dict[str, Any]],
        answer_knowledge_question: Callable[[KnowledgeQARequest], dict[str, Any]],
        create_task_breakdown: Callable[[TaskBreakdownRequest], dict[str, Any]],
        run_tool_calling_agent: Callable[[str], dict[str, Any]],
        reply: Callable[[str], str],
    ) -> None:
        """保存统一路由所需的模型调用函数和能力函数。"""
        self.invoke_json = invoke_json
        self.render_router_prompt = render_router_prompt
        self.create_study_plan = create_study_plan
        self.answer_knowledge_question = answer_knowledge_question
        self.create_task_breakdown = create_task_breakdown
        self.run_tool_calling_agent = run_tool_calling_agent
        self.reply = reply
        self.allowed_intents = {
            "study_plan",
            "knowledge_qa",
            "task_breakdown",
            "tool_agent",
            "general_chat",
        }

    def _validate_user_input(self, user_input: str) -> str:
        """校验统一入口收到的用户输入。"""
        cleaned = user_input.strip()
        if not cleaned:
            raise ValueError("Agent v1 的输入不能为空。")
        return cleaned

    def _make_decision(self, user_input: str) -> AgentV1Decision:
        """先让模型判断当前问题最适合路由到哪个主项目能力。"""
        prompt = self.render_router_prompt(user_input)
        result = self.invoke_json(prompt, 0.1)

        intent = str(result.get("intent", "")).strip()
        reason = str(result.get("reason", "")).strip()
        rewritten_input = str(result.get("rewritten_input", "")).strip()

        if intent not in self.allowed_intents:
            raise ValueError(f"Agent v1 收到了未知意图：{intent}")
        if not rewritten_input:
            raise ValueError("Agent v1 路由结果缺少 rewritten_input。")
        if not reason:
            raise ValueError("Agent v1 路由结果缺少 reason。")

        return AgentV1Decision(
            intent=intent,
            reason=reason,
            rewritten_input=rewritten_input,
        )

    def handle(self, user_input: str) -> dict[str, Any]:
        """根据统一入口的路由结果，调用对应的主项目能力。"""
        cleaned_input = self._validate_user_input(user_input)
        decision = self._make_decision(cleaned_input)

        if decision.intent == "study_plan":
            request = parse_study_plan_request(decision.rewritten_input)
            result = self.create_study_plan(request)
            return {
                "intent": decision.intent,
                "reason": decision.reason,
                "result_type": "json",
                "result": result,
            }

        if decision.intent == "knowledge_qa":
            result = self.answer_knowledge_question(
                KnowledgeQARequest(question=decision.rewritten_input)
            )
            return {
                "intent": decision.intent,
                "reason": decision.reason,
                "result_type": "knowledge_qa",
                "result": result,
            }

        if decision.intent == "task_breakdown":
            request = parse_task_breakdown_request(decision.rewritten_input)
            result = self.create_task_breakdown(request)
            return {
                "intent": decision.intent,
                "reason": decision.reason,
                "result_type": "json",
                "result": result,
            }

        if decision.intent == "tool_agent":
            result = self.run_tool_calling_agent(decision.rewritten_input)
            return {
                "intent": decision.intent,
                "reason": decision.reason,
                "result_type": "tool_agent",
                "result": result,
            }

        answer = self.reply(decision.rewritten_input)
        return {
            "intent": decision.intent,
            "reason": decision.reason,
            "result_type": "text",
            "result": answer,
        }
