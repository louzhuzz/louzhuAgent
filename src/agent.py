import re
import json
import time
from typing import Any

from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from agent_v1 import AgentV1Service
from config import Settings
from langchain_helpers import to_langchain_messages
from knowledge_qa import KnowledgeQARequest, KnowledgeQAService
from output_parsers import OutputParserError, parse_json_output, parse_text_output
from prompts import (
    Message,
    build_chat_messages,
    render_agent_v1_router_prompt,
    render_knowledge_qa_prompt,
    load_system_prompt,
    render_react_agent_prompt,
    render_study_plan_prompt,
    render_task_breakdown_prompt,
    render_tool_agent_decision_prompt,
    render_tool_learning_prompt,
)
from study_plan import StudyPlanRequest, StudyPlanService
from task_breakdown import TaskBreakdownRequest, TaskBreakdownService
from tools import execute_tool_call, get_tool_schemas, list_notes, read_note


class LearningAgent:
    """主项目核心 Agent，对外暴露聊天、问答、任务拆解和工具调用等能力。"""

    def __init__(self, settings: Settings):
        """初始化聊天模型、系统提示词和会话历史。"""
        self.settings = settings
        self.model = ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=0.7,
            request_timeout=30,
            max_retries=0,
        )
        if not settings.embedding_model:
            raise ValueError("缺少 ARK_EMBEDDING_MODEL，主项目知识点问答需要真实 embedding 配置。")
        self.system_prompt = load_system_prompt()
        self.history: list[Message] = []
        self.study_plan_service = StudyPlanService(
            invoke_json=self._invoke_json,
            render_prompt=render_study_plan_prompt,
        )
        self.task_breakdown_service = TaskBreakdownService(
            invoke_json=self._invoke_json,
            render_prompt=render_task_breakdown_prompt,
        )
        self.knowledge_qa_service = KnowledgeQAService(
            list_notes=self.list_notes_tool,
            read_note=self.read_note_tool,
            invoke_text=self._invoke_text,
            render_prompt=render_knowledge_qa_prompt,
            api_key=settings.api_key,
            base_url=settings.base_url,
            embedding_model=settings.embedding_model,
            chroma_persist_directory=settings.chroma_persist_directory,
        )
        self.agent_v1_service = AgentV1Service(
            invoke_json=self._invoke_json,
            render_router_prompt=render_agent_v1_router_prompt,
            create_study_plan=self.create_study_plan,
            answer_knowledge_question=self.answer_knowledge_question,
            create_task_breakdown=self.create_task_breakdown,
            run_tool_calling_agent=self.run_tool_calling_agent,
            reply=self.reply,
        )

    def _build_messages(self, user_input: str) -> list[Message]:
        """把系统提示、历史消息和当前输入组装成一次请求的消息列表。"""
        return build_chat_messages(
            system_prompt=self.system_prompt,
            history=self.history,
            user_input=user_input,
        )

    def _extract_query_terms(self, text: str) -> set[str]:
        """提取问题中的英文词和中文短语，用于选择更相关的知识点文件。"""
        english_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        chinese_blocks = re.findall(r"[\u4e00-\u9fff]+", text)

        chinese_terms: set[str] = set()
        for block in chinese_blocks:
            cleaned = block.strip()
            if len(cleaned) >= 2:
                chinese_terms.add(cleaned)
            for index in range(len(cleaned) - 1):
                chinese_terms.add(cleaned[index : index + 2])

        return {term for term in english_terms | chinese_terms if term}

    def _score_note_name(self, question: str, file_name: str) -> float:
        """根据问题关键词和文件名的重叠程度，给知识点文件打分。"""
        query_terms = self._extract_query_terms(question)
        if not query_terms:
            return 0.0

        lowered_name = file_name.lower()
        hit_count = sum(1 for term in query_terms if term in lowered_name)
        return hit_count / len(query_terms)

    def _select_note_for_question(self, question: str) -> str:
        """从全部知识点文件中选出当前最可能相关的一篇。"""
        note_names = self.list_notes_tool()
        if not note_names:
            raise ValueError("当前没有可用的知识点文件。")

        scored_notes = sorted(
            note_names,
            key=lambda file_name: (self._score_note_name(question, file_name), file_name),
            reverse=True,
        )
        return scored_notes[0]

    def _save_turn(self, user_input: str, answer: str) -> None:
        """把一轮对话的用户输入和助手回答保存到历史记录中。"""
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": answer})

    def _invoke_text(self, user_input: str, temperature: float) -> str:
        """调用聊天模型，并把返回结果解析成普通文本。

        这里会对常见瞬时异常做有限重试，避免网络抖动或短暂限流直接打崩主链路。
        """
        messages = to_langchain_messages(self._build_messages(user_input))
        retryable_errors = (APIConnectionError, APITimeoutError, RateLimitError, APIError)
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.model.bind(temperature=temperature).invoke(messages)
                return parse_text_output(response.content)
            except retryable_errors as exc:
                if attempt == max_attempts:
                    raise RuntimeError(f"模型服务暂时不可用：{exc}") from exc
                time.sleep(0.8 * attempt)
            except Exception as exc:
                raise RuntimeError(f"模型调用失败：{exc}") from exc

        raise RuntimeError("模型调用失败：未知异常。")

    def _invoke_json(self, user_input: str, temperature: float) -> dict[str, Any]:
        """调用模型并把返回结果解析成 JSON 对象。

        这里额外做了一层“自动修复重试”：
        1. 先按正常结构化任务去生成 JSON
        2. 如果模型返回了近似 JSON 但格式有错误，再让模型只做 JSON 修复

        这样可以降低评测或批量任务里因为一次格式波动而整条链路中断的概率。
        """
        raw_text = self._invoke_text(user_input, temperature=temperature)
        try:
            return parse_json_output(raw_text)
        except OutputParserError:
            repair_prompt = (
                "你是一个 JSON 修复器。"
                "下面会给你一段本来想返回 JSON、但格式不合法的文本。"
                "请在不改变原始语义的前提下，把它修复成一个合法 JSON 对象。"
                "只输出最终 JSON，不要输出解释，不要加 Markdown 代码块。\n\n"
                f"待修复文本：\n{raw_text}"
            )
            repaired_text = self._invoke_text(repair_prompt, temperature=0.0)
            return parse_json_output(repaired_text)

    def reply(self, user_input: str) -> str:
        """执行一轮普通对话。"""
        answer = self._invoke_text(user_input, temperature=0.7)
        self._save_turn(user_input, answer)
        return answer

    def create_study_plan(self, request: StudyPlanRequest) -> dict:
        """根据项目化的请求对象生成结构化学习计划。"""
        plan = self.study_plan_service.generate(request)
        self._save_turn(
            f"/plan {request.topic} | {request.current_level} | {request.days} | {request.goal}",
            json.dumps(plan, ensure_ascii=False),
        )
        return plan

    def create_task_breakdown(self, request: TaskBreakdownRequest) -> dict:
        """根据项目化的请求对象生成结构化任务拆解。"""
        result = self.task_breakdown_service.generate(request)
        self._save_turn(
            f"/breakdown {request.goal} | {request.current_level} | {request.available_days} | {request.output_style}",
            json.dumps(result, ensure_ascii=False),
        )
        return result

    def answer_knowledge_question(self, request: KnowledgeQARequest) -> dict[str, object]:
        """执行主项目版知识点问答，并把结果保存到会话历史。"""
        result = self.knowledge_qa_service.answer(request)
        self._save_turn(f"/qa {request.question}", str(result["answer"]))
        return result

    def run_agent_v1(self, user_input: str) -> dict[str, Any]:
        """运行主项目的统一入口，让系统先判断意图再路由到对应能力。"""
        return self.agent_v1_service.handle(user_input)

    def list_notes_tool(self) -> list[str]:
        """执行列出知识点文件的本地工具。"""
        return list_notes()

    def read_note_tool(self, file_name: str) -> str:
        """执行读取指定知识点文件内容的本地工具。"""
        return read_note(file_name)

    def get_tool_schemas(self) -> dict:
        """返回当前 Agent 可用的工具 schema。"""
        return get_tool_schemas()

    def execute_tool(self, tool_name: str, arguments: dict) -> object:
        """按统一 schema 校验并执行指定工具。"""
        return execute_tool_call(tool_name, arguments)

    def _build_tool_agent_scratchpad(self, steps: list[dict[str, Any]]) -> str:
        """把已有工具调用记录整理成供下一轮决策使用的执行上下文。"""
        if not steps:
            return "暂无执行记录。"

        parts: list[str] = []
        for step in steps:
            tool_name = step["tool_name"]
            arguments = json.dumps(step["arguments"], ensure_ascii=False)
            result = step["result"]
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, indent=2)

            parts.append(
                f"步骤 {step['step']}:\n"
                f"- reason: {step['reason']}\n"
                f"- tool_name: {tool_name}\n"
                f"- arguments: {arguments}\n"
                f"- result:\n{result}"
            )

        return "\n\n".join(parts)

    def _build_react_scratchpad(self, steps: list[dict[str, Any]]) -> str:
        """把 ReAct 过程中的 thought / action / observation 整理成上下文。"""
        if not steps:
            return "暂无思考与观察记录。"

        parts: list[str] = []
        for step in steps:
            arguments = json.dumps(step["arguments"], ensure_ascii=False)
            observation = step["observation"]
            if not isinstance(observation, str):
                observation = json.dumps(observation, ensure_ascii=False, indent=2)

            parts.append(
                f"步骤 {step['step']}:\n"
                f"Thought: {step['thought']}\n"
                f"Action: {step['tool_name']}\n"
                f"Arguments: {arguments}\n"
                f"Observation:\n{observation}"
            )

        return "\n\n".join(parts)

    def answer_with_note_tool(self, question: str) -> tuple[str, str]:
        """先选择知识点文件，再读取文件内容，并基于该工具结果回答问题。"""
        file_name = self._select_note_for_question(question)
        note_content = self.read_note_tool(file_name)
        user_input = render_tool_learning_prompt(file_name, note_content, question)
        answer = self._invoke_text(user_input, temperature=0.3)
        self._save_turn(question, answer)
        return file_name, answer

    def run_tool_calling_agent(
        self,
        question: str,
        max_steps: int = 3,
    ) -> dict[str, Any]:
        """运行一个由模型决定是否调用工具的最小 Agent 循环。"""
        tool_schemas_json = json.dumps(
            self.get_tool_schemas(),
            ensure_ascii=False,
            indent=2,
        )
        steps: list[dict[str, Any]] = []

        for step_number in range(1, max_steps + 1):
            scratchpad = self._build_tool_agent_scratchpad(steps) # 把之前的工具调用记录整理成上下文，供模型决策时参考
            prompt = render_tool_agent_decision_prompt( # 生成决策提示词，包含问题、工具信息和执行上下文
                question=question,
                tool_schemas_json=tool_schemas_json,
                scratchpad=scratchpad,
            )
            decision = self._invoke_json(prompt, temperature=0.1) # 调用模型并解析成结构化决策

            action = decision.get("action", "")
            reason = str(decision.get("reason", "")).strip()

            if action == "final_answer": # 模型认为当前信息足够，可以直接回答问题了
                answer = str(decision.get("answer", "")).strip()
                if not answer:
                    raise ValueError("模型选择了 final_answer，但没有返回 answer。")

                self._save_turn(question, answer)
                return {"steps": steps, "answer": answer}

            if action != "tool_call":
                raise ValueError(f"模型返回了未知 action：{action}")

            tool_name = str(decision.get("tool_name", "")).strip()
            arguments = decision.get("arguments", {})
            try:
                result = self.execute_tool(tool_name, arguments)
            except Exception as exc:
                result = f"TOOL_ERROR: {exc}"
            steps.append(
                {
                    "step": step_number,
                    "reason": reason,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                }
            )

        final_answer = "工具调用已达到最大步数，当前工具结果不足以完整回答。"
        self._save_turn(question, final_answer)
        return {"steps": steps, "answer": final_answer}

    def run_react_agent(
        self,
        question: str,
        max_steps: int = 4,
    ) -> dict[str, Any]:
        """运行一个显式暴露 Thought/Action/Observation 的最小 ReAct Agent。"""
        tool_schemas_json = json.dumps(
            self.get_tool_schemas(),
            ensure_ascii=False,
            indent=2,
        )
        steps: list[dict[str, Any]] = []

        for step_number in range(1, max_steps + 1):
            scratchpad = self._build_react_scratchpad(steps)
            prompt = render_react_agent_prompt(
                question=question,
                tool_schemas_json=tool_schemas_json,
                scratchpad=scratchpad,
            )
            decision = self._invoke_json(prompt, temperature=0.1)

            thought = str(decision.get("thought", "")).strip()
            action = str(decision.get("action", "")).strip()

            if action == "final_answer":
                answer = str(decision.get("answer", "")).strip()
                if not answer:
                    raise ValueError("模型选择了 final_answer，但没有返回 answer。")

                self._save_turn(question, answer)
                return {"steps": steps, "answer": answer}

            if action != "tool_call":
                raise ValueError(f"模型返回了未知 action：{action}")

            tool_name = str(decision.get("tool_name", "")).strip()
            arguments = decision.get("arguments", {})
            try:
                observation = self.execute_tool(tool_name, arguments)
            except Exception as exc:
                observation = f"TOOL_ERROR: {exc}"
            steps.append(
                {
                    "step": step_number,
                    "thought": thought,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "observation": observation,
                }
            )

        final_answer = "ReAct Agent 已达到最大步数，当前观察结果不足以完整回答。"
        self._save_turn(question, final_answer)
        return {"steps": steps, "answer": final_answer}

    def clear_history(self) -> None:
        """清空当前会话历史。"""
        self.history.clear()
