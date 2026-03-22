import re

from langchain_openai import ChatOpenAI

from config import Settings
from langchain_helpers import to_langchain_messages
from output_parsers import parse_json_output, parse_text_output
from prompts import (
    Message,
    build_chat_messages,
    load_system_prompt,
    render_study_plan_prompt,
    render_tool_learning_prompt,
)
from tools import execute_tool_call, get_tool_schemas, list_notes, read_note


class LearningAgent:
    def __init__(self, settings: Settings):
        """初始化聊天模型、系统提示词和会话历史。"""
        self.settings = settings
        self.model = ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=0.7,
        )
        self.system_prompt = load_system_prompt()
        self.history: list[Message] = []

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
        """调用聊天模型，并把返回结果解析成普通文本。"""
        response = self.model.bind(temperature=temperature).invoke(
            to_langchain_messages(self._build_messages(user_input))
        )
        return parse_text_output(response.content)

    def reply(self, user_input: str) -> str:
        """执行一轮普通对话。"""
        answer = self._invoke_text(user_input, temperature=0.7)
        self._save_turn(user_input, answer)
        return answer

    def create_study_plan(self, topic: str) -> dict:
        """根据学习主题生成结构化学习计划。"""
        user_input = render_study_plan_prompt(topic)
        raw_answer = self._invoke_text(user_input, temperature=0.3)
        plan = parse_json_output(raw_answer)
        self._save_turn(user_input, raw_answer)
        return plan

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

    def answer_with_note_tool(self, question: str) -> tuple[str, str]:
        """先选择知识点文件，再读取文件内容，并基于该工具结果回答问题。"""
        file_name = self._select_note_for_question(question)
        note_content = self.read_note_tool(file_name)
        user_input = render_tool_learning_prompt(file_name, note_content, question)
        answer = self._invoke_text(user_input, temperature=0.3)
        self._save_turn(question, answer)
        return file_name, answer

    def clear_history(self) -> None:
        """清空当前会话历史。"""
        self.history.clear()
