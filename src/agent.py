from langchain_openai import ChatOpenAI

from config import Settings
from langchain_helpers import to_langchain_messages
from output_parsers import parse_json_output, parse_text_output
from prompts import Message, build_chat_messages, load_system_prompt, render_study_plan_prompt


class LearningAgent:
    def __init__(self, settings: Settings):
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
        return build_chat_messages(
            system_prompt=self.system_prompt,
            history=self.history,
            user_input=user_input,
        )

    def _save_turn(self, user_input: str, answer: str) -> None:
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": answer})

    def _invoke_text(self, user_input: str, temperature: float) -> str:
        response = self.model.bind(temperature=temperature).invoke(
            to_langchain_messages(self._build_messages(user_input))
        )
        return parse_text_output(response.content)

    def reply(self, user_input: str) -> str:
        answer = self._invoke_text(user_input, temperature=0.7)
        self._save_turn(user_input, answer)
        return answer

    def create_study_plan(self, topic: str) -> dict:
        user_input = render_study_plan_prompt(topic)
        raw_answer = self._invoke_text(user_input, temperature=0.3)
        plan = parse_json_output(raw_answer)
        self._save_turn(user_input, raw_answer)
        return plan

    def clear_history(self) -> None:
        self.history.clear()
