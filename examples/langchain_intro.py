import sys

from _bootstrap import setup_example_path
from langchain_openai import ChatOpenAI

setup_example_path()

from config import load_settings
from output_parsers import parse_text_output
from prompts import load_system_prompt
from langchain_core.messages import HumanMessage, SystemMessage


def main() -> None:
    user_input = "请用 3 点说明 LangChain 是什么，以及它解决了什么问题。"
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:]).strip() or user_input

    settings = load_settings()
    model = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0.7,
    )

    messages = [
        SystemMessage(content=load_system_prompt()),
        HumanMessage(content=user_input),
    ]

    response = model.invoke(messages)
    print("LangChain 示例已启动。")
    print(f"\n你: {user_input}")
    print(f"\nLangChain Agent: {parse_text_output(response.content)}")


if __name__ == "__main__":
    main()
