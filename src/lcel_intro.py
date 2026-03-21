import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import load_settings
from prompts import render_lcel_summary_prompt


def main() -> None:
    topic = "LangChain 的核心价值"
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:]).strip() or topic

    settings = load_settings()
    model = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个严谨的 AI 学习教练，回答要简洁、结构化。",
            ),
            ("human", "{user_input}"),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | model | parser

    result = chain.invoke({"user_input": render_lcel_summary_prompt(topic)})
    print("LCEL 示例已启动。")
    print(f"\n主题: {topic}")
    print(f"\n结果:\n{result}")


if __name__ == "__main__":
    main()
