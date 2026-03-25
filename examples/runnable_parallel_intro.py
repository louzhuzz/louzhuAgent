import json
import sys

from _bootstrap import setup_example_path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

setup_example_path()

from config import load_settings
from prompts import render_parallel_core_prompt, render_parallel_pitfalls_prompt

# lambda函数示例：
# 这个示例展示了如何使用 lambda 函数来定义一个简单的函数，该函数接受一个输入并返回一个字符串。
# 例如，我们可以定义一个 lambda 函数来计算一个数的平方：
# square = lambda x: x * x # 第一个x是输入参数，第二个x是函数体中的变量，表示输入参数的值。
# print(f"5 的平方是: {square(5)}")  # 输出: 5 的平方是: 25
# 这个示例展示了如何使用 RunnableParallel 来并行生成核心内容和常见陷阱的提示。
def main() -> None:
    topic = "RAG"
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
            ("system", "你是一个严谨的 AI 学习教练，回答要简洁、结构化。"),
            ("human", "{user_input}"),
        ]
    )
    parser = StrOutputParser()

    core_chain = prompt | model | parser
    pitfalls_chain = prompt | model | parser

    parallel_chain = RunnableParallel(
        core_summary=lambda data: core_chain.invoke( # data是一个字典，包含了主题信息，我们通过data["topic"]来获取主题，并生成核心内容的提示。
            {"user_input": render_parallel_core_prompt(data["topic"])} # 根据主题生成核心内容的提示
        ),
        common_pitfalls=lambda data: pitfalls_chain.invoke(
            {"user_input": render_parallel_pitfalls_prompt(data["topic"])}
        ),
    )

    result = parallel_chain.invoke({"topic": topic})
    print("RunnableParallel 示例已启动。")
    print(f"\n主题: {topic}")
    print("\n并行结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
