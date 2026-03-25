import json

from agent import LearningAgent
from config import load_settings
from knowledge_qa import KnowledgeQARequest
from study_plan import parse_study_plan_request
from task_breakdown import parse_task_breakdown_request


def main() -> None:
    """启动命令行版学习助理，并处理基础命令和工具命令。"""
    settings = load_settings()
    agent = LearningAgent(settings)

    print("个人学习助理 Agent 已启动。")
    print(
        "输入 exit 退出，输入 /clear 清空对话记忆，"
        "输入 /plan 主题 | 当前基础 | 学习天数 | 学习目标 生成结构化学习计划，"
        "输入 /breakdown 目标 | 当前基础 | 可用天数 | 输出风格 生成结构化任务拆解，"
        "输入 /qa 问题 运行主项目版知识点问答，"
        "输入 /notes 查看知识点文件，输入 /read 文件名 读取笔记，"
        "输入 /tools 查看工具 schema，输入 /tool 工具名 JSON参数 手动执行工具，"
        "输入 /study 问题 让程序自动选知识点并回答，"
        "输入 /agent 问题 让模型自己决定是否调用工具，"
        "输入 /react 问题 运行 ReAct Agent。"
    )

    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            print("请输入问题。")
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("已退出。")
            break

        if user_input == "/clear":
            agent.clear_history()
            print("已清空当前对话记忆。")
            continue

        if user_input.startswith("/plan "):
            raw_request = user_input[6:].strip()
            if not raw_request:
                print("请在 /plan 后面补充内容，例如 /plan LangChain | 零基础 | 5 | 做出一个可运行 Demo")
                continue

            try:
                request = parse_study_plan_request(raw_request)
            except ValueError as exc:
                print(f"学习计划请求不合法：{exc}")
                continue

            plan = agent.create_study_plan(request)
            print("\nAgent(JSON):")
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            continue

        if user_input.startswith("/breakdown "):
            raw_request = user_input[11:].strip()
            if not raw_request:
                print("请在 /breakdown 后面补充内容，例如 /breakdown 做一个个人学习助理 | 零基础 | 14 | 可执行步骤")
                continue

            try:
                request = parse_task_breakdown_request(raw_request)
            except ValueError as exc:
                print(f"任务拆解请求不合法：{exc}")
                continue

            result = agent.create_task_breakdown(request)
            print("\nAgent(JSON):")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            continue

        if user_input.startswith("/qa "):
            question = user_input[4:].strip()
            if not question:
                print("请在 /qa 后面补充问题。")
                continue

            try:
                result = agent.answer_knowledge_question(
                    KnowledgeQARequest(question=question)
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"知识点问答执行失败：{exc}")
                continue

            print("\n本次选中的知识点文件:")
            for file_name in result["selected_notes"]:
                print(f"- {file_name}")
            print("\n命中的资料片段:")
            for chunk in result["retrieved_chunks"]:
                print(
                    f"- {chunk['file_name']} "
                    f"({chunk['chunk_start']}, {chunk['chunk_end']})"
                )
            print(f"\nAgent(QA): {result['answer']}")
            continue

        if user_input == "/notes":
            notes = agent.list_notes_tool()
            print("\n知识点文件:")
            for note in notes:
                print(f"- {note}")
            continue

        if user_input == "/tools":
            print("\n工具 Schema:")
            print(json.dumps(agent.get_tool_schemas(), ensure_ascii=False, indent=2))
            continue

        if user_input.startswith("/read "):
            file_name = user_input[6:].strip()
            if not file_name:
                print("请在 /read 后面补充文件名，例如 /read 13-输出解析器.md")
                continue

            try:
                content = agent.read_note_tool(file_name)
            except (FileNotFoundError, ValueError) as exc:
                print(f"读取失败：{exc}")
                continue

            print(f"\n文件内容：{file_name}\n")
            print(content)
            continue

        if user_input.startswith("/tool "):
            payload = user_input[6:].strip()
            if not payload:
                print('请使用 /tool 工具名 JSON参数，例如 /tool read_note {"file_name":"13-输出解析器.md"}')
                continue

            tool_name, separator, raw_arguments = payload.partition(" ")
            if not separator:
                print('请补充 JSON 参数，例如 /tool read_note {"file_name":"13-输出解析器.md"}')
                continue

            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                print(f"工具参数不是合法 JSON：{exc}")
                continue

            try:
                result = agent.execute_tool(tool_name, arguments)
            except (FileNotFoundError, ValueError) as exc:
                print(f"工具执行失败：{exc}")
                continue

            print("\n工具执行结果:")
            if isinstance(result, str):
                print(result)
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            continue

        if user_input.startswith("/study "):
            question = user_input[7:].strip()
            if not question:
                print("请在 /study 后面补充问题。")
                continue

            try:
                file_name, answer = agent.answer_with_note_tool(question)
            except (FileNotFoundError, ValueError) as exc:
                print(f"工具型问答执行失败：{exc}")
                continue

            print(f"\n本次自动选择的知识点文件: {file_name}")
            print(f"\nAgent(Tool): {answer}")
            continue

        if user_input.startswith("/agent "):
            question = user_input[7:].strip()
            if not question:
                print("请在 /agent 后面补充问题。")
                continue

            try:
                result = agent.run_tool_calling_agent(question)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Tool Calling Agent 执行失败：{exc}")
                continue

            print("\nTool Calling 步骤:")
            for step in result["steps"]:
                print(f"- 第 {step['step']} 步: {step['tool_name']} {step['arguments']}")
                print(f"  决策原因: {step['reason']}")
            print(f"\nAgent(Tool Calling): {result['answer']}")
            continue

        if user_input.startswith("/react "):
            question = user_input[7:].strip()
            if not question:
                print("请在 /react 后面补充问题。")
                continue

            try:
                result = agent.run_react_agent(question)
            except (FileNotFoundError, ValueError) as exc:
                print(f"ReAct Agent 执行失败：{exc}")
                continue

            print("\nReAct 步骤:")
            for step in result["steps"]:
                print(f"- 第 {step['step']} 步")
                print(f"  Thought: {step['thought']}")
                print(f"  Action: {step['tool_name']} {step['arguments']}")
            print(f"\nAgent(ReAct): {result['answer']}")
            continue

        answer = agent.reply(user_input)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()
