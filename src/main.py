import json

from agent import LearningAgent
from config import load_settings


def main() -> None:
    """启动命令行版学习助理，并处理基础命令和工具命令。"""
    settings = load_settings()
    agent = LearningAgent(settings)

    print("个人学习助理 Agent 已启动。")
    print(
        "输入 exit 退出，输入 /clear 清空对话记忆，"
        "输入 /plan 主题 生成结构化学习计划，"
        "输入 /notes 查看知识点文件，输入 /read 文件名 读取笔记，"
        "输入 /tools 查看工具 schema，输入 /tool 工具名 JSON参数 手动执行工具，"
        "输入 /study 问题 让程序自动选知识点并回答。"
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
            topic = user_input[6:].strip()
            if not topic:
                print("请在 /plan 后面补充学习主题。")
                continue

            plan = agent.create_study_plan(topic)
            print("\nAgent(JSON):")
            print(json.dumps(plan, ensure_ascii=False, indent=2))
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

        answer = agent.reply(user_input)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()
