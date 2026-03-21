import json

from agent import LearningAgent
from config import load_settings


def main() -> None:
    settings = load_settings()
    agent = LearningAgent(settings)

    print("个人学习助理 Agent 已启动。")
    print("输入 exit 退出，输入 /clear 清空对话记忆，输入 /plan 主题 生成结构化学习计划。")

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

        answer = agent.reply(user_input)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()
