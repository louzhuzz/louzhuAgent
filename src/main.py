import json

from agent import LearningAgent
from app_runtime import handle_user_input, render_cli_result
from config import load_settings
from langsmith_observer import LangSmithObserver
from runtime_logger import RuntimeLogger


def main() -> None:
    """启动命令行版学习助理，并处理基础命令和工具命令。"""
    settings = load_settings()
    agent = LearningAgent(settings)
    logger = RuntimeLogger(settings.log_directory)
    observer = LangSmithObserver.from_settings(settings)

    print("个人学习助理 Agent 已启动。")
    print(
        "输入 exit 退出，输入 /clear 清空对话记忆，"
        "输入 /plan 主题 | 当前基础 | 学习天数 | 学习目标 生成结构化学习计划，"
        "输入 /breakdown 目标 | 当前基础 | 可用天数 | 输出风格 生成结构化任务拆解，"
        "输入 /qa 问题 运行主项目版知识点问答，"
        "输入 /v1 问题 运行个人学习助理 Agent v1 统一入口，"
        "输入 /notes 查看知识点文件，输入 /read 文件名 读取笔记，"
        "输入 /tools 查看工具 schema，输入 /tool 工具名 JSON参数 手动执行工具，"
        "输入 /study 问题 让程序自动选知识点并回答，"
        "输入 /agent 问题 让模型自己决定是否调用工具，"
        "输入 /react 问题 运行 ReAct Agent。"
    )

    while True:
        user_input = input("\n你: ").strip()
        result = handle_user_input(
            agent,
            user_input,
            logger=logger,
            observer=observer,
            session_id="cli",
        )
        print(render_cli_result(result))
        if result["status"] == "exit":
            break


if __name__ == "__main__":
    main()
