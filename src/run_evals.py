import json
import sys

from agent import LearningAgent
from config import load_settings
from evals import EvalRunner


def main() -> None:
    """运行主项目内置评测，并输出汇总结果。"""
    settings = load_settings()
    agent = LearningAgent(settings)
    runner = EvalRunner(agent)

    case_file = sys.argv[1] if len(sys.argv) > 1 else "evals/agent_eval_cases.json"
    cases = runner.load_cases(case_file)
    report = runner.run(cases)

    print("评测完成。")
    print(f"总样例数: {report['total_cases']}")
    print(f"通过样例数: {report['passed_cases']}")
    print(f"通过率: {report['pass_rate']:.2%}")
    print("\n逐条结果:")
    for item in report["cases"]:
        print(f"- {item['case_id']}: {'PASS' if item['passed'] else 'FAIL'}")

    print("\n结果文件:")
    print("data/evals/latest_eval_report.json")


if __name__ == "__main__":
    main()
