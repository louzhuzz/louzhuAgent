from project_metrics import ProjectMetricsBuilder


def main() -> None:
    """生成项目包装指标摘要，并把结果打印到终端。"""
    builder = ProjectMetricsBuilder()
    summary = builder.build()

    eval_metrics = summary["eval_metrics"]
    runtime_metrics = summary["runtime_metrics"]

    print("项目指标摘要已生成。")
    print(f"最近评测通过率: {eval_metrics['pass_rate']:.2%}")
    print(f"QA 预期笔记命中率: {eval_metrics['qa_note_hit_rate']:.2%}")
    print(f"Agent v1 意图匹配率: {eval_metrics['v1_intent_match_rate']:.2%}")
    print(f"累计用户输入数: {runtime_metrics['user_input_count']}")
    print("\n结果文件:")
    print("data/metrics/latest_project_metrics.json")
    print("data/metrics/latest_project_metrics.md")


if __name__ == "__main__":
    main()
