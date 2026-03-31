import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProjectMetricsPaths:
    """描述项目指标摘要器需要读写的路径。"""

    eval_report_path: str = "data/evals/latest_eval_report.json"
    runtime_log_path: str = "data/logs/agent_runtime.jsonl"
    output_directory: str = "data/metrics"


class ProjectMetricsBuilder:
    """把评测结果和运行日志整理成更适合项目包装的指标摘要。"""

    def __init__(self, paths: ProjectMetricsPaths | None = None) -> None:
        """保存路径配置，并准备输出目录。"""
        self.paths = paths or ProjectMetricsPaths()
        self.output_directory = Path(self.paths.output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _load_json_file(self, file_path: str) -> dict[str, Any]:
        """读取一个 JSON 文件。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到文件：{file_path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_jsonl_records(self, file_path: str) -> list[dict[str, Any]]:
        """读取 JSONL 结构化日志。"""
        path = Path(file_path)
        if not path.exists():
            return []

        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
        return records

    def _build_eval_metrics(self, report: dict[str, Any]) -> dict[str, Any]:
        """从评测报告里提炼核心指标。"""
        cases = report.get("cases", [])
        qa_cases = [item for item in cases if item.get("mode") == "qa"]
        v1_cases = [item for item in cases if item.get("mode") == "v1"]

        qa_note_hit_rate = 0.0
        if qa_cases:
            hit_count = sum(1 for item in qa_cases if item.get("note_hit"))
            qa_note_hit_rate = round(hit_count / len(qa_cases), 4)

        v1_intent_match_rate = 0.0
        if v1_cases:
            hit_count = sum(1 for item in v1_cases if item.get("intent_match"))
            v1_intent_match_rate = round(hit_count / len(v1_cases), 4)

        v1_result_type_match_rate = 0.0
        if v1_cases:
            hit_count = sum(1 for item in v1_cases if item.get("result_type_match"))
            v1_result_type_match_rate = round(hit_count / len(v1_cases), 4)

        return {
            "generated_at": report.get("generated_at"),
            "total_cases": report.get("total_cases", 0),
            "passed_cases": report.get("passed_cases", 0),
            "pass_rate": report.get("pass_rate", 0.0),
            "qa_case_count": len(qa_cases),
            "qa_note_hit_rate": qa_note_hit_rate,
            "v1_case_count": len(v1_cases),
            "v1_intent_match_rate": v1_intent_match_rate,
            "v1_result_type_match_rate": v1_result_type_match_rate,
        }

    def _build_runtime_metrics(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """从运行日志中提炼调用规模和事件分布。"""
        if not records:
            return {
                "total_events": 0,
                "total_sessions": 0,
                "user_input_count": 0,
                "completion_event_count": 0,
                "error_event_count": 0,
                "top_events": [],
            }

        event_counter = Counter(item.get("event_type", "unknown") for item in records)
        session_ids = {item.get("session_id", "default") for item in records}
        user_input_count = event_counter.get("user_input_received", 0)
        completion_event_count = sum(
            count for event_type, count in event_counter.items() if event_type.endswith("_completed")
        )
        error_event_count = sum(
            count for event_type, count in event_counter.items() if event_type.endswith("_error")
        )

        return {
            "total_events": len(records),
            "total_sessions": len(session_ids),
            "user_input_count": user_input_count,
            "completion_event_count": completion_event_count,
            "error_event_count": error_event_count,
            "top_events": event_counter.most_common(10),
        }

    def _build_resume_bullets(
        self,
        eval_metrics: dict[str, Any],
        runtime_metrics: dict[str, Any],
    ) -> list[str]:
        """生成更适合简历和项目讲述的指标话术。"""
        bullets = [
            (
                f"构建主项目最小回归评测体系，当前固定样例 {eval_metrics['total_cases']} 条，"
                f"最近一次通过率 {eval_metrics['pass_rate']:.2%}。"
            ),
            (
                f"围绕知识点问答建立检索命中指标，当前 `qa` 样例的预期笔记命中率为 "
                f"{eval_metrics['qa_note_hit_rate']:.2%}。"
            ),
            (
                f"围绕 Agent v1 建立统一入口路由评测，当前意图匹配率为 "
                f"{eval_metrics['v1_intent_match_rate']:.2%}，结果类型匹配率为 "
                f"{eval_metrics['v1_result_type_match_rate']:.2%}。"
            ),
            (
                f"本地结构化日志已累计记录 {runtime_metrics['total_events']} 条事件、"
                f"{runtime_metrics['user_input_count']} 次用户输入，便于后续继续沉淀调用规模、错误分布和性能指标。"
            ),
        ]
        return bullets

    def build(self) -> dict[str, Any]:
        """生成完整项目指标摘要，并落盘保存。"""
        eval_report = self._load_json_file(self.paths.eval_report_path)
        runtime_records = self._load_jsonl_records(self.paths.runtime_log_path)

        eval_metrics = self._build_eval_metrics(eval_report)
        runtime_metrics = self._build_runtime_metrics(runtime_records)
        summary = {
            "eval_metrics": eval_metrics,
            "runtime_metrics": runtime_metrics,
            "resume_bullets": self._build_resume_bullets(eval_metrics, runtime_metrics),
        }

        json_path = self.output_directory / "latest_project_metrics.json"
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        markdown_lines = [
            "# 项目指标摘要",
            "",
            "## 核心指标",
            f"- 最近评测样例数：{eval_metrics['total_cases']}",
            f"- 最近评测通过率：{eval_metrics['pass_rate']:.2%}",
            f"- `qa` 预期笔记命中率：{eval_metrics['qa_note_hit_rate']:.2%}",
            f"- `v1` 意图匹配率：{eval_metrics['v1_intent_match_rate']:.2%}",
            f"- `v1` 结果类型匹配率：{eval_metrics['v1_result_type_match_rate']:.2%}",
            f"- 累计结构化日志事件数：{runtime_metrics['total_events']}",
            f"- 累计用户输入数：{runtime_metrics['user_input_count']}",
            "",
            "## 项目包装可直接使用的表达",
        ]
        for bullet in summary["resume_bullets"]:
            markdown_lines.append(f"- {bullet}")

        md_path = self.output_directory / "latest_project_metrics.md"
        md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
        return summary
