import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent import LearningAgent
from knowledge_qa import KnowledgeQARequest


@dataclass
class EvalCase:
    """描述一条最小评测样例。"""

    case_id: str
    mode: str
    input_text: str
    expected_intent: str | None = None
    expected_result_type: str | None = None
    expected_notes: list[str] | None = None
    expected_keywords: list[str] | None = None


class EvalRunner:
    """执行当前主项目的最小回归评测。"""

    def __init__(self, agent: LearningAgent, output_directory: str = "data/evals") -> None:
        """保存 Agent 并准备评测结果输出目录。"""
        self.agent = agent
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def load_cases(self, file_path: str) -> list[EvalCase]:
        """从 JSON 文件加载评测样例。"""
        raw_cases = json.loads(Path(file_path).read_text(encoding="utf-8"))
        return [EvalCase(**item) for item in raw_cases]

    def _keyword_coverage(self, text: str, keywords: list[str]) -> float:
        """计算答案对关键词的覆盖率。"""
        if not keywords:
            return 1.0
        hit_count = sum(1 for keyword in keywords if keyword in text)
        return hit_count / len(keywords)

    def _evaluate_qa_case(self, case: EvalCase) -> dict[str, Any]:
        """执行一条知识点问答评测。"""
        result = self.agent.answer_knowledge_question(KnowledgeQARequest(question=case.input_text))
        selected_notes = result["selected_notes"]
        answer = str(result["answer"])

        expected_notes = case.expected_notes or []
        note_hit = not expected_notes or any(note in selected_notes for note in expected_notes)
        keyword_coverage = self._keyword_coverage(answer, case.expected_keywords or [])
        passed = note_hit and keyword_coverage >= 0.5

        return {
            "case_id": case.case_id,
            "mode": case.mode,
            "input_text": case.input_text,
            "passed": passed,
            "selected_notes": selected_notes,
            "expected_notes": expected_notes,
            "note_hit": note_hit,
            "keyword_coverage": round(keyword_coverage, 4),
            "answer_preview": answer[:200],
        }

    def _extract_v1_answer_text(self, result: dict[str, Any]) -> str:
        """从 Agent v1 结果里提取适合做关键词匹配的文本。"""
        result_type = result["result_type"]
        payload = result["result"]

        if result_type == "json":
            return json.dumps(payload, ensure_ascii=False)
        if result_type == "knowledge_qa":
            return str(payload["answer"])
        if result_type == "tool_agent":
            return str(payload["answer"])
        return str(payload)

    def _evaluate_v1_case(self, case: EvalCase) -> dict[str, Any]:
        """执行一条 Agent v1 统一入口评测。"""
        result = self.agent.run_agent_v1(case.input_text)
        answer_text = self._extract_v1_answer_text(result)

        intent_match = case.expected_intent is None or result["intent"] == case.expected_intent
        result_type_match = (
            case.expected_result_type is None
            or result["result_type"] == case.expected_result_type
        )
        note_hit = True
        selected_notes: list[str] = []
        if result["result_type"] == "knowledge_qa":
            selected_notes = result["result"]["selected_notes"]
            expected_notes = case.expected_notes or []
            note_hit = not expected_notes or any(note in selected_notes for note in expected_notes)

        keyword_coverage = self._keyword_coverage(answer_text, case.expected_keywords or [])
        passed = intent_match and result_type_match and note_hit and keyword_coverage >= 0.5

        return {
            "case_id": case.case_id,
            "mode": case.mode,
            "input_text": case.input_text,
            "passed": passed,
            "intent": result["intent"],
            "expected_intent": case.expected_intent,
            "intent_match": intent_match,
            "result_type": result["result_type"],
            "expected_result_type": case.expected_result_type,
            "result_type_match": result_type_match,
            "selected_notes": selected_notes,
            "expected_notes": case.expected_notes or [],
            "note_hit": note_hit,
            "keyword_coverage": round(keyword_coverage, 4),
            "answer_preview": answer_text[:200],
        }

    def evaluate_case(self, case: EvalCase) -> dict[str, Any]:
        """根据评测模式执行单条样例。

        评测体系不应该因为某一条样例异常就整体中断，所以这里会把异常转成失败结果。
        """
        try:
            if case.mode == "qa":
                return self._evaluate_qa_case(case)
            if case.mode == "v1":
                return self._evaluate_v1_case(case)
            raise ValueError(f"当前不支持的评测模式：{case.mode}")
        except Exception as exc:
            return {
                "case_id": case.case_id,
                "mode": case.mode,
                "input_text": case.input_text,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def run(self, cases: list[EvalCase]) -> dict[str, Any]:
        """执行一组评测，并生成汇总报告。"""
        case_results = [self.evaluate_case(case) for case in cases]
        total = len(case_results)
        passed = sum(1 for item in case_results if item["passed"])
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total_cases": total,
            "passed_cases": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "cases": case_results,
        }

        latest_path = self.output_directory / "latest_eval_report.json"
        latest_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report
