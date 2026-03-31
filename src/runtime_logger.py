import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RuntimeLogger:
    """把主项目运行过程按 JSONL 形式落到本地文件。"""

    def __init__(self, log_directory: str) -> None:
        """初始化日志目录和默认日志文件路径。"""
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_directory / "agent_runtime.jsonl"

    def log_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        session_id: str = "default",
    ) -> None:
        """记录一条结构化事件日志。"""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "event_type": event_type,
            "payload": payload,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
