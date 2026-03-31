from contextlib import contextmanager, nullcontext
from typing import Iterator

from config import Settings

try:
    import langsmith as ls
except ImportError:  # pragma: no cover - 依赖缺失时走降级
    ls = None


class LangSmithObserver:
    """为主项目提供可选开启的 LangSmith tracing 包装层。"""

    def __init__(self, enabled: bool, project_name: str) -> None:
        """保存 tracing 开关和项目名。"""
        self.enabled = enabled
        self.project_name = project_name

    @classmethod
    def from_settings(cls, settings: Settings) -> "LangSmithObserver":
        """根据项目配置创建 LangSmith 观察器。"""
        return cls(
            enabled=settings.langsmith_tracing,
            project_name=settings.langsmith_project,
        )

    @contextmanager
    def request_context(self, session_id: str, user_input: str) -> Iterator[None]:
        """为单次请求建立可选的 tracing 上下文。

        注意：
        1. 这里不会替代本地 JSONL 日志，而是补充一个外部可视化追踪层。
        2. 如果没有安装 `langsmith`，或没有开启 tracing，就自动降级成空上下文。
        """
        if not self.enabled or ls is None:
            with nullcontext():
                yield
            return

        with ls.tracing_context(
            enabled=True,
            project_name=self.project_name,
        ):
            yield
