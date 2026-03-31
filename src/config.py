import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    """把环境变量解析成布尔值。"""
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    api_key: str
    base_url: str
    model: str
    embedding_model: str | None
    chroma_persist_directory: str
    log_directory: str
    langsmith_tracing: bool
    langsmith_project: str


def load_settings() -> Settings:
    api_key = os.getenv("ARK_API_KEY", "").strip()
    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip()
    model = os.getenv("ARK_MODEL", "").strip()
    embedding_model = os.getenv("ARK_EMBEDDING_MODEL", "").strip() or None
    chroma_persist_directory = os.getenv(
        "CHROMA_PERSIST_DIRECTORY",
        "data/chroma/knowledge_qa",
    ).strip()
    log_directory = os.getenv("LOG_DIRECTORY", "data/logs").strip()
    langsmith_tracing = _env_bool("LANGSMITH_TRACING", default=False)
    langsmith_project = os.getenv("LANGSMITH_PROJECT", "personal-learning-agent").strip()

    if not api_key:
        raise ValueError("缺少 ARK_API_KEY，请先配置 .env 文件。")

    if not model:
        raise ValueError("缺少 ARK_MODEL，请先配置 .env 文件。")

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        embedding_model=embedding_model,
        chroma_persist_directory=chroma_persist_directory,
        log_directory=log_directory,
        langsmith_tracing=langsmith_tracing,
        langsmith_project=langsmith_project,
    )


def require_embedding_model(settings: Settings) -> str:
    if not settings.embedding_model:
        raise ValueError("缺少 ARK_EMBEDDING_MODEL，请先在 .env 中配置真实 embedding 模型或接入点。")
    return settings.embedding_model
