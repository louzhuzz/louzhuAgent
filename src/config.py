import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    api_key: str
    base_url: str
    model: str
    embedding_model: str | None


def load_settings() -> Settings:
    api_key = os.getenv("ARK_API_KEY", "").strip()
    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip()
    model = os.getenv("ARK_MODEL", "").strip()
    embedding_model = os.getenv("ARK_EMBEDDING_MODEL", "").strip() or None

    if not api_key:
        raise ValueError("缺少 ARK_API_KEY，请先配置 .env 文件。")

    if not model:
        raise ValueError("缺少 ARK_MODEL，请先配置 .env 文件。")

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        embedding_model=embedding_model,
    )


def require_embedding_model(settings: Settings) -> str:
    if not settings.embedding_model:
        raise ValueError("缺少 ARK_EMBEDDING_MODEL，请先在 .env 中配置真实 embedding 模型或接入点。")
    return settings.embedding_model
