import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    api_key: str
    base_url: str
    model: str


def load_settings() -> Settings:
    api_key = os.getenv("ARK_API_KEY", "").strip()
    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip()
    model = os.getenv("ARK_MODEL", "").strip()

    if not api_key:
        raise ValueError("缺少 ARK_API_KEY，请先配置 .env 文件。")

    if not model:
        raise ValueError("缺少 ARK_MODEL，请先配置 .env 文件。")

    return Settings(api_key=api_key, base_url=base_url, model=model)
