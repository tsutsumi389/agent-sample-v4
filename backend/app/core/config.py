"""アプリ設定 (pydantic-settings)。環境変数は APP_ プレフィックス、.env はリポジトリルート。"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py -> core -> app -> backend -> リポジトリルート
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://postgres:postgres@localhost:5432/agent"
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "gpt-oss"
    memory_model: str = "qwen3"
    embed_model: str = "nomic-embed-text"
    num_ctx: int = 32768
    reasoning_effort: str = "medium"
    cors_origins: list[str] = ["http://localhost:5173"]
    mcp_config_path: str = "mcp_servers.json"
    reflection_delay_seconds: int = 30
    # テスト用: lifespan での pool/agent/MCP 構築をスキップする (APP_SKIP_STARTUP=1)
    skip_startup: bool = False

    @property
    def mcp_config_file(self) -> Path:
        path = Path(self.mcp_config_path)
        return path if path.is_absolute() else BACKEND_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
