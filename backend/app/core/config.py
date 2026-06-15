"""アプリ設定 (pydantic-settings)。環境変数は APP_ プレフィックス、.env はリポジトリルート。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
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
    # LLM プロバイダ: "ollama" (ローカル) / "openai" (ChatGPT・OpenAI互換API)。
    # openai は構造化出力 (Structured Outputs) をネイティブ対応するため、判断系ノード
    # (orchestrator/planner/evaluator) でテキストパースの代わりに with_structured_output を使う。
    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "gpt-oss"
    memory_model: str = "qwen3"
    embed_model: str = "nomic-embed-text"
    num_ctx: int = 32768
    reasoning_effort: str = "medium"
    # ---- OpenAI / ChatGPT (llm_provider="openai" のとき使用) ----
    openai_api_key: str = ""
    # OpenAI 互換エンドポイント (Azure OpenAI / ローカル LLM サーバ等) を使う場合のみ設定。
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"  # responder/executor/synthesizer (自由文生成)
    openai_control_model: str = "gpt-4o-mini"  # orchestrator/planner/evaluator (構造化判断)
    cors_origins: list[str] = ["http://localhost:5173"]
    mcp_config_path: str = "mcp_servers.json"
    reflection_delay_seconds: int = 30

    # ---- マルチエージェントグラフ ----
    # 構造化判断系 (orchestrator/planner/evaluator) のモデル。memory_model はリフレクション用に温存。
    control_model: str = "qwen3"
    control_num_ctx: int = 8192
    # ハードリミット (無限ループ防止)
    max_plan_steps: int = Field(5, ge=1)  # 計画ステップ数上限 (超過切捨て)
    max_step_retries: int = Field(1, ge=0)  # 同一ステップの retry 上限
    max_replans: int = Field(1, ge=0)  # planner 再突入上限
    max_executor_runs: int = Field(8, ge=1)  # 大域停止条件 (実行ステップ数の累計上限)
    max_parallel_executors: int = Field(3, ge=1)  # 1ラウンドで同時実行する executor 数の上限
    executor_recursion_limit: int = Field(12, ge=2)  # executor サブグラフ内 ReAct 反復上限
    graph_recursion_limit: int = Field(80, ge=10)  # 親グラフ recursion_limit (二重防御)
    # コンテキスト管理 (num_ctx に収めるための切詰め)
    router_skip_under_chars: int = 20  # これ未満の入力は LLM 分類スキップで direct
    goal_max_chars: int = 2000
    step_result_max_chars: int = 1500
    executor_history_max_chars: int = 6000
    feedback_max_chars: int = 800
    # テスト用: lifespan での pool/agent/MCP 構築をスキップする (APP_SKIP_STARTUP=1)
    skip_startup: bool = False

    @model_validator(mode="after")
    def _derive_graph_recursion_limit(self) -> "Settings":
        # 予算超過時に GraphRecursionError ではなく必ず synthesizer 経由の graceful な
        # 打ち切りになるよう、親グラフの recursion_limit を予算から導出した下限で補正する。
        # 1 executor 通過 ≒ executor+evaluator の 2 superstep。orchestrator/planner/synthesizer
        # と replan 分を加えた余裕を持たせる。
        needed = 2 * self.max_executor_runs + self.max_replans + 10
        if self.graph_recursion_limit < needed:
            self.graph_recursion_limit = needed
        return self

    @property
    def supports_structured_output(self) -> bool:
        """構造化出力 (with_structured_output) を信頼して使えるプロバイダか。

        openai は Structured Outputs (strict json_schema) でスキーマ準拠を保証する。
        ollama (gpt-oss/qwen3) は不安定 (langchain#33116) なためテキストパースに委ねる。
        """
        return self.llm_provider == "openai"

    @property
    def mcp_config_file(self) -> Path:
        path = Path(self.mcp_config_path)
        return path if path.is_absolute() else BACKEND_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
