"""チャットモデルのファクトリ (ChatModel 生成の唯一の場所)。

プロバイダは settings.llm_provider で切り替わる:

- "ollama" (既定): ローカル LLM。
  - "chat": gpt-oss — 自由文生成・ツール ReAct (responder/executor/synthesizer)
  - "control": qwen3 — 判断系 (orchestrator/planner/evaluator)。
    gpt-oss/qwen3 は構造化出力が不安定 (langchain#33116) なため、判断系は
    テキストパース (parsing.parse_with_retry / 1語+正規表現) で堅牢化する。
- "openai": ChatGPT / OpenAI互換API。
  - 判断系は Structured Outputs (with_structured_output) でスキーマ準拠を保証できる
    (settings.supports_structured_output == True)。

tags=["nostream"] を渡すと langgraph の StreamMessagesHandler が発生源で
トークン配信を抑制する (TAG_NOSTREAM)。
"""

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings


def _ollama_model(kind: str, settings: Settings, tags: list[str]) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    if kind == "control":
        return ChatOllama(
            model=settings.control_model,
            base_url=settings.ollama_base_url,
            num_ctx=settings.control_num_ctx,
            reasoning=False,  # qwen3 の thinking トグルは boolean
            temperature=0,
            tags=tags,
            validate_model_on_init=True,
        )
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        num_ctx=settings.num_ctx,
        reasoning=settings.reasoning_effort,  # gpt-oss: "low" / "medium" / "high"
        temperature=0,
        tags=tags,
        validate_model_on_init=True,
    )


def _openai_model(kind: str, settings: Settings, tags: list[str]) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    model = settings.openai_control_model if kind == "control" else settings.openai_chat_model
    # base_url=None は OpenAI 本家。互換エンドポイント利用時のみ設定する。
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
        temperature=0,
        tags=tags,
    )


def default_model_factory(kind: str, settings: Settings, tags: list[str]) -> BaseChatModel:
    if settings.llm_provider == "openai":
        return _openai_model(kind, settings, tags)
    return _ollama_model(kind, settings, tags)
