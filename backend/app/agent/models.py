"""チャットモデルのファクトリ (ChatOllama 生成の唯一の場所)。

- "chat": gpt-oss — 自由文生成・ツール ReAct (responder/executor/synthesizer)
- "control": qwen3 — 構造化判断系 (orchestrator/planner/evaluator)。
  gpt-oss は構造化出力が不安定 (langchain#33116) なため判断系には使わない。

tags=["nostream"] を渡すと langgraph の StreamMessagesHandler が発生源で
トークン配信を抑制する (TAG_NOSTREAM)。
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from app.core.config import Settings


def default_model_factory(kind: str, settings: Settings, tags: list[str]) -> BaseChatModel:
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
