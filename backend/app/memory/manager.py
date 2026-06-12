"""LangMem バックグラウンド統合 (ReflectionExecutor + qwen3 抽出モデル)。"""

import logging
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.store.base import BaseStore
from langmem import ReflectionExecutor, create_memory_store_manager

from app.core.config import Settings
from app.memory.tools import MEMORY_NAMESPACE

logger = logging.getLogger(__name__)


def build_reflection_executor(store: BaseStore, settings: Settings) -> ReflectionExecutor:
    # qwen3 の thinking トグルは boolean (gpt-oss の effort 文字列は不可)
    extraction_model = ChatOllama(
        model=settings.memory_model,
        base_url=settings.ollama_base_url,
        reasoning=False,
        num_ctx=8192,
        temperature=0,
    )
    manager = create_memory_store_manager(
        extraction_model,
        namespace=MEMORY_NAMESPACE,
        enable_inserts=True,
        store=store,
    )
    return ReflectionExecutor(manager, store=store)


def schedule_reflection(
    executor: Any,
    *,
    user_id: str,
    thread_id: str,
    messages: list[BaseMessage],
    after_seconds: int,
) -> None:
    """ターン完了後の記憶統合を debounce 付きでスケジュールする (best-effort)。

    thread_id を executor.submit へ渡すことで、同一スレッドの保留中タスクが
    キャンセルされる (debounce)。キャンセルされたターンの内容が失われないよう、
    呼び出し側はそのターンの差分ではなくスレッドの全会話メッセージを渡すこと。
    """
    try:
        executor.submit(
            {"messages": messages},
            config={"configurable": {"langgraph_user_id": user_id}},
            thread_id=thread_id,
            after_seconds=after_seconds,
        )
    except Exception:
        logger.exception(
            "リフレクションのスケジュールに失敗しました (user_id=%s, thread_id=%s)",
            user_id,
            thread_id,
        )
