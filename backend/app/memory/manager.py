"""LangMem バックグラウンド統合 (ReflectionExecutor + qwen3 抽出モデル)。"""

import logging
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.store.base import BaseStore
from langmem import ReflectionExecutor, create_memory_store_manager

from app.core.config import Settings
from app.memory.profile import PROFILE_NAMESPACE, UserProfile
from app.memory.tools import MEMORY_NAMESPACE

logger = logging.getLogger(__name__)

# 意味記憶 (プロファイル) 抽出の指針。一時的な話題はエピソード記憶側に任せる。
PROFILE_INSTRUCTIONS = """\
あなたはユーザーの「意味記憶 (安定したプロファイル)」を維持する担当です。
会話から、名前・職業・恒常的な好み・守るべき制約・好む回答文体といった、長期的に
安定した属性のみを抽出し、単一のプロファイルを更新してください。
一時的な話題・その場限りの出来事・作業ログは対象外です (それらは別系統が扱います)。
既存のプロファイルが古い・誤っている場合は、新しい情報で上書き更新してください。"""


def _build_extraction_model(settings: Settings) -> ChatOllama:
    # qwen3 の thinking トグルは boolean (gpt-oss の effort 文字列は不可)
    return ChatOllama(
        model=settings.memory_model,
        base_url=settings.ollama_base_url,
        reasoning=False,
        num_ctx=8192,
        temperature=0,
    )


def build_reflection_executor(store: BaseStore, settings: Settings) -> ReflectionExecutor:
    """エピソード記憶 (自由テキスト Collection) のバックグラウンド統合 executor。"""
    manager = create_memory_store_manager(
        _build_extraction_model(settings),
        namespace=MEMORY_NAMESPACE,
        enable_inserts=True,
        store=store,
    )
    return ReflectionExecutor(manager, store=store)


def build_profile_reflection_executor(
    store: BaseStore, settings: Settings
) -> ReflectionExecutor:
    """意味記憶 (構造化ユーザープロファイル) のバックグラウンド統合 executor。

    enable_inserts=False + 単一スキーマ + default で「default キーの 1 ドキュメントを
    update し続ける」Profile パターンにする (UUID 乱立する Collection と対比)。
    """
    manager = create_memory_store_manager(
        _build_extraction_model(settings),
        namespace=PROFILE_NAMESPACE,
        schemas=[UserProfile],
        instructions=PROFILE_INSTRUCTIONS,
        default=UserProfile(),
        enable_inserts=False,
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
