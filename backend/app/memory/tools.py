"""LangMem hot-path ツール (エージェントが会話中に記憶を保存/検索/忘却)。"""

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from app.memory.forget import (
    DEFAULT_MAX_KEYS,
    build_confirmation_payload,
    select_candidates,
)
from app.memory.store_query import (
    batch_delete_memories,
    search_forget_candidates,
    verify_forgotten,
)

# namespace は manager.py / store_query.py とバイト単位で一致させること (langmem#140)
MEMORY_NAMESPACE = ("memories", "{langgraph_user_id}")

# forget ツールが一度に検索する候補の上限。
FORGET_SEARCH_LIMIT = 30


def _user_id_from_config(config: RunnableConfig | None) -> str:
    """実行 config から langgraph_user_id を取り出す (hot-path ツールと同じ規約)。"""
    configurable = (config or {}).get("configurable", {}) if config else {}
    return configurable.get("langgraph_user_id", "default-user")


def build_forget_tool(store: BaseStore) -> BaseTool:
    """エージェント主導の「スコープ付き一括忘却」ツールを作る。

    破壊的・不可逆なので 2 段階プロトコルにする:
      - confirm=False (既定): 関連記憶の候補一覧を返すだけ。1 件も削除しない。
      - confirm=True:         候補を実削除し、削除後に Absence 検証して結果を返す。
    """

    @tool
    async def forget_memories(
        query: str,
        confirm: bool = False,
        config: RunnableConfig = None,
    ) -> str:
        """ユーザーが「〜を忘れて」「〜の話はもうしないで」と明示的に依頼したときだけ使う長期記憶の一括忘却ツール。

        まず confirm=False で呼び、削除候補の一覧をユーザーに提示して明示的な承認を得ること。
        ユーザーが承認したら、同じ query を confirm=True で渡して実際に削除する。
        承認なしに confirm=True を呼んではならない (削除は不可逆)。依頼が曖昧なときは対象を絞る質問をすること。

        Args:
            query: 忘れる対象を表す自然文 (例: "仕事", "私の住所")。
            confirm: True で実削除。False (既定) は候補提示のみ。
        """
        user_id = _user_id_from_config(config)
        candidates = await search_forget_candidates(
            store, user_id, query, limit=FORGET_SEARCH_LIMIT
        )
        selected = select_candidates(candidates, max_keys=DEFAULT_MAX_KEYS)
        if not selected:
            return f"「{query}」に関連する記憶は見つかりませんでした。"

        if not confirm:
            payload = build_confirmation_payload(selected)
            lines = "\n".join(f"- {it['content']}" for it in payload["items"])
            return (
                f"以下の{payload['count']}件の記憶を削除しようとしています。"
                "よろしければ承認してください (confirm=true で再実行)。\n"
                f"{lines}"
            )

        keys = [c["key"] for c in selected]
        result = await batch_delete_memories(store, user_id, keys)
        verification = await verify_forgotten(
            store, user_id, result["deleted_keys"]
        )
        if verification["ok"]:
            return f"{result['deleted_count']}件の記憶を忘れました。"
        return (
            f"{result['deleted_count']}件を削除しましたが、一部の記憶が残っている"
            f"可能性があります (leaked: {verification['leaked_keys']})。"
        )

    return forget_memories


def langmem_hotpath_tools(store: BaseStore) -> list[BaseTool]:
    return [
        create_manage_memory_tool(namespace=MEMORY_NAMESPACE, store=store),
        create_search_memory_tool(namespace=MEMORY_NAMESPACE, store=store),
        build_forget_tool(store),
    ]
