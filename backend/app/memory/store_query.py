"""メモリパネル用の store 問い合わせヘルパ。"""

from datetime import datetime, timezone
from typing import Any

from langgraph.store.base import BaseStore


def _to_text(value: Any) -> str:
    """LangMem の値形状 ({"kind":..., "content": {...}} 等) を人間可読テキストへ。"""
    if isinstance(value, dict):
        if "content" in value:
            return _to_text(value["content"])
        return str(value)
    return str(value)


def _iso_z(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


async def list_memories(
    store: BaseStore, user_id: str, query: str | None, limit: int
) -> list[dict]:
    namespace = ("memories", user_id)
    if query:
        items = await store.asearch(namespace, query=query, limit=limit)
    else:
        # クエリ無しは更新日時の降順を保証する (asearch は順序を保証しない)
        items = await store.asearch(namespace, limit=limit + 50)
        items = sorted(items, key=lambda i: i.updated_at, reverse=True)[:limit]
    return [
        {
            "key": item.key,
            "content": _to_text(item.value),
            "namespace": list(item.namespace),
            "updated_at": _iso_z(item.updated_at),
            "score": item.score if query else None,
        }
        for item in items
    ]


async def delete_memory(store: BaseStore, user_id: str, key: str) -> None:
    await store.adelete(("memories", user_id), key)


async def search_forget_candidates(
    store: BaseStore, user_id: str, query: str, limit: int = 20
) -> list[dict]:
    """忘却対象の候補をスコープ検索する (セマンティック検索)。

    list_memories と違い「削除候補」用途なので、各候補に key/content/score/updated_at
    を付けて返す。実際に消すかは呼び出し側 (確認ゲート) が決める。
    """
    namespace = ("memories", user_id)
    items = await store.asearch(namespace, query=query, limit=limit)
    return [
        {
            "key": item.key,
            "content": _to_text(item.value),
            "score": item.score,
            "updated_at": _iso_z(item.updated_at),
        }
        for item in items
    ]


async def batch_delete_memories(
    store: BaseStore, user_id: str, keys: list[str]
) -> dict:
    """指定した key 群を一括削除する。Postgres 単一ストアなので孤児は発生しない。"""
    namespace = ("memories", user_id)
    deleted: list[str] = []
    for key in keys:
        await store.adelete(namespace, key)
        deleted.append(key)
    return {"deleted_count": len(deleted), "deleted_keys": deleted}


async def verify_forgotten(
    store: BaseStore, user_id: str, deleted_keys: list[str]
) -> dict:
    """削除が本当に効いたかを Absence 検証する。

    各 key を namespace から直接 ``aget`` し、まだ取得できる (=消えていない) ものを
    leaked として報告する。セマンティック検索の top-k に依存しないため、関連記憶が
    多いユーザーや汎用的な内容でも取りこぼさず確実に「消えたこと」を確認できる。
    """
    namespace = ("memories", user_id)
    leaked: list[str] = []
    for key in deleted_keys:
        if await store.aget(namespace, key) is not None:
            leaked.append(key)
    return {"ok": not leaked, "leaked_keys": leaked}
