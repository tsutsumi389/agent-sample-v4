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
