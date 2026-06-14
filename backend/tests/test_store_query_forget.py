"""一括忘却の Store I/O (app/memory/store_query.py) のテスト。"""

import pytest

from app.memory.store_query import (
    batch_delete_memories,
    search_forget_candidates,
    verify_forgotten,
)
from tests.fakes import FakeStore

NS = ("memories", "default-user")


def _store_with(items: dict[str, tuple[object, float]]) -> FakeStore:
    store = FakeStore()
    for key, (value, score) in items.items():
        store.put_item(NS, key, value, score=score)
    return store


class TestSearchForgetCandidates:
    async def test_returns_candidate_shape(self):
        store = _store_with(
            {
                "k1": ({"content": "残業が多い"}, 0.9),
                "k2": ({"content": "コーヒーが好き"}, 0.3),
            }
        )
        cands = await search_forget_candidates(store, "default-user", "仕事", limit=10)
        # スコア降順
        assert [c["key"] for c in cands] == ["k1", "k2"]
        first = cands[0]
        assert first["key"] == "k1"
        assert first["content"] == "残業が多い"  # _to_text で入れ子を展開
        assert first["score"] == 0.9
        assert "updated_at" in first

    async def test_searches_correct_namespace_and_limit(self):
        store = _store_with({"k1": ({"content": "x"}, 0.5)})
        await search_forget_candidates(store, "alice", "q", limit=7)
        ns, query, limit = store.search_calls[-1]
        assert ns == ("memories", "alice")
        assert query == "q"
        assert limit == 7

    async def test_empty_when_no_memories(self):
        assert await search_forget_candidates(FakeStore(), "default-user", "q") == []


class TestBatchDeleteMemories:
    async def test_deletes_each_key(self):
        store = _store_with(
            {"k1": ({"content": "a"}, 1.0), "k2": ({"content": "b"}, 1.0)}
        )
        result = await batch_delete_memories(store, "default-user", ["k1", "k2"])
        assert result["deleted_count"] == 2
        assert result["deleted_keys"] == ["k1", "k2"]
        assert store.deleted == [(NS, "k1"), (NS, "k2")]

    async def test_empty_keys_is_noop(self):
        store = _store_with({"k1": ({"content": "a"}, 1.0)})
        result = await batch_delete_memories(store, "default-user", [])
        assert result == {"deleted_count": 0, "deleted_keys": []}
        assert store.deleted == []


class TestVerifyForgotten:
    async def test_ok_when_all_gone(self):
        store = _store_with(
            {"k1": ({"content": "残業"}, 0.9), "k2": ({"content": "趣味"}, 0.9)}
        )
        await batch_delete_memories(store, "default-user", ["k1"])
        result = await verify_forgotten(store, "default-user", ["k1"])
        assert result == {"ok": True, "leaked_keys": []}

    async def test_detects_leaked_key(self):
        # k1 を「削除したつもり」だが実際には残っている (= leak) ケース
        store = _store_with({"k1": ({"content": "残業"}, 0.9)})
        result = await verify_forgotten(store, "default-user", ["k1"])
        assert result["ok"] is False
        assert result["leaked_keys"] == ["k1"]

    async def test_empty_deleted_keys_is_ok(self):
        store = _store_with({"k1": ({"content": "x"}, 0.9)})
        result = await verify_forgotten(store, "default-user", [])
        assert result == {"ok": True, "leaked_keys": []}

    async def test_exact_lookup_not_topk_search(self):
        # 直接照会 (aget) なので、検索の top-k 窓の外にある残存記憶も漏れなく検知する。
        # FakeStore.asearch は put 順に依存しないが、aget は key を直接引くため確実。
        store = FakeStore()
        for i in range(60):
            store.put_item(NS, f"other{i}", {"content": "無関係"}, score=0.99)
        store.put_item(NS, "residual", {"content": "残った記憶"}, score=0.01)
        # residual を「削除したつもり」だが実際には残っている
        result = await verify_forgotten(store, "default-user", ["residual"])
        assert result["ok"] is False
        assert result["leaked_keys"] == ["residual"]
