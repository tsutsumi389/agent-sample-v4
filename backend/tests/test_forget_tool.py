"""エージェント主導 forget ツール (app/memory/tools.py) のテスト。"""

import pytest

from app.memory.tools import build_forget_tool
from tests.fakes import FakeStore

NS = ("memories", "default-user")
CONFIG = {"configurable": {"langgraph_user_id": "default-user"}}


def _store() -> FakeStore:
    store = FakeStore()
    store.put_item(NS, "k1", {"content": "残業が多い"}, score=0.9)
    store.put_item(NS, "k2", {"content": "上司はAさん"}, score=0.8)
    return store


class TestForgetTool:
    async def test_preview_does_not_delete(self):
        # confirm 省略 (=False) では1件も削除せず候補のみ返す (誤削除防止)
        store = _store()
        tool = build_forget_tool(store)
        out = await tool.ainvoke({"query": "仕事"}, config=CONFIG)
        assert store.deleted == []
        assert "残業が多い" in out
        assert "上司はAさん" in out

    async def test_confirm_deletes_and_verifies(self):
        store = _store()
        tool = build_forget_tool(store)
        out = await tool.ainvoke(
            {"query": "仕事", "confirm": True}, config=CONFIG
        )
        assert {k for _, k in store.deleted} == {"k1", "k2"}
        assert "忘れ" in out

    async def test_no_candidates_message(self):
        store = FakeStore()
        tool = build_forget_tool(store)
        out = await tool.ainvoke(
            {"query": "存在しない話題", "confirm": True}, config=CONFIG
        )
        assert store.deleted == []
        assert "見つかりません" in out

    async def test_uses_user_id_from_config(self):
        store = FakeStore()
        store.put_item(("memories", "alice"), "a1", {"content": "秘密"}, score=0.9)
        tool = build_forget_tool(store)
        await tool.ainvoke(
            {"query": "秘密", "confirm": True},
            config={"configurable": {"langgraph_user_id": "alice"}},
        )
        assert store.deleted == [(("memories", "alice"), "a1")]
