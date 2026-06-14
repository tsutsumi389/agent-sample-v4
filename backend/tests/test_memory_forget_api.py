"""忘却 API (POST /api/memory/forget/preview, /confirm) のテスト。"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.fakes import FakeStore

NS = ("memories", "default-user")


def _client_with_store(store: FakeStore) -> TestClient:
    client = TestClient(create_app())
    client.__enter__()  # lifespan を起動 (skip_startup により store=None)
    client.app.state.store = store  # フェイク Store を差し込む
    return client


def _seed() -> FakeStore:
    store = FakeStore()
    store.put_item(NS, "k1", {"content": "残業が多い"}, score=0.9)
    store.put_item(NS, "k2", {"content": "上司はAさん"}, score=0.8)
    return store


class TestForgetPreview:
    def test_preview_returns_candidates_without_deleting(self):
        store = _seed()
        client = _client_with_store(store)
        try:
            res = client.post(
                "/api/memory/forget/preview",
                json={"user_id": "default-user", "query": "仕事"},
            )
        finally:
            client.__exit__(None, None, None)
        assert res.status_code == 200
        body = res.json()
        keys = [c["key"] for c in body["candidates"]]
        assert keys == ["k1", "k2"]
        assert store.deleted == []  # preview は破壊しない


class TestForgetConfirm:
    def test_confirm_deletes_verifies_and_reports(self):
        store = _seed()
        client = _client_with_store(store)
        try:
            res = client.post(
                "/api/memory/forget/confirm",
                json={"user_id": "default-user", "keys": ["k1", "k2"]},
            )
        finally:
            client.__exit__(None, None, None)
        assert res.status_code == 200
        body = res.json()
        assert body["deleted_count"] == 2
        assert body["verified"] is True
        assert body["leaked_keys"] == []
        assert {k for _, k in store.deleted} == {"k1", "k2"}

    def test_confirm_with_no_keys_deletes_nothing(self):
        store = _seed()
        client = _client_with_store(store)
        try:
            res = client.post(
                "/api/memory/forget/confirm",
                json={"user_id": "default-user", "keys": []},
            )
        finally:
            client.__exit__(None, None, None)
        assert res.status_code == 200
        assert res.json()["deleted_count"] == 0
        assert store.deleted == []

    def test_confirm_rejects_too_many_keys(self):
        # 大量削除の安全弁: DEFAULT_MAX_KEYS 超過は 422 で拒否し、何も削除しない
        from app.memory.forget import DEFAULT_MAX_KEYS

        store = _seed()
        client = _client_with_store(store)
        try:
            res = client.post(
                "/api/memory/forget/confirm",
                json={
                    "user_id": "default-user",
                    "keys": [f"k{i}" for i in range(DEFAULT_MAX_KEYS + 1)],
                },
            )
        finally:
            client.__exit__(None, None, None)
        assert res.status_code == 422
        assert store.deleted == []
