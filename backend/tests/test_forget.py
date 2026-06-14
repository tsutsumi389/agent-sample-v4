"""忘却の純粋関数 (app/memory/forget.py) のテスト。"""

import pytest

from app.memory.forget import (
    DEFAULT_MAX_KEYS,
    build_confirmation_payload,
    select_candidates,
)


def _cand(key: str, content: str = "", score: float | None = None) -> dict:
    return {"key": key, "content": content, "score": score, "updated_at": None}


class TestSelectCandidates:
    def test_empty_input_returns_empty(self):
        assert select_candidates([]) == []

    def test_keeps_all_when_no_threshold(self):
        cands = [_cand("a", score=0.1), _cand("b", score=0.9)]
        assert [c["key"] for c in select_candidates(cands)] == ["a", "b"]

    def test_excludes_below_score_threshold(self):
        cands = [_cand("a", score=0.2), _cand("b", score=0.8)]
        out = select_candidates(cands, score_threshold=0.5)
        assert [c["key"] for c in out] == ["b"]

    def test_none_score_passes_threshold(self):
        # score 不明 (None) の記憶は閾値で誤って消さない (保守的に残す)
        cands = [_cand("a", score=None), _cand("b", score=0.1)]
        out = select_candidates(cands, score_threshold=0.5)
        assert [c["key"] for c in out] == ["a"]

    def test_clips_to_max_keys(self):
        cands = [_cand(str(i), score=1.0) for i in range(30)]
        out = select_candidates(cands, max_keys=5)
        assert len(out) == 5
        assert [c["key"] for c in out] == ["0", "1", "2", "3", "4"]

    def test_default_max_keys_caps_catastrophic_deletion(self):
        cands = [_cand(str(i), score=1.0) for i in range(DEFAULT_MAX_KEYS + 10)]
        assert len(select_candidates(cands)) == DEFAULT_MAX_KEYS


class TestBuildConfirmationPayload:
    def test_payload_shape(self):
        cands = [_cand("k1", "コーヒーが好き", 0.9), _cand("k2", "朝型である", 0.7)]
        payload = build_confirmation_payload(cands)
        assert payload["count"] == 2
        assert payload["keys"] == ["k1", "k2"]
        assert payload["items"][0] == {
            "key": "k1",
            "content": "コーヒーが好き",
            "score": 0.9,
        }

    def test_empty(self):
        payload = build_confirmation_payload([])
        assert payload == {"count": 0, "keys": [], "items": []}
