"""スコープ付き一括忘却の純粋関数 (副作用なし・TDD 対象)。

長期記憶の「忘れる機能」は、誤削除を防ぐため次の段階に分けて設計している:

  1. search_forget_candidates() で候補を検索 (store_query.py)
  2. select_candidates()        で上限・(任意)閾値により削除対象を選別 (本モジュール)
  3. build_confirmation_payload() で確認 UI 用に整形 (本モジュール)
  4. (ユーザー/エージェントの承認)
  5. batch_delete_memories()    で実削除 (store_query.py)
  6. verify_forgotten()         で削除済みか (key 直接照会で) 検証 (store_query.py)

ここに置くのは 2/3 の決定論的な純粋関数のみ。Store への I/O は store_query.py。
"""

from typing import Any

# 1 回の忘却で削除できる上限。暴走的な大量削除を防ぐ主たる安全弁。
# preview/confirm の API スキーマと forget ツールの双方がこの値を共有する。
DEFAULT_MAX_KEYS = 25


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    score_threshold: float | None = None,
    max_keys: int = DEFAULT_MAX_KEYS,
) -> list[dict[str, Any]]:
    """候補から削除対象を選別する。

    - 最大 ``max_keys`` 件にクリップする (大量削除の主たる安全弁)。
    - ``score_threshold`` は任意の補助フィルタ。指定された場合、それ未満のスコアの
      候補を除外する。ただしスコアが ``None`` (関連度不明) の候補は保守的に残す
      (誤って消さない)。閾値の最適値は Store の距離尺度に依存するため既定では無効
      (None) とし、必要に応じて呼び出し側が渡す。
    """
    selected: list[dict[str, Any]] = []
    for cand in candidates:
        score = cand.get("score")
        if score_threshold is not None and score is not None and score < score_threshold:
            continue
        selected.append(cand)
    return selected[:max_keys]


def build_confirmation_payload(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """削除前の確認用ペイロードを整形する (件数・key 一覧・人間可読な項目)。"""
    return {
        "count": len(candidates),
        "keys": [c["key"] for c in candidates],
        "items": [
            {
                "key": c["key"],
                "content": c.get("content", ""),
                "score": c.get("score"),
            }
            for c in candidates
        ],
    }
