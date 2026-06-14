"""ノード共通ヘルパー。"""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer

from app.agent.parsing import content_to_text

# executor が例外・反復上限で打ち切られたことを evaluator に伝えるマーカー
EXECUTION_FAILED_MARKER = "(実行打ち切り: "


def safe_stream_writer() -> Callable[[Any], None]:
    """get_stream_writer はランタイム外で例外を投げるため、ユニットテストでも安全な no-op を返す。"""
    try:
        return get_stream_writer()
    except Exception:
        return lambda _payload: None


def last_human_text(state: dict) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return content_to_text(msg.content)
    return ""


def sanitize_dependencies(plan: list[dict]) -> None:
    """plan の depends_on を破壊的に正規化する (planner 出力の堅牢化)。

    - 自己参照・重複・存在しない id 参照を除去。
    - 循環があれば Kahn 法で検出し、循環に残ったノードの依存をクリアして前進可能にする
      (循環は永遠に ready にならず停止性は routing で保たれるが、結果が出ないため壊す)。
    """
    by_id = {s["id"]: s for s in plan}
    for s in plan:
        s["depends_on"] = [
            d for d in dict.fromkeys(s.get("depends_on") or []) if d in by_id and d != s["id"]
        ]
    indeg = {sid: len(s["depends_on"]) for sid, s in by_id.items()}
    nexts: dict[int, list[int]] = {sid: [] for sid in by_id}
    for s in plan:
        for d in s["depends_on"]:
            nexts[d].append(s["id"])
    queue = [sid for sid, deg in indeg.items() if deg == 0]
    seen = 0
    while queue:
        sid = queue.pop()
        seen += 1
        for nx in nexts[sid]:
            indeg[nx] -= 1
            if indeg[nx] == 0:
                queue.append(nx)
    if seen < len(by_id):
        # 循環あり: indeg>0 のノード (循環本体＋それに間接依存する下流ノード) の依存を
        # 安全側で全クリアし、全ステップを即時実行可能にする。下流の非循環依存も巻き込まれ
        # 順序保証は失うが、前進性 (必ず ready になる) を優先する。発火は planner(LLM) が
        # 循環を出した異常時のみ。
        for sid, deg in indeg.items():
            if deg > 0:
                by_id[sid]["depends_on"] = []


def ready_step_indices(plan: list[dict]) -> list[int]:
    """依存 (depends_on) が全て done で、まだ実行待ち (pending) のステップの index 群。"""
    done = {s["id"] for s in plan if s.get("status") == "done"}
    return [
        i
        for i, s in enumerate(plan)
        if s.get("status") == "pending" and all(d in done for d in (s.get("depends_on") or []))
    ]
