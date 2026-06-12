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
