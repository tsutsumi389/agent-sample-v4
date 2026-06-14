"""テスト用フェイク (DB / Ollama なしで動く)。"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage


class ScriptedModel:
    """ainvoke ごとにスクリプトを順に返すフェイクチャットモデル。

    要素が Exception ならその呼び出しで raise する。スクリプトが尽きたら raise。
    """

    def __init__(self, outputs: list[str | Exception]) -> None:
        self.outputs = list(outputs)
        self.calls: list[Any] = []

    async def ainvoke(self, messages: Any, config: Any = None) -> AIMessage:
        self.calls.append(messages)
        if not self.outputs:
            raise RuntimeError("ScriptedModel のスクリプトが尽きました")
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return AIMessage(content=out)


class ScriptedExecutorAgent:
    """executor_agent.ainvoke のフェイク。スクリプトの文字列を最終 AIMessage として返す。"""

    def __init__(self, outputs: list[str | Exception]) -> None:
        self.outputs = list(outputs)
        self.configs: list[Any] = []

    async def ainvoke(self, payload: dict, config: Any = None) -> dict:
        self.configs.append(config)
        if not self.outputs:
            raise RuntimeError("ScriptedExecutorAgent のスクリプトが尽きました")
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return {"messages": [*payload.get("messages", []), AIMessage(content=out)]}


@dataclass
class FakeSearchItem:
    """langgraph BaseStore の SearchItem を模したフェイク。"""

    key: str
    value: Any
    score: float | None = None
    namespace: tuple = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None


class FakeStore:
    """BaseStore のインメモリフェイク (DB / 埋め込みなしで動く)。

    asearch はクエリの意味的類似を再現しない (埋め込みがないため) 代わりに、
    namespace 内の全アイテムを score 降順で返す。削除が検索結果へ反映されるため、
    verify_forgotten の Absence 検証 (削除後に残っていないか) をテストできる。
    adelete は呼び出しを記録し、対象アイテムを取り除く。
    """

    def __init__(self) -> None:
        self._ns: dict[tuple, dict[str, FakeSearchItem]] = {}
        self.deleted: list[tuple[tuple, str]] = []
        self.search_calls: list[tuple[tuple, str | None, int]] = []

    def put_item(
        self, namespace: tuple, key: str, value: Any, score: float | None = None
    ) -> None:
        item = FakeSearchItem(
            key=key,
            value=value,
            score=score,
            namespace=tuple(namespace),
            updated_at=datetime.now(timezone.utc),
        )
        self._ns.setdefault(tuple(namespace), {})[key] = item

    async def asearch(
        self,
        namespace: tuple,
        *,
        query: str | None = None,
        filter: Any = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[FakeSearchItem]:
        self.search_calls.append((tuple(namespace), query, limit))
        items = list(self._ns.get(tuple(namespace), {}).values())
        items.sort(key=lambda i: (i.score is not None, i.score or 0.0), reverse=True)
        return items[offset : offset + limit]

    async def aget(self, namespace: tuple, key: str) -> FakeSearchItem | None:
        return self._ns.get(tuple(namespace), {}).get(key)

    async def adelete(self, namespace: tuple, key: str) -> None:
        self.deleted.append((tuple(namespace), key))
        self._ns.get(tuple(namespace), {}).pop(key, None)
