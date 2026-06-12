"""テスト用フェイク (DB / Ollama なしで動く)。"""

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
