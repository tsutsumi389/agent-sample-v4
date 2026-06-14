"""チャット内GUI(生成的UI)のUIリソース封筒(エンベロープ)。

設計書: docs/guide/11-generative-ui.html を参照。

UIリソースは LangChain 標準の ``ToolMessage.artifact`` を唯一の封筒とする。
新しいデコレータや基底クラスは発明しない。UIツールは

    @tool(response_format="content_and_artifact")
    def render_table(...):
        return ui("table", summary="...", title=..., columns=..., rows=...)

のように ``(summary, 封筒)`` のタプルを返すだけでよい。``summary`` (=content) のみが
LLM の後続文脈へ渡り、巨大な ``props`` 辞書はコンテキストを汚さない。封筒は artifact
として checkpoint に自動永続化される (別テーブル・別保存経路は不要)。

判定/正規化ロジックは ``is_ui_artifact`` / ``coerce_ui`` に一元化し、
streaming.py (ライブ SSE) と history.py (履歴リロード復元) の両方から呼ぶことで、
ストリームと再水和の完全一致を保証する。
"""

from typing import Any

UI_SCHEMA_VERSION = 1


def is_ui_artifact(artifact: object) -> bool:
    """artifact が UI 封筒かどうかを判定する。

    kind=="ui" かつ component が文字列・props が dict のときだけ True。
    ネイティブツールでも MCP ツールでも、この形さえ満たせば同一経路で UI 化される。
    """
    return (
        isinstance(artifact, dict)
        and artifact.get("kind") == "ui"
        and isinstance(artifact.get("component"), str)
        and isinstance(artifact.get("props"), dict)
    )


def coerce_ui(artifact: object, tool_call_id: str | None) -> dict[str, Any] | None:
    """artifact を UI リソース封筒へ正規化する。UI でなければ None。

    返す形は SSE の ``ui_resource`` data と履歴 API の ``MessageOut.ui`` で共通。
    ``id`` は封筒の ``ui_id`` を優先し、無ければ ``tool_call_id`` を流用する。
    """
    if not is_ui_artifact(artifact):
        return None
    a: dict[str, Any] = artifact  # type: ignore[assignment]
    ui_id = a.get("ui_id") or tool_call_id
    if not ui_id:
        # id を確定できないUIは描画側の突合・React key が破綻するため出さない。
        return None
    return {
        "id": ui_id,
        "component": a["component"],
        "mode": a.get("mode", "declarative"),
        "props": a["props"],
        "v": a.get("v", UI_SCHEMA_VERSION),
    }


def ui(
    component: str,
    *,
    summary: str,
    mode: str = "declarative",
    ui_id: str | None = None,
    **props: Any,
) -> tuple[str, dict[str, Any]]:
    """UIツールの戻り値 ``(content, artifact)`` を組み立てる薄いヘルパー。

    第1要素 ``summary`` は LLM 文脈用の要約 (必ず意味のある日本語にすること)。
    第2要素が UI 封筒で、``@tool(response_format="content_and_artifact")`` の
    artifact としてそのまま ToolMessage.artifact に格納される。
    """
    envelope: dict[str, Any] = {
        "v": UI_SCHEMA_VERSION,
        "kind": "ui",
        "component": component,
        "mode": mode,
        "props": props,
    }
    if ui_id is not None:
        envelope["ui_id"] = ui_id
    return summary, envelope
