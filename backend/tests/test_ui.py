"""UI封筒 (app/agent/ui.py) のユニットテスト — 判定・正規化・ヘルパー。"""

from app.agent.ui import UI_SCHEMA_VERSION, coerce_ui, is_ui_artifact, ui


def test_is_ui_artifact_accepts_valid_envelope():
    assert is_ui_artifact(
        {"kind": "ui", "component": "table", "props": {"rows": []}}
    )


def test_is_ui_artifact_rejects_non_ui():
    assert not is_ui_artifact(None)
    assert not is_ui_artifact("文字列")
    assert not is_ui_artifact({"kind": "other", "component": "table", "props": {}})
    assert not is_ui_artifact({"kind": "ui", "component": 123, "props": {}})
    assert not is_ui_artifact({"kind": "ui", "component": "table", "props": "x"})


def test_coerce_ui_normalizes_and_uses_tool_call_id():
    artifact = {"kind": "ui", "component": "card", "props": {"title": "T"}}
    out = coerce_ui(artifact, "call_1")
    assert out == {
        "id": "call_1",
        "component": "card",
        "mode": "declarative",
        "props": {"title": "T"},
        "v": UI_SCHEMA_VERSION,
    }


def test_coerce_ui_prefers_explicit_ui_id_and_mode():
    artifact = {
        "kind": "ui",
        "component": "chart",
        "mode": "iframe",
        "ui_id": "explicit",
        "props": {},
        "v": 7,
    }
    out = coerce_ui(artifact, "call_1")
    assert out["id"] == "explicit"
    assert out["mode"] == "iframe"
    assert out["v"] == 7


def test_coerce_ui_returns_none_for_non_ui():
    assert coerce_ui(None, "call_1") is None
    assert coerce_ui("plain", "call_1") is None


def test_coerce_ui_returns_none_when_id_cannot_be_determined():
    # ui_id 無し かつ tool_call_id 無し → 描画側の突合が破綻するため出さない
    artifact = {"kind": "ui", "component": "table", "props": {}}
    assert coerce_ui(artifact, None) is None
    assert coerce_ui(artifact, "") is None


def test_ui_helper_builds_content_and_artifact():
    content, artifact = ui("table", summary="2行の表", title="t", rows=[[1], [2]])
    assert content == "2行の表"
    assert artifact["kind"] == "ui"
    assert artifact["component"] == "table"
    assert artifact["mode"] == "declarative"
    assert artifact["v"] == UI_SCHEMA_VERSION
    assert artifact["props"] == {"title": "t", "rows": [[1], [2]]}
    assert "ui_id" not in artifact
    # ヘルパー出力はそのまま封筒として正規化できる
    assert coerce_ui(artifact, "c")["component"] == "table"
