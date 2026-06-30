"""streaming.py (SSE ブリッジ) のテスト — 3タプル対応・フィルタ・dedupe・progress 透過。"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

import app.services.streaming as streaming
from app.core.config import Settings
from app.services import threads as threads_service


class FakeAgent:
    """astream が事前定義のタプル列を流すフェイク。"""

    def __init__(self, parts, state_messages=None):
        self.parts = parts
        self.state_messages = state_messages or []

    def astream(self, *args, **kwargs):
        async def gen():
            for part in self.parts:
                yield part

        return gen()

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": self.state_messages})


class FakeReflectionExecutor:
    pass


@pytest.fixture(autouse=True)
def _stub_title(monkeypatch):
    async def fake_touch_and_title(pool, thread_id, title):
        return "タイトル"

    monkeypatch.setattr(threads_service, "touch_and_title", fake_touch_and_title)


async def _collect(agent, *, reflection_executor=None):
    events = []
    async for ev in streaming.stream_agent(
        agent=agent,
        pool=None,
        reflection_executor=reflection_executor,
        message="テストメッセージです",
        thread_id="t1",
        user_id="u1",
        reflection_delay_seconds=0,
        settings=Settings(),
    ):
        events.append(ev)
    return events


def _ai_with_tool_call(tc_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "calc", "args": {"x": 1}, "id": tc_id, "type": "tool_call"}],
    )


async def test_token_filtering_by_user_facing_nodes():
    parts = [
        (("responder:run1",), "messages", (AIMessageChunk(content="やあ"), {"langgraph_node": "model"})),
        ((), "messages", (AIMessageChunk(content="統合"), {"langgraph_node": "synthesizer"})),
        (("executor:run2",), "messages", (AIMessageChunk(content="内部"), {"langgraph_node": "model"})),
        ((), "messages", (AIMessageChunk(content="思考"), {"langgraph_node": "planner"})),
    ]
    events = await _collect(FakeAgent(parts))
    tokens = [e for e in events if e.event == "token"]
    assert [t.data["content"] for t in tokens] == ["やあ", "統合"]
    assert [t.data["node"] for t in tokens] == ["responder", "synthesizer"]
    assert events[-1].event == "done"


async def test_tool_events_dedupe_and_responder_parent_skip():
    ai = _ai_with_tool_call("t1")
    tool_msg = ToolMessage(content="42", tool_call_id="t1", name="calc")
    parts = [
        # responder サブグラフ内 updates (一次ソース)
        (("responder:run1",), "updates", {"model": {"messages": [ai]}}),
        (("responder:run1",), "updates", {"tools": {"messages": [tool_msg]}}),
        # responder の親レベル update は全履歴の再掲 → スキップされる
        ((), "updates", {"responder": {"messages": [ai, tool_msg, AIMessage(content="回答")]}}),
        # executor 内部の tool イベントは進捗として表出する
        (("executor:run9",), "updates", {"model": {"messages": [_ai_with_tool_call("t2")]}}),
    ]
    events = await _collect(FakeAgent(parts))
    tool_calls = [e for e in events if e.event == "tool_call"]
    tool_results = [e for e in events if e.event == "tool_result"]
    assert [e.data["id"] for e in tool_calls] == ["t1", "t2"]
    assert [e.data["id"] for e in tool_results] == ["t1"]


def _ui_tool_message(tc_id: str, *, component: str = "table") -> ToolMessage:
    """UI 封筒 (artifact) を持つ ToolMessage を作る。"""
    return ToolMessage(
        content="テーブルを表示しました",
        tool_call_id=tc_id,
        name="render_table",
        artifact={
            "v": 1,
            "kind": "ui",
            "component": component,
            "mode": "declarative",
            "props": {"title": "比較表", "columns": ["a"], "rows": [["1"]]},
        },
    )


async def test_ui_resource_emitted_before_tool_result():
    ai = _ai_with_tool_call("t1")
    ui_msg = _ui_tool_message("t1")
    parts = [
        (("responder:run1",), "updates", {"model": {"messages": [ai]}}),
        (("responder:run1",), "updates", {"tools": {"messages": [ui_msg]}}),
    ]
    events = await _collect(FakeAgent(parts))
    names = [e.event for e in events]
    # 同一 id に対し tool_call → ui_resource → tool_result の順で届く
    assert names.index("tool_call") < names.index("ui_resource")
    assert names.index("ui_resource") < names.index("tool_result")
    ui_events = [e for e in events if e.event == "ui_resource"]
    assert len(ui_events) == 1
    data = ui_events[0].data
    assert data["id"] == "t1"
    assert data["component"] == "table"
    assert data["mode"] == "declarative"
    assert data["name"] == "render_table"
    assert data["props"]["title"] == "比較表"


async def test_ui_resource_dedupe_on_parent_replay():
    ai = _ai_with_tool_call("t1")
    ui_msg = _ui_tool_message("t1")
    parts = [
        (("responder:run1",), "updates", {"model": {"messages": [ai]}}),
        (("responder:run1",), "updates", {"tools": {"messages": [ui_msg]}}),
        # responder 親レベル update は全履歴の再掲 → ui_resource も重複させない
        ((), "updates", {"responder": {"messages": [ai, ui_msg, AIMessage(content="回答")]}}),
    ]
    events = await _collect(FakeAgent(parts))
    assert len([e for e in events if e.event == "ui_resource"]) == 1


async def test_non_ui_tool_message_emits_no_ui_resource():
    ai = _ai_with_tool_call("t1")
    tool_msg = ToolMessage(content="42", tool_call_id="t1", name="calc")
    parts = [
        (("responder:run1",), "updates", {"model": {"messages": [ai]}}),
        (("responder:run1",), "updates", {"tools": {"messages": [tool_msg]}}),
    ]
    events = await _collect(FakeAgent(parts))
    assert not [e for e in events if e.event == "ui_resource"]
    assert [e.event for e in events if e.event == "tool_result"]


async def test_progress_passthrough_and_legacy_compat():
    parts = [
        (
            (),
            "custom",
            {
                "status": "計画を作成しました (2ステップ)",
                "phase": "plan",
                "plan": [{"id": 1, "description": "調べる"}],
            },
        ),
        ("custom", {"status": "旧2タプル形式"}),  # subgraphs=False 形式の防御的互換
        ((), "custom", "素の文字列"),
    ]
    events = await _collect(FakeAgent(parts))
    progress = [e for e in events if e.event == "progress"]
    assert progress[0].data["status"] == "計画を作成しました (2ステップ)"
    assert progress[0].data["phase"] == "plan"
    assert progress[0].data["plan"][0]["description"] == "調べる"
    assert progress[1].data == {"status": "旧2タプル形式"}
    assert progress[2].data == {"status": "素の文字列"}


async def test_reflection_excludes_executor_scratchpad(monkeypatch):
    captured = {}

    def fake_schedule(executor, *, user_id, thread_id, messages, after_seconds):
        captured["messages"] = messages

    monkeypatch.setattr(streaming, "schedule_reflection", fake_schedule)

    ai = _ai_with_tool_call("t1")
    tool_msg = ToolMessage(content="42", tool_call_id="t1", name="calc")
    parts = [
        (("responder:run1",), "updates", {"model": {"messages": [ai]}}),
        (("responder:run1",), "updates", {"tools": {"messages": [tool_msg]}}),
        # executor の中間メッセージはリフレクションに含めない
        (("executor:run9",), "updates", {"model": {"messages": [AIMessage(content="内部作業")]}}),
        ((), "updates", {"synthesizer": {"messages": [AIMessage(content="最終回答")]}}),
    ]
    await _collect(FakeAgent(parts), reflection_executor=FakeReflectionExecutor())

    contents = [getattr(m, "content", None) for m in captured["messages"]]
    assert "テストメッセージです" in contents  # HumanMessage
    assert "最終回答" in contents
    assert "内部作業" not in contents


async def test_real_graph_direct_path_end_to_end():
    """実グラフ astream × 実ブリッジの結合テスト — タプル形状・ns 規約・nostream 抑制の
    契約が langgraph 更新で変わったとき検出する。"""
    from itertools import cycle

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent.graph import build_agent

    agent = build_agent(
        settings=Settings(),
        native_tools=[],
        langmem_tools=[],
        mcp_tools=[],
        checkpointer=InMemorySaver(),
        store=None,
        model_factory=lambda kind, s, tags: GenericFakeChatModel(
            messages=cycle([AIMessage(content="やあ！")])
        ),
    )
    # メッセージは router_skip_under_chars 未満 → direct 経路 (responder)
    events = []
    async for ev in streaming.stream_agent(
        agent=agent,
        pool=None,
        reflection_executor=None,
        message="こんにちは",
        thread_id="t1",
        user_id="u1",
        reflection_delay_seconds=0,
        settings=Settings(),
    ):
        events.append(ev)

    tokens = [e for e in events if e.event == "token"]
    assert tokens, "responder のトークンが SSE に流れること"
    assert {t.data["node"] for t in tokens} == {"responder"}
    assert "".join(t.data["content"] for t in tokens) == "やあ！"
    assert events[-1].event == "done"
    assert not [e for e in events if e.event == "error"]


async def test_real_graph_plan_path_end_to_end():
    """plan 経路: 計画進捗 progress が流れ、トークンは synthesizer 由来のみであること。"""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent.graph import build_agent

    control = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content='{"route": "plan", "goal": "対象を調査する"}'),
                AIMessage(content='{"steps": ["調査する"]}'),
                AIMessage(content='{"scores": {"goal": 5, "accuracy": 5, "completeness": 5}, "flawed": false, "feedback": ""}'),
            ]
        )
    )
    responder = GenericFakeChatModel(messages=iter([]))
    synthesizer = GenericFakeChatModel(messages=iter([AIMessage(content="最終回答")]))
    executor = GenericFakeChatModel(messages=iter([AIMessage(content="調査結果")]))
    untagged = [responder, synthesizer]

    def factory(kind, s, tags):
        if kind == "control":
            return control
        return executor if "nostream" in tags else untagged.pop(0)

    agent = build_agent(
        settings=Settings(),
        native_tools=[],
        langmem_tools=[],
        mcp_tools=[],
        checkpointer=InMemorySaver(),
        store=None,
        model_factory=factory,
    )
    events = []
    async for ev in streaming.stream_agent(
        agent=agent,
        pool=None,
        reflection_executor=None,
        message="東京と大阪の明日の天気を調べて比較表にまとめてください",
        thread_id="t2",
        user_id="u1",
        reflection_delay_seconds=0,
        settings=Settings(),
    ):
        events.append(ev)

    assert not [e for e in events if e.event == "error"]
    assert events[-1].event == "done"
    # 計画・ステップ進捗が progress として流れる
    phases = {e.data.get("phase") for e in events if e.event == "progress"}
    assert {"routing", "plan", "step", "evaluate", "synthesize"} <= phases
    # トークンは synthesizer 由来のみ (orchestrator/planner/evaluator/executor は非表出)
    tokens = [e for e in events if e.event == "token"]
    assert tokens, "synthesizer のトークンが SSE に流れること"
    assert {t.data["node"] for t in tokens} == {"synthesizer"}
    assert "".join(t.data["content"] for t in tokens) == "最終回答"


async def test_error_event_on_stream_failure():
    class BrokenAgent(FakeAgent):
        def astream(self, *args, **kwargs):
            async def gen():
                yield ((), "custom", {"status": "開始"})
                raise RuntimeError("途中で死亡")

            return gen()

    events = await _collect(BrokenAgent([]))
    assert events[-1].event == "error"
    assert "途中で死亡" in events[-1].data["message"]
