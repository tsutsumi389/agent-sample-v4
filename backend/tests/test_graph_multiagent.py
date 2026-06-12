"""マルチエージェントグラフの E2E テスト (フェイクモデル + InMemorySaver、DB / Ollama なし)。"""

from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.context import AgentContext
from app.agent.graph import build_agent
from app.core.config import Settings

LONG_GOAL = "東京と大阪の明日の天気を調べて、移動手段ごとの所要時間と合わせて比較表にまとめてください"


def _fake_factory(*, control, responder, synthesizer, executor):
    """kind と tags でフェイクを配る (nostream 付き chat = executor、なし = responder→synthesizer 順)。

    呼び出し列は factory.calls に記録され、nostream タグの配線リグレッションを検出できる。
    """
    untagged_chat = [responder, synthesizer]
    calls: list[tuple[str, tuple[str, ...]]] = []

    def factory(kind, settings, tags):
        calls.append((kind, tuple(tags)))
        if kind == "control":
            return control
        if "nostream" in tags:
            return executor
        return untagged_chat.pop(0)

    factory.calls = calls
    return factory


def _build(factory, settings=None):
    return build_agent(
        settings=settings or Settings(),
        native_tools=[],
        langmem_tools=[],
        mcp_tools=[],
        checkpointer=InMemorySaver(),
        store=None,
        model_factory=factory,
    )


def _config(thread_id: str = "t1") -> dict:
    return {"configurable": {"thread_id": thread_id, "langgraph_user_id": "u1"}}


def test_nostream_tag_wiring():
    """内部思考系 (control/executor) のみ nostream。responder/synthesizer に付くと
    回答トークンが SSE に一切流れなくなるため、配線をここで固定する。"""
    factory = _fake_factory(
        control=GenericFakeChatModel(messages=iter([])),
        responder=GenericFakeChatModel(messages=iter([])),
        synthesizer=GenericFakeChatModel(messages=iter([])),
        executor=GenericFakeChatModel(messages=iter([])),
    )
    _build(factory)
    assert factory.calls == [
        ("chat", ()),  # responder: トークン表出
        ("control", ("nostream",)),  # orchestrator/planner/evaluator
        ("chat", ()),  # synthesizer: トークン表出
        ("chat", ("nostream",)),  # executor
    ]


async def test_direct_path_returns_ai_message():
    factory = _fake_factory(
        control=GenericFakeChatModel(messages=iter([])),  # 短文なので呼ばれない
        responder=GenericFakeChatModel(messages=iter([AIMessage(content="こんにちは！")])),
        synthesizer=GenericFakeChatModel(messages=iter([])),
        executor=GenericFakeChatModel(messages=iter([])),
    )
    agent = _build(factory)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "こんにちは"}]},
        config=_config(),
        context=AgentContext(user_id="u1"),
    )
    assert result["messages"][-1].content == "こんにちは！"
    assert result["route"] == "direct"


async def test_plan_path_runs_full_loop():
    factory = _fake_factory(
        control=GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="PLAN"),
                    AIMessage(content='{"steps": ["天気を調べる", "比較表を作る"]}'),
                    AIMessage(content='{"verdict": "pass", "feedback": ""}'),
                    AIMessage(content='{"verdict": "pass", "feedback": ""}'),
                ]
            )
        ),
        responder=GenericFakeChatModel(messages=iter([])),
        synthesizer=GenericFakeChatModel(messages=iter([AIMessage(content="統合された最終回答")])),
        executor=GenericFakeChatModel(
            messages=iter([AIMessage(content="結果1"), AIMessage(content="結果2")])
        ),
    )
    agent = _build(factory)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": LONG_GOAL}]},
        config=_config(),
        context=AgentContext(user_id="u1"),
    )
    assert result["messages"][-1].content == "統合された最終回答"
    assert [s["status"] for s in result["plan"]] == ["done", "done"]
    assert result["executor_runs"] == 2


async def test_all_llm_outputs_broken_still_reaches_end():
    """全 LLM が壊れた出力を返しても、フォールバック経路で必ず回答が返る。"""
    broken = lambda: GenericFakeChatModel(  # noqa: E731
        messages=cycle([AIMessage(content="PLAN かもしれない {壊れたjson: ")])
    )
    factory = _fake_factory(
        control=broken(),  # orchestrator は PLAN に分類 → planner/evaluator は全てパース失敗
        responder=broken(),
        synthesizer=broken(),
        executor=broken(),
    )
    agent = _build(factory)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": LONG_GOAL}]},
        config=_config(),
        context=AgentContext(user_id="u1"),
    )
    # planner → 単一ステップ計画 / evaluator → pass 前進 / synthesizer → 出力 (壊れていても文字列)
    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content  # 空でない回答が必ず返る


async def test_scratch_state_resets_between_turns():
    factory = _fake_factory(
        control=GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="PLAN"),
                    AIMessage(content='{"steps": ["調べる"]}'),
                    AIMessage(content='{"verdict": "pass", "feedback": ""}'),
                ]
            )
        ),
        responder=GenericFakeChatModel(messages=iter([AIMessage(content="2ターン目の回答")])),
        synthesizer=GenericFakeChatModel(messages=iter([AIMessage(content="1ターン目の回答")])),
        executor=GenericFakeChatModel(messages=iter([AIMessage(content="結果")])),
    )
    agent = _build(factory)
    config = _config("t-reset")
    context = AgentContext(user_id="u1")

    first = await agent.ainvoke(
        {"messages": [{"role": "user", "content": LONG_GOAL}]}, config=config, context=context
    )
    assert first["plan"]  # 1ターン目は計画が残る

    second = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "こんにちは"}]}, config=config, context=context
    )
    # 2ターン目 (direct) では orchestrator がスクラッチをリセットしている
    assert second["route"] == "direct"
    assert second["plan"] == []
    assert second["executor_runs"] == 0
    assert second["messages"][-1].content == "2ターン目の回答"


async def test_global_budget_terminates_retry_loop():
    """evaluator が retry を出し続けても max_executor_runs で必ず終了する。"""
    settings = Settings(max_executor_runs=3, max_step_retries=99)
    factory = _fake_factory(
        control=GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="PLAN"),
                    AIMessage(content='{"steps": ["終わらないタスク"]}'),
                    AIMessage(content='{"verdict": "retry", "feedback": "やり直し"}'),
                    AIMessage(content='{"verdict": "retry", "feedback": "やり直し"}'),
                    AIMessage(content='{"verdict": "retry", "feedback": "やり直し"}'),
                ]
            )
        ),
        responder=GenericFakeChatModel(messages=iter([])),
        synthesizer=GenericFakeChatModel(messages=iter([AIMessage(content="打ち切り回答")])),
        executor=GenericFakeChatModel(messages=cycle([AIMessage(content="不十分な結果")])),
    )
    agent = _build(factory, settings)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": LONG_GOAL}]},
        config=_config(),
        context=AgentContext(user_id="u1"),
    )
    assert result["executor_runs"] == 3
    assert result["messages"][-1].content == "打ち切り回答"
