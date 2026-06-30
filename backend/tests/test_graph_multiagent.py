"""マルチエージェントグラフの E2E テスト (フェイクモデル + InMemorySaver、DB / Ollama なし)。"""

from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.context import AgentContext
from app.agent.graph import build_agent
from app.core.config import Settings

LONG_GOAL = "東京と大阪の明日の天気を調べて、移動手段ごとの所要時間と合わせて比較表にまとめてください"


def _fake_factory(*, control, responder, synthesizer, executor, screen=None):
    """kind と tags でフェイクを配る。

    nostream なし chat = responder→synthesizer 順、nostream 付き chat = executor→screen 順
    (生成順に対応)。呼び出し列は factory.calls に記録され、nostream 配線リグレッションを検出できる。
    """
    untagged_chat = [responder, synthesizer]
    nostream_chat = [executor, screen if screen is not None else GenericFakeChatModel(messages=iter([]))]
    calls: list[tuple[str, tuple[str, ...]]] = []

    def factory(kind, settings, tags):
        calls.append((kind, tuple(tags)))
        if kind == "control":
            return control
        if "nostream" in tags:
            return nostream_chat.pop(0)
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
        ("chat", ("nostream",)),  # screen: 構造化データ選別 (内部処理のため nostream)
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
                    # orchestrator は route + 文脈化 goal を JSON で返す
                    AIMessage(content='{"route": "plan", "goal": "東京と大阪の天気と所要時間を比較表にまとめる"}'),
                    AIMessage(content='{"steps": ["天気を調べる", "比較表を作る"]}'),
                    AIMessage(content='{"scores": {"goal": 5, "accuracy": 5, "completeness": 5}, "flawed": false, "feedback": ""}'),
                    AIMessage(content='{"scores": {"goal": 5, "accuracy": 5, "completeness": 5}, "flawed": false, "feedback": ""}'),
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
        control=broken(),  # orchestrator はパース全滅 → direct + 生 goal にフォールバック
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
    # orchestrator が壊れた出力で direct へフォールバックしても、responder が (壊れていても) 文字列を返す
    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content  # 空でない回答が必ず返る


async def test_scratch_state_resets_between_turns():
    factory = _fake_factory(
        control=GenericFakeChatModel(
            messages=iter(
                [
                    # 1ターン目: plan 分類 + plan ループ (orchestrator / planner / evaluator)
                    AIMessage(content='{"route": "plan", "goal": "LONG_GOAL を文脈化"}'),
                    AIMessage(content='{"steps": ["調べる"]}'),
                    AIMessage(content='{"scores": {"goal": 5, "accuracy": 5, "completeness": 5}, "flawed": false, "feedback": ""}'),
                    # 2ターン目: 履歴があるため短文「こんにちは」でも orchestrator が呼ばれる → direct 分類
                    AIMessage(content='{"route": "direct", "goal": "こんにちは"}'),
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


async def test_followup_goal_is_contextualized_into_state():
    """2ターン目のフォローアップで、orchestrator が会話履歴を踏まえて goal を文脈化し、
    その文脈化済み goal が state に載る (= 下流 plan 経路へ伝播する) ことを固定する。"""
    factory = _fake_factory(
        control=GenericFakeChatModel(
            messages=iter(
                [
                    # 1ターン目: direct で軽く返す
                    AIMessage(content='{"route": "direct", "goal": "Pythonのデコレータについて詳しく教えてください"}'),
                    # 2ターン目: 「もっと例を」を履歴で補完した自己完結 goal を返す
                    AIMessage(content='{"route": "direct", "goal": "Pythonのデコレータの例をもっと挙げる"}'),
                ]
            )
        ),
        responder=GenericFakeChatModel(
            messages=iter([AIMessage(content="デコレータの解説"), AIMessage(content="追加の例")])
        ),
        synthesizer=GenericFakeChatModel(messages=iter([])),
        executor=GenericFakeChatModel(messages=iter([])),
    )
    agent = _build(factory)
    config = _config("t-ctx")
    context = AgentContext(user_id="u1")

    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Pythonのデコレータについて詳しく教えてください"}]},
        config=config,
        context=context,
    )
    second = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "もっと例を"}]}, config=config, context=context
    )
    # 生入力「もっと例を」ではなく、履歴で補完された自己完結 goal が state に載る
    assert second["goal"] == "Pythonのデコレータの例をもっと挙げる"


async def test_parallel_dag_execution():
    """依存DAG計画で、独立ステップ(1,2)が1ラウンドで並列実行され、依存ステップ(3)が
    その後のラウンドで実行される。全2ラウンドで executor_runs=3 に到達する。"""
    dag_plan = (
        '{"steps": ['
        '{"id": 1, "description": "東京の天気", "depends_on": []}, '
        '{"id": 2, "description": "大阪の天気", "depends_on": []}, '
        '{"id": 3, "description": "比較表を作る", "depends_on": [1, 2]}'
        "]}"
    )
    factory = _fake_factory(
        control=GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content='{"route": "plan", "goal": "東京と大阪の天気を比較する"}'),
                    AIMessage(content=dag_plan),
                    # round1: ステップ1,2 を並列評価 / round2: ステップ3 を評価
                    AIMessage(content='{"scores": {"goal": 5, "accuracy": 5, "completeness": 5}, "flawed": false, "feedback": ""}'),
                    AIMessage(content='{"scores": {"goal": 5, "accuracy": 5, "completeness": 5}, "flawed": false, "feedback": ""}'),
                    AIMessage(content='{"scores": {"goal": 5, "accuracy": 5, "completeness": 5}, "flawed": false, "feedback": ""}'),
                ]
            )
        ),
        responder=GenericFakeChatModel(messages=iter([])),
        synthesizer=GenericFakeChatModel(messages=iter([AIMessage(content="比較表です")])),
        executor=GenericFakeChatModel(
            messages=cycle([AIMessage(content="ステップ結果")])
        ),
    )
    agent = _build(factory)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": LONG_GOAL}]},
        config=_config(),
        context=AgentContext(user_id="u1"),
    )
    assert result["messages"][-1].content == "比較表です"
    assert [s["status"] for s in result["plan"]] == ["done", "done", "done"]
    assert [s["depends_on"] for s in result["plan"]] == [[], [], [1, 2]]
    assert result["executor_runs"] == 3


async def test_global_budget_terminates_retry_loop():
    """evaluator が retry を出し続けても max_executor_runs で必ず終了する。"""
    settings = Settings(max_executor_runs=3, max_step_retries=99)
    factory = _fake_factory(
        control=GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content='{"route": "plan", "goal": "終わらないタスクを実行する"}'),
                    AIMessage(content='{"steps": ["終わらないタスク"]}'),
                    AIMessage(content='{"scores": {"goal": 2, "accuracy": 2, "completeness": 2}, "flawed": false, "feedback": "やり直し"}'),
                    AIMessage(content='{"scores": {"goal": 2, "accuracy": 2, "completeness": 2}, "flawed": false, "feedback": "やり直し"}'),
                    AIMessage(content='{"scores": {"goal": 2, "accuracy": 2, "completeness": 2}, "flawed": false, "feedback": "やり直し"}'),
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
