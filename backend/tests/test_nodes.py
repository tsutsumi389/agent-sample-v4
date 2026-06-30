"""各ノードの単体テスト (フェイクモデル注入、DB / Ollama なし)。"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.nodes.common import (
    EXECUTION_FAILED_MARKER,
    apply_data_selection,
    recent_history_text,
    screen_step_data,
)
from app.agent.nodes.evaluator import make_evaluator_node
from app.agent.nodes.executor import make_executor_node
from app.agent.nodes.orchestrator import make_orchestrator_node
from app.agent.nodes.planner import make_planner_node
from app.agent.nodes.synthesizer import make_synthesizer_node
from app.agent.parsing import (
    DataSelectionSchema,
    PlanSchema,
    RouteSchema,
    RubricScores,
    VerdictSchema,
)
from app.core.config import Settings
from tests.fakes import (
    FakeScreenModel,
    FakeStore,
    ScriptedExecutorAgent,
    ScriptedModel,
    StructuredModel,
)

SETTINGS = Settings()
OPENAI_SETTINGS = Settings(llm_provider="openai", openai_api_key="x")
SCREEN = FakeScreenModel()  # no-op スクリーニング (全量素通り)
LONG_GOAL = "東京と大阪の明日の天気を調べて、移動手段ごとの所要時間と合わせて比較表にまとめてください"


def _human_state(text: str) -> dict:
    return {"messages": [HumanMessage(content=text)]}


def _step(
    desc: str = "調査する",
    *,
    id: int = 1,
    depends_on: list[int] | None = None,
    result: str = "",
    attempts: int = 0,
    status: str = "pending",
    feedback_history: list[str] | None = None,
    instruction: str | None = None,
) -> dict:
    step = {
        "id": id,
        "description": desc,
        "depends_on": depends_on or [],
        "status": status,
        "result": result,
        "attempts": attempts,
    }
    if feedback_history is not None:
        step["feedback_history"] = feedback_history
    if instruction is not None:
        step["instruction"] = instruction
    return step


# ---- orchestrator ----


async def test_orchestrator_short_input_skips_llm():
    model = ScriptedModel([])  # 呼ばれたら raise する
    node = make_orchestrator_node(model, SETTINGS)
    out = await node(_human_state("こんにちは"), {})
    assert out["route"] == "direct"
    assert model.calls == []


async def test_orchestrator_classifies_plan():
    # ollama 経路は JSON テキストパース (think ブロックは除去される)。route と goal を両取り。
    model = ScriptedModel(['<think>複雑そう</think>{"route": "plan", "goal": "東京と大阪の天気を比較"}'])
    node = make_orchestrator_node(model, SETTINGS)
    out = await node(_human_state(LONG_GOAL), {})
    assert out["route"] == "plan"


async def test_orchestrator_falls_back_to_direct_on_exception():
    model = ScriptedModel([RuntimeError("接続断")])
    node = make_orchestrator_node(model, SETTINGS)
    out = await node(_human_state(LONG_GOAL), {})
    assert out["route"] == "direct"


async def test_orchestrator_separates_system_and_user_roles():
    model = ScriptedModel(["DIRECT"])
    node = make_orchestrator_node(model, SETTINGS)
    await node(_human_state(LONG_GOAL), {})
    system, human = model.calls[0]
    assert isinstance(system, SystemMessage)
    assert isinstance(human, HumanMessage)
    assert "タスク分類器" in system.content
    assert LONG_GOAL in human.content  # ユーザー要求はデータとして human 側に隔離


async def test_orchestrator_structured_output_openai():
    # openai 経路: with_structured_output で RouteSchema を直接得る
    settings = Settings(llm_provider="openai", openai_api_key="x")
    model = StructuredModel([RouteSchema(route="plan")])
    node = make_orchestrator_node(model, settings)
    out = await node(_human_state(LONG_GOAL), {})
    assert out["route"] == "plan"
    assert model.bound_schema is RouteSchema


async def test_orchestrator_structured_output_direct_openai():
    settings = Settings(llm_provider="openai", openai_api_key="x")
    model = StructuredModel([RouteSchema(route="direct")])
    node = make_orchestrator_node(model, settings)
    out = await node(_human_state(LONG_GOAL), {})
    assert out["route"] == "direct"


async def test_orchestrator_resets_scratch():
    model = ScriptedModel(["DIRECT"])
    node = make_orchestrator_node(model, SETTINGS)
    stale = _human_state(LONG_GOAL) | {
        "plan": [_step()],
        "executor_runs": 7,
        "replan_count": 1,
        "needs_replan": True,
        "failure_notes": ["古いメモ"],
    }
    out = await node(stale, {})
    assert out["plan"] == []
    assert out["executor_runs"] == 0
    assert out["replan_count"] == 0
    assert out["needs_replan"] is False
    assert out["failure_notes"] == []


# orchestrator: 会話履歴を踏まえた goal 文脈化


def _history_state(history_msgs: list, current: str) -> dict:
    """履歴 + 今ターンの HumanMessage を持つ state を作る。"""
    return {"messages": [*history_msgs, HumanMessage(content=current)]}


async def test_orchestrator_short_followup_with_history_calls_llm():
    # 履歴があれば短文フォローアップでも LLM を通して文脈化する (skip しない)。
    model = ScriptedModel(['{"route": "plan", "goal": "Pythonのデコレータの例をもっと挙げる"}'])
    node = make_orchestrator_node(model, SETTINGS)
    state = _history_state(
        [HumanMessage(content="Pythonのデコレータを教えて"), AIMessage(content="デコレータは関数を…")],
        "もっと例を",  # 20文字未満だが履歴あり
    )
    out = await node(state, {})
    assert model.calls != []  # スキップされず LLM が呼ばれた
    assert out["route"] == "plan"
    assert out["goal"] == "Pythonのデコレータの例をもっと挙げる"  # リライト採用


async def test_orchestrator_short_first_turn_skips_llm():
    # 履歴なしの短文 (初回挨拶) は従来どおり LLM を呼ばず direct。
    model = ScriptedModel([])  # 呼ばれたら raise
    node = make_orchestrator_node(model, SETTINGS)
    out = await node(_human_state("やあ"), {})
    assert out["route"] == "direct"
    assert model.calls == []


async def test_orchestrator_rewrites_goal_via_history_openai():
    model = StructuredModel([RouteSchema(route="direct", goal="東京の明日の天気を教えて")])
    node = make_orchestrator_node(model, OPENAI_SETTINGS)
    out = await node(_human_state(LONG_GOAL), {})
    assert out["goal"] == "東京の明日の天気を教えて"  # 文脈反映済み goal を採用
    assert out["route"] == "direct"


async def test_orchestrator_blank_rewrite_falls_back_to_raw_goal():
    model = StructuredModel([RouteSchema(route="plan", goal="")])
    node = make_orchestrator_node(model, OPENAI_SETTINGS)
    out = await node(_human_state(LONG_GOAL), {})
    assert out["goal"] == LONG_GOAL  # 空リライトは生 goal にフォールバック
    assert out["route"] == "plan"


async def test_orchestrator_oversized_rewrite_falls_back_to_raw_goal():
    huge = "あ" * (OPENAI_SETTINGS.goal_max_chars + 50)
    model = StructuredModel([RouteSchema(route="direct", goal=huge)])
    node = make_orchestrator_node(model, OPENAI_SETTINGS)
    out = await node(_human_state(LONG_GOAL), {})
    assert out["goal"] == LONG_GOAL  # 過長リライトは棄却し生 goal を維持


async def test_orchestrator_ollama_parses_route_and_goal_from_json():
    model = ScriptedModel(['{"route": "plan", "goal": "文脈反映済みの要求"}'])
    node = make_orchestrator_node(model, SETTINGS)
    out = await node(_human_state(LONG_GOAL), {})
    assert out["route"] == "plan"
    assert out["goal"] == "文脈反映済みの要求"


async def test_orchestrator_history_in_human_message():
    model = ScriptedModel(['{"route": "direct", "goal": "x"}'])
    node = make_orchestrator_node(model, SETTINGS)
    state = _history_state(
        [HumanMessage(content="デコレータの話"), AIMessage(content="デコレータの解説")],
        "もっと詳しく説明して",
    )
    await node(state, {})
    _, human = model.calls[0]
    assert "conversation_history" in human.content  # 履歴セクションが User 側に隔離されて入る
    assert "デコレータの解説" in human.content


# ---- recent_history_text ----


async def test_recent_history_excludes_current_human():
    state = _history_state(
        [HumanMessage(content="前の質問"), AIMessage(content="前の回答")],
        "今の質問",
    )
    out = recent_history_text(state)
    assert "前の質問" in out and "前の回答" in out
    assert "今の質問" not in out  # 今ターンの入力は履歴に含めない


async def test_recent_history_skips_tool_and_empty_ai():
    from langchain_core.messages import ToolMessage

    state = {
        "messages": [
            HumanMessage(content="調べて"),
            AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "t1"}]),
            ToolMessage(content="ツール結果", tool_call_id="t1"),
            AIMessage(content="調べた結果です"),
            HumanMessage(content="今の質問"),
        ]
    }
    out = recent_history_text(state)
    assert "ユーザー: 調べて" in out
    assert "アシスタント: 調べた結果です" in out
    assert "ツール結果" not in out  # ToolMessage は除外
    assert out.count("アシスタント:") == 1  # 本文空の中間 AIMessage は除外


async def test_recent_history_empty_when_no_prior():
    # 今ターンの HumanMessage のみ (履歴なし) → ""
    assert recent_history_text(_human_state("最初の質問")) == ""


# ---- planner ----


async def test_planner_builds_plan():
    model = ScriptedModel(['{"steps": ["天気を調べる", "比較表を作る"]}'])
    node = make_planner_node(model, [], SETTINGS, FakeStore(), SCREEN)
    out = await node({"goal": LONG_GOAL}, {})
    assert [s["description"] for s in out["plan"]] == ["天気を調べる", "比較表を作る"]
    # 旧形式 (list[str]) は依存なしの並列ステップとして取り込まれる。instruction は空
    # (executor が description にフォールバックする)。
    assert out["plan"][0] == {
        "id": 1,
        "description": "天気を調べる",
        "instruction": "",
        "depends_on": [],
        "status": "pending",
        "result": "",
        "attempts": 0,
    }


async def test_planner_builds_dependency_dag():
    model = ScriptedModel(
        [
            '{"steps": ['
            '{"id": 1, "description": "東京の天気", "depends_on": []}, '
            '{"id": 2, "description": "大阪の天気", "depends_on": []}, '
            '{"id": 3, "description": "比較表", "depends_on": [1, 2]}'
            "]}"
        ]
    )
    node = make_planner_node(model, [], SETTINGS, FakeStore(), SCREEN)
    out = await node({"goal": LONG_GOAL}, {})
    assert [s["depends_on"] for s in out["plan"]] == [[], [], [1, 2]]


async def test_planner_structured_output_openai_remaps_and_sanitizes():
    # openai 経路: with_structured_output で PlanSchema を直接受け取っても、id リマップと
    # depends_on サニタイズが従来どおり機能することを保証する。
    # PlanSchema の before バリデータは dict 入力を正規化する (実際の構造化出力と同じ経路)。
    model = StructuredModel(
        [
            PlanSchema(
                steps=[
                    {"id": 10, "description": "A", "depends_on": []},
                    {"id": 20, "description": "B", "depends_on": [10]},
                ]
            )
        ]
    )
    node = make_planner_node(model, [], OPENAI_SETTINGS, FakeStore(), SCREEN)
    out = await node({"goal": LONG_GOAL}, {})
    assert [s["id"] for s in out["plan"]] == [1, 2]  # 元 id 10/20 → 出現順 1/2 へリマップ
    assert [s["depends_on"] for s in out["plan"]] == [[], [1]]
    assert model.bound_schema is PlanSchema


async def test_planner_remaps_zero_based_and_sparse_ids():
    # LLM が 0始まり / 飛び番 id を出しても depends_on の参照が壊れない
    model = ScriptedModel(
        [
            '{"steps": ['
            '{"id": 0, "description": "A", "depends_on": []}, '
            '{"id": 10, "description": "B", "depends_on": [0]}, '
            '{"id": 20, "description": "C", "depends_on": [0, 10]}'
            "]}"
        ]
    )
    node = make_planner_node(model, [], SETTINGS, FakeStore(), SCREEN)
    out = await node({"goal": LONG_GOAL}, {})
    assert [s["id"] for s in out["plan"]] == [1, 2, 3]
    # 0→1, 10→2, 20→3 にリマップされ、依存関係が保存される
    assert [s["depends_on"] for s in out["plan"]] == [[], [1], [1, 2]]


async def test_planner_remaps_deps_when_middle_step_dropped():
    # 中間の空 description ステップが間引かれても後続の depends_on がズレない
    model = ScriptedModel(
        [
            '{"steps": ['
            '{"id": 1, "description": "A", "depends_on": []}, '
            '{"id": 2, "description": "   ", "depends_on": []}, '  # 空 → 間引かれる
            '{"id": 3, "description": "C", "depends_on": [1]}, '
            '{"id": 4, "description": "D", "depends_on": [3]}'
            "]}"
        ]
    )
    node = make_planner_node(model, [], SETTINGS, FakeStore(), SCREEN)
    out = await node({"goal": LONG_GOAL}, {})
    assert [s["description"] for s in out["plan"]] == ["A", "C", "D"]
    # A=1, C=2, D=3。C→A(1) と D→C(2) の依存が正しく保存される
    assert [s["depends_on"] for s in out["plan"]] == [[], [1], [2]]


async def test_planner_sanitizes_invalid_and_cyclic_deps():
    # 自己参照(2)・範囲外(99)・循環(3→4→3) を含む計画
    model = ScriptedModel(
        [
            '{"steps": ['
            '{"id": 1, "description": "A", "depends_on": [99]}, '
            '{"id": 2, "description": "B", "depends_on": [2]}, '
            '{"id": 3, "description": "C", "depends_on": [4]}, '
            '{"id": 4, "description": "D", "depends_on": [3]}'
            "]}"
        ]
    )
    node = make_planner_node(model, [], SETTINGS, FakeStore(), SCREEN)
    out = await node({"goal": LONG_GOAL}, {})
    deps = {s["id"]: s["depends_on"] for s in out["plan"]}
    assert deps[1] == []  # 範囲外 99 は除去
    assert deps[2] == []  # 自己参照は除去
    # 循環 3↔4 は壊されて少なくとも片方が空になり、全ステップが最終的に実行可能になる
    assert deps[3] == [] or deps[4] == []


async def test_planner_truncates_to_max_steps():
    steps = [f"手順{i}" for i in range(1, 9)]
    model = ScriptedModel([f'{{"steps": {steps}}}'.replace("'", '"')])
    node = make_planner_node(model, [], SETTINGS, FakeStore(), SCREEN)
    out = await node({"goal": LONG_GOAL}, {})
    assert len(out["plan"]) == SETTINGS.max_plan_steps


async def test_planner_falls_back_to_single_step_plan():
    model = ScriptedModel(["壊れた出力", "また壊れた出力"])
    node = make_planner_node(model, [], SETTINGS, FakeStore(), SCREEN)
    out = await node({"goal": LONG_GOAL}, {})
    assert [s["description"] for s in out["plan"]] == [LONG_GOAL]
    assert out["plan"][0]["depends_on"] == []


async def test_planner_propagates_instruction_to_plan():
    # planner は各ステップの instruction (executor 向けの具体的・パーソナライズ済み手順) を
    # そのまま plan に伝播する。description は短い成果物名のまま保たれる。
    model = ScriptedModel(
        [
            '{"steps": ['
            '{"id": 1, "description": "カフェを探す", '
            '"instruction": "駅近・禁煙・予算1000円以下のカフェを3件。カフェイン控えめを優先", '
            '"depends_on": []}'
            "]}"
        ]
    )
    node = make_planner_node(model, [], SETTINGS, FakeStore(), SCREEN)
    out = await node({"goal": LONG_GOAL}, {})
    assert out["plan"][0]["description"] == "カフェを探す"
    assert "カフェイン控えめを優先" in out["plan"][0]["instruction"]


async def test_planner_replan_increments_count_and_includes_failures():
    model = ScriptedModel(['{"steps": ["やり直す"]}'])
    node = make_planner_node(model, [], SETTINGS, FakeStore(), SCREEN)
    state = {
        "goal": LONG_GOAL,
        "plan": [_step("旧ステップ", result="旧結果", status="done")],
        "replan_count": 0,
        "failure_notes": ["ステップ2「比較表」: 計画が粗すぎる"],
    }
    out = await node(state, {})
    assert out["replan_count"] == 1
    assert out["needs_replan"] is False
    # ロール分離: 指示は SystemMessage、要求・再計画情報 (データ) は HumanMessage に入る
    system, human = model.calls[0]
    assert isinstance(system, SystemMessage)
    assert isinstance(human, HumanMessage)
    assert "あなたはタスク計画立案者です" in system.content
    assert "前回の計画は失敗しました" in human.content
    assert "ステップ2「比較表」: 計画が粗すぎる" in human.content


# ---- executor ----


async def test_executor_runs_step_and_records_result():
    agent = ScriptedExecutorAgent(["<think>作業中</think>調査結果です"])
    node = make_executor_node(agent, SETTINGS, SCREEN)
    state = {"goal": LONG_GOAL, "plan": [_step()], "executor_runs": 0}
    out = await node(state, {"configurable": {"thread_id": "t1"}})
    assert out["plan"][0]["result"] == "調査結果です"
    assert out["plan"][0]["attempts"] == 1
    assert out["plan"][0]["status"] == "running"  # 評価待ち
    assert out["executor_runs"] == 1
    # recursion_limit と configurable が伝搬している
    assert agent.configs[0]["recursion_limit"] == SETTINGS.executor_recursion_limit
    assert agent.configs[0]["configurable"]["thread_id"] == "t1"


async def test_executor_runs_independent_steps_in_parallel():
    agent = ScriptedExecutorAgent(["結果A", "結果B", "結果C"])
    node = make_executor_node(agent, SETTINGS, SCREEN)  # max_parallel_executors=3
    plan = [
        _step("A", id=1),
        _step("B", id=2),
        _step("C", id=3),
    ]
    out = await node({"goal": LONG_GOAL, "plan": plan, "executor_runs": 0}, {})
    # 依存なし3ステップが1ラウンドで全て実行される
    assert [s["status"] for s in out["plan"]] == ["running", "running", "running"]
    assert {s["result"] for s in out["plan"]} == {"結果A", "結果B", "結果C"}
    assert out["executor_runs"] == 3


async def test_executor_respects_parallel_cap():
    agent = ScriptedExecutorAgent(["1", "2", "3", "4", "5"])
    settings = Settings(max_parallel_executors=2)
    node = make_executor_node(agent, settings, SCREEN)
    plan = [_step(f"S{i}", id=i) for i in range(1, 5)]  # 依存なし4ステップ
    out = await node({"goal": LONG_GOAL, "plan": plan, "executor_runs": 0}, {})
    # 1ラウンドでは上限2件だけ実行される (残りは次ラウンド)
    running = [s for s in out["plan"] if s["status"] == "running"]
    assert len(running) == 2
    assert out["executor_runs"] == 2


async def test_executor_skips_steps_with_unmet_dependencies():
    agent = ScriptedExecutorAgent(["Aの結果"])
    node = make_executor_node(agent, SETTINGS, SCREEN)
    plan = [
        _step("A", id=1),  # 依存なし → 実行可能
        _step("B", id=2, depends_on=[1]),  # 1 が未 done → 実行されない
    ]
    out = await node({"goal": LONG_GOAL, "plan": plan, "executor_runs": 0}, {})
    assert out["plan"][0]["status"] == "running"
    assert out["plan"][1]["status"] == "pending"  # 依存未解決でスキップ
    assert out["executor_runs"] == 1


async def test_executor_passes_dependency_results_in_prompt():
    agent = ScriptedExecutorAgent(["統合完了"])
    node = make_executor_node(agent, SETTINGS, SCREEN)
    plan = [
        _step("A", id=1, result="Aの成果", status="done"),
        _step("統合", id=2, depends_on=[1]),
    ]
    await node({"goal": LONG_GOAL, "plan": plan, "executor_runs": 0}, {})
    prompt = agent.payloads[0]["messages"][0].content
    assert "Aの成果" in prompt  # 依存ステップの結果がプロンプトに渡る


async def test_executor_prefers_instruction_over_description():
    # executor は instruction (planner がプロファイルを反映した手順) を「今回のタスク」に使う
    agent = ScriptedExecutorAgent(["完了"])
    node = make_executor_node(agent, SETTINGS, SCREEN)
    plan = [_step("カフェを探す", id=1, instruction="カフェイン控えめの店を優先すること")]
    await node({"goal": LONG_GOAL, "plan": plan, "executor_runs": 0}, {})
    prompt = agent.payloads[0]["messages"][0].content
    assert "今回のタスク: カフェイン控えめの店を優先すること" in prompt
    assert "今回のタスク: カフェを探す" not in prompt


async def test_executor_falls_back_to_description_without_instruction():
    # instruction 欠落 (旧 plan) なら description にフォールバックして従来どおり動く
    agent = ScriptedExecutorAgent(["完了"])
    node = make_executor_node(agent, SETTINGS, SCREEN)
    plan = [_step("カフェを探す", id=1)]  # instruction なし
    await node({"goal": LONG_GOAL, "plan": plan, "executor_runs": 0}, {})
    prompt = agent.payloads[0]["messages"][0].content
    assert "今回のタスク: カフェを探す" in prompt


async def test_executor_never_raises_on_agent_failure():
    agent = ScriptedExecutorAgent([RuntimeError("ツール大爆発")])
    node = make_executor_node(agent, SETTINGS, SCREEN)
    state = {"goal": LONG_GOAL, "plan": [_step()]}
    out = await node(state, {})
    assert out["plan"][0]["result"].startswith(EXECUTION_FAILED_MARKER)


async def test_executor_without_ready_steps_still_advances_runs():
    agent = ScriptedExecutorAgent([])
    node = make_executor_node(agent, SETTINGS, SCREEN)
    out = await node({"plan": [], "executor_runs": 2}, {})
    assert out == {"executor_runs": 3}


# ---- evaluator ----


def _running(**kw) -> dict:
    kw.setdefault("status", "running")
    return _step(**kw)


# evaluator のルーブリック採点 JSON (ScriptedModel=ollama テキスト経路用)。
# pass=満点(100)、retry=低スコア(40 < eval_pass_threshold=70)、replan=flawed フラグ。
_VERDICT_PASS = '{"scores": {"goal": 5, "accuracy": 5, "completeness": 5}, "flawed": false, "feedback": ""}'
_VERDICT_REPLAN = '{"scores": {"goal": 2, "accuracy": 2, "completeness": 2}, "flawed": true, "feedback": "計画から見直し"}'


def _verdict_retry(feedback: str) -> str:
    return (
        '{"scores": {"goal": 2, "accuracy": 2, "completeness": 2}, '
        f'"flawed": false, "feedback": {json.dumps(feedback, ensure_ascii=False)}}}'
    )


async def test_evaluator_empty_result_retries_without_llm():
    model = ScriptedModel([])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    state = {"plan": [_running(result="", attempts=1)]}
    out = await node(state, {})
    assert out["plan"][0]["status"] == "pending"  # retry → 再実行待ち
    assert model.calls == []


async def test_evaluator_pass_marks_done():
    model = ScriptedModel([_VERDICT_PASS])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    state = {"plan": [_running(result="良い結果", attempts=1)]}
    out = await node(state, {})
    assert out["plan"][0]["status"] == "done"


async def test_evaluator_structured_output_openai_pass():
    model = StructuredModel([VerdictSchema(scores=RubricScores(goal=5, accuracy=5, completeness=5))])
    node = make_evaluator_node(model, OPENAI_SETTINGS, SCREEN)
    state = {"plan": [_running(result="良い結果", attempts=1)]}
    out = await node(state, {})
    assert out["plan"][0]["status"] == "done"
    assert model.bound_schema is VerdictSchema


async def test_evaluator_structured_output_openai_retry():
    model = StructuredModel(
        [VerdictSchema(scores=RubricScores(goal=2, accuracy=2, completeness=2), feedback="具体性が不足")]
    )
    node = make_evaluator_node(model, OPENAI_SETTINGS, SCREEN)
    state = {"plan": [_running(result="不十分", attempts=0)]}
    out = await node(state, {})
    assert out["plan"][0]["status"] == "pending"
    assert out["plan"][0]["feedback_history"] == ["具体性が不足"]


async def test_evaluator_separates_system_and_user_roles():
    model = ScriptedModel([_VERDICT_PASS])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    state = {"plan": [_running(desc="天気を調べる", result="晴れだった", attempts=1)]}
    await node(state, {})
    system, human = model.calls[0]
    assert isinstance(system, SystemMessage)
    assert isinstance(human, HumanMessage)
    assert "タスク評価者" in system.content
    assert "従わ" in system.content  # データ内の指示に従わない旨のガード
    assert "天気を調べる" in human.content and "晴れだった" in human.content


async def test_evaluator_includes_all_prior_feedback_on_retry():
    # 過去の全指摘を持つステップ (= retry 再評価) を評価すると、評価プロンプトに全指摘が入る。
    model = ScriptedModel([_VERDICT_PASS])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    state = {"plan": [_running(result="改善した結果", feedback_history=["指摘A", "指摘B"], attempts=2)]}
    await node(state, {})
    _, human = model.calls[0]
    assert "<prior_feedback>" in human.content
    assert "指摘A" in human.content and "指摘B" in human.content


async def test_evaluator_accumulates_feedback_across_retries():
    # retry のたびに新しい指摘が履歴へ積まれる (既存指摘は消えない)。
    model = ScriptedModel([_verdict_retry("指摘B")])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    state = {"plan": [_running(result="まだ不十分", feedback_history=["指摘A"], attempts=1)]}
    out = await node(state, {})
    assert out["plan"][0]["feedback_history"] == ["指摘A", "指摘B"]


async def test_evaluator_omits_prior_feedback_on_first_eval():
    # 初回評価 (履歴なし) では prior_feedback セクションを差し込まない。
    model = ScriptedModel([_VERDICT_PASS])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    state = {"plan": [_running(result="良い結果", attempts=1)]}
    await node(state, {})
    _, human = model.calls[0]
    assert "<prior_feedback>" not in human.content


async def test_evaluator_retry_sets_feedback_and_pends():
    model = ScriptedModel([_verdict_retry("もっと具体的に")])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    state = {"plan": [_running(result="不十分", attempts=1)]}
    out = await node(state, {})
    assert out["plan"][0]["status"] == "pending"
    assert out["plan"][0]["feedback_history"] == ["もっと具体的に"]


async def test_evaluator_retry_budget_downgrades_to_fail():
    model = ScriptedModel([_verdict_retry("もう一度")])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    # attempts=2 > max_step_retries=1 → fail に格下げして前進
    state = {"plan": [_running(result="不十分", attempts=2)]}
    out = await node(state, {})
    assert out["plan"][0]["status"] == "failed"
    assert out["failure_notes"]


async def test_evaluator_replan_sets_needs_replan():
    model = ScriptedModel([_VERDICT_REPLAN])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    state = {"plan": [_running(result="ずれた結果", attempts=1)], "replan_count": 0}
    out = await node(state, {})
    assert out["needs_replan"] is True
    assert out["failure_notes"]


async def test_evaluator_replan_budget_downgrades_to_fail():
    model = ScriptedModel([_VERDICT_REPLAN])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    state = {
        "plan": [_running(result="ずれた結果", attempts=1)],
        "replan_count": SETTINGS.max_replans,
    }
    out = await node(state, {})
    assert out["needs_replan"] is False
    assert out["plan"][0]["status"] == "failed"


async def test_evaluator_evaluates_round_in_parallel():
    # 2ステップ同時評価。両方 pass を返す
    model = ScriptedModel([_VERDICT_PASS, _VERDICT_PASS])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    plan = [
        _running(id=1, result="結果1", attempts=1),
        _running(id=2, result="結果2", attempts=1),
    ]
    out = await node({"plan": plan}, {})
    assert [s["status"] for s in out["plan"]] == ["done", "done"]
    assert len(model.calls) == 2


async def test_evaluator_score_threshold_decides_pass_or_retry():
    # 4/4/3=合計11 → 73点 ≥ 70 で pass、3/3/3=60点 < 70 で retry。閾値で機械的に分岐する。
    near_pass = '{"scores": {"goal": 4, "accuracy": 4, "completeness": 3}, "flawed": false, "feedback": ""}'
    just_under = '{"scores": {"goal": 3, "accuracy": 3, "completeness": 3}, "flawed": false, "feedback": "もう少し"}'
    node = make_evaluator_node(ScriptedModel([near_pass]), SETTINGS, SCREEN)
    out = await node({"plan": [_running(result="ほぼ良い", attempts=1)]}, {})
    assert out["plan"][0]["status"] == "done"

    node = make_evaluator_node(ScriptedModel([just_under]), SETTINGS, SCREEN)
    out = await node({"plan": [_running(result="そこそこ", attempts=0)]}, {})
    assert out["plan"][0]["status"] == "pending"


async def test_evaluator_threshold_is_configurable():
    # 同じ 60点でも、閾値を 50 に下げれば pass になる (config で調整可能であることの確認)。
    just_60 = '{"scores": {"goal": 3, "accuracy": 3, "completeness": 3}, "flawed": false, "feedback": ""}'
    lenient = Settings(eval_pass_threshold=50)
    node = make_evaluator_node(ScriptedModel([just_60]), lenient, SCREEN)
    out = await node({"plan": [_running(result="そこそこ", attempts=1)]}, {})
    assert out["plan"][0]["status"] == "done"


async def test_evaluator_ignores_non_running_steps():
    model = ScriptedModel([])  # 評価対象がなければ LLM は呼ばれない
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    state = {"plan": [_step(result="x", status="done"), _step(status="pending")]}
    out = await node(state, {})
    assert out == {}
    assert model.calls == []


async def test_evaluator_parse_failure_falls_back_to_pass():
    model = ScriptedModel(["壊れた出力", "また壊れた出力"])
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    state = {"plan": [_running(result="それなりの結果", attempts=1)]}
    out = await node(state, {})
    assert out["plan"][0]["status"] == "done"


async def test_evaluator_survives_unexpected_exception_in_round():
    """eval_one 内で想定外例外 (ここでは description キー欠落) が出ても、gather が node 全体を
    巻き込まず pass で前進する (executor.run_one と対称な「例外を漏らさない」契約)。"""
    model = ScriptedModel([])  # LLM 到達前に例外が出るので呼ばれない
    node = make_evaluator_node(model, SETTINGS, SCREEN)
    broken = {"id": 1, "depends_on": [], "status": "running", "result": "x", "attempts": 1}
    out = await node({"plan": [broken]}, {})
    assert out["plan"][0]["status"] == "done"  # pass フォールバックで確定
    assert model.calls == []


# ---- synthesizer ----


async def test_synthesizer_returns_ai_message():
    model = ScriptedModel(["最終回答です"])
    node = make_synthesizer_node(model, SETTINGS, FakeStore(), SCREEN)
    state = {"goal": LONG_GOAL, "plan": [_step(result="結果A", status="done")]}
    out = await node(state, {})
    assert isinstance(out["messages"][0], AIMessage)
    assert out["messages"][0].content == "最終回答です"


async def test_synthesizer_separates_system_and_user_roles():
    model = ScriptedModel(["最終回答です"])
    node = make_synthesizer_node(model, SETTINGS, FakeStore(), SCREEN)
    state = {"goal": LONG_GOAL, "plan": [_step("天気を調べる", result="晴れ", status="done")]}
    await node(state, {})
    system, human = model.calls[0]
    assert isinstance(system, SystemMessage)
    assert isinstance(human, HumanMessage)
    assert "最終回答者" in system.content
    assert LONG_GOAL in human.content and "晴れ" in human.content


async def test_synthesizer_falls_back_to_mechanical_concat():
    model = ScriptedModel([RuntimeError("生成失敗")])
    node = make_synthesizer_node(model, SETTINGS, FakeStore(), SCREEN)
    state = {
        "goal": LONG_GOAL,
        "plan": [_step("天気を調べる", result="晴れ", status="done")],
        "failure_notes": ["ステップ2: 未達成"],
    }
    out = await node(state, {})
    text = out["messages"][0].content
    assert "天気を調べる" in text
    assert "晴れ" in text
    assert "ステップ2: 未達成" in text


# ---- screening (構造化データの絞り込み) ----

_NEWS_DATA = [
    {
        "tool": "news_search",
        "artifact": [
            {"id": "n001", "title": "A", "summary": "sa", "url": "ua", "body": "ba"},
            {"id": "n002", "title": "B", "summary": "sb", "url": "ub", "body": "bb"},
            {"id": "n003", "title": "C", "summary": "sc", "url": "uc", "body": "bc"},
        ],
    }
]


def test_apply_data_selection_projects_fields_keeping_values():
    # keep_fields 指定 → 各 dict が id/title だけになり、値は元のまま (書き換えない)
    sel = DataSelectionSchema(selections=[{"index": 0, "keep_fields": ["id", "title"]}])
    out = apply_data_selection(_NEWS_DATA, sel)
    assert out == [
        {
            "tool": "news_search",
            "artifact": [
                {"id": "n001", "title": "A"},
                {"id": "n002", "title": "B"},
                {"id": "n003", "title": "C"},
            ],
        }
    ]


def test_apply_data_selection_filters_items_and_ignores_out_of_range():
    # keep_items で上位2件のみ + 範囲外 index(99) と未知フィールド(nope) は無視
    sel = DataSelectionSchema(
        selections=[{"index": 0, "keep_fields": ["id", "nope"], "keep_items": [0, 1, 99]}]
    )
    out = apply_data_selection(_NEWS_DATA, sel)
    assert out == [{"tool": "news_search", "artifact": [{"id": "n001"}, {"id": "n002"}]}]


def test_apply_data_selection_empty_selection_returns_full():
    # 空選択 → 全量フォールバック (誤って全ドロップしない)
    assert apply_data_selection(_NEWS_DATA, DataSelectionSchema(selections=[])) == _NEWS_DATA


def test_apply_data_selection_unlisted_entry_dropped_else_full_fallback():
    multi = [
        {"tool": "a", "artifact": [{"x": 1}]},
        {"tool": "b", "artifact": [{"y": 2}]},
    ]
    # index 1 のみ選択 → エントリ0 はドロップ
    out = apply_data_selection(multi, DataSelectionSchema(selections=[{"index": 1}]))
    assert out == [{"tool": "b", "artifact": [{"y": 2}]}]


async def test_screen_step_data_applies_llm_selection():
    # LLM が選択指定 JSON を返す → コードが決定論的に射影 (値は LLM 由来でなく元データ)
    model = ScriptedModel(['{"selections": [{"index": 0, "keep_fields": ["id"]}]}'])
    out = await screen_step_data(model, _NEWS_DATA, purpose="記事IDだけ必要", settings=SETTINGS)
    assert out == [
        {"tool": "news_search", "artifact": [{"id": "n001"}, {"id": "n002"}, {"id": "n003"}]}
    ]


async def test_screen_step_data_empty_skips_llm():
    model = ScriptedModel([])  # 呼ばれたら raise
    assert await screen_step_data(model, None, purpose="x", settings=SETTINGS) is None
    assert await screen_step_data(model, [], purpose="x", settings=SETTINGS) == []
    assert model.calls == []


async def test_screen_step_data_falls_back_to_full_on_failure():
    # 選択取得が全滅 (空スクリプトで例外) → 全量フォールバック (前進性)
    model = ScriptedModel([RuntimeError("接続断"), RuntimeError("接続断")])
    out = await screen_step_data(model, _NEWS_DATA, purpose="x", settings=SETTINGS)
    assert out == _NEWS_DATA
