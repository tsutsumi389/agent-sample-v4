"""各ノードの単体テスト (フェイクモデル注入、DB / Ollama なし)。"""

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.common import EXECUTION_FAILED_MARKER
from app.agent.nodes.evaluator import make_evaluator_node
from app.agent.nodes.executor import make_executor_node
from app.agent.nodes.orchestrator import make_orchestrator_node
from app.agent.nodes.planner import make_planner_node
from app.agent.nodes.synthesizer import make_synthesizer_node
from app.core.config import Settings
from tests.fakes import ScriptedExecutorAgent, ScriptedModel

SETTINGS = Settings()
LONG_GOAL = "東京と大阪の明日の天気を調べて、移動手段ごとの所要時間と合わせて比較表にまとめてください"


def _human_state(text: str) -> dict:
    return {"messages": [HumanMessage(content=text)]}


def _step(desc: str = "調査する", *, result: str = "", attempts: int = 0, status: str = "pending") -> dict:
    return {"id": 1, "description": desc, "status": status, "result": result, "attempts": attempts}


# ---- orchestrator ----


async def test_orchestrator_short_input_skips_llm():
    model = ScriptedModel([])  # 呼ばれたら raise する
    node = make_orchestrator_node(model, SETTINGS)
    out = await node(_human_state("こんにちは"), {})
    assert out["route"] == "direct"
    assert model.calls == []


async def test_orchestrator_classifies_plan():
    model = ScriptedModel(["<think>複雑そう</think>PLAN"])
    node = make_orchestrator_node(model, SETTINGS)
    out = await node(_human_state(LONG_GOAL), {})
    assert out["route"] == "plan"


async def test_orchestrator_falls_back_to_direct_on_exception():
    model = ScriptedModel([RuntimeError("接続断")])
    node = make_orchestrator_node(model, SETTINGS)
    out = await node(_human_state(LONG_GOAL), {})
    assert out["route"] == "direct"


async def test_orchestrator_resets_scratch():
    model = ScriptedModel(["DIRECT"])
    node = make_orchestrator_node(model, SETTINGS)
    stale = _human_state(LONG_GOAL) | {
        "plan": [_step()],
        "current_step": 3,
        "executor_runs": 7,
        "replan_count": 1,
        "failure_notes": ["古いメモ"],
    }
    out = await node(stale, {})
    assert out["plan"] == []
    assert out["current_step"] == 0
    assert out["executor_runs"] == 0
    assert out["replan_count"] == 0
    assert out["failure_notes"] == []


# ---- planner ----


async def test_planner_builds_plan():
    model = ScriptedModel(['{"steps": ["天気を調べる", "比較表を作る"]}'])
    node = make_planner_node(model, [], SETTINGS)
    out = await node({"goal": LONG_GOAL}, {})
    assert [s["description"] for s in out["plan"]] == ["天気を調べる", "比較表を作る"]
    assert out["plan"][0] == {
        "id": 1,
        "description": "天気を調べる",
        "status": "pending",
        "result": "",
        "attempts": 0,
    }
    assert out["current_step"] == 0


async def test_planner_truncates_to_max_steps():
    steps = [f"手順{i}" for i in range(1, 9)]
    model = ScriptedModel([f'{{"steps": {steps}}}'.replace("'", '"')])
    node = make_planner_node(model, [], SETTINGS)
    out = await node({"goal": LONG_GOAL}, {})
    assert len(out["plan"]) == SETTINGS.max_plan_steps


async def test_planner_falls_back_to_single_step_plan():
    model = ScriptedModel(["壊れた出力", "また壊れた出力"])
    node = make_planner_node(model, [], SETTINGS)
    out = await node({"goal": LONG_GOAL}, {})
    assert [s["description"] for s in out["plan"]] == [LONG_GOAL]


async def test_planner_replan_increments_count_and_includes_failures():
    model = ScriptedModel(['{"steps": ["やり直す"]}'])
    node = make_planner_node(model, [], SETTINGS)
    state = {
        "goal": LONG_GOAL,
        "plan": [_step("旧ステップ", result="旧結果", status="done")],
        "replan_count": 0,
        "failure_notes": ["ステップ2が失敗"],
        "evaluation": {"verdict": "replan", "feedback": "計画が粗すぎる"},
    }
    out = await node(state, {})
    assert out["replan_count"] == 1
    prompt = model.calls[0][0].content
    assert "前回の計画は失敗しました" in prompt
    assert "ステップ2が失敗" in prompt
    assert "計画が粗すぎる" in prompt


# ---- executor ----


async def test_executor_runs_step_and_records_result():
    agent = ScriptedExecutorAgent(["<think>作業中</think>調査結果です"])
    node = make_executor_node(agent, SETTINGS)
    state = {"goal": LONG_GOAL, "plan": [_step()], "current_step": 0, "executor_runs": 0}
    out = await node(state, {"configurable": {"thread_id": "t1"}})
    assert out["plan"][0]["result"] == "調査結果です"
    assert out["plan"][0]["attempts"] == 1
    assert out["executor_runs"] == 1
    # recursion_limit と configurable が伝搬している
    assert agent.configs[0]["recursion_limit"] == SETTINGS.executor_recursion_limit
    assert agent.configs[0]["configurable"]["thread_id"] == "t1"


async def test_executor_never_raises_on_agent_failure():
    agent = ScriptedExecutorAgent([RuntimeError("ツール大爆発")])
    node = make_executor_node(agent, SETTINGS)
    state = {"goal": LONG_GOAL, "plan": [_step()], "current_step": 0}
    out = await node(state, {})
    assert out["plan"][0]["result"].startswith(EXECUTION_FAILED_MARKER)


async def test_executor_without_steps_still_advances_runs():
    agent = ScriptedExecutorAgent([])
    node = make_executor_node(agent, SETTINGS)
    out = await node({"plan": [], "current_step": 0, "executor_runs": 2}, {})
    assert out == {"executor_runs": 3}


# ---- evaluator ----


async def test_evaluator_empty_result_retries_without_llm():
    model = ScriptedModel([])
    node = make_evaluator_node(model, SETTINGS)
    state = {"plan": [_step(result="", attempts=1)], "current_step": 0}
    out = await node(state, {})
    assert out["evaluation"]["verdict"] == "retry"
    assert model.calls == []


async def test_evaluator_pass_advances_step():
    model = ScriptedModel(['{"verdict": "pass", "feedback": ""}'])
    node = make_evaluator_node(model, SETTINGS)
    state = {"plan": [_step(result="良い結果", attempts=1)], "current_step": 0}
    out = await node(state, {})
    assert out["evaluation"]["verdict"] == "pass"
    assert out["current_step"] == 1
    assert out["plan"][0]["status"] == "done"


async def test_evaluator_retry_budget_downgrades_to_fail():
    model = ScriptedModel(['{"verdict": "retry", "feedback": "もう一度"}'])
    node = make_evaluator_node(model, SETTINGS)
    # attempts=2 > max_step_retries=1 → fail に格下げして前進
    state = {"plan": [_step(result="不十分", attempts=2)], "current_step": 0}
    out = await node(state, {})
    assert out["evaluation"]["verdict"] == "fail"
    assert out["current_step"] == 1
    assert out["plan"][0]["status"] == "failed"
    assert out["failure_notes"]


async def test_evaluator_replan_budget_downgrades_to_fail():
    model = ScriptedModel(['{"verdict": "replan", "feedback": "計画から見直し"}'])
    node = make_evaluator_node(model, SETTINGS)
    state = {
        "plan": [_step(result="ずれた結果", attempts=1)],
        "current_step": 0,
        "replan_count": SETTINGS.max_replans,
    }
    out = await node(state, {})
    assert out["evaluation"]["verdict"] == "fail"


async def test_evaluator_parse_failure_falls_back_to_pass():
    model = ScriptedModel(["壊れた出力", "また壊れた出力"])
    node = make_evaluator_node(model, SETTINGS)
    state = {"plan": [_step(result="それなりの結果", attempts=1)], "current_step": 0}
    out = await node(state, {})
    assert out["evaluation"]["verdict"] == "pass"
    assert out["current_step"] == 1


# ---- synthesizer ----


async def test_synthesizer_returns_ai_message():
    model = ScriptedModel(["最終回答です"])
    node = make_synthesizer_node(model, SETTINGS)
    state = {"goal": LONG_GOAL, "plan": [_step(result="結果A", status="done")]}
    out = await node(state, {})
    assert isinstance(out["messages"][0], AIMessage)
    assert out["messages"][0].content == "最終回答です"


async def test_synthesizer_falls_back_to_mechanical_concat():
    model = ScriptedModel([RuntimeError("生成失敗")])
    node = make_synthesizer_node(model, SETTINGS)
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
