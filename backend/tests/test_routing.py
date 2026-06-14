"""routing.py (条件付きエッジ純関数) のテスト — 停止性の境界値を固定する。"""

from app.agent.routing import Limits, route_after_evaluation, route_after_orchestrator

LIMITS = Limits(max_executor_runs=8, max_step_retries=1, max_replans=1)


def _plan(statuses: list[str], deps: list[list[int]] | None = None) -> list[dict]:
    deps = deps or [[] for _ in statuses]
    return [
        {
            "id": i + 1,
            "description": f"s{i + 1}",
            "depends_on": deps[i],
            "status": st,
            "result": "",
            "attempts": 0,
            "feedback": "",
        }
        for i, st in enumerate(statuses)
    ]


def test_orchestrator_route_plan():
    assert route_after_orchestrator({"route": "plan"}) == "planner"


def test_orchestrator_route_missing_or_unknown_falls_back_to_responder():
    assert route_after_orchestrator({}) == "responder"
    assert route_after_orchestrator({"route": "direct"}) == "responder"
    assert route_after_orchestrator({"route": "???"}) == "responder"


def test_evaluation_budget_exhausted_always_synthesizer():
    # 予算到達が最優先 — pending が残っていても needs_replan でも synthesizer
    state = {"executor_runs": 8, "needs_replan": True, "plan": _plan(["pending"])}
    assert route_after_evaluation(state, limits=LIMITS) == "synthesizer"


def test_ready_pending_step_goes_to_executor():
    # 依存解決済みの pending (retry で戻されたものを含む) が残れば次ラウンドへ
    state = {"executor_runs": 1, "plan": _plan(["done", "pending"])}
    assert route_after_evaluation(state, limits=LIMITS) == "executor"


def test_needs_replan_goes_to_planner():
    state = {"executor_runs": 1, "needs_replan": True, "replan_count": 0, "plan": _plan(["running"])}
    assert route_after_evaluation(state, limits=LIMITS) == "planner"


def test_needs_replan_over_budget_ignored():
    # replan 予算超過時は needs_replan を無視し ready 判定へ (evaluator 側で fail 済みのはず)
    state = {
        "executor_runs": 1,
        "needs_replan": True,
        "replan_count": LIMITS.max_replans,
        "plan": _plan(["done", "failed"]),
    }
    assert route_after_evaluation(state, limits=LIMITS) == "synthesizer"


def test_all_done_goes_to_synthesizer():
    state = {"executor_runs": 2, "plan": _plan(["done", "done"])}
    assert route_after_evaluation(state, limits=LIMITS) == "synthesizer"


def test_pending_with_unmet_dependency_deadlock_goes_to_synthesizer():
    # 依存先が failed で永遠に ready にならない pending は前進不能 → synthesizer で停止
    state = {"executor_runs": 3, "plan": _plan(["failed", "pending"], deps=[[], [1]])}
    assert route_after_evaluation(state, limits=LIMITS) == "synthesizer"


def test_missing_plan_moves_forward_to_synthesizer():
    state = {"executor_runs": 1, "plan": []}
    assert route_after_evaluation(state, limits=LIMITS) == "synthesizer"
