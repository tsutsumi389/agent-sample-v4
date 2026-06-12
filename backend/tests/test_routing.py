"""routing.py (条件付きエッジ純関数) のテスト — 停止性の境界値を固定する。"""

from app.agent.routing import Limits, route_after_evaluation, route_after_orchestrator

LIMITS = Limits(max_executor_runs=8, max_step_retries=1, max_replans=1)


def _plan(n: int) -> list[dict]:
    return [
        {"id": i + 1, "description": f"s{i+1}", "status": "pending", "result": "", "attempts": 0}
        for i in range(n)
    ]


def test_orchestrator_route_plan():
    assert route_after_orchestrator({"route": "plan"}) == "planner"


def test_orchestrator_route_missing_or_unknown_falls_back_to_responder():
    assert route_after_orchestrator({}) == "responder"
    assert route_after_orchestrator({"route": "direct"}) == "responder"
    assert route_after_orchestrator({"route": "???"}) == "responder"


def test_evaluation_budget_exhausted_always_synthesizer():
    state = {
        "executor_runs": 8,
        "evaluation": {"verdict": "retry"},  # verdict に関係なく予算が最優先
        "plan": _plan(3),
        "current_step": 0,
    }
    assert route_after_evaluation(state, limits=LIMITS) == "synthesizer"


def test_evaluation_retry_goes_to_executor():
    state = {"executor_runs": 1, "evaluation": {"verdict": "retry"}, "plan": _plan(2)}
    assert route_after_evaluation(state, limits=LIMITS) == "executor"


def test_evaluation_replan_goes_to_planner():
    state = {"executor_runs": 1, "evaluation": {"verdict": "replan"}, "plan": _plan(2)}
    assert route_after_evaluation(state, limits=LIMITS) == "planner"


def test_evaluation_pass_with_remaining_steps_goes_to_executor():
    state = {
        "executor_runs": 1,
        "evaluation": {"verdict": "pass"},
        "plan": _plan(2),
        "current_step": 1,
    }
    assert route_after_evaluation(state, limits=LIMITS) == "executor"


def test_evaluation_pass_on_last_step_goes_to_synthesizer():
    state = {
        "executor_runs": 2,
        "evaluation": {"verdict": "pass"},
        "plan": _plan(2),
        "current_step": 2,
    }
    assert route_after_evaluation(state, limits=LIMITS) == "synthesizer"


def test_evaluation_missing_evaluation_moves_forward():
    state = {"executor_runs": 1, "plan": _plan(1), "current_step": 1}
    assert route_after_evaluation(state, limits=LIMITS) == "synthesizer"
