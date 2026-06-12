"""マルチエージェントグラフの State 定義。

messages のみが意味的に永続 (threads/history API・LangMem reflection と互換)。
それ以外はターン内スクラッチで、orchestrator が毎ターン fresh_scratch() で必ず上書きする
(チェックポイントに残留した前ターンの値を決定論的に潰す)。
全制御フィールドは NotRequired のため、旧チェックポイント (単一エージェント時代の
{"messages": [...]} のみの state) もそのまま読める。
"""

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

Route = Literal["direct", "plan"]
Verdict = Literal["pass", "retry", "replan", "fail"]


class PlanStep(TypedDict):
    id: int
    description: str  # 実行タスク (自然文)
    status: Literal["pending", "running", "done", "failed"]
    result: str  # 実行結果要約 (step_result_max_chars で切詰め済み)
    attempts: int  # このステップの executor 実行回数


class AgentGraphState(TypedDict):
    # ---- 永続チャネル ----
    messages: Annotated[list[AnyMessage], add_messages]

    # ---- ターン内スクラッチ (毎ターン orchestrator がリセット) ----
    route: NotRequired[Route]
    goal: NotRequired[str]  # 今ターンのユーザー要求 (goal_max_chars で切詰め済み)
    plan: NotRequired[list[PlanStep]]
    current_step: NotRequired[int]
    replan_count: NotRequired[int]  # <= settings.max_replans
    executor_runs: NotRequired[int]  # <= settings.max_executor_runs (大域停止条件)
    evaluation: NotRequired[dict]  # {"verdict": Verdict, "feedback": str}
    evaluator_feedback: NotRequired[str]  # retry 時に executor へ渡す
    failure_notes: NotRequired[list[str]]  # 諦めたステップの記録 (synthesizer が未完了を明示する)


def fresh_scratch(goal: str) -> dict:
    """ターン開始時のスクラッチ初期化。前ターンのチェックポイント残留値を必ず潰す。"""
    return {
        "route": "direct",
        "goal": goal,
        "plan": [],
        "current_step": 0,
        "replan_count": 0,
        "executor_runs": 0,
        "evaluation": {},
        "evaluator_feedback": "",
        "failure_notes": [],
    }
