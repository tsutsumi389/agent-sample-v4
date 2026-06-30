"""マルチエージェントグラフの State 定義。

messages のみが意味的に永続 (threads/history API・LangMem reflection と互換)。
それ以外はターン内スクラッチで、orchestrator が毎ターン fresh_scratch() で必ず上書きする
(チェックポイントに残留した前ターンの値を決定論的に潰す)。
全制御フィールドは NotRequired のため、旧チェックポイント (単一エージェント時代の
{"messages": [...]} のみの state) もそのまま読める。

実行は「ラウンド」単位。1ラウンドで依存 (depends_on) が解決済みの pending ステップ群を
executor が並列実行し、evaluator がその結果群を並列評価する。進行状態は plan の各ステップの
status / depends_on から純関数で導けるため、逐次時代の current_step は廃止した。
"""

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

Route = Literal["direct", "plan"]
Verdict = Literal["pass", "retry", "replan", "fail"]


class PlanStep(TypedDict):
    id: int
    description: str  # 実行タスク (自然文)。UI 表示・要約・依存見出し用の短い成果物名。
    # executor 向けの具体的・パーソナライズ済み実行手順。executor はユーザープロファイルを
    # 参照しないため、planner が制約・好みを実行条件として落とし込んでここに書く。
    # 空/欠落なら executor は description にフォールバックする (旧 plan との後方互換)。
    instruction: NotRequired[str]
    depends_on: list[int]  # 先行ステップ ID 群 (全て done で実行可能 / 空なら即時実行可)
    status: Literal["pending", "running", "done", "failed"]
    result: str  # 実行結果要約 (step_result_max_chars で切詰め済み / LLM・人間向けテキスト)
    attempts: int  # このステップの executor 実行回数
    # retry 毎に評価者が積む、このステップ専用の改善指示の履歴 (古い→新しい)。executor は
    # 全指摘を改善対象として読み、evaluator は再評価時に全指摘の反映度を判断材料にする。
    feedback_history: NotRequired[list[str]]
    # ツール (content_and_artifact 形式) が返した構造化データの回収先。executor が
    # ReAct 実行中の ToolMessage.artifact を集めてここへ格納する。後続ノードは result
    # (テキスト要約) に加え、機械処理用にこちらも読む (executor の依存渡し / evaluator /
    # synthesizer / planner 再計画)。切り詰めると不正確になるため全量を保持し、後続へ渡す際は
    # 各エージェントが必要な分だけを LLM スクリーニングで抜き出す (common.screen_step_data。
    # 値は変えず選別のみ)。整形は common.format_step_data。各要素は {"tool", "artifact"}。
    data: NotRequired[list[dict]]


class AgentGraphState(TypedDict):
    # ---- 永続チャネル ----
    messages: Annotated[list[AnyMessage], add_messages]

    # ---- ターン内スクラッチ (毎ターン orchestrator がリセット) ----
    route: NotRequired[Route]
    goal: NotRequired[str]  # 今ターンのユーザー要求 (goal_max_chars で切詰め済み)
    plan: NotRequired[list[PlanStep]]
    replan_count: NotRequired[int]  # <= settings.max_replans
    executor_runs: NotRequired[int]  # 実行したステップ数の累計 (<= settings.max_executor_runs)
    needs_replan: NotRequired[bool]  # 直近ラウンドで replan 判定が出た (routing が planner へ振る)
    failure_notes: NotRequired[list[str]]  # 諦めたステップの記録 (synthesizer が未完了を明示する)


def fresh_scratch(goal: str) -> dict:
    """ターン開始時のスクラッチ初期化。前ターンのチェックポイント残留値を必ず潰す。"""
    return {
        "route": "direct",
        "goal": goal,
        "plan": [],
        "replan_count": 0,
        "executor_runs": 0,
        "needs_replan": False,
        "failure_notes": [],
    }
