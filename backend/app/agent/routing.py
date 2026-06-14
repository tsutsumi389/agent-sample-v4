"""条件付きエッジの判定関数 (LLM を一切呼ばない純関数)。

停止性の保証はここに集約する:
- executor 通過ごとに executor_runs が (実行ステップ数分) 単調増加し、上限到達で無条件に
  synthesizer へ抜ける。executor は ready がある限り必ず 1 件以上実行するため前進が保証される。
- retry / replan の予算超過は evaluator ノード側で fail にダウングレード済み。
- 実行可能 (依存解決済み pending) なステップが残らなくなれば synthesizer へ抜ける。
  循環・依存先 failed で詰んだステップは ready にならないため、ここで自然に停止する。
- synthesizer / responder は END 直結。
よって LLM がどんな出力をしても有限ステップで END に到達する。
"""

from dataclasses import dataclass

from app.agent.nodes.common import ready_step_indices
from app.core.config import Settings


@dataclass(frozen=True)
class Limits:
    max_executor_runs: int
    max_step_retries: int
    max_replans: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "Limits":
        return cls(
            max_executor_runs=settings.max_executor_runs,
            max_step_retries=settings.max_step_retries,
            max_replans=settings.max_replans,
        )


def route_after_orchestrator(state: dict) -> str:
    """不明値・欠損は必ず responder (フェイルセーフ: 既存単一エージェント挙動へ縮退)。"""
    return "planner" if state.get("route") == "plan" else "responder"


def route_after_evaluation(state: dict, *, limits: Limits) -> str:
    # 1) 大域予算が最優先 (停止性の保証)
    if state.get("executor_runs", 0) >= limits.max_executor_runs:
        return "synthesizer"
    # 2) replan 要求 (予算内のみ — 予算超過分は evaluator が fail にダウングレード済み)
    if state.get("needs_replan") and state.get("replan_count", 0) < limits.max_replans:
        return "planner"
    # 3) 実行可能 (依存解決済み pending) なステップが残れば次ラウンドへ。retry で pending に
    #    戻されたステップもここに含まれる。詰んだステップは ready にならず synthesizer へ抜ける。
    if ready_step_indices(state.get("plan") or []):
        return "executor"
    return "synthesizer"
