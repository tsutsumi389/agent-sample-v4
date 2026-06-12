"""条件付きエッジの判定関数 (LLM を一切呼ばない純関数)。

停止性の保証はここに集約する:
- executor 通過ごとに executor_runs が単調増加し、上限到達で無条件に synthesizer へ抜ける。
- retry / replan の予算超過は evaluator ノード側で fail にダウングレード済み。
- synthesizer / responder は END 直結。
よって LLM がどんな出力をしても有限ステップで END に到達する。
"""

from dataclasses import dataclass

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
    verdict = (state.get("evaluation") or {}).get("verdict", "pass")  # 欠損は前進
    if verdict == "retry":
        return "executor"  # 予算ダウングレードは evaluator ノード側で適用済み
    if verdict == "replan":
        return "planner"
    # pass / fail → evaluator が current_step を進めてある
    if state.get("current_step", 0) < len(state.get("plan") or []):
        return "executor"
    return "synthesizer"
