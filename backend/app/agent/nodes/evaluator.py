"""エヴァリュエーター: ステップ結果を判定し、retry / replan の予算管理を行う。

- 空結果・打ち切りは LLM を呼ばず決定論的に retry / fail。
- パース全滅・例外時は pass (前進フォールバック — 評価器が壊れても全体は完走する)。
- retry / replan の予算超過はここで fail にダウングレードする (routing.py は純関数のまま)。
"""

import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from app.agent.nodes.common import EXECUTION_FAILED_MARKER, safe_stream_writer
from app.agent.parsing import VerdictSchema, parse_with_retry
from app.agent.prompts import EVALUATOR_PROMPT
from app.core.config import Settings

logger = logging.getLogger(__name__)


def make_evaluator_node(model, settings: Settings):
    async def evaluator_node(state: dict, config: RunnableConfig) -> dict:
        plan = [dict(s) for s in state.get("plan") or []]
        idx = state.get("current_step", 0)
        if idx >= len(plan):
            # 防御: 評価対象なし → 前進 (route_after_evaluation が synthesizer へ抜く)
            return {"evaluation": {"verdict": "pass", "feedback": ""}}

        step = plan[idx]
        result = step.get("result") or ""

        # 決定論プレチェック (LLM スキップ)
        if not result.strip() or result.startswith(EXECUTION_FAILED_MARKER):
            verdict, feedback = (
                "retry",
                "前回の実行が失敗または空の結果でした。別のアプローチで再実行してください。",
            )
        else:
            parsed = await parse_with_retry(
                model,
                [
                    HumanMessage(
                        content=EVALUATOR_PROMPT.format(
                            step_description=step["description"],
                            result=result,
                        )
                    )
                ],
                VerdictSchema,
                fallback=VerdictSchema(verdict="pass", feedback=""),
            )
            verdict, feedback = parsed.verdict, parsed.feedback

        # 予算ダウングレード (決定論コード — 停止性の保証)
        if verdict == "retry" and step.get("attempts", 0) > settings.max_step_retries:
            verdict = "fail"
        if verdict == "replan" and state.get("replan_count", 0) >= settings.max_replans:
            verdict = "fail"

        failure_notes = list(state.get("failure_notes") or [])
        note = f"ステップ{idx + 1}「{step['description'][:100]}」: {feedback[:200] or '達成できませんでした'}"
        updates: dict
        if verdict == "pass":
            step["status"] = "done"
            plan[idx] = step
            updates = {"plan": plan, "current_step": idx + 1, "evaluator_feedback": ""}
        elif verdict == "retry":
            updates = {"evaluator_feedback": feedback[: settings.feedback_max_chars]}
        elif verdict == "replan":
            failure_notes.append(note)
            updates = {"failure_notes": failure_notes}
        else:  # fail: ステップを諦めて前進
            step["status"] = "failed"
            plan[idx] = step
            failure_notes.append(note)
            updates = {
                "plan": plan,
                "current_step": idx + 1,
                "failure_notes": failure_notes,
                "evaluator_feedback": "",
            }
        updates["evaluation"] = {"verdict": verdict, "feedback": feedback}

        writer = safe_stream_writer()
        writer(
            {
                "status": f"ステップ {idx + 1} を評価: {verdict}",
                "phase": "evaluate",
                "verdict": verdict,
            }
        )
        return updates

    return evaluator_node
