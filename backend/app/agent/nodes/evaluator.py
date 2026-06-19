"""エヴァリュエーター: 直近ラウンドで実行されたステップ群を並列に判定する。

executor が status="running" にしたステップを asyncio.gather で並列評価し、それぞれに
retry / replan の予算管理を適用する。
- 空結果・打ち切りは LLM を呼ばず決定論的に retry / fail。
- パース全滅・例外時は pass (前進フォールバック — 評価器が壊れても全体は完走する)。
- retry / replan の予算超過はここで fail にダウングレードする (routing.py は純関数のまま)。
- retry はステップを pending に戻し (次ラウンドで再実行)、step 専用 feedback を残す。
"""

import asyncio
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.nodes.common import (
    EXECUTION_FAILED_MARKER,
    format_step_data,
    safe_stream_writer,
    screen_step_data,
)
from app.agent.parsing import VerdictSchema, structured_or_parse
from app.agent.prompts import EVALUATOR_SYSTEM, evaluator_user
from app.core.config import Settings

logger = logging.getLogger(__name__)


def make_evaluator_node(model, settings: Settings, screen_model):
    async def evaluator_node(state: dict, config: RunnableConfig) -> dict:
        plan = [dict(s) for s in state.get("plan") or []]
        running = [i for i, s in enumerate(plan) if s.get("status") == "running"]
        if not running:
            # 防御: 評価対象なし → 何も変えず routing が plan 状態で前進判断する
            return {}

        replan_budget_left = state.get("replan_count", 0) < settings.max_replans

        async def eval_one(idx: int) -> tuple[int, str, str]:
            # 例外を漏らさない契約 (executor.run_one と対称)。gather は return_exceptions=False の
            # ため、1件でも例外を出すと他コルーチンを巻き込んで node 全体が落ち、executor が
            # running にしたステップ群が未確定のまま消える。想定外の例外も pass で前進させる。
            step = plan[idx]
            try:
                result = step.get("result") or ""
                # 決定論プレチェック (LLM スキップ)
                if not result.strip() or result.startswith(EXECUTION_FAILED_MARKER):
                    verdict, feedback = (
                        "retry",
                        "前回の実行が失敗または空の結果でした。別のアプローチで再実行してください。",
                    )
                else:
                    # 構造化データは判定に必要な分だけへ絞ってから補助材料として渡す (値は変えず選別のみ)。
                    screened = await screen_step_data(
                        screen_model,
                        step.get("data"),
                        purpose=f"ステップ「{step['description']}」の実行結果が目的を達成しているかの評価",
                        settings=settings,
                    )
                    parsed = await structured_or_parse(
                        model,
                        [
                            SystemMessage(content=EVALUATOR_SYSTEM),
                            HumanMessage(
                                content=evaluator_user(
                                    step_description=step["description"],
                                    result=result,
                                    data=format_step_data(screened),
                                )
                            ),
                        ],
                        VerdictSchema,
                        use_structured=settings.supports_structured_output,
                        fallback=VerdictSchema(verdict="pass", feedback=""),
                    )
                    verdict, feedback = parsed.verdict, parsed.feedback
                # 予算ダウングレード (決定論コード — 停止性の保証)
                if verdict == "retry" and step.get("attempts", 0) > settings.max_step_retries:
                    verdict = "fail"
                if verdict == "replan" and not replan_budget_left:
                    verdict = "fail"
                return idx, verdict, feedback
            except Exception:
                logger.exception("ステップ評価に失敗したため pass で前進します (step=%s)", step.get("id"))
                return idx, "pass", ""

        verdicts = await asyncio.gather(*(eval_one(i) for i in running))

        failure_notes = list(state.get("failure_notes") or [])
        needs_replan = bool(state.get("needs_replan"))
        writer = safe_stream_writer()

        def _note(step: dict, feedback: str) -> str:
            desc = (step.get("description") or "")[:100]
            return f"ステップ{step.get('id')}「{desc}」: {feedback[:200] or '達成できませんでした'}"

        for idx, verdict, feedback in verdicts:
            step = plan[idx]
            if verdict == "pass":
                step["status"] = "done"
                step["feedback"] = ""
            elif verdict == "retry":
                step["status"] = "pending"  # 次ラウンドで再実行 (依存は既に done)
                step["feedback"] = feedback[: settings.feedback_max_chars]
            elif verdict == "replan":
                needs_replan = True
                failure_notes.append(_note(step, feedback))  # status は running (再計画で置換)
            else:  # fail: ステップを諦めて前進
                step["status"] = "failed"
                failure_notes.append(_note(step, feedback))
            writer(
                {
                    "status": f"ステップ {step['id']} を評価: {verdict}",
                    "phase": "evaluate",
                    "step": step["id"],
                    "verdict": verdict,
                }
            )

        return {"plan": plan, "failure_notes": failure_notes, "needs_replan": needs_replan}

    return evaluator_node
