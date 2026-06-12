"""プランナー: 要求をステップ列に分解する。

会話全履歴は渡さない (goal＋ツールカタログ＋再計画時の失敗情報のみ)。
JSON パース全滅時は単一ステップ計画 [goal] に縮退し、必ず前進する。
"""

import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from app.agent.nodes.common import safe_stream_writer
from app.agent.parsing import PlanSchema, parse_with_retry
from app.agent.prompts import PLANNER_PROMPT, PLANNER_REPLAN_SECTION
from app.agent.state import PlanStep
from app.core.config import Settings

logger = logging.getLogger(__name__)

_CATALOG_DESC_MAX = 80
_REPLAN_SUMMARY_MAX = 1500


def make_planner_node(model, tools: list[BaseTool], settings: Settings):
    catalog = (
        ", ".join(f"{t.name} ({(t.description or '').strip()[:_CATALOG_DESC_MAX]})" for t in tools)
        or "(なし)"
    )

    async def planner_node(state: dict, config: RunnableConfig) -> dict:
        writer = safe_stream_writer()
        goal = state.get("goal", "")
        prev_plan = state.get("plan") or []
        is_replan = bool(prev_plan)

        replan_section = ""
        if is_replan:
            done = (
                " / ".join(
                    f"{s['description'][:100]} → {s.get('result', '')[:200]}"
                    for s in prev_plan
                    if s.get("status") == "done"
                )
                or "(なし)"
            )
            notes = " / ".join(state.get("failure_notes") or []) or "(なし)"
            feedback = ((state.get("evaluation") or {}).get("feedback") or "")[
                : settings.feedback_max_chars
            ]
            replan_section = PLANNER_REPLAN_SECTION.format(
                done_summaries=done[:_REPLAN_SUMMARY_MAX],
                failure_notes=notes[:_REPLAN_SUMMARY_MAX],
                feedback=feedback,
            )

        writer({"status": "実行計画を作成中", "phase": "plan"})
        prompt = PLANNER_PROMPT.format(
            max_steps=settings.max_plan_steps,
            tool_catalog=catalog,
            goal=goal,
            replan_section=replan_section,
        )
        parsed = await parse_with_retry(
            model,
            [HumanMessage(content=prompt)],
            PlanSchema,
            fallback=PlanSchema(steps=[]),
        )
        steps = [s for s in parsed.steps if s.strip()][: settings.max_plan_steps]
        if not steps:
            # 最終フォールバック: 単一ステップ計画 (executor 1回の単一 ReAct 実行に縮退)
            steps = [goal or "ユーザーの要求に回答する"]
            logger.warning("計画の生成に失敗したため単一ステップ計画に縮退します")

        plan: list[PlanStep] = [
            {"id": i + 1, "description": desc, "status": "pending", "result": "", "attempts": 0}
            for i, desc in enumerate(steps)
        ]
        writer(
            {
                "status": f"計画を作成しました ({len(plan)}ステップ)",
                "phase": "plan",
                "plan": [{"id": s["id"], "description": s["description"]} for s in plan],
            }
        )
        updates: dict = {"plan": plan, "current_step": 0, "evaluator_feedback": ""}
        if is_replan:
            updates["replan_count"] = state.get("replan_count", 0) + 1
        return updates

    return planner_node
