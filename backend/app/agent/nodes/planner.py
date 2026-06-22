"""プランナー: 要求をステップ列に分解する。

会話全履歴は渡さない (goal＋ツールカタログ＋再計画時の失敗情報のみ)。
JSON パース全滅時は単一ステップ計画 [goal] に縮退し、必ず前進する。
"""

import asyncio
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from app.agent.nodes.common import (
    format_step_data,
    safe_stream_writer,
    sanitize_dependencies,
    screen_step_data,
)
from app.agent.parsing import PlanSchema, structured_or_parse
from app.agent.prompts import (
    PLANNER_REPLAN_SECTION,
    PLANNER_SYSTEM,
    planner_user,
    profile_section,
)
from app.agent.state import PlanStep
from app.core.config import Settings
from app.memory.profile import get_profile_text
from app.memory.tools import _user_id_from_config

logger = logging.getLogger(__name__)

_CATALOG_DESC_MAX = 80
_REPLAN_SUMMARY_MAX = 1500


def make_planner_node(
    model, tools: list[BaseTool], settings: Settings, store: BaseStore, screen_model
):
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
            done_steps = [s for s in prev_plan if s.get("status") == "done"]
            done = (
                " / ".join(
                    f"{s['description'][:100]} → {s.get('result', '')[:200]}" for s in done_steps
                )
                or "(なし)"
            )
            # result(テキスト) は _REPLAN_SUMMARY_MAX で切り詰めるが、構造化データ (artifact) は
            # 機械処理用のため切り詰めない。再計画に必要な分だけへ絞ってから (値は変えず選別のみ) 別途付ける。
            done_summaries = done[:_REPLAN_SUMMARY_MAX]
            screened = await asyncio.gather(
                *(screen_step_data(screen_model, s.get("data"), purpose=goal, settings=settings) for s in done_steps)
            )
            data_blocks = [
                f"ステップ{s['id']} のデータ: {dt}"
                for s, d in zip(done_steps, screened)
                if (dt := format_step_data(d))
            ]
            if data_blocks:
                done_summaries += "\n構造化データ:\n" + "\n".join(data_blocks)
            # 評価者の retry/replan feedback は failure_notes に「ステップ名: 指摘」として
            # 取り込まれているため、replan の理由はここに一本化する。
            notes = " / ".join(state.get("failure_notes") or []) or "(なし)"
            replan_section = PLANNER_REPLAN_SECTION.format(
                done_summaries=done_summaries,
                failure_notes=notes[:_REPLAN_SUMMARY_MAX],
            )

        writer({"status": "実行計画を作成中", "phase": "plan"})
        # 意味記憶 (制約・好み) を System 末尾へ注入し、計画立案=ステップ分解に反映する。
        profile_text = await get_profile_text(store, _user_id_from_config(config))
        system = PLANNER_SYSTEM.format(max_steps=settings.max_plan_steps) + profile_section(
            profile_text
        )
        user = planner_user(goal=goal, tool_catalog=catalog, replan_section=replan_section)
        parsed = await structured_or_parse(
            model,
            [SystemMessage(content=system), HumanMessage(content=user)],
            PlanSchema,
            use_structured=settings.supports_structured_output,
            fallback=PlanSchema(steps=[]),
        )
        # 空 description を除いた生存ステップに、LLM の元 id (無ければ間引き前の出現位置) を
        # キーとして付ける。これを基準に depends_on をリマップするため、間引き前の位置で振る。
        survivors: list[tuple[int, str, str, list[int]]] = []
        for pos, s in enumerate(parsed.steps, start=1):
            desc = s.description.strip()
            if not desc:
                continue
            old_key = s.id if s.id is not None else pos
            survivors.append((old_key, desc, s.instruction, s.depends_on))
        survivors = survivors[: settings.max_plan_steps]

        if not survivors:
            # 最終フォールバック: 単一ステップ計画 (executor 1回の単一 ReAct 実行に縮退)
            plan: list[PlanStep] = [
                {
                    "id": 1,
                    "description": goal or "ユーザーの要求に回答する",
                    "instruction": "",
                    "depends_on": [],
                    "status": "pending",
                    "result": "",
                    "attempts": 0,
                    "feedback": "",
                }
            ]
            logger.warning("計画の生成に失敗したため単一ステップ計画に縮退します")
        else:
            # 元 id → 出現順 1..N へリマップ。これにより LLM が 0始まり/飛び番 id を出しても、
            # 間引き・max_steps 切捨てがあっても depends_on の参照が壊れない。生存ステップ外
            # (間引かれた/範囲外) を指す依存は除去する。重複 id は後勝ち (sanitize が最終防御)。
            remap = {old_key: i + 1 for i, (old_key, _, _, _) in enumerate(survivors)}
            plan = [
                {
                    "id": i + 1,
                    "description": desc,
                    "instruction": instruction,
                    "depends_on": [remap[d] for d in deps if d in remap],
                    "status": "pending",
                    "result": "",
                    "attempts": 0,
                    "feedback": "",
                }
                for i, (_, desc, instruction, deps) in enumerate(survivors)
            ]
        sanitize_dependencies(plan)
        writer(
            {
                "status": f"計画を作成しました ({len(plan)}ステップ)",
                "phase": "plan",
                "plan": [
                    {"id": s["id"], "description": s["description"], "depends_on": s["depends_on"]}
                    for s in plan
                ],
            }
        )
        updates: dict = {"plan": plan, "needs_replan": False}
        if is_replan:
            updates["replan_count"] = state.get("replan_count", 0) + 1
        return updates

    return planner_node
