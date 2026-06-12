"""エグゼキューター: 現在のステップをツール付き ReAct サブエージェントで実行する。

毎ステップ新規のスクラッチパッド (スレッド履歴非共有) で実行し、例外・反復上限は
打ち切りマーカー付きの結果として必ず正常リターンする (例外を漏らさない契約)。
"""

import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from app.agent.nodes.common import EXECUTION_FAILED_MARKER, safe_stream_writer
from app.agent.parsing import content_to_text, strip_think
from app.agent.state import PlanStep
from app.core.config import Settings

logger = logging.getLogger(__name__)


def _scoped_prompt(state: dict, plan: list[PlanStep], idx: int, settings: Settings) -> str:
    """スレッド履歴を入れず、goal＋過去ステップ結果 (直近優先・上限付き) ＋今回タスクのみで構築。"""
    lines = [f"最終目標: {state.get('goal', '')}"]
    history: list[str] = []
    budget = settings.executor_history_max_chars
    for step in reversed(plan[:idx]):  # 直近優先で予算内に詰める
        result = step.get("result") or ""
        if not result:
            continue
        entry = f"- ステップ{step['id']}「{step['description'][:100]}」の結果: {result}"
        if len(entry) > budget:
            break
        history.append(entry)
        budget -= len(entry)
    if history:
        lines.append("これまでの結果:")
        lines.extend(reversed(history))
    lines.append(f"今回のタスク: {plan[idx]['description']}")
    feedback = state.get("evaluator_feedback") or ""
    if feedback:
        lines.append(f"前回試行への評価者からの指摘 (必ず改善すること): {feedback}")
    return "\n".join(lines)


def make_executor_node(executor_agent, settings: Settings):
    async def executor_node(state: dict, config: RunnableConfig) -> dict:
        plan = [dict(s) for s in state.get("plan") or []]
        runs = state.get("executor_runs", 0) + 1
        idx = state.get("current_step", 0)
        if idx >= len(plan):
            # 防御: 実行対象がない場合も runs を進めて evaluator → synthesizer に抜ける
            return {"executor_runs": runs}

        step = plan[idx]
        writer = safe_stream_writer()
        writer(
            {
                "status": f"ステップ {idx + 1}/{len(plan)} を実行中: {step['description'][:40]}",
                "phase": "step",
                "step": idx + 1,
                "total": len(plan),
            }
        )
        try:
            result = await executor_agent.ainvoke(
                {"messages": [HumanMessage(content=_scoped_prompt(state, plan, idx, settings))]},
                config={**(config or {}), "recursion_limit": settings.executor_recursion_limit},
            )
            messages = (result or {}).get("messages") or []
            text = strip_think(content_to_text(messages[-1].content)) if messages else ""
            result_text = text[: settings.step_result_max_chars]
        except Exception as exc:
            logger.exception("ステップ実行に失敗しました (step=%s)", idx + 1)
            # 例外文字列は HTTP ボディ等で巨大になり得るため型名＋短い要約に縮約し、
            # 正常経路と同じ文字数予算に収める (synthesizer 経由でユーザーに表出し得る)
            result_text = f"{EXECUTION_FAILED_MARKER}{type(exc).__name__}: {str(exc)[:200]})"[
                : settings.step_result_max_chars
            ]

        step["result"] = result_text
        step["attempts"] = step.get("attempts", 0) + 1
        step["status"] = "running"
        plan[idx] = step
        return {"plan": plan, "executor_runs": runs}

    return executor_node
