"""エグゼキューター: 依存解決済みのステップ群を並列にツール付き ReAct で実行する。

1ラウンドで、depends_on が全て done になった pending ステップを最大
settings.max_parallel_executors 件まで asyncio.gather で同時実行する。各ステップは新規の
スクラッチパッド (スレッド履歴非共有) で実行され、例外・反復上限は打ち切りマーカー付きの
結果として必ず正常リターンする (例外を漏らさない契約 — gather には伝播させない)。
"""

import asyncio
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from app.agent.nodes.common import (
    EXECUTION_FAILED_MARKER,
    ready_step_indices,
    safe_stream_writer,
)
from app.agent.parsing import content_to_text, strip_think
from app.agent.state import PlanStep
from app.core.config import Settings

logger = logging.getLogger(__name__)


def _scoped_prompt(state: dict, plan: list[PlanStep], idx: int, settings: Settings) -> str:
    """スレッド履歴を入れず、goal＋依存ステップ(depends_on)の結果＋今回タスクのみで構築。

    並列実行では逐次の「直前まで」という前提が崩れるため、履歴は依存関係で明示された
    先行ステップ (depends_on) の結果に限定する (独立ステップには履歴を渡さない)。
    """
    step = plan[idx]
    lines = [f"最終目標: {state.get('goal', '')}"]
    by_id = {s["id"]: s for s in plan}
    history: list[str] = []
    budget = settings.executor_history_max_chars
    for dep_id in step.get("depends_on") or []:
        dep = by_id.get(dep_id)
        result = (dep or {}).get("result") or ""
        if not dep or not result:
            continue
        entry = f"- ステップ{dep['id']}「{dep['description'][:100]}」の結果: {result}"
        if len(entry) > budget:
            break
        history.append(entry)
        budget -= len(entry)
    if history:
        lines.append("依存タスクの結果:")
        lines.extend(history)
    lines.append(f"今回のタスク: {step['description']}")
    feedback = step.get("feedback") or ""
    if feedback:
        lines.append(f"前回試行への評価者からの指摘 (必ず改善すること): {feedback}")
    return "\n".join(lines)


def make_executor_node(executor_agent, settings: Settings):
    async def executor_node(state: dict, config: RunnableConfig) -> dict:
        plan = [dict(s) for s in state.get("plan") or []]
        runs = state.get("executor_runs", 0)
        # 依存解決済みの pending を同時実行上限まで。スライスで並列度を制御する。
        ready = ready_step_indices(plan)[: settings.max_parallel_executors]
        if not ready:
            # 防御: 実行対象がない (デッドロック等) 場合も runs を進めて evaluator → 前進させる
            return {"executor_runs": runs + 1}

        writer = safe_stream_writer()
        writer(
            {
                "status": f"{len(ready)}件のステップを並列実行中",
                "phase": "step",
                "parallel": len(ready),
                "total": len(plan),
            }
        )

        async def run_one(idx: int) -> None:
            step = plan[idx]
            writer(
                {
                    "status": f"ステップ {step['id']} を実行中: {step['description'][:40]}",
                    "phase": "step",
                    "step": step["id"],
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
                logger.exception("ステップ実行に失敗しました (step=%s)", step["id"])
                # 例外文字列は HTTP ボディ等で巨大になり得るため型名＋短い要約に縮約し、
                # 正常経路と同じ文字数予算に収める (synthesizer 経由でユーザーに表出し得る)
                result_text = f"{EXECUTION_FAILED_MARKER}{type(exc).__name__}: {str(exc)[:200]})"[
                    : settings.step_result_max_chars
                ]
            # 各コルーチンは異なる idx の dict のみ更新するため競合しない (単一イベントループ)
            step["result"] = result_text
            step["attempts"] = step.get("attempts", 0) + 1
            step["status"] = "running"  # 評価待ち。evaluator が done/pending/failed へ確定する

        # ready は max_parallel_executors 件以下なので gather がそのまま同時実行上限になる
        await asyncio.gather(*(run_one(i) for i in ready))
        return {"plan": plan, "executor_runs": runs + len(ready)}

    return executor_node
