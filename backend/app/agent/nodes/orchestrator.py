"""オーケストレーター: ターン開始時のスクラッチリセットと「goal の文脈化 + ルーティング分類」。

指示 (SystemMessage) とユーザー入力・会話履歴 (HumanMessage) をロール分離する (プロンプト
インジェクション緩和)。会話履歴を踏まえて (1) goal を自己完結な要求文へ言い換え、(2) direct/plan を
分類する。両方を 1 回の structured_or_parse で得る (openai=構造化出力 / ollama=JSON テキストパース、
どちらも RouteSchema の validator が揺れを吸収)。

失敗・例外・空/過長のリライト・履歴なし短文はすべて direct + 生 goal へ graceful フォールバックし、
回答ゼロや情報欠落を起こさない。
"""

import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.nodes.common import last_human_text, recent_history_text, safe_stream_writer
from app.agent.parsing import RouteSchema, structured_or_parse
from app.agent.prompts import (
    ORCHESTRATOR_SYSTEM,
    orchestrator_history_section,
    orchestrator_user,
)
from app.agent.state import fresh_scratch
from app.core.config import Settings

logger = logging.getLogger(__name__)


def make_orchestrator_node(model, settings: Settings):
    async def orchestrator_node(state: dict, config: RunnableConfig) -> dict:
        writer = safe_stream_writer()
        raw_goal = last_human_text(state)[: settings.goal_max_chars]
        scratch = fresh_scratch(raw_goal)
        writer({"status": "リクエストを分析中", "phase": "routing"})

        history = recent_history_text(state)
        # 決定論プレチェック: 短い入力 かつ 履歴なし のときだけ LLM を省く (初回の挨拶等のゼロレイテンシ)。
        # 履歴があれば短文でも文脈化が要る (「もっと例を」等のフォローアップ救済) ため必ず LLM を通す。
        if len(raw_goal) < settings.router_skip_under_chars and not history:
            return scratch

        history_section = orchestrator_history_section(history)
        messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM),
            HumanMessage(content=orchestrator_user(raw_goal, history_section)),
        ]
        result = await structured_or_parse(
            model,
            messages,
            RouteSchema,
            use_structured=settings.supports_structured_output,
            fallback=RouteSchema(route="direct", goal=raw_goal),
        )
        scratch["route"] = result.route
        # リライトが妥当なら採用、空・異常膨張は生 goal にフォールバック (中途切断せず棄却し情報欠落を防ぐ)。
        rewritten = result.goal.strip()
        if rewritten and len(rewritten) <= settings.goal_max_chars:
            scratch["goal"] = rewritten
        return scratch

    return orchestrator_node
