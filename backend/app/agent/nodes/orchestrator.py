"""オーケストレーター: ターン開始時のスクラッチリセットとルーティング分類。

指示 (SystemMessage) とユーザー要求 (HumanMessage) をロール分離する (プロンプト
インジェクション緩和)。分類は:
- 構造化出力可 (openai): with_structured_output で RouteSchema を直接得る。
- 不可 (ollama): 1語出力＋正規表現部分一致 (JSON を要求しない最も壊れにくい形)。
失敗・例外・短文はすべて direct (既存単一エージェント挙動) へフォールバックする。
"""

import logging
import re

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.nodes.common import last_human_text, safe_stream_writer
from app.agent.parsing import RouteSchema, content_to_text, strip_think
from app.agent.prompts import ORCHESTRATOR_SYSTEM, orchestrator_user
from app.agent.state import fresh_scratch
from app.core.config import Settings

logger = logging.getLogger(__name__)

_PLAN_RE = re.compile(r"\bPLAN\b")


def make_orchestrator_node(model, settings: Settings):
    async def orchestrator_node(state: dict, config: RunnableConfig) -> dict:
        writer = safe_stream_writer()
        goal = last_human_text(state)[: settings.goal_max_chars]
        scratch = fresh_scratch(goal)
        writer({"status": "リクエストを分析中", "phase": "routing"})
        # 決定論プレチェック: 短い入力は LLM を呼ばず direct (挨拶等のレイテンシゼロ化)
        if len(goal) < settings.router_skip_under_chars:
            return scratch
        messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM),
            HumanMessage(content=orchestrator_user(goal)),
        ]
        try:
            if settings.supports_structured_output:
                result = await model.with_structured_output(RouteSchema).ainvoke(messages)
                if isinstance(result, RouteSchema) and result.route == "plan":
                    scratch["route"] = "plan"
            else:
                response = await model.ainvoke(messages)
                text = strip_think(content_to_text(response.content))
                if _PLAN_RE.search(text.upper()):
                    scratch["route"] = "plan"
        except Exception:
            logger.exception("分類に失敗したため direct へフォールバックします")
        return scratch

    return orchestrator_node
