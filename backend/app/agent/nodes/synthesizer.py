"""シンセサイザー: ステップ結果を統合して最終回答 (AIMessage) を生成する。

LLM 失敗時はステップ結果の機械的な連結で回答する — 回答ゼロで終わることが構造上ない。
"""

import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.nodes.common import safe_stream_writer
from app.agent.parsing import content_to_text
from app.agent.prompts import SYNTHESIZER_SYSTEM, synthesizer_user
from app.core.config import Settings

logger = logging.getLogger(__name__)

_STATUS_LABEL = {"done": "完了", "failed": "失敗", "running": "中断", "pending": "未着手"}


def make_synthesizer_node(model, settings: Settings):
    async def synthesizer_node(state: dict, config: RunnableConfig) -> dict:
        writer = safe_stream_writer()
        writer({"status": "回答を作成中", "phase": "synthesize"})

        goal = state.get("goal", "")
        plan = state.get("plan") or []
        step_summaries = (
            "\n".join(
                f"{s['id']}. [{_STATUS_LABEL.get(s.get('status', 'pending'), s.get('status'))}] "
                f"{s['description']}\n   結果: {s.get('result') or '(なし)'}"
                for s in plan
            )
            or "(実行されたステップはありません)"
        )
        failure_notes = state.get("failure_notes") or []
        failure_section = (
            "未完了・失敗した項目:\n" + "\n".join(f"- {n}" for n in failure_notes)
            if failure_notes
            else ""
        )
        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM),
            HumanMessage(
                content=synthesizer_user(
                    goal=goal, step_summaries=step_summaries, failure_section=failure_section
                )
            ),
        ]
        try:
            response = await model.ainvoke(messages, config)
            text = content_to_text(response.content).strip()
            if not text:
                raise ValueError("synthesizer の出力が空でした")
            message = AIMessage(content=text)
        except Exception:
            logger.exception("最終回答の生成に失敗したため、結果の機械的な連結で回答します")
            parts = ["各ステップの実行結果:", step_summaries]
            if failure_section:
                parts.append(failure_section)
            message = AIMessage(content="\n\n".join(parts))
        return {"messages": [message]}

    return synthesizer_node
