"""シンセサイザー: ステップ結果を統合して最終回答 (AIMessage) を生成する。

LLM 失敗時はステップ結果の機械的な連結で回答する — 回答ゼロで終わることが構造上ない。
"""

import asyncio
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.store.base import BaseStore

from app.agent.nodes.common import format_step_data, safe_stream_writer, screen_step_data
from app.agent.parsing import content_to_text
from app.agent.prompts import SYNTHESIZER_SYSTEM, profile_section, synthesizer_user
from app.core.config import Settings
from app.memory.profile import get_profile_text
from app.memory.tools import _user_id_from_config

logger = logging.getLogger(__name__)

_STATUS_LABEL = {"done": "完了", "failed": "失敗", "running": "中断", "pending": "未着手"}


def make_synthesizer_node(model, settings: Settings, store: BaseStore, screen_model):
    async def synthesizer_node(state: dict, config: RunnableConfig) -> dict:
        writer = safe_stream_writer()
        writer({"status": "回答を作成中", "phase": "synthesize"})

        goal = state.get("goal", "")
        plan = state.get("plan") or []

        # 各ステップの構造化データを最終回答に必要な分だけへ絞る (値は変えず選別のみ)。並列実行。
        screened_data = await asyncio.gather(
            *(screen_step_data(screen_model, s.get("data"), purpose=goal, settings=settings) for s in plan)
        )

        def _summary(s: dict, data: list[dict] | None) -> str:
            label = _STATUS_LABEL.get(s.get("status", "pending"), s.get("status"))
            line = f"{s['id']}. [{label}] {s['description']}\n   結果: {s.get('result') or '(なし)'}"
            # 絞り込み後の構造化データを併記する (機械処理用のため無切詰め)。
            data_text = format_step_data(data)
            if data_text:
                line += f"\n   構造化データ: {data_text}"
            return line

        step_summaries = (
            "\n".join(_summary(s, d) for s, d in zip(plan, screened_data))
            or "(実行されたステップはありません)"
        )
        failure_notes = state.get("failure_notes") or []
        failure_section = (
            "未完了・失敗した項目:\n" + "\n".join(f"- {n}" for n in failure_notes)
            if failure_notes
            else ""
        )
        # 意味記憶 (プロファイル) を System 末尾へ注入し、最終回答をパーソナライズする。
        profile_text = await get_profile_text(store, _user_id_from_config(config))
        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM + profile_section(profile_text)),
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
