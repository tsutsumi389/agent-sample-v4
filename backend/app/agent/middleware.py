"""responder へ意味記憶 (ユーザープロファイル) を注入する middleware。

create_agent の system_prompt は静的なので、ターン開始時の動的注入はモデル呼び出しを
ラップする middleware で行う。runtime.context (AgentContext) から user_id を得て
プロファイルをロードし、システムメッセージ末尾に profile_section を足す。

planner / synthesizer はノード関数が SystemMessage を自前で組むため、そちら側で直接
注入する (この middleware は responder 専用)。
"""

import logging

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from langgraph.store.base import BaseStore

from app.agent.prompts import SYSTEM_PROMPT, profile_section
from app.memory.profile import get_profile_text

logger = logging.getLogger(__name__)


class UserProfileMiddleware(AgentMiddleware):
    """responder のモデル呼び出し直前に意味記憶を System プロンプト末尾へ注入する。"""

    def __init__(self, store: BaseStore) -> None:
        super().__init__()
        self._store = store

    async def awrap_model_call(self, request, handler):
        context = getattr(request.runtime, "context", None)
        user_id = getattr(context, "user_id", None)
        if not user_id:
            return await handler(request)
        try:
            profile_text = await get_profile_text(self._store, user_id)
        except Exception:
            logger.exception("プロファイルのロードに失敗しました (user_id=%s)", user_id)
            profile_text = ""
        section = profile_section(profile_text)
        if not section:
            # プロファイルが空なら素通し (注入による差分なし = キャッシュも維持)
            return await handler(request)
        base = request.system_prompt or SYSTEM_PROMPT
        request = request.override(system_message=SystemMessage(content=base + section))
        return await handler(request)
