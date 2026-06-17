"""アプリケーション共有状態 (app.state.deps) の型定義とアクセサ。

ルーターは request.app.state.* を動的属性で触る代わりに get_state(request) を経由する
ことで、IDE 補完・型チェックの効く形で依存 (pool/store/agent 等) を参照できる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.store.base import BaseStore
    from langmem import ReflectionExecutor
    from psycopg_pool import AsyncConnectionPool

    from app.core.config import Settings


@dataclass
class AppState:
    """lifespan で組み立て app.state.deps に格納する依存一式。

    skip_startup (テスト) や起動前は settings 以外が None。実運用では lifespan が
    _build_dependencies の結果で置き換える。
    """

    settings: Settings
    pool: AsyncConnectionPool | None = None
    store: BaseStore | None = None
    # build_agent は LangGraph の CompiledStateGraph を返すが未公開型のため Any。
    agent: Any | None = None
    mcp_client: MultiServerMCPClient | None = None
    reflection_executor: ReflectionExecutor | None = None
    # 意味記憶 (構造化プロファイル) 統合用。エピソード記憶用と並走する。
    profile_reflection_executor: ReflectionExecutor | None = None
    tool_info: list[dict] = field(default_factory=list)


def get_state(request: Request) -> AppState:
    """リクエストから型付きの AppState を取り出す。"""
    return request.app.state.deps
