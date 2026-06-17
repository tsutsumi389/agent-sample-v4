"""FastAPI アプリケーションファクトリ。"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.graph import build_agent
from app.core.config import Settings, get_settings
from app.core.db import build_pool, ensure_threads_table, init_persistence
from app.core.state import AppState
from app.mcp.loader import build_mcp_client, get_mcp_tools, load_mcp_config
from app.memory.manager import (
    build_profile_reflection_executor,
    build_reflection_executor,
)
from app.memory.tools import langmem_hotpath_tools
from app.routers import chat, health, memory, threads, tools
from app.tools.registry import build_registry

logger = logging.getLogger(__name__)

API_PREFIX = "/api"

# /api 配下にまとめて登録するルーター。新規追加時はここへ 1 要素足すだけでよい。
ROUTERS = (
    health.router,
    chat.router,
    threads.router,
    memory.router,
    tools.router,
)


async def _build_dependencies(pool, settings: Settings) -> AppState:
    """開いた pool 上に永続化・レジストリ・MCP・エージェント・executor を組み立てる。

    pool の open/close と executor の shutdown は呼び出し側 (lifespan) が管理する。
    ここでは構築のみを担い、結果を型付きの AppState として返す。
    """
    checkpointer, store = await init_persistence(pool, settings)
    await ensure_threads_table(pool)

    registry = build_registry()
    langmem_tools = langmem_hotpath_tools(store)

    mcp_client = None
    mcp_tools = []
    try:
        mcp_client = build_mcp_client(load_mcp_config(settings.mcp_config_file))
        mcp_tools = await get_mcp_tools(mcp_client)
    except Exception:
        logger.exception(
            "MCP ツールの取得に失敗しました。ネイティブツールのみで続行します"
        )
        mcp_client = None

    agent = build_agent(
        settings=settings,
        native_tools=registry.all(),
        langmem_tools=langmem_tools,
        mcp_tools=mcp_tools,
        checkpointer=checkpointer,
        store=store,
    )
    reflection_executor = build_reflection_executor(store, settings)
    profile_reflection_executor = build_profile_reflection_executor(store, settings)

    tool_info = (
        registry.describe()
        + [
            {"name": t.name, "description": t.description, "source": "langmem"}
            for t in langmem_tools
        ]
        + [
            {"name": t.name, "description": t.description, "source": "mcp"}
            for t in mcp_tools
        ]
    )
    return AppState(
        settings=settings,
        pool=pool,
        store=store,
        agent=agent,
        mcp_client=mcp_client,
        reflection_executor=reflection_executor,
        profile_reflection_executor=profile_reflection_executor,
        tool_info=tool_info,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # settings は create_app で生成済み。二重に get_settings() せず共有状態から参照する。
    settings = app.state.deps.settings

    if settings.skip_startup:
        # テスト用: Postgres / Ollama / MCP なしで起動する (空の AppState のまま)
        yield
        return

    state: AppState | None = None
    pool = build_pool(settings.database_url)
    await pool.open()
    try:
        state = await _build_dependencies(pool, settings)
        app.state.deps = state
        yield
    finally:
        # 起動が途中で失敗しても必ず後片付けする (executor → pool の順)
        if state is not None and state.reflection_executor is not None:
            state.reflection_executor.shutdown(wait=False, cancel_futures=True)
        if state is not None and state.profile_reflection_executor is not None:
            state.profile_reflection_executor.shutdown(wait=False, cancel_futures=True)
        await pool.close()


def _configure_cors(app: FastAPI, settings: Settings) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="agent-sample", lifespan=lifespan)
    # 起動前・skip_startup 用の空状態。lifespan が実依存で置き換える。
    app.state.deps = AppState(settings=settings)
    _configure_cors(app, settings)
    for router in ROUTERS:
        app.include_router(router, prefix=API_PREFIX)
    return app


app = create_app()
