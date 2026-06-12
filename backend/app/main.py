"""FastAPI アプリケーションファクトリ。"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.graph import build_agent
from app.core.config import get_settings
from app.core.db import build_pool, ensure_threads_table, init_persistence
from app.mcp.loader import build_mcp_client, get_mcp_tools, load_mcp_config
from app.memory.manager import build_reflection_executor
from app.memory.tools import langmem_hotpath_tools
from app.routers import chat, health, memory, threads, tools
from app.tools.registry import build_registry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    if settings.skip_startup:
        # テスト用: Postgres / Ollama / MCP なしで起動する
        app.state.pool = None
        app.state.agent = None
        app.state.store = None
        app.state.reflection_executor = None
        app.state.tool_info = []
        yield
        return

    pool = build_pool(settings.database_url)
    await pool.open()
    reflection_executor = None
    try:
        checkpointer, store = await init_persistence(pool, settings)
        await ensure_threads_table(pool)

        registry = build_registry()
        langmem_tools = langmem_hotpath_tools(store)

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

        app.state.pool = pool
        app.state.store = store
        app.state.agent = agent
        app.state.mcp_client = mcp_client
        app.state.reflection_executor = reflection_executor
        app.state.tool_info = (
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

        yield
    finally:
        # 起動が途中で失敗しても必ず後片付けする (executor → pool の順)
        if reflection_executor is not None:
            reflection_executor.shutdown(wait=False, cancel_futures=True)
        await pool.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="agent-sample", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    prefix = "/api"
    app.include_router(health.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)
    app.include_router(threads.router, prefix=prefix)
    app.include_router(memory.router, prefix=prefix)
    app.include_router(tools.router, prefix=prefix)
    return app


app = create_app()
