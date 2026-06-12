"""DB 接続プールと永続化レイヤ (checkpointer / store / threads テーブル)。"""

from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import Settings


def build_pool(database_url: str) -> AsyncConnectionPool:
    """単一の共有 AsyncConnectionPool を作る (open は呼び出し側で await)。"""
    return AsyncConnectionPool(
        conninfo=database_url,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )


async def init_persistence(
    pool: AsyncConnectionPool, settings: Settings
) -> tuple[AsyncPostgresSaver, AsyncPostgresStore]:
    """checkpointer と store を共有プール上に構築し、setup() を実行する (冪等)。"""
    checkpointer = AsyncPostgresSaver(pool)
    store = AsyncPostgresStore(
        pool,
        index={
            "dims": 768,
            "embed": OllamaEmbeddings(
                model=settings.embed_model,
                base_url=settings.ollama_base_url,
                num_ctx=8192,
            ),
            "fields": ["$"],
        },
    )
    await checkpointer.setup()
    await store.setup()
    return checkpointer, store


async def ensure_threads_table(pool: AsyncConnectionPool) -> None:
    """スレッドメタ情報テーブル (チェックポインタには無い) を作成する。"""
    async with pool.connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                thread_id  TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                title      TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
