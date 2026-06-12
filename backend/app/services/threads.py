"""threads テーブル CRUD。"""

import uuid

from psycopg_pool import AsyncConnectionPool

DEFAULT_TITLE = "新しい会話"
TITLE_MAX_LEN = 40


def derive_title(message: str) -> str:
    title = message.strip().splitlines()[0] if message.strip() else DEFAULT_TITLE
    return title[:TITLE_MAX_LEN] or DEFAULT_TITLE


async def create_thread(pool: AsyncConnectionPool, user_id: str, title: str | None) -> dict:
    thread_id = f"t_{uuid.uuid4().hex}"
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            INSERT INTO threads (thread_id, user_id, title)
            VALUES (%s, %s, %s)
            RETURNING thread_id, title, created_at, updated_at
            """,
            (thread_id, user_id, title or DEFAULT_TITLE),
        )
        return await cur.fetchone()


async def list_threads(pool: AsyncConnectionPool, user_id: str) -> list[dict]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM threads WHERE user_id = %s
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )
        return await cur.fetchall()


async def get_thread(pool: AsyncConnectionPool, thread_id: str, user_id: str) -> dict | None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM threads "
            "WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        return await cur.fetchone()


async def delete_thread(pool: AsyncConnectionPool, thread_id: str, user_id: str) -> bool:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "DELETE FROM threads WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        return cur.rowcount > 0


async def touch_and_title(
    pool: AsyncConnectionPool, thread_id: str, candidate_title: str
) -> str | None:
    """updated_at を更新し、タイトルが初期値ならユーザーメッセージ由来の候補に差し替える。

    返り値は更新後のタイトル (スレッドが無ければ None)。
    """
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT title FROM threads WHERE thread_id = %s", (thread_id,)
        )
        row = await cur.fetchone()
        if row is None:
            return None
        title = candidate_title if row["title"] == DEFAULT_TITLE else row["title"]
        await conn.execute(
            "UPDATE threads SET title = %s, updated_at = now() WHERE thread_id = %s",
            (title, thread_id),
        )
        return title
