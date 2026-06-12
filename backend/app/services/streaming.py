"""エージェントの astream を SSE イベント列へ変換するブリッジ。

FastAPI 0.136 のネイティブ SSE (response_class=EventSourceResponse) を使う。
keep-alive コメント (15秒ごと) は FastAPI のルーティング層が自動挿入する。
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi.sse import ServerSentEvent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from app.agent.context import AgentContext
from app.memory.manager import schedule_reflection
from app.services import threads as threads_service

logger = logging.getLogger(__name__)

TOOL_RESULT_MAX_CHARS = 2000


def _content_to_text(content: Any) -> str:
    """message content (str | list[block]) をプレーンテキストへ。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


async def stream_agent(
    *,
    agent,
    pool,
    reflection_executor,
    message: str,
    thread_id: str,
    user_id: str,
    reflection_delay_seconds: int,
) -> AsyncIterator[ServerSentEvent]:
    # タイトルはユーザーメッセージだけから決まるため、ストリーム開始前に確定する
    # (途中で stop されてもタイトルが残るように)
    title = await threads_service.touch_and_title(
        pool, thread_id, threads_service.derive_title(message)
    )
    completed = False
    turn_messages: list[Any] = [HumanMessage(content=message)]
    try:
        async for mode, chunk in agent.astream(
            {"messages": [{"role": "user", "content": message}]},
            config={
                "configurable": {"thread_id": thread_id, "langgraph_user_id": user_id}
            },
            context=AgentContext(user_id=user_id),
            stream_mode=["messages", "updates", "custom"],
        ):
            if mode == "messages":
                msg_chunk, metadata = chunk
                if isinstance(msg_chunk, AIMessageChunk):
                    text = _content_to_text(msg_chunk.content)
                    if text:
                        yield ServerSentEvent(
                            event="token",
                            data={
                                "content": text,
                                "node": metadata.get("langgraph_node", "model"),
                            },
                        )
            elif mode == "updates":
                if not isinstance(chunk, dict):
                    continue
                for update in chunk.values():
                    if not isinstance(update, dict):
                        continue
                    for msg in update.get("messages") or []:
                        if isinstance(msg, AIMessage):
                            turn_messages.append(msg)
                            for tc in msg.tool_calls:
                                yield ServerSentEvent(
                                    event="tool_call",
                                    data={
                                        "id": tc.get("id"),
                                        "name": tc.get("name"),
                                        "args": tc.get("args"),
                                    },
                                )
                        elif isinstance(msg, ToolMessage):
                            turn_messages.append(msg)
                            yield ServerSentEvent(
                                event="tool_result",
                                data={
                                    "id": msg.tool_call_id,
                                    "name": msg.name,
                                    "content": _content_to_text(msg.content)[
                                        :TOOL_RESULT_MAX_CHARS
                                    ],
                                },
                            )
            elif mode == "custom":
                status = (
                    chunk.get("status") if isinstance(chunk, dict) else str(chunk)
                )
                yield ServerSentEvent(event="progress", data={"status": status})
        completed = True
        yield ServerSentEvent(event="done", data={"thread_id": thread_id, "title": title})
    except Exception as exc:
        logger.exception("ストリーミング中にエラーが発生しました (thread_id=%s)", thread_id)
        yield ServerSentEvent(event="error", data={"message": str(exc)})
    finally:
        # ターンが完了した場合のみ記憶のリフレクションをスケジュールする。
        # thread_id による debounce で保留中タスクがキャンセルされても内容が
        # 失われないよう、今ターンの差分ではなくスレッドの全会話メッセージ
        # (エージェント最終状態) を渡す。
        if completed and reflection_executor is not None and len(turn_messages) > 1:
            reflection_messages = turn_messages
            try:
                snapshot = await agent.aget_state(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "langgraph_user_id": user_id,
                        }
                    }
                )
                state_messages = (snapshot.values or {}).get("messages") or []
                if state_messages:
                    reflection_messages = state_messages
            except Exception:
                logger.exception(
                    "リフレクション用の状態取得に失敗しました (thread_id=%s)。"
                    "今ターンのメッセージのみで続行します",
                    thread_id,
                )
            schedule_reflection(
                reflection_executor,
                user_id=user_id,
                thread_id=thread_id,
                messages=reflection_messages,
                after_seconds=reflection_delay_seconds,
            )
