from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.schemas.chat import ChatRequest
from app.services import threads as threads_service
from app.services.streaming import stream_agent

router = APIRouter()


async def _ensure_thread(body: ChatRequest, request: Request) -> ChatRequest:
    """SSE 開始前 (依存解決時) に 404 を返す。スレッドは POST /api/threads で事前作成必須。"""
    thread = await threads_service.get_thread(
        request.app.state.pool, body.thread_id, body.user_id
    )
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return body


@router.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(
    request: Request, body: ChatRequest = Depends(_ensure_thread)
) -> AsyncIterator[ServerSentEvent]:
    state = request.app.state
    async for event in stream_agent(
        agent=state.agent,
        pool=state.pool,
        reflection_executor=state.reflection_executor,
        message=body.message,
        thread_id=body.thread_id,
        user_id=body.user_id,
        reflection_delay_seconds=state.settings.reflection_delay_seconds,
    ):
        yield event
