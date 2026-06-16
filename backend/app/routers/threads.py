from fastapi import APIRouter, HTTPException, Request

from app.core.state import get_state
from app.schemas.chat import (
    DeletedOut,
    MessagesOut,
    ThreadCreate,
    ThreadListOut,
    ThreadOut,
)
from app.services import threads as threads_service
from app.services.history import get_thread_messages

router = APIRouter()


@router.get("/threads", response_model=ThreadListOut)
async def list_threads(request: Request, user_id: str = "default-user"):
    rows = await threads_service.list_threads(get_state(request).pool, user_id)
    return {"threads": rows}


@router.post("/threads", status_code=201, response_model=ThreadOut)
async def create_thread(body: ThreadCreate, request: Request):
    return await threads_service.create_thread(
        get_state(request).pool, body.user_id, body.title
    )


@router.get(
    "/threads/{thread_id}/messages",
    response_model=MessagesOut,
    response_model_exclude_none=True,
)
async def thread_messages(
    thread_id: str, request: Request, user_id: str = "default-user"
):
    state = get_state(request)
    thread = await threads_service.get_thread(state.pool, thread_id, user_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    messages = await get_thread_messages(state.agent, thread_id, user_id)
    return {"thread_id": thread_id, "messages": messages}


@router.delete("/threads/{thread_id}", response_model=DeletedOut)
async def delete_thread(
    thread_id: str, request: Request, user_id: str = "default-user"
):
    deleted = await threads_service.delete_thread(
        get_state(request).pool, thread_id, user_id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"deleted": True}
