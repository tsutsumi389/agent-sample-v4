from fastapi import APIRouter, Request

from app.memory.store_query import delete_memory, list_memories
from app.schemas.chat import DeletedOut, MemoriesOut

router = APIRouter()


@router.get("/memory", response_model=MemoriesOut)
async def get_memories(
    request: Request,
    user_id: str = "default-user",
    query: str | None = None,
    limit: int = 20,
):
    memories = await list_memories(request.app.state.store, user_id, query, limit)
    return {"user_id": user_id, "memories": memories}


@router.delete("/memory/{key}", response_model=DeletedOut)
async def remove_memory(key: str, request: Request, user_id: str = "default-user"):
    await delete_memory(request.app.state.store, user_id, key)
    return {"deleted": True}
