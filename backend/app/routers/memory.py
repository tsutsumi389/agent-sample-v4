from fastapi import APIRouter, Request

from app.memory.forget import select_candidates
from app.memory.store_query import (
    batch_delete_memories,
    delete_memory,
    list_memories,
    search_forget_candidates,
    verify_forgotten,
)
from app.schemas.chat import (
    DeletedOut,
    ForgetConfirmIn,
    ForgetConfirmOut,
    ForgetPreviewIn,
    ForgetPreviewOut,
    MemoriesOut,
)

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


@router.post("/memory/forget/preview", response_model=ForgetPreviewOut)
async def forget_preview(body: ForgetPreviewIn, request: Request):
    """スコープ付き一括忘却の候補を返す (非破壊・確認用)。"""
    store = request.app.state.store
    candidates = await search_forget_candidates(
        store, body.user_id, body.query, body.limit
    )
    selected = select_candidates(candidates)
    return {"candidates": selected}


@router.post("/memory/forget/confirm", response_model=ForgetConfirmOut)
async def forget_confirm(body: ForgetConfirmIn, request: Request):
    """承認済みの key 群を一括削除し、削除後に Absence 検証して結果を返す。

    注意: 本アプリは無認証・単一ユーザー前提のため、user_id/keys はクライアント任意
    パラメータ (既存の DELETE /memory/{key} と同じ信頼モデル)。確認ゲートの強制は
    フロント/エージェントの 2 段階プロトコルと keys の上限 (ForgetConfirmIn) で担保する。
    """
    store = request.app.state.store
    result = await batch_delete_memories(store, body.user_id, body.keys)
    verification = await verify_forgotten(store, body.user_id, result["deleted_keys"])
    return {
        "deleted_count": result["deleted_count"],
        "verified": verification["ok"],
        "leaked_keys": verification["leaked_keys"],
    }
