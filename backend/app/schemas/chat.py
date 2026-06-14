"""API スキーマ (api_contract.md に厳密一致)。"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_serializer

from app.memory.forget import DEFAULT_MAX_KEYS


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_id: str = "default-user"


class ThreadCreate(BaseModel):
    user_id: str = "default-user"
    title: str | None = None


class ThreadOut(BaseModel):
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def _ser_dt(self, dt: datetime) -> str:
        return _iso_z(dt)


class ThreadListOut(BaseModel):
    threads: list[ThreadOut]


class MessageOut(BaseModel):
    role: str
    content: str
    id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class MessagesOut(BaseModel):
    thread_id: str
    messages: list[MessageOut]


class MemoryOut(BaseModel):
    key: str
    content: str
    namespace: list[str]
    updated_at: str | None
    score: float | None = None


class MemoriesOut(BaseModel):
    user_id: str
    memories: list[MemoryOut]


class ToolOut(BaseModel):
    name: str
    description: str
    source: str


class ToolsOut(BaseModel):
    tools: list[ToolOut]


class DeletedOut(BaseModel):
    deleted: bool = True


class ForgetPreviewIn(BaseModel):
    query: str
    user_id: str = "default-user"
    limit: int = 20


class ForgetCandidate(BaseModel):
    key: str
    content: str
    score: float | None = None
    updated_at: str | None = None


class ForgetPreviewOut(BaseModel):
    candidates: list[ForgetCandidate]


class ForgetConfirmIn(BaseModel):
    # max_length は confirm 経路でも大量削除の安全弁 (DEFAULT_MAX_KEYS) を効かせるため。
    # 超過時は 422 で拒否し、唯一の不可逆パスを保護する。
    keys: list[str] = Field(max_length=DEFAULT_MAX_KEYS)
    user_id: str = "default-user"


class ForgetConfirmOut(BaseModel):
    deleted_count: int
    verified: bool
    leaked_keys: list[str]
