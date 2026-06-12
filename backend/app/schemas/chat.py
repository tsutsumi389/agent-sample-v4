"""API スキーマ (api_contract.md に厳密一致)。"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, field_serializer


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
