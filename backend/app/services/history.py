"""チェックポインタからの会話履歴復元。"""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def _content_to_text(content: Any) -> str:
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


def _to_api_message(msg: BaseMessage) -> dict | None:
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": _content_to_text(msg.content), "id": msg.id}
    if isinstance(msg, AIMessage):
        out: dict = {
            "role": "assistant",
            "content": _content_to_text(msg.content),
            "id": msg.id,
        }
        if msg.tool_calls:
            out["tool_calls"] = [
                {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args")}
                for tc in msg.tool_calls
            ]
        return out
    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": _content_to_text(msg.content),
            "id": msg.id,
            "tool_call_id": msg.tool_call_id,
            "name": msg.name,
        }
    return None  # SystemMessage 等は API へ出さない


async def get_thread_messages(agent, thread_id: str, user_id: str) -> list[dict]:
    config = {
        "configurable": {"thread_id": thread_id, "langgraph_user_id": user_id}
    }
    snapshot = await agent.aget_state(config)
    messages = (snapshot.values or {}).get("messages", [])
    return [m for m in (_to_api_message(msg) for msg in messages) if m is not None]
