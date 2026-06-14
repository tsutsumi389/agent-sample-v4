"""チェックポインタからの会話履歴復元。"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.agent.parsing import content_to_text as _content_to_text
from app.agent.ui import coerce_ui


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
        out = {
            "role": "tool",
            "content": _content_to_text(msg.content),
            "id": msg.id,
            "tool_call_id": msg.tool_call_id,
            "name": msg.name,
        }
        # ライブ stream と同じ coerce_ui を共有し、リロード復元でも UI を再水和する。
        ui = coerce_ui(getattr(msg, "artifact", None), msg.tool_call_id or msg.id)
        if ui is not None:
            out["ui"] = ui
        return out
    return None  # SystemMessage 等は API へ出さない


async def get_thread_messages(agent, thread_id: str, user_id: str) -> list[dict]:
    config = {
        "configurable": {"thread_id": thread_id, "langgraph_user_id": user_id}
    }
    snapshot = await agent.aget_state(config)
    messages = (snapshot.values or {}).get("messages", [])
    return [m for m in (_to_api_message(msg) for msg in messages) if m is not None]
