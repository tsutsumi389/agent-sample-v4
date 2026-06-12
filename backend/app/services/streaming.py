"""エージェントの astream を SSE イベント列へ変換するブリッジ。

FastAPI 0.136 のネイティブ SSE (response_class=EventSourceResponse) を使う。
keep-alive コメント (15秒ごと) は FastAPI のルーティング層が自動挿入する。

マルチエージェントグラフ対応:
- subgraphs=True で responder / executor サブグラフ内のイベントも受信する
  (この場合 astream は (ns, mode, chunk) の3タプルを返す)。
- トークンは USER_FACING_NODES (responder / synthesizer) 由来のみ配信する。
  一次防御はモデル側の tags=["nostream"] で、これは二重防御。
- responder はサブグラフ内 updates と親レベル updates (全履歴の再掲) の両方で
  メッセージが届くため、親レベル側はスキップし、tool_call/tool_result は id で dedupe する。
- custom イベント (計画・ステップ進捗) は dict をそのまま progress の data として透過する
  (必ず "status" を含む規約のため、旧フロントは data.status だけ読めば従来どおり動く)。
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi.sse import ServerSentEvent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from app.agent.context import AgentContext
from app.agent.parsing import content_to_text as _content_to_text
from app.core.config import Settings
from app.memory.manager import schedule_reflection
from app.services import threads as threads_service

logger = logging.getLogger(__name__)

TOOL_RESULT_MAX_CHARS = 2000

# トークンを SSE に流すノード (それ以外のノードの LLM 出力は内部思考として非表出)
USER_FACING_NODES = {"responder", "synthesizer"}


async def stream_agent(
    *,
    agent,
    pool,
    reflection_executor,
    message: str,
    thread_id: str,
    user_id: str,
    reflection_delay_seconds: int,
    settings: Settings,
) -> AsyncIterator[ServerSentEvent]:
    # タイトルはユーザーメッセージだけから決まるため、ストリーム開始前に確定する
    # (途中で stop されてもタイトルが残るように)
    title = await threads_service.touch_and_title(
        pool, thread_id, threads_service.derive_title(message)
    )
    completed = False
    turn_messages: list[Any] = [HumanMessage(content=message)]
    seen_tool_calls: set[str] = set()
    seen_tool_results: set[str] = set()
    try:
        async for part in agent.astream(
            {"messages": [{"role": "user", "content": message}]},
            config={
                "configurable": {"thread_id": thread_id, "langgraph_user_id": user_id},
                "recursion_limit": settings.graph_recursion_limit,
            },
            context=AgentContext(user_id=user_id),
            stream_mode=["messages", "updates", "custom"],
            subgraphs=True,
        ):
            # subgraphs=True → (ns, mode, chunk) / False → (mode, chunk) の防御的両対応
            if isinstance(part, tuple) and len(part) == 3:
                ns, mode, chunk = part
            else:
                mode, chunk = part
                ns = ()
            top = str(ns[0]).split(":")[0] if ns else None

            if mode == "messages":
                msg_chunk, metadata = chunk
                if isinstance(msg_chunk, AIMessageChunk):
                    origin = top or metadata.get("langgraph_node", "model")
                    if origin not in USER_FACING_NODES:
                        continue
                    text = _content_to_text(msg_chunk.content)
                    if text:
                        yield ServerSentEvent(
                            event="token",
                            data={"content": text, "node": origin},
                        )
            elif mode == "updates":
                if not isinstance(chunk, dict):
                    continue
                for node_name, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    # responder の親レベル update はサブグラフ最終状態 (全履歴の再掲) の
                    # ため処理しない。実際のイベントはサブグラフ内 updates から拾う。
                    if ns == () and node_name == "responder":
                        continue
                    # turn_messages (リフレクション用) は親レベルと responder 由来のみ。
                    # executor の中間メッセージはステップ毎のスクラッチパッドなので含めない。
                    accumulate = ns == () or top == "responder"
                    for msg in update.get("messages") or []:
                        if isinstance(msg, AIMessage):
                            if accumulate:
                                turn_messages.append(msg)
                            for tc in msg.tool_calls:
                                key = str(tc.get("id") or f"{node_name}:{msg.id}")
                                if key in seen_tool_calls:
                                    continue
                                seen_tool_calls.add(key)
                                yield ServerSentEvent(
                                    event="tool_call",
                                    data={
                                        "id": tc.get("id"),
                                        "name": tc.get("name"),
                                        "args": tc.get("args"),
                                    },
                                )
                        elif isinstance(msg, ToolMessage):
                            if accumulate:
                                turn_messages.append(msg)
                            key = str(msg.tool_call_id or msg.id)
                            if key in seen_tool_results:
                                continue
                            seen_tool_results.add(key)
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
                if isinstance(chunk, dict):
                    status = chunk.get("status")
                    data = (
                        chunk
                        if isinstance(status, str)
                        else {**chunk, "status": str(status or "")}
                    )
                else:
                    data = {"status": str(chunk)}
                yield ServerSentEvent(event="progress", data=data)
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
