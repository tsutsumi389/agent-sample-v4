import { useCallback, useEffect, useRef, useState } from "react";
import { createThread, getMessages } from "../api/client";
import type { DoneEvent, MessageOut, Thread } from "../api/types";
import { useChatStream } from "../hooks/useChatStream";
import { Composer } from "./Composer";
import { MessageBubble, type UIMessage } from "./MessageBubble";
import type { UIToolCall } from "./ToolCallChip";
import type { UIAction } from "./UIResource";

/** UI操作を人間可読＋機械可読な1メッセージへ整形し、chat/stream へ環流する (案2)。 */
function formatUiAction(a: UIAction): string {
  return `[ui-action] ${a.component}#${a.uiId} ${a.action}: ${JSON.stringify(
    a.payload,
  )}`;
}

interface Props {
  threadId: string | null;
  userId: string;
  /** スレッド未選択で送信した際に新規作成されたスレッドを通知する。 */
  onThreadCreated: (thread: Thread) => void;
  /** done イベント受信時 (タイトル更新・メモリ再取得のトリガ)。 */
  onDone: (e: DoneEvent) => void;
}

/** 履歴 API のメッセージを表示用モデルへ変換。tool メッセージは call_id で突合して chip に畳み込む。 */
function toUIMessages(messages: MessageOut[]): UIMessage[] {
  const out: UIMessage[] = [];
  const callIndex = new Map<string, UIToolCall>();
  for (const m of messages) {
    if (m.role === "tool") {
      const call = m.tool_call_id ? callIndex.get(m.tool_call_id) : undefined;
      if (call) {
        call.result = m.content;
        if (m.ui) call.ui = m.ui;
      } else {
        // 対応する tool_call が無い孤立 tool メッセージ (念のため)
        out.push({
          id: m.id,
          role: "assistant",
          content: "",
          toolCalls: [
            {
              id: m.tool_call_id ?? m.id,
              name: m.name ?? "tool",
              args: {},
              result: m.content,
              ui: m.ui,
            },
          ],
        });
      }
      continue;
    }
    const toolCalls: UIToolCall[] = (m.tool_calls ?? []).map((tc) => ({
      id: tc.id,
      name: tc.name,
      args: tc.args,
    }));
    for (const tc of toolCalls) callIndex.set(tc.id, tc);
    out.push({ id: m.id, role: m.role, content: m.content, toolCalls });
  }
  return out;
}

function updateMessage(
  messages: UIMessage[],
  id: string,
  fn: (m: UIMessage) => UIMessage,
): UIMessage[] {
  return messages.map((m) => (m.id === id ? fn(m) : m));
}

export function ChatView({ threadId, userId, onThreadCreated, onDone }: Props) {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // ストリーム中の UI (カーソル等) を所有スレッドに限定するための状態。
  const [streamingThreadId, setStreamingThreadId] = useState<string | null>(null);
  const { streaming, send, stop } = useChatStream();

  // 送信時に自分で作ったスレッドへの切替では履歴ロードをスキップする
  // (ロードするとローカルで組み立て中のメッセージが消えるため)。
  const skipLoadRef = useRef<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  // ハンドラから現在表示中のスレッドを参照するためのミラー。
  const threadIdRef = useRef(threadId);
  threadIdRef.current = threadId;
  // ストリームを所有するスレッド (effect からの同期参照用)。
  const streamingThreadIdRef = useRef<string | null>(null);
  // createThread 中 (streaming=true になる前) の二重送信を防ぐ同期ガード。
  const sendingRef = useRef(false);

  useEffect(() => {
    // 別スレッドへの切替時は進行中のストリームを中断する
    // (自分で作ったスレッドへの切替では中断しない)。
    if (streamingThreadIdRef.current !== threadId) {
      stop();
    }
    setError(null);
    setProgress(null);
    if (threadId && skipLoadRef.current === threadId) {
      skipLoadRef.current = null;
      return;
    }
    if (!threadId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    setLoadingHistory(true);
    getMessages(threadId, userId)
      .then((res) => {
        if (!cancelled) setMessages(toUIMessages(res.messages));
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setMessages([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingHistory(false);
      });
    return () => {
      cancelled = true;
    };
  }, [threadId, userId, stop]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, progress]);

  const handleSend = useCallback(
    async (text: string) => {
      if (streaming || sendingRef.current) return;
      sendingRef.current = true;
      try {
        setError(null);

        // 契約: 未知の thread_id は 404。送信前にスレッドの存在を保証する。
        let tid = threadId;
        if (!tid) {
          try {
            const thread = await createThread(userId);
            tid = thread.thread_id;
            skipLoadRef.current = tid;
            onThreadCreated(thread);
          } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
            return;
          }
        }

        // ストリームの所有スレッドを記録し、別スレッドの UI への漏れを防ぐ。
        streamingThreadIdRef.current = tid;
        setStreamingThreadId(tid);
        const owns = () => threadIdRef.current === tid;

        const now = Date.now();
        const assistantId = `local-a-${now}`;
        setMessages((prev) => [
          ...prev,
          { id: `local-u-${now}`, role: "user", content: text, toolCalls: [] },
          { id: assistantId, role: "assistant", content: "", toolCalls: [] },
        ]);

        await send(
          { message: text, thread_id: tid, user_id: userId },
          {
            onToken: (e) => {
              if (!owns()) return;
              setMessages((prev) =>
                updateMessage(prev, assistantId, (m) => ({
                  ...m,
                  content: m.content + e.content,
                })),
              );
            },
            onToolCall: (e) => {
              if (!owns()) return;
              setMessages((prev) =>
                updateMessage(prev, assistantId, (m) => ({
                  ...m,
                  toolCalls: [
                    ...m.toolCalls,
                    { id: e.id, name: e.name, args: e.args },
                  ],
                })),
              );
            },
            onToolResult: (e) => {
              if (!owns()) return;
              setMessages((prev) =>
                updateMessage(prev, assistantId, (m) => ({
                  ...m,
                  toolCalls: m.toolCalls.map((tc) =>
                    tc.id === e.id ? { ...tc, result: e.content } : tc,
                  ),
                })),
              );
            },
            onUIResource: (e) => {
              if (!owns()) return;
              // tool_call → ui_resource の順で届くため、対応する toolCall に後付けする。
              // 念のため未着なら新規 toolCall を生成して UI を保持する。
              setMessages((prev) =>
                updateMessage(prev, assistantId, (m) => {
                  const exists = m.toolCalls.some((tc) => tc.id === e.id);
                  return {
                    ...m,
                    toolCalls: exists
                      ? m.toolCalls.map((tc) =>
                          tc.id === e.id ? { ...tc, ui: e } : tc,
                        )
                      : [
                          ...m.toolCalls,
                          { id: e.id, name: e.name ?? e.component, args: {}, ui: e },
                        ],
                  };
                }),
              );
            },
            onProgress: (e) => {
              if (owns()) setProgress(e.status);
            },
            onDone: (e) => {
              if (owns()) setProgress(null);
              onDone(e);
            },
            onError: (e) => {
              if (!owns()) return;
              setProgress(null);
              setError(e.message);
            },
          },
        );
        if (owns()) setProgress(null);
      } finally {
        sendingRef.current = false;
      }
    },
    [threadId, userId, streaming, send, onThreadCreated, onDone],
  );

  // 生成的UIの操作を chat/stream へ環流する。ストリーミング中は送れないため、
  // 受理可否を boolean で返す (UI 側はこれを見て送信済みロックを判断する)。
  const handleUIAction = useCallback(
    (a: UIAction): boolean => {
      if (streaming || sendingRef.current) return false;
      void handleSend(formatUiAction(a));
      return true;
    },
    [streaming, handleSend],
  );

  return (
    <main className="chat-view">
      <div className="message-list">
        {loadingHistory && <div className="chat-note">履歴を読み込み中...</div>}
        {!loadingHistory && messages.length === 0 && (
          <div className="chat-empty">
            <h2>まだ何も憶えていません</h2>
            <p>最初のひとことから、記憶の輪がひとつずつ結ばれていきます。</p>
          </div>
        )}
        {messages.map((m, i) => (
          <MessageBubble
            key={m.id}
            message={m}
            continued={i > 0 && messages[i - 1].role === m.role}
            streaming={
              streaming &&
              threadId === streamingThreadId &&
              i === messages.length - 1 &&
              m.role === "assistant"
            }
            onUIAction={handleUIAction}
          />
        ))}
        {progress && <div className="progress-line">{progress}</div>}
        {error && <div className="chat-error">エラー: {error}</div>}
        <div ref={bottomRef} />
      </div>
      <Composer
        streaming={streaming}
        onSend={(text) => void handleSend(text)}
        onStop={stop}
      />
    </main>
  );
}
