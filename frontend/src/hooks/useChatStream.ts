import { useCallback, useRef, useState } from "react";
import type {
  ChatStreamRequest,
  DoneEvent,
  ErrorEvent,
  ProgressEvent,
  TokenEvent,
  ToolCallEvent,
  ToolResultEvent,
  UIResourceEvent,
} from "../api/types";

export interface StreamHandlers {
  onToken: (e: TokenEvent) => void;
  onToolCall: (e: ToolCallEvent) => void;
  onToolResult: (e: ToolResultEvent) => void;
  onUIResource?: (e: UIResourceEvent) => void;
  onProgress?: (e: ProgressEvent) => void;
  onDone: (e: DoneEvent) => void;
  onError: (e: ErrorEvent) => void;
}

/**
 * POST /api/chat/stream の SSE を fetch + ReadableStream で受信する手書きパーサ。
 * - TextDecoder({stream:true}) で UTF-8 マルチバイトをチャンク境界で安全に結合
 * - "event:" / "data:" 行を蓄積し、空行で 1 イベント確定・ディスパッチ
 * - ":" 始まりの keep-alive コメント行は無視
 * - CRLF ("\r\n") も処理
 * - AbortController による stop()
 *
 * 注意: send は必ずイベントハンドラから呼ぶこと (useEffect から呼ばない)。
 * React 19 StrictMode は effect を二重実行するため、effect 起点のストリームは二重送信になる。
 */
export function useChatStream() {
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const send = useCallback(
    async (body: ChatStreamRequest, handlers: StreamHandlers): Promise<void> => {
      // 進行中のストリームがある間の再入を拒否する (abortRef の上書き・二重送信防止)。
      if (abortRef.current) return;
      const controller = new AbortController();
      abortRef.current = controller;
      setStreaming(true);

      try {
        const res = await fetch("/api/chat/stream", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok) {
          // 契約: 未知の thread_id は SSE 開始前に 404 {"detail": "thread not found"}
          let detail = `HTTP ${res.status}`;
          try {
            const errBody = (await res.json()) as { detail?: string };
            if (typeof errBody.detail === "string") detail = errBody.detail;
          } catch {
            // JSON でないボディは無視
          }
          handlers.onError({ message: detail });
          return;
        }
        if (!res.body) {
          handlers.onError({ message: "レスポンスボディがありません" });
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let eventName = "";
        let dataLines: string[] = [];

        const dispatch = () => {
          const name = eventName;
          const raw = dataLines.join("\n");
          eventName = "";
          dataLines = [];
          if (raw === "") return;
          let data: unknown;
          try {
            data = JSON.parse(raw);
          } catch {
            return; // 不正な data 行は捨てる
          }
          switch (name) {
            case "token":
              handlers.onToken(data as TokenEvent);
              break;
            case "tool_call":
              handlers.onToolCall(data as ToolCallEvent);
              break;
            case "tool_result":
              handlers.onToolResult(data as ToolResultEvent);
              break;
            case "ui_resource":
              handlers.onUIResource?.(data as UIResourceEvent);
              break;
            case "progress":
              handlers.onProgress?.(data as ProgressEvent);
              break;
            case "done":
              handlers.onDone(data as DoneEvent);
              break;
            case "error":
              handlers.onError(data as ErrorEvent);
              break;
            default:
              break; // 未知イベントは無視
          }
        };

        const processLine = (line: string) => {
          if (line === "") {
            dispatch(); // 空行 = イベント区切り
            return;
          }
          if (line.startsWith(":")) {
            return; // keep-alive コメント
          }
          const colon = line.indexOf(":");
          const field = colon === -1 ? line : line.slice(0, colon);
          let value = colon === -1 ? "" : line.slice(colon + 1);
          if (value.startsWith(" ")) value = value.slice(1);
          if (field === "event") {
            eventName = value;
          } else if (field === "data") {
            dataLines.push(value);
          }
          // その他のフィールド (id, retry など) は無視
        };

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let nl: number;
          while ((nl = buffer.indexOf("\n")) !== -1) {
            let line = buffer.slice(0, nl);
            buffer = buffer.slice(nl + 1);
            if (line.endsWith("\r")) line = line.slice(0, -1); // CRLF
            processLine(line);
          }
        }
        // ストリーム終端: デコーダのフラッシュと残バッファの処理
        buffer += decoder.decode();
        if (buffer.length > 0) {
          let line = buffer;
          if (line.endsWith("\r")) line = line.slice(0, -1);
          processLine(line);
        }
        dispatch();
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          // ユーザーによる停止。エラー扱いしない。
        } else {
          handlers.onError({
            message: err instanceof Error ? err.message : String(err),
          });
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [],
  );

  return { streaming, send, stop };
}
